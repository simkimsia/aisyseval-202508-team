#!/usr/bin/env python3
"""
Modal deployment for flexible_pipeline.py with full SWE-bench Docker support.
Supports large models (20B-30B) with A100 80GB GPUs.
"""

import modal
import json
from typing import List, Dict, Optional
from pathlib import Path

# Create Modal app
app = modal.App("swebench-large-models")

# SWE-bench image with ML model support
swebench_image = (
    modal.Image.debian_slim(python_version="3.10")

    # Install system dependencies
    .apt_install([
        "git", "build-essential", "curl", "wget"
    ])

    # Install Python ML stack
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "anthropic>=0.25.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0"
    ])

    # Clone and setup SWE-bench
    .run_commands([
        "git clone https://github.com/princeton-nlp/SWE-bench.git /opt/swebench",
        "cd /opt/swebench && pip install -e ."
    ])

    # Add flexible_pipeline.py to the image
    .add_local_file("flexible_pipeline.py", "/opt/flexible_pipeline.py")
)

# Shared volume for temporary files and results
volume = modal.Volume.from_name("swebench-results", create_if_missing=True)

@app.function(
    image=swebench_image,
    gpu="A100-80GB",  # Single A100 80GB for 20B-30B models
    memory=128*1024,  # 128GB RAM
    timeout=10800,    # 3 hours max per batch
    volumes={"/results": volume},
)
def run_swebench_evaluation(
    model_name: str,
    instance_ids: List[str],
    max_tokens: int = 150,
    device: str = "cuda",
    api_key: Optional[str] = None
) -> Dict:
    """
    Run SWE-bench evaluation with ML model inference.

    Args:
        model_name: HuggingFace model name or Anthropic model
        instance_ids: List of SWE-bench instance IDs
        max_tokens: Maximum tokens to generate
        device: Device to use ('cuda', 'cpu', 'auto')
        api_key: API key for Anthropic models

    Returns:
        Dictionary with evaluation results
    """
    import subprocess
    import sys
    import os
    import time

    print(f"🚀 Starting evaluation on Modal GPU")
    print(f"Model: {model_name}")
    print(f"Instances: {instance_ids}")
    print(f"Max tokens: {max_tokens}")

    # Note: Docker not needed for basic SWE-bench evaluation
    print("📦 Running without Docker (basic evaluation mode)")

    # Verify GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ No GPU detected - using CPU")
            device = "cpu"
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")
        device = "cpu"

    # Add flexible_pipeline to Python path
    sys.path.insert(0, "/opt")

    # Import and run flexible_pipeline
    try:
        from flexible_pipeline import FlexibleConfidenceDetector

        # Initialize detector
        detector = FlexibleConfidenceDetector(
            model_name=model_name,
            device=device,
            api_key=api_key
        )

        # Load model
        print("📥 Loading model...")
        if not detector.load_model():
            raise Exception("Failed to load model")

        # Load SWE-bench data
        print("📚 Loading SWE-bench dataset...")
        if not detector.load_swebench_data():
            raise Exception("Failed to load SWE-bench data")

        # Run pipeline with full Docker support
        print("🔄 Running evaluation pipeline...")
        results = detector.run_pipeline(
            instance_ids=instance_ids,
            output_file=None  # We'll handle output manually
        )

        # Save results to shared volume AND return results directly
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_safe = model_name.replace("/", "_").replace("-", "_")
        output_file = f"/results/results_{model_safe}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to {output_file}")

        # Also store results in the summary for direct return
        summary['full_results'] = results

        # Generate summary
        if results:
            total_instances = len(results)
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            accuracy = correct_count / total_instances * 100

            # Calculate confidence stats for models that provide them
            results_with_confidence = [r for r in results
                                     if r.get('analysis', {}).get('has_confidence_data')]

            summary = {
                'model_name': model_name,
                'total_instances': total_instances,
                'correct_count': correct_count,
                'accuracy_percent': accuracy,
                'instances_processed': instance_ids,
                'has_confidence_data': len(results_with_confidence) > 0,
                'timestamp': timestamp,
                'output_file': output_file
            }

            if results_with_confidence:
                avg_confidence = sum(r['analysis']['avg_confidence']
                                   for r in results_with_confidence) / len(results_with_confidence)
                summary['avg_confidence'] = avg_confidence
                summary['confidence_instances'] = len(results_with_confidence)

            print("\n" + "="*60)
            print("📊 EVALUATION SUMMARY")
            print("="*60)
            print(f"Model: {model_name}")
            print(f"Accuracy: {correct_count}/{total_instances} ({accuracy:.1f}%)")
            if summary['has_confidence_data']:
                print(f"Avg Confidence: {summary.get('avg_confidence', 0):.4f}")
            print(f"Results file: {output_file}")

            return summary
        else:
            raise Exception("No results generated")

    except Exception as e:
        error_summary = {
            'error': str(e),
            'model_name': model_name,
            'instances_attempted': instance_ids,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        print(f"❌ Evaluation failed: {e}")
        return error_summary

@app.function(
    image=swebench_image,
    timeout=300,  # 5 minutes for batch coordination
)
def run_batch_evaluation(
    model_name: str,
    instance_ids: List[str],
    batch_size: int = 3,
    max_tokens: int = 150,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Run evaluation in batches to handle large instance lists.
    """
    print(f"📦 Running batch evaluation: {len(instance_ids)} instances in batches of {batch_size}")

    # Split instances into batches
    batches = [instance_ids[i:i + batch_size]
              for i in range(0, len(instance_ids), batch_size)]

    print(f"Created {len(batches)} batches")

    all_results = []
    for i, batch in enumerate(batches):
        print(f"\n🔄 Processing batch {i+1}/{len(batches)}: {batch}")

        # Run each batch on separate GPU instance
        try:
            result = run_swebench_evaluation.remote(
                model_name=model_name,
                instance_ids=batch,
                max_tokens=max_tokens,
                api_key=api_key
            )
            all_results.append(result)
            print(f"✅ Batch {i+1} completed")
        except Exception as e:
            print(f"❌ Batch {i+1} failed: {e}")
            all_results.append({
                'error': str(e),
                'batch': batch,
                'batch_number': i+1
            })

    return all_results

@app.function(
    image=swebench_image,
    volumes={"/results": volume},
    timeout=60
)
def list_results() -> str:
    """List all files in the results volume"""
    import os
    try:
        files = os.listdir("/results")
        return f"Files in /results: {files}"
    except Exception as e:
        return f"Error listing files: {e}"

@app.function(
    image=swebench_image,
    volumes={"/results": volume},
    timeout=60
)
def download_results(filename: str) -> str:
    """Download results file from Modal volume"""
    import os
    if os.path.exists(f"/results/{filename}"):
        with open(f"/results/{filename}", 'r') as f:
            return f.read()
    else:
        # List available files for debugging
        files = os.listdir("/results") if os.path.exists("/results") else []
        return f"File {filename} not found. Available files: {files}"

# CLI interface
@app.local_entrypoint()
def main(
    model: str = "microsoft/CodeGPT-small-py",
    instances: str = "django__django-10097",
    max_tokens: int = 150,
    batch_size: int = 3,
    api_key: str = None
):
    """
    Run SWE-bench evaluation on Modal with large model support.

    Examples:
        modal run modal_swebench.py --model="gpt2" --instances="django__django-10097"
        modal run modal_swebench.py --model="microsoft/CodeGPT-small-py" --instances="django__django-10097 requests__requests-863"
        modal run modal_swebench.py --model="claude-3-5-haiku-20241022" --instances="django__django-10097" --api-key="your-key"
    """
    import os

    # Parse instance list
    instance_list = instances.strip().split()
    print(f"🎯 Target instances: {instance_list}")

    # Get API key from environment if not provided
    if not api_key and model.startswith("claude"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️ Warning: Anthropic model selected but no API key provided")
            print("Set ANTHROPIC_API_KEY environment variable or use --api-key")

    # Choose single instance or batch mode
    if len(instance_list) <= batch_size:
        print(f"🚀 Running single batch evaluation")
        result = run_swebench_evaluation.remote(
            model_name=model,
            instance_ids=instance_list,
            max_tokens=max_tokens,
            api_key=api_key
        )
        print(f"📋 Final result: {result}")

        # Auto-save results locally if evaluation succeeded
        if 'output_file' in result and result['output_file'] and 'full_results' in result:
            output_filename = result['output_file'].split('/')[-1]  # Extract filename
            print(f"💾 Saving results to local file: {output_filename}")
            try:
                # Create a clean results dict without numpy objects for JSON serialization
                clean_result = {k: v for k, v in result.items() if k != 'full_results'}
                clean_result['results'] = result['full_results']

                with open(output_filename, 'w') as f:
                    json.dump(clean_result, f, indent=2, default=str)
                print(f"✅ Results saved locally as {output_filename}")
            except Exception as e:
                print(f"⚠️ Failed to save results locally: {e}")
                print(f"💡 You can manually download with: modal run modal_swebench.py::download_results --filename='{output_filename}'")

    else:
        print(f"📦 Running batch evaluation ({len(instance_list)} instances)")
        results = run_batch_evaluation.remote(
            model_name=model,
            instance_ids=instance_list,
            batch_size=batch_size,
            max_tokens=max_tokens,
            api_key=api_key
        )
        print(f"📋 Batch results: {results}")

        # Auto-download batch results
        for i, result in enumerate(results):
            if 'output_file' in result and result['output_file']:
                output_filename = result['output_file'].split('/')[-1]
                print(f"📥 Downloading batch {i+1} results: {output_filename}")
                try:
                    content = download_results.remote(output_filename)
                    with open(output_filename, 'w') as f:
                        f.write(content)
                    print(f"✅ Batch {i+1} results saved locally as {output_filename}")
                except Exception as e:
                    print(f"⚠️ Failed to download batch {i+1} results: {e}")

if __name__ == "__main__":
    # For local testing
    print("Use: modal run modal_swebench.py --help")