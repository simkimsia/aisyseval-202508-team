#!/usr/bin/env python3
"""
Modal deployment for flexible_pipeline.py with full SWE-bench Docker support.
Supports large models (20B-30B) with A100 80GB GPUs.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import modal

# Create Modal app
app = modal.App("swebench-large-models")

# SWE-bench image with ML model support
swebench_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install system dependencies
    .apt_install(["git", "build-essential", "curl", "wget"])
    # Install Python ML stack
    .pip_install(
        [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.12.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.39.0",
            "anthropic>=0.25.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "swebench>=2.0.0",
            "docker>=6.0.0",
            "scikit-learn>=1.0.0",
            "safetensors>=0.3.0",
            "mistral-common>=1.0.0",
            "huggingface-hub>=0.16.0",
        ]
    )
    # Clone and setup SWE-bench
    .run_commands(
        [
            "git clone https://github.com/princeton-nlp/SWE-bench.git /opt/swebench",
            "cd /opt/swebench && pip install -e .",
        ]
    )
    # Add flexible_pipeline.py to the image
    .add_local_file("flexible_pipeline.py", "/opt/flexible_pipeline.py")
)

# Shared volume for temporary files and results
volume = modal.Volume.from_name("swebench-results", create_if_missing=True)


@app.function(
    image=swebench_image,
    gpu="A100-80GB",  # Single A100 80GB for 20B-30B models
    memory=128 * 1024,  # 128GB RAM
    timeout=10800,  # 3 hours max per batch
    volumes={"/results": volume},
)
def run_swebench_evaluation(
    model_name: str,
    instance_ids: List[str],
    max_tokens: int = 1000000,
    device: str = "cuda",
    api_key: Optional[str] = None,
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

    print("🚀 Starting evaluation on Modal GPU")
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
            model_name=model_name, device=device, api_key=api_key
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
            output_file=None,  # We'll handle output manually
            max_tokens=max_tokens,
        )

        # Save results to shared volume AND return results directly
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_safe = model_name.replace("/", "_").replace("-", "_")
        output_file = f"/results/results_{model_safe}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Results saved to {output_file}")

        # Generate summary
        if results:
            total_instances = len(results)
            correct_count = sum(1 for r in results if r.get("is_correct", False))
            accuracy = correct_count / total_instances * 100

            # Calculate confidence stats for models that provide them
            results_with_confidence = [
                r for r in results if r.get("analysis", {}).get("has_confidence_data")
            ]

            summary = {
                "model_name": model_name,
                "total_instances": total_instances,
                "correct_count": correct_count,
                "accuracy_percent": accuracy,
                "instances_processed": instance_ids,
                "has_confidence_data": len(results_with_confidence) > 0,
                "timestamp": timestamp,
                "output_file": output_file,
            }

            if results_with_confidence:
                avg_confidence = sum(
                    r["analysis"]["avg_confidence"] for r in results_with_confidence
                ) / len(results_with_confidence)
                summary["avg_confidence"] = avg_confidence
                summary["confidence_instances"] = len(results_with_confidence)

            print("\n" + "=" * 60)
            print("📊 EVALUATION SUMMARY")
            print("=" * 60)
            print(f"Model: {model_name}")
            print(f"Accuracy: {correct_count}/{total_instances} ({accuracy:.1f}%)")
            if summary["has_confidence_data"]:
                print(f"Avg Confidence: {summary.get('avg_confidence', 0):.4f}")
            print(f"Results file: {output_file}")

            # Add full results to summary for direct return
            summary["full_results"] = results

            return summary
        else:
            raise Exception("No results generated")

    except Exception as e:
        error_summary = {
            "error": str(e),
            "model_name": model_name,
            "instances_attempted": instance_ids,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
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
    max_tokens: int = 1000000,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Run evaluation in batches to handle large instance lists.
    """
    print(
        f"📦 Running batch evaluation: {len(instance_ids)} instances in batches of {batch_size}"
    )

    # Split instances into batches
    batches = [
        instance_ids[i : i + batch_size]
        for i in range(0, len(instance_ids), batch_size)
    ]

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
                api_key=api_key,
            )
            all_results.append(result)
            print(f"✅ Batch {i+1} completed")
        except Exception as e:
            print(f"❌ Batch {i+1} failed: {e}")
            all_results.append({"error": str(e), "batch": batch, "batch_number": i + 1})

    return all_results


@app.function(image=swebench_image, volumes={"/results": volume}, timeout=60)
def list_results() -> str:
    """List all files in the results volume"""
    try:
        files = os.listdir("/results")
        return f"Files in /results: {files}"
    except Exception as e:
        return f"Error listing files: {e}"


@app.function(image=swebench_image, volumes={"/results": volume}, timeout=60)
def download_results(filename: str) -> str:
    """Download results file from Modal volume"""
    if os.path.exists(f"/results/{filename}"):
        with open(f"/results/{filename}", "r") as f:
            return f.read()
    else:
        # List available files for debugging
        files = os.listdir("/results") if os.path.exists("/results") else []
        return f"File {filename} not found. Available files: {files}"


@app.function(
    image=swebench_image,
    volumes={"/results": volume},
    timeout=3600,  # 1 hour for SWE-bench evaluation
    memory=32 * 1024,  # 32GB for evaluation
)
def run_swebench_official_evaluation(
    predictions_data: Dict, run_id: str = "modal_eval"
) -> Dict:
    """
    Run official SWE-bench evaluation using their Modal integration.

    Args:
        predictions_data: The prediction results from generation
        run_id: Unique identifier for this evaluation run

    Returns:
        Combined results with both generation and evaluation data
    """
    import tempfile

    print("🔍 Starting official SWE-bench evaluation...")
    print(f"Run ID: {run_id}")

    try:
        # Prepare predictions in SWE-bench format
        swebench_predictions = []
        for result in predictions_data.get("results", []):
            swebench_pred = {
                "instance_id": result["instance_id"],
                "model_patch": result["model_patch"],
                "model_name_or_path": result.get(
                    "model_name_or_path",
                    result.get(
                        "model_name", predictions_data.get("model_name", "unknown")
                    ),
                ),
            }
            swebench_predictions.append(swebench_pred)

        # Write predictions to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(swebench_predictions, f, indent=2)
            predictions_file = f.name

        print(f"📝 Created predictions file: {predictions_file}")
        print(f"📊 Evaluating {len(swebench_predictions)} instances")

        # Add SWE-bench to Python path
        sys.path.insert(0, "/opt/swebench")

        # Run SWE-bench evaluation (already on Modal, don't use --modal flag)
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "princeton-nlp/SWE-bench",
            "--predictions_path",
            predictions_file,
            "--run_id",
            run_id,
            "--max_workers",
            "1",
        ]

        print(f"🚀 Running SWE-bench command: {' '.join(cmd)}")

        # Change to SWE-bench directory
        os.chdir("/opt/swebench")

        # Run evaluation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000,  # 50 minutes
            cwd="/opt/swebench",
        )

        print(
            f"📋 SWE-bench evaluation completed with return code: {result.returncode}"
        )

        if result.returncode == 0:
            print("✅ SWE-bench evaluation successful!")
            print("📄 STDOUT:", result.stdout[-1000:])  # Last 1000 chars

            # Try to find and parse results
            log_dir = f"/opt/swebench/logs/{run_id}"
            report_file = f"{log_dir}/report.json"

            evaluation_results = {}
            if os.path.exists(report_file):
                try:
                    with open(report_file, "r") as f:
                        evaluation_results = json.load(f)
                    print(
                        f"📊 Loaded evaluation report: {len(evaluation_results)} instances"
                    )
                except Exception as e:
                    print(f"⚠️ Failed to load report: {e}")
            else:
                print(f"⚠️ Report file not found at: {report_file}")
                # List available files for debugging
                if os.path.exists(log_dir):
                    files = os.listdir(log_dir)
                    print(f"📁 Available files in {log_dir}: {files}")

            # Combine generation and evaluation results
            combined_results = {
                "generation_data": predictions_data,
                "evaluation_results": evaluation_results,
                "swebench_stdout": result.stdout,
                "swebench_return_code": result.returncode,
                "evaluation_timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "run_id": run_id,
            }

            # Add pass/fail info to each instance
            for gen_result in combined_results["generation_data"].get("results", []):
                instance_id = gen_result["instance_id"]
                if instance_id in evaluation_results:
                    gen_result["swebench_resolved"] = evaluation_results[
                        instance_id
                    ].get("resolved", False)
                    gen_result["swebench_details"] = evaluation_results[instance_id]
                else:
                    gen_result["swebench_resolved"] = False
                    gen_result["swebench_details"] = {
                        "error": "No evaluation result found"
                    }

            # Save combined results to volume
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            combined_file = f"/results/combined_results_{run_id}_{timestamp}.json"

            with open(combined_file, "w") as f:
                json.dump(combined_results, f, indent=2, default=str)

            print(f"💾 Combined results saved to {combined_file}")

            return combined_results

        else:
            print("❌ SWE-bench evaluation failed!")
            print(f"📄 STDERR: {result.stderr}")
            print(f"📄 STDOUT: {result.stdout}")

            return {
                "error": "SWE-bench evaluation failed",
                "return_code": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "generation_data": predictions_data,
            }

    except subprocess.TimeoutExpired:
        print("⏰ SWE-bench evaluation timed out")
        return {
            "error": "SWE-bench evaluation timed out",
            "generation_data": predictions_data,
        }
    except Exception as e:
        print(f"❌ Error during SWE-bench evaluation: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e), "generation_data": predictions_data}
    finally:
        # Clean up temp file
        if "predictions_file" in locals() and os.path.exists(predictions_file):
            os.unlink(predictions_file)


# CLI interface
@app.local_entrypoint()
def main(
    model: str = "microsoft/CodeGPT-small-py",
    instances: str = "django__django-10097",
    max_tokens: int = 1000000,
    batch_size: int = 3,
    api_key: str = None,
    run_swebench_eval: bool = False,
):
    """
    Run SWE-bench evaluation on Modal with large model support.

    Examples:
        # Step 1: Generate predictions only
        modal run modal_swebench.py --model="gpt2" --instances="django__django-10097"

        # Step 2: Run evaluation separately (recommended)
        modal run modal_swebench.py::run_swebench_official_evaluation --predictions-file="results_gpt2_*.json" --run-id="eval_test"

        # Combined (may have conflicts)
        modal run modal_swebench.py --model="openai/gpt-oss-20b" --instances="django__django-10097" --run-swebench-eval=true
    """
    import json

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
        print("🚀 Running generation only (no SWE-bench evaluation)")
        result = run_swebench_evaluation.remote(
            model_name=model,
            instance_ids=instance_list,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        print(f"📋 Generation result: {result}")

        # Skip SWE-bench evaluation - user will run separately
        if False and run_swebench_eval and "full_results" in result:
            print("\n🔍 Running official SWE-bench evaluation...")
            run_id = f"eval_{model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"

            # Prepare data for evaluation (convert numpy arrays to lists)

            # Deep copy and convert numpy arrays to regular Python objects
            def clean_numpy_data(obj):
                """Recursively convert numpy arrays to lists for JSON serialization"""
                import numpy as np

                if hasattr(obj, "tolist"):
                    return obj.tolist()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: clean_numpy_data(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [clean_numpy_data(item) for item in obj]
                else:
                    return obj

            clean_results = []
            for res in result["full_results"]:
                clean_result = clean_numpy_data(res)
                clean_results.append(clean_result)

            eval_data = {"model_name": model, "results": clean_results}

            # Run official evaluation
            eval_result = run_swebench_official_evaluation.remote(
                predictions_data=eval_data, run_id=run_id
            )

            print("🎯 SWE-bench evaluation completed!")

            # Save combined results locally
            if "error" not in eval_result:
                combined_filename = f"combined_results_{model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                print(f"💾 Saving combined results to: {combined_filename}")
                try:
                    with open(combined_filename, "w") as f:
                        json.dump(eval_result, f, indent=2, default=str)
                    print(f"✅ Combined results saved as {combined_filename}")

                    # Print summary
                    if (
                        "generation_data" in eval_result
                        and "results" in eval_result["generation_data"]
                    ):
                        results_list = eval_result["generation_data"]["results"]
                        total = len(results_list)
                        resolved = sum(
                            1 for r in results_list if r.get("swebench_resolved", False)
                        )
                        print("\n📊 FINAL SUMMARY:")
                        print(f"Model: {model}")
                        print(
                            f"SWE-bench Resolution: {resolved}/{total} ({resolved/total*100:.1f}%)"
                        )

                        for r in results_list:
                            status = (
                                "✅ RESOLVED"
                                if r.get("swebench_resolved")
                                else "❌ FAILED"
                            )
                            print(f"  {r['instance_id']}: {status}")

                except Exception as e:
                    print(f"⚠️ Failed to save combined results: {e}")
            else:
                print(
                    f"❌ SWE-bench evaluation failed: {eval_result.get('error', 'Unknown error')}"
                )
                # Still save generation results
                result = eval_result.get("generation_data", result)

        # CRITICAL: Auto-save generation results locally - this is the only way to get results!
        if "full_results" in result:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_safe = model.replace("/", "_").replace("-", "_")
            local_filename = f"results_{model_safe}_{timestamp}.json"

            print(f"💾 CRITICAL: Saving results locally as {local_filename}")
            try:
                # Create a clean results dict without numpy objects for JSON serialization
                def clean_numpy_data(obj):
                    """Recursively convert numpy arrays to lists for JSON serialization"""
                    import numpy as np

                    if hasattr(obj, "tolist"):
                        return obj.tolist()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, dict):
                        return {k: clean_numpy_data(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [clean_numpy_data(item) for item in obj]
                    else:
                        return obj

                # Clean and prepare results
                clean_result = clean_numpy_data(
                    {
                        "model_name": model,
                        "total_instances": result.get(
                            "total_instances", len(instance_list)
                        ),
                        "correct_count": result.get("correct_count", 0),
                        "accuracy_percent": result.get("accuracy_percent", 0),
                        "instances_processed": instance_list,
                        "timestamp": timestamp,
                        "results": result[
                            "full_results"
                        ],  # This contains logits, confidences, etc.
                    }
                )

                with open(local_filename, "w") as f:
                    json.dump(clean_result, f, indent=2, default=str)
                print(f"✅ SUCCESS: Results with logits saved as {local_filename}")
                print(
                    f"📊 File contains {len(clean_result['results'])} instances with logits and confidence data"
                )
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Failed to save results locally: {e}")
                import traceback

                traceback.print_exc()
                # Try to save at least the raw result
                try:
                    with open(f"emergency_results_{timestamp}.json", "w") as f:
                        json.dump(str(result), f)
                    print(
                        f"🆘 Emergency backup saved as emergency_results_{timestamp}.json"
                    )
                except Exception as e2:
                    print(f"💀 Complete failure to save: {e2}")
        else:
            print("⚠️ No results to save - check for errors above")

    else:
        print(f"📦 Running batch evaluation ({len(instance_list)} instances)")
        results = run_batch_evaluation.remote(
            model_name=model,
            instance_ids=instance_list,
            batch_size=batch_size,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        print(f"📋 Batch results: {results}")

        # Auto-download batch results
        for i, result in enumerate(results):
            if "output_file" in result and result["output_file"]:
                output_filename = result["output_file"].split("/")[-1]
                print(f"📥 Downloading batch {i+1} results: {output_filename}")
                try:
                    content = download_results.remote(output_filename)
                    with open(output_filename, "w") as f:
                        f.write(content)
                    print(f"✅ Batch {i+1} results saved locally as {output_filename}")
                except Exception as e:
                    print(f"⚠️ Failed to download batch {i+1} results: {e}")


@app.local_entrypoint()
def evaluate_only(predictions_file: str, run_id: str = "modal_eval"):
    """
    Run only the official SWE-bench evaluation on existing predictions.

    Args:
        predictions_file: Path to JSON file with generation results
        run_id: Unique identifier for this evaluation run
    """
    import json

    print(f"🔍 Loading predictions from: {predictions_file}")

    # Load the predictions file
    try:
        with open(predictions_file, "r") as f:
            predictions_data = json.load(f)

        print(f"📊 Loaded {len(predictions_data.get('results', []))} predictions")

        # Run official evaluation
        result = run_swebench_official_evaluation.remote(
            predictions_data=predictions_data, run_id=run_id
        )

        print("✅ Evaluation completed!")
        print(f"📋 Result: {result}")

        # Save results locally if successful
        if "error" not in result:
            output_filename = (
                f"evaluation_results_{run_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_filename, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Evaluation results saved as: {output_filename}")

    except FileNotFoundError:
        print(f"❌ Predictions file not found: {predictions_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # For local testing
    print("Use: modal run modal_swebench.py --help")
    print("Or: modal run modal_swebench.py::evaluate_only --help")
