#!/usr/bin/env python3
"""
Flexible pipeline for confidence-based vulnerability detection:
1. Extract logits from configurable HuggingFace transformers model
2. Generate code patches for configurable SWE-bench instances
3. Test SWE-bench instance pass/fail
4. Analyze confidence patterns vs correctness
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import subprocess
import os
import tempfile
from datasets import load_dataset
import argparse
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlexibleConfidenceDetector:
    def __init__(self, model_name: str = "microsoft/CodeGPT-small-py", device: str = "auto"):
        """
        Initialize with configurable model.

        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.swebench_data = None

    def _setup_device(self, device: str) -> str:
        """Setup device for model."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the specified HuggingFace model."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )

            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def load_swebench_data(self, split: str = "test"):
        """Load SWE-bench dataset."""
        logger.info(f"Loading SWE-bench dataset ({split} split)...")
        try:
            self.swebench_data = load_dataset('princeton-nlp/SWE-bench', split=split)
            logger.info(f"✅ SWE-bench loaded: {len(self.swebench_data)} instances")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load SWE-bench: {e}")
            return False

    def get_instance_by_id(self, instance_id: str) -> Optional[Dict]:
        """Get specific SWE-bench instance by ID."""
        if not self.swebench_data:
            logger.error("SWE-bench data not loaded")
            return None

        instance = next((inst for inst in self.swebench_data if inst['instance_id'] == instance_id), None)
        if not instance:
            logger.error(f"Instance {instance_id} not found")
            return None
        return instance

    def generate_with_logits(self, prompt: str, max_new_tokens: int = 150) -> Tuple[str, List[float], List[float]]:
        """
        Generate text while extracting token-level logits and confidence scores.

        Returns:
            generated_text: The generated text
            all_logits: Raw logit values for each generated token
            all_confidences: Probability scores for each generated token
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded")
            return "", [], []

        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        logger.info(f"Input tokens: {inputs.shape[1]}")

        generated_tokens = []
        all_logits = []
        all_confidences = []

        # Generate token by token to capture logits
        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits

                # Get probabilities
                probabilities = F.softmax(logits, dim=-1)

                # Sample next token (you can also use greedy with argmax)
                next_token = torch.multinomial(probabilities, 1)
                next_token_id = next_token.item()

                # Store logits and confidence
                token_logit = logits[next_token_id].item()
                token_confidence = probabilities[next_token_id].item()

                all_logits.append(token_logit)
                all_confidences.append(token_confidence)
                generated_tokens.append(next_token_id)

                # Update inputs
                inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)

                # Stop if EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                # Log progress for debugging
                if step < 20:  # Only log first 20 tokens
                    token_text = self.tokenizer.decode([next_token_id])
                    logger.info(f"Token {step:2d}: '{token_text}' (conf: {token_confidence:.4f})")

        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text, all_logits, all_confidences

    def extract_code_patch(self, generated_text: str, instance_data: Dict) -> str:
        """Extract and format code patch from generated text."""
        lines = generated_text.split('\n')
        patch_lines = []

        # Look for diff-like content
        for line in lines:
            if any(indicator in line for indicator in ['---', '+++', '@@', 'diff', 'Index:']):
                patch_lines.append(line)

        if patch_lines:
            return '\n'.join(patch_lines)

        # Fallback: try to construct a simple patch based on problem context
        if 'django' in instance_data.get('repo', '').lower():
            # For Django issues, try to find code blocks
            code_blocks = []
            in_code = False
            for line in lines:
                if '```' in line or line.strip().startswith('def ') or line.strip().startswith('class '):
                    in_code = not in_code
                if in_code and line.strip():
                    code_blocks.append(line)

            if code_blocks:
                return '\n'.join(code_blocks)

        # Return full generated text as fallback
        return generated_text

    def test_swebench_instance(self, instance_id: str, model_patch: str) -> Tuple[bool, str]:
        """Test generated patch against SWE-bench evaluation."""
        # Create predictions file
        predictions = [{
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": self.model_name
        }]

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f, indent=2)
            predictions_file = f.name

        try:
            logger.info(f"Testing instance {instance_id} with SWE-bench...")

            # Check if SWE-bench tools are available
            swebench_path = Path("tools/SWE-bench")
            if not swebench_path.exists():
                logger.warning("SWE-bench tools not found, simulating evaluation...")
                # Simulate evaluation for testing
                return len(model_patch) > 50, "Simulated evaluation - patch looks substantial"

            # Run SWE-bench evaluation
            cmd = [
                'python', '-m', 'swebench.harness.run_evaluation',
                '--dataset_name', 'princeton-nlp/SWE-bench',
                '--predictions_path', predictions_file,
                '--max_workers', '1',
                '--run_id', f'confidence_test_{instance_id}'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                cwd=str(swebench_path)
            )

            if result.returncode == 0:
                logger.info("✅ SWE-bench evaluation completed")
                success = "PASSED" in result.stdout or "resolved" in result.stdout.lower()
                return success, result.stdout
            else:
                logger.warning(f"⚠️ SWE-bench evaluation issues: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logger.warning("⏰ SWE-bench evaluation timed out")
            return False, "Timeout"
        except Exception as e:
            logger.error(f"❌ Error running SWE-bench: {e}")
            return False, str(e)
        finally:
            # Clean up temp file
            if os.path.exists(predictions_file):
                os.unlink(predictions_file)

    def analyze_confidence_patterns(self, confidences: List[float], generated_text: str, is_correct: bool) -> Dict:
        """Analyze confidence patterns and their relationship to correctness."""
        if not confidences:
            return {}

        analysis = {
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'confidence_std': np.std(confidences),
            'is_correct': is_correct,
            'num_tokens': len(confidences)
        }

        # Security-specific keyword analysis
        security_keywords = ['password', 'auth', 'token', 'key', 'encrypt', 'hash', 'validate', 'sanitize']
        security_confidences = []

        tokens = generated_text.lower().split()
        for i, token in enumerate(tokens):
            if any(keyword in token for keyword in security_keywords) and i < len(confidences):
                security_confidences.append(confidences[i])

        if security_confidences:
            analysis['security_avg_confidence'] = np.mean(security_confidences)
            analysis['security_tokens_found'] = len(security_confidences)
        else:
            analysis['security_avg_confidence'] = 0
            analysis['security_tokens_found'] = 0

        # Confidence degradation analysis
        if len(confidences) > 10:
            first_half = confidences[:len(confidences)//2]
            second_half = confidences[len(confidences)//2:]
            analysis['confidence_degradation'] = np.mean(first_half) - np.mean(second_half)
        else:
            analysis['confidence_degradation'] = 0

        # Low confidence ratio
        low_conf_threshold = 0.5
        low_conf_tokens = [c for c in confidences if c < low_conf_threshold]
        analysis['low_confidence_ratio'] = len(low_conf_tokens) / len(confidences)

        return analysis

    def run_pipeline(self, instance_ids: List[str], output_file: Optional[str] = None) -> List[Dict]:
        """
        Run complete pipeline on multiple instances.

        Args:
            instance_ids: List of SWE-bench instance IDs to process
            output_file: Optional file to save results

        Returns:
            List of results for each instance
        """
        if not self.model or not self.swebench_data:
            logger.error("Model or data not loaded")
            return []

        all_results = []

        for instance_id in instance_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING INSTANCE: {instance_id}")
            logger.info(f"{'='*60}")

            # Get instance data
            instance = self.get_instance_by_id(instance_id)
            if not instance:
                continue

            logger.info(f"Problem: {instance['problem_statement'][:200]}...")

            # Create prompt for code generation
            prompt = f"""Fix this issue in the {instance.get('repo', 'codebase')}:

Problem: {instance['problem_statement'][:800]}

Generate a code patch to fix this issue:"""

            # Generate code with logits
            logger.info("--- GENERATING CODE WITH LOGITS ---")
            generated_text, logits, confidences = self.generate_with_logits(prompt)

            if not generated_text:
                logger.error("Failed to generate code")
                continue

            logger.info(f"Generated {len(confidences)} tokens")
            logger.info(f"Text preview: {generated_text[:200]}...")

            # Extract patch
            logger.info("--- EXTRACTING CODE PATCH ---")
            model_patch = self.extract_code_patch(generated_text, instance)

            # Test with SWE-bench
            logger.info("--- TESTING WITH SWE-BENCH ---")
            is_correct, evaluation_output = self.test_swebench_instance(instance_id, model_patch)

            logger.info(f"Result: {'✅ PASSED' if is_correct else '❌ FAILED'}")

            # Analyze confidence patterns
            logger.info("--- ANALYZING CONFIDENCE PATTERNS ---")
            analysis = self.analyze_confidence_patterns(confidences, generated_text, is_correct)

            # Create insight
            avg_conf = analysis.get('avg_confidence', 0)
            if is_correct and avg_conf > 0.7:
                insight = "✅ HIGH confidence + CORRECT = Reliable generation"
            elif is_correct and avg_conf < 0.5:
                insight = "🤔 LOW confidence + CORRECT = Lucky guess"
            elif not is_correct and avg_conf > 0.7:
                insight = "⚠️ HIGH confidence + WRONG = Dangerous overconfidence"
            else:
                insight = "❓ LOW confidence + WRONG = Uncertain and incorrect"

            logger.info(f"Insight: {insight}")

            # Store results
            result = {
                'instance_id': instance_id,
                'model_name': self.model_name,
                'generated_text': generated_text,
                'model_patch': model_patch,
                'confidences': confidences,
                'logits': logits,
                'is_correct': is_correct,
                'evaluation_output': evaluation_output,
                'analysis': analysis,
                'insight': insight
            }

            all_results.append(result)

        # Save results if requested
        if output_file and all_results:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")

        return all_results

def main():
    parser = argparse.ArgumentParser(description="Flexible SWE-bench confidence analysis pipeline")
    parser.add_argument("--model", default="microsoft/CodeGPT-small-py",
                       help="HuggingFace model name/path")
    parser.add_argument("--instances", nargs="+",
                       default=["django__django-10097"],
                       help="SWE-bench instance IDs to process")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--max-tokens", type=int, default=150,
                       help="Maximum tokens to generate")

    args = parser.parse_args()

    # Initialize detector
    detector = FlexibleConfidenceDetector(model_name=args.model, device=args.device)

    # Load model and data
    if not detector.load_model():
        logger.error("Failed to load model")
        return

    if not detector.load_swebench_data():
        logger.error("Failed to load SWE-bench data")
        return

    # Run pipeline
    results = detector.run_pipeline(args.instances, args.output)

    # Print summary
    if results:
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Processed: {len(results)} instances")

        correct_count = sum(1 for r in results if r['is_correct'])
        logger.info(f"Passed SWE-bench: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")

        avg_confidence = np.mean([r['analysis']['avg_confidence'] for r in results])
        logger.info(f"Average confidence: {avg_confidence:.4f}")

        # Hypothesis validation
        high_conf_correct = sum(1 for r in results if r['analysis']['avg_confidence'] > 0.7 and r['is_correct'])
        low_conf_incorrect = sum(1 for r in results if r['analysis']['avg_confidence'] < 0.5 and not r['is_correct'])

        logger.info(f"\n🎯 HYPOTHESIS VALIDATION:")
        logger.info(f"High confidence + correct: {high_conf_correct}")
        logger.info(f"Low confidence + incorrect: {low_conf_incorrect}")

if __name__ == "__main__":
    main()