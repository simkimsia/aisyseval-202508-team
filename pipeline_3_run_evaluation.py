#!/usr/bin/env python3
"""
Stage 3: Run SWE-bench evaluation

This script runs the official SWE-bench evaluation harness.
Outputs: evaluation.json for each instance
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SWEBenchEvaluator:
    """Runs SWE-bench evaluation on predictions."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from run directory."""
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            return json.load(f)

    def run_evaluation(self) -> Dict:
        """Run SWE-bench evaluation on all predictions."""
        logger.info(f"🚀 Running SWE-bench evaluation on {self.run_dir}")

        # Check for aggregated predictions file
        predictions_path = self.run_dir / "predictions_all.json"
        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            logger.error("Run stage 2 first: pipeline_2_create_predictions.py")
            raise FileNotFoundError(f"Predictions not found: {predictions_path}")

        # Load predictions to check count
        with open(predictions_path, "r") as f:
            predictions = json.load(f)

        logger.info(f"Found {len(predictions)} predictions to evaluate")

        if len(predictions) == 0:
            logger.error("No predictions to evaluate")
            return {
                "stage": "3_run_evaluation",
                "status": "error",
                "error": "No predictions to evaluate",
            }

        # Create run ID for this evaluation
        run_id = f"{self.config['model_name'].replace('/', '_')}_{self.config['timestamp']}"

        start_time = time.time()

        try:
            # Build SWE-bench evaluation command
            cmd = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                self.config["swebench_dataset"],
                "--predictions_path",
                str(predictions_path),
                "--max_workers",
                str(self.config.get("max_workers", 1)),
                "--run_id",
                run_id,
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            # Run evaluation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hours max
            )

            elapsed_time = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"❌ SWE-bench evaluation failed")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "stage": "3_run_evaluation",
                    "status": "error",
                    "error": f"Evaluation failed with return code {result.returncode}",
                    "stderr": result.stderr,
                    "elapsed_time": elapsed_time,
                }

            logger.info(f"✅ SWE-bench evaluation completed in {elapsed_time:.1f}s")

            # Find and parse evaluation results
            eval_results_path = Path(f"{run_id}.json")
            if not eval_results_path.exists():
                # Try alternative naming
                eval_results_path = Path(f"{self.config['model_name'].replace('/', '_')}.{run_id}.json")

            if not eval_results_path.exists():
                logger.warning(f"Evaluation results file not found: {eval_results_path}")
                # Try to find it in current directory
                possible_files = list(Path(".").glob(f"*{run_id}*.json"))
                if possible_files:
                    eval_results_path = possible_files[0]
                    logger.info(f"Found results at: {eval_results_path}")
                else:
                    logger.error("Could not find evaluation results file")
                    return {
                        "stage": "3_run_evaluation",
                        "status": "error",
                        "error": "Evaluation results file not found",
                        "elapsed_time": elapsed_time,
                    }

            # Load evaluation results
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)

            # Copy individual evaluation results to instance directories
            self._distribute_results(eval_results)

            # Move aggregated results to run directory
            aggregated_eval_path = self.run_dir / "evaluation_results.json"
            eval_results_path.rename(aggregated_eval_path)

            summary = {
                "stage": "3_run_evaluation",
                "status": "success",
                "elapsed_time": elapsed_time,
                "total_instances": eval_results.get("submitted_instances", 0),
                "resolved": eval_results.get("resolved_instances", 0),
                "unresolved": eval_results.get("unresolved_instances", 0),
                "error_instances": eval_results.get("error_instances", 0),
                "resolved_ids": eval_results.get("resolved_ids", []),
                "unresolved_ids": eval_results.get("unresolved_ids", []),
                "error_ids": eval_results.get("error_ids", []),
                "aggregated_results_path": str(aggregated_eval_path),
            }

            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 3 SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Total instances: {summary['total_instances']}")
            logger.info(f"✅ Resolved: {summary['resolved']}")
            logger.info(f"❌ Unresolved: {summary['unresolved']}")
            logger.info(f"⚠️ Errors: {summary['error_instances']}")
            logger.info(f"⏱️ Time: {elapsed_time:.1f}s")

            return summary

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logger.error(f"⏱️ Evaluation timed out after {elapsed_time:.1f}s")
            return {
                "stage": "3_run_evaluation",
                "status": "timeout",
                "error": "Evaluation timed out",
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Evaluation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "stage": "3_run_evaluation",
                "status": "error",
                "error": str(e),
                "elapsed_time": elapsed_time,
            }

    def _distribute_results(self, eval_results: Dict):
        """Distribute evaluation results to individual instance directories."""
        logger.info("Distributing evaluation results to instance directories")

        # SWE-bench results don't have per-instance details in the main JSON
        # We mark each instance as resolved/unresolved based on the lists
        resolved_ids = set(eval_results.get("resolved_ids", []))
        unresolved_ids = set(eval_results.get("unresolved_ids", []))
        error_ids = set(eval_results.get("error_ids", []))

        for instance_dir in self.run_dir.iterdir():
            if not instance_dir.is_dir():
                continue

            instance_id = instance_dir.name
            eval_path = instance_dir / "evaluation.json"

            # Determine status
            if instance_id in resolved_ids:
                status = "resolved"
            elif instance_id in unresolved_ids:
                status = "unresolved"
            elif instance_id in error_ids:
                status = "error"
            else:
                status = "not_evaluated"

            # Create evaluation result
            eval_result = {
                "instance_id": instance_id,
                "status": status,
                "resolved": instance_id in resolved_ids,
            }

            # Save to instance directory
            with open(eval_path, "w") as f:
                json.dump(eval_result, f, indent=2)

            logger.info(f"  {instance_id}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Run SWE-bench evaluation"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to run directory (e.g., output/claude-sonnet-4-20250514/20250930_0928)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Run evaluation
    evaluator = SWEBenchEvaluator(run_dir)
    summary = evaluator.run_evaluation()

    # Save stage summary
    summary_path = run_dir / "stage3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Stage 3 complete! Summary saved to {summary_path}")

    # Exit with error code if evaluation failed
    if summary.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()