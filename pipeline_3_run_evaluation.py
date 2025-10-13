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
from typing import Dict, List

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

    def _discover_runs(self) -> List[int]:
        """Discover all run numbers from predictions_all_run{N}.json files.

        Returns:
            Sorted list of run numbers found, or empty list for legacy single-run structure
        """
        runs = []
        for item in self.run_dir.iterdir():
            if item.is_file() and item.name.startswith("predictions_all_run") and item.name.endswith(".json"):
                try:
                    # Extract run number from "predictions_all_run{N}.json"
                    run_num = int(item.name.replace("predictions_all_run", "").replace(".json", ""))
                    runs.append(run_num)
                except ValueError:
                    logger.warning(f"Invalid predictions file name: {item.name}")
                    continue
        return sorted(runs)

    def _run_evaluation_for_run(self, run_number: int = None) -> Dict:
        """Run SWE-bench evaluation for a specific run.

        Args:
            run_number: Run number (None for legacy single-run structure)

        Returns:
            Dictionary with evaluation results and status
        """
        run_info = f" (run {run_number})" if run_number else ""
        logger.info(f"🚀 Running SWE-bench evaluation{run_info}")

        # Determine predictions path based on run_number
        if run_number is not None:
            predictions_path = self.run_dir / f"predictions_all_run{run_number}.json"
        else:
            predictions_path = self.run_dir / "predictions_all.json"

        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            return {
                "status": "error",
                "error": f"Predictions not found: {predictions_path}",
                "run_number": run_number,
            }

        # Load predictions to check count
        with open(predictions_path, "r") as f:
            predictions = json.load(f)

        logger.info(f"Found {len(predictions)} predictions to evaluate")

        if len(predictions) == 0:
            logger.error("No predictions to evaluate")
            return {
                "status": "error",
                "error": "No predictions to evaluate",
                "run_number": run_number,
            }

        # Create unique run ID for this evaluation
        base_run_id = f"{self.config['model_name'].replace('/', '_')}_{self.config['timestamp']}"
        if run_number is not None:
            run_id = f"{base_run_id}_run{run_number}"
        else:
            run_id = base_run_id

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
                "--namespace",
                "none",
                "--force_rebuild",
                "True",
                "--cache_level",
                "none",
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
                        "status": "error",
                        "error": "Evaluation results file not found",
                        "elapsed_time": elapsed_time,
                        "run_number": run_number,
                    }

            # Load evaluation results
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)

            # Copy individual evaluation results to instance directories
            self._distribute_results(eval_results, run_number=run_number)

            # Move aggregated results to run directory
            if run_number is not None:
                aggregated_eval_path = self.run_dir / f"evaluation_results_run{run_number}.json"
            else:
                aggregated_eval_path = self.run_dir / "evaluation_results.json"
            eval_results_path.rename(aggregated_eval_path)

            result = {
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
                "run_number": run_number,
            }

            logger.info(f"\n{'='*80}")
            if run_number is not None:
                logger.info(f"📊 EVALUATION RESULTS (RUN {run_number})")
            else:
                logger.info(f"📊 EVALUATION RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Total instances: {result['total_instances']}")
            logger.info(f"✅ Resolved: {result['resolved']}")
            logger.info(f"❌ Unresolved: {result['unresolved']}")
            logger.info(f"⚠️ Errors: {result['error_instances']}")
            logger.info(f"⏱️ Time: {elapsed_time:.1f}s")

            return result

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logger.error(f"⏱️ Evaluation timed out after {elapsed_time:.1f}s")
            return {
                "status": "timeout",
                "error": "Evaluation timed out",
                "elapsed_time": elapsed_time,
                "run_number": run_number,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Evaluation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e),
                "elapsed_time": elapsed_time,
                "run_number": run_number,
            }

    def run_evaluation(self, specific_run: int = None) -> Dict:
        """Run SWE-bench evaluation on all predictions.

        This method supports both single-run (legacy) and multi-run structures.
        - Single-run: predictions_all.json exists
        - Multi-run: predictions_all_run{N}.json files exist

        Args:
            specific_run: If specified, only evaluate this run number. If None, evaluate all runs.

        Returns:
            Dictionary with aggregated evaluation results
        """
        logger.info(f"🚀 Running SWE-bench evaluation on {self.run_dir}")

        # Discover runs
        runs = self._discover_runs()

        # If specific run requested, validate and use only that run
        if specific_run is not None:
            if runs and specific_run not in runs:
                logger.error(f"Run {specific_run} not found. Available runs: {runs}")
                return {
                    "stage": "3_run_evaluation",
                    "status": "error",
                    "error": f"Run {specific_run} not found",
                }
            logger.info(f"Evaluating specific run: {specific_run}")
            runs = [specific_run]

        if not runs:
            # Legacy single-run structure
            logger.info("Detected single-run structure")
            result = self._run_evaluation_for_run(run_number=None)

            # Add stage marker for backward compatibility
            result["stage"] = "3_run_evaluation"

            return result
        else:
            # Multi-run structure
            logger.info(f"Detected multi-run structure with {len(runs)} runs")

            all_results = []
            for run_num in runs:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 Processing run {run_num} of {len(runs)}")
                logger.info(f"{'='*80}")

                result = self._run_evaluation_for_run(run_number=run_num)
                all_results.append(result)

            # Calculate aggregated statistics
            total_instances = sum(r.get("total_instances", 0) for r in all_results if r.get("status") == "success")
            total_resolved = sum(r.get("resolved", 0) for r in all_results if r.get("status") == "success")
            total_unresolved = sum(r.get("unresolved", 0) for r in all_results if r.get("status") == "success")
            total_errors = sum(r.get("error_instances", 0) for r in all_results if r.get("status") == "success")
            total_time = sum(r.get("elapsed_time", 0) for r in all_results)

            # Group statistics by run
            per_run_stats = {}
            for result in all_results:
                run_num = result.get("run_number")
                if run_num is not None:
                    per_run_stats[f"run_{run_num}"] = {
                        "status": result.get("status"),
                        "total_instances": result.get("total_instances", 0),
                        "resolved": result.get("resolved", 0),
                        "unresolved": result.get("unresolved", 0),
                        "error_instances": result.get("error_instances", 0),
                        "elapsed_time": result.get("elapsed_time", 0),
                    }

            summary = {
                "stage": "3_run_evaluation",
                "status": "success",
                "num_runs": len(runs),
                "total_instances": total_instances,
                "total_resolved": total_resolved,
                "total_unresolved": total_unresolved,
                "total_errors": total_errors,
                "total_elapsed_time": total_time,
                "per_run_stats": per_run_stats,
                "all_results": all_results,
            }

            # Display summary
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 3 SUMMARY (ALL RUNS)")
            logger.info(f"{'='*80}")
            logger.info(f"Number of runs: {len(runs)}")
            logger.info(f"Total instances evaluated: {total_instances}")
            logger.info(f"✅ Total resolved: {total_resolved}")
            logger.info(f"❌ Total unresolved: {total_unresolved}")
            logger.info(f"⚠️ Total errors: {total_errors}")
            logger.info(f"⏱️ Total time: {total_time:.1f}s")

            # Show per-run breakdown
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 PER-RUN BREAKDOWN")
            logger.info(f"{'='*80}")
            for run_key, stats in per_run_stats.items():
                logger.info(f"\n{run_key}:")
                logger.info(f"  Status: {stats['status']}")
                if stats['status'] == "success":
                    logger.info(f"  ✅ Resolved: {stats['resolved']}")
                    logger.info(f"  ❌ Unresolved: {stats['unresolved']}")
                    logger.info(f"  ⚠️ Errors: {stats['error_instances']}")

            return summary

    def _distribute_results(self, eval_results: Dict, run_number: int = None):
        """Distribute evaluation results to individual instance directories.

        Args:
            eval_results: Evaluation results from SWE-bench harness
            run_number: Run number (None for legacy single-run structure)
        """
        run_info = f" (run {run_number})" if run_number else ""
        logger.info(f"Distributing evaluation results to instance directories{run_info}")

        # SWE-bench results don't have per-instance details in the main JSON
        # We mark each instance as resolved/unresolved based on the lists
        resolved_ids = set(eval_results.get("resolved_ids", []))
        unresolved_ids = set(eval_results.get("unresolved_ids", []))
        error_ids = set(eval_results.get("error_ids", []))

        for instance_dir in self.run_dir.iterdir():
            if not instance_dir.is_dir():
                continue

            instance_id = instance_dir.name

            # Determine evaluation path based on run_number
            if run_number is not None:
                # Multi-run: save to run_N subdirectory
                eval_path = instance_dir / f"run_{run_number}" / "evaluation.json"
                # Create run directory if it doesn't exist
                eval_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Legacy single-run: save directly in instance directory
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
            if run_number is not None:
                eval_result["run_number"] = run_number

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
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Evaluate specific run number only (default: None = evaluate all runs)",
    )
    parser.add_argument(
        "--clean_logs",
        action="store_true",
        help="Delete cached logs before running evaluation (forces fresh evaluation)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Clean logs if requested
    if args.clean_logs:
        config_path = run_dir / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        run_id = f"{config['model_name'].replace('/', '_')}_{config['timestamp']}"
        logs_dir = Path("logs/run_evaluation") / run_id

        if logs_dir.exists():
            logger.info(f"🗑️  Deleting cached logs: {logs_dir}")
            import shutil
            shutil.rmtree(logs_dir)
            logger.info("✅ Cached logs deleted")
        else:
            logger.info(f"No cached logs found at {logs_dir}")

    # Run evaluation
    evaluator = SWEBenchEvaluator(run_dir)
    summary = evaluator.run_evaluation(specific_run=args.run_number)

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