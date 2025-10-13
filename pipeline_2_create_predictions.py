#!/usr/bin/env python3
"""
Stage 2: Create SWE-bench predictions

This script converts generated patches to SWE-bench prediction format.
Outputs: prediction.json for each instance
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from pipeline_minisweagent_config import PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PredictionCreator:
    """Creates SWE-bench prediction files from patches."""

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

    def _discover_runs(self, instance_dir: Path) -> List[int]:
        """Discover all run_N directories for an instance.

        Args:
            instance_dir: Path to instance directory

        Returns:
            Sorted list of run numbers found, or empty list if no run directories
        """
        runs = []
        if not instance_dir.exists():
            return runs

        for item in instance_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    run_num = int(item.name.split("_")[1])
                    runs.append(run_num)
                except (IndexError, ValueError):
                    logger.warning(f"Invalid run directory name: {item.name}")
                    continue
        return sorted(runs)

    def create_prediction_for_instance(self, instance_id: str, run_number: int = None) -> Dict:
        """Create prediction JSON for a single instance.

        Args:
            instance_id: SWE-bench instance identifier
            run_number: Optional run number. If provided, uses run_N subdirectory

        Returns:
            Dictionary with status and result information
        """
        run_info = f" (run {run_number})" if run_number else ""
        logger.info(f"Creating prediction for {instance_id}{run_info}")

        # Determine instance directory based on run_number
        instance_dir = self.run_dir / instance_id
        if run_number is not None:
            instance_dir = instance_dir / f"run_{run_number}"

        patch_path = instance_dir / "patch.diff"
        prediction_path = instance_dir / "prediction.json"
        metadata_path = instance_dir / "metadata.json"

        # Load metadata to check status
        if not metadata_path.exists():
            logger.error(f"Metadata not found for {instance_id}")
            result = {"instance_id": instance_id, "status": "error", "error": "Metadata not found"}
            if run_number is not None:
                result["run_number"] = run_number
            return result

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check if patch generation was successful
        if metadata.get("status") != "completed":
            logger.warning(f"Skipping {instance_id}: status = {metadata.get('status')}")
            result = {
                "instance_id": instance_id,
                "status": "skipped",
                "reason": f"Patch generation {metadata.get('status')}",
            }
            if run_number is not None:
                result["run_number"] = run_number
            return result

        # Check if patch exists
        if not patch_path.exists():
            logger.error(f"Patch not found: {patch_path}")
            result = {
                "instance_id": instance_id,
                "status": "error",
                "error": "Patch file not found",
            }
            if run_number is not None:
                result["run_number"] = run_number
            return result

        # Read patch
        with open(patch_path, "r") as f:
            model_patch = f.read()

        if not model_patch.strip():
            logger.error(f"Empty patch for {instance_id}")
            result = {
                "instance_id": instance_id,
                "status": "error",
                "error": "Empty patch",
            }
            if run_number is not None:
                result["run_number"] = run_number
            return result

        # Create prediction in SWE-bench format
        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": self.config["model_name"],
        }

        # Save prediction
        with open(prediction_path, "w") as f:
            json.dump(prediction, f, indent=2)

        logger.info(f"✅ Prediction saved: {prediction_path}")

        result = {
            "instance_id": instance_id,
            "status": "success",
            "prediction_path": str(prediction_path),
        }
        if run_number is not None:
            result["run_number"] = run_number
        return result

    def create_all_predictions(self) -> Dict:
        """Create predictions for all instances."""
        logger.info(f"🚀 Creating predictions for instances in {self.run_dir}")

        # Find all instance directories
        instance_dirs = [d for d in self.run_dir.iterdir() if d.is_dir()]
        instance_ids = [d.name for d in instance_dirs]

        logger.info(f"Found {len(instance_ids)} instances")

        # Create predictions - handle both single-run and multi-run structures
        results = []
        for instance_id in instance_ids:
            instance_dir = self.run_dir / instance_id
            runs = self._discover_runs(instance_dir)

            if not runs:
                # Legacy single-run structure (no run_N subdirectories)
                logger.info(f"Processing {instance_id} (single-run structure)")
                result = self.create_prediction_for_instance(instance_id, run_number=None)
                results.append(result)
            else:
                # Multi-run structure (run_1, run_2, etc.)
                logger.info(f"Processing {instance_id} with {len(runs)} runs")
                for run_num in runs:
                    result = self.create_prediction_for_instance(instance_id, run_number=run_num)
                    results.append(result)

        # Create aggregated predictions file(s) for SWE-bench harness
        # Group results by run_number
        run_results = {}
        for result in results:
            run_num = result.get("run_number", None)
            if run_num not in run_results:
                run_results[run_num] = []
            run_results[run_num].append(result)

        aggregated_paths = []

        for run_num, run_specific_results in run_results.items():
            successful_predictions = []
            for result in run_specific_results:
                if result["status"] == "success":
                    instance_id = result["instance_id"]
                    if run_num is not None:
                        # Multi-run structure
                        pred_path = self.run_dir / instance_id / f"run_{run_num}" / "prediction.json"
                    else:
                        # Legacy single-run structure
                        pred_path = self.run_dir / instance_id / "prediction.json"

                    with open(pred_path, "r") as f:
                        pred = json.load(f)
                        successful_predictions.append(pred)

            # Save aggregated predictions per run
            if run_num is not None:
                # For multi-run, save predictions_all.json at run_dir level with run identifier
                # This allows evaluation stage to find them easily
                aggregated_path = self.run_dir / f"predictions_all_run{run_num}.json"
            else:
                # For single-run, keep legacy behavior
                aggregated_path = self.run_dir / "predictions_all.json"

            with open(aggregated_path, "w") as f:
                json.dump(successful_predictions, f, indent=2)

            aggregated_paths.append(str(aggregated_path))
            logger.info(f"📄 Saved aggregated predictions: {aggregated_path}")

        # Summary - calculate overall and per-run statistics
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errored = sum(1 for r in results if r["status"] == "error")

        # Group statistics by run
        per_run_stats = {}
        for run_num, run_specific_results in run_results.items():
            run_success = sum(1 for r in run_specific_results if r["status"] == "success")
            run_skipped = sum(1 for r in run_specific_results if r["status"] == "skipped")
            run_errored = sum(1 for r in run_specific_results if r["status"] == "error")

            run_key = f"run_{run_num}" if run_num is not None else "single_run"
            per_run_stats[run_key] = {
                "success": run_success,
                "skipped": run_skipped,
                "errored": run_errored,
                "total": len(run_specific_results),
            }

        num_runs = len([k for k in run_results.keys() if k is not None])
        if num_runs == 0:
            num_runs = 1  # Legacy single-run

        summary = {
            "stage": "2_create_predictions",
            "num_runs": num_runs,
            "total_predictions": len(results),
            "success": success,
            "skipped": skipped,
            "errored": errored,
            "aggregated_predictions_paths": aggregated_paths,
            "per_run_stats": per_run_stats,
            "results": results,
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE 2 SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Number of runs: {num_runs}")
        logger.info(f"Total predictions: {summary['total_predictions']}")
        logger.info(f"✅ Success: {success}")
        logger.info(f"⏭️ Skipped: {skipped}")
        logger.info(f"❌ Errored: {errored}")

        # Show per-run breakdown if multiple runs
        if num_runs > 1:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 PER-RUN BREAKDOWN")
            logger.info(f"{'='*80}")
            for run_key, stats in per_run_stats.items():
                logger.info(f"\n{run_key}:")
                logger.info(f"  ✅ Success: {stats['success']}")
                logger.info(f"  ⏭️ Skipped: {stats['skipped']}")
                logger.info(f"  ❌ Errored: {stats['errored']}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Create SWE-bench predictions from patches"
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

    # Create predictions
    creator = PredictionCreator(run_dir)
    summary = creator.create_all_predictions()

    # Save stage summary
    summary_path = run_dir / "stage2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Stage 2 complete! Summary saved to {summary_path}")

    # Exit with error code if any instances errored
    if summary["errored"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()