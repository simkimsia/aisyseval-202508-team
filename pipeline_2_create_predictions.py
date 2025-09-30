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

    def create_prediction_for_instance(self, instance_id: str) -> Dict:
        """Create prediction JSON for a single instance."""
        logger.info(f"Creating prediction for {instance_id}")

        instance_dir = self.run_dir / instance_id
        patch_path = instance_dir / "patch.diff"
        prediction_path = instance_dir / "prediction.json"
        metadata_path = instance_dir / "metadata.json"

        # Load metadata to check status
        if not metadata_path.exists():
            logger.error(f"Metadata not found for {instance_id}")
            return {"instance_id": instance_id, "status": "error", "error": "Metadata not found"}

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check if patch generation was successful
        if metadata.get("status") != "completed":
            logger.warning(f"Skipping {instance_id}: status = {metadata.get('status')}")
            return {
                "instance_id": instance_id,
                "status": "skipped",
                "reason": f"Patch generation {metadata.get('status')}",
            }

        # Check if patch exists
        if not patch_path.exists():
            logger.error(f"Patch not found: {patch_path}")
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": "Patch file not found",
            }

        # Read patch
        with open(patch_path, "r") as f:
            model_patch = f.read()

        if not model_patch.strip():
            logger.error(f"Empty patch for {instance_id}")
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": "Empty patch",
            }

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

        return {
            "instance_id": instance_id,
            "status": "success",
            "prediction_path": str(prediction_path),
        }

    def create_all_predictions(self) -> Dict:
        """Create predictions for all instances."""
        logger.info(f"🚀 Creating predictions for instances in {self.run_dir}")

        # Find all instance directories
        instance_dirs = [d for d in self.run_dir.iterdir() if d.is_dir()]
        instance_ids = [d.name for d in instance_dirs]

        logger.info(f"Found {len(instance_ids)} instances")

        # Create predictions
        results = []
        for instance_id in instance_ids:
            result = self.create_prediction_for_instance(instance_id)
            results.append(result)

        # Create aggregated predictions file for SWE-bench harness
        successful_predictions = []
        for result in results:
            if result["status"] == "success":
                instance_id = result["instance_id"]
                pred_path = self.run_dir / instance_id / "prediction.json"
                with open(pred_path, "r") as f:
                    pred = json.load(f)
                    successful_predictions.append(pred)

        # Save aggregated predictions
        aggregated_path = self.run_dir / "predictions_all.json"
        with open(aggregated_path, "w") as f:
            json.dump(successful_predictions, f, indent=2)

        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errored = sum(1 for r in results if r["status"] == "error")

        summary = {
            "stage": "2_create_predictions",
            "total_instances": len(results),
            "success": success,
            "skipped": skipped,
            "errored": errored,
            "aggregated_predictions_path": str(aggregated_path),
            "results": results,
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE 2 SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total instances: {summary['total_instances']}")
        logger.info(f"✅ Success: {success}")
        logger.info(f"⏭️ Skipped: {skipped}")
        logger.info(f"❌ Errored: {errored}")
        logger.info(f"📄 Aggregated predictions: {aggregated_path}")

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