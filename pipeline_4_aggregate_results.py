#!/usr/bin/env python3
"""
Stage 4: Aggregate and analyze results

This script aggregates all results from previous stages and creates:
- run_summary.json: Complete summary of the entire run
- results.csv: CSV format for easy analysis
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates results from all pipeline stages."""

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

    def _load_stage_summary(self, stage: str) -> Dict:
        """Load summary from a specific stage."""
        summary_path = self.run_dir / f"stage{stage}_summary.json"
        if not summary_path.exists():
            logger.warning(f"Stage {stage} summary not found: {summary_path}")
            return {}

        with open(summary_path, "r") as f:
            return json.load(f)

    def aggregate_instance_results(self) -> List[Dict]:
        """Aggregate results for all instances."""
        logger.info("Aggregating instance results")

        results = []

        for instance_dir in self.run_dir.iterdir():
            if not instance_dir.is_dir():
                continue

            instance_id = instance_dir.name
            logger.info(f"  Processing {instance_id}")

            # Load all files for this instance
            metadata_path = instance_dir / "metadata.json"
            evaluation_path = instance_dir / "evaluation.json"
            patch_path = instance_dir / "patch.diff"

            instance_result = {
                "instance_id": instance_id,
                "patch_exists": patch_path.exists(),
                "patch_size": patch_path.stat().st_size if patch_path.exists() else 0,
            }

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                instance_result.update({
                    "generation_status": metadata.get("status"),
                    "cost": metadata.get("cost", 0.0),
                    "api_calls": metadata.get("api_calls", 0),
                    "generation_time": metadata.get("elapsed_time", 0.0),
                    "exit_status": metadata.get("exit_status"),
                    "error": metadata.get("error"),
                })
            else:
                instance_result.update({
                    "generation_status": "unknown",
                    "cost": 0.0,
                    "api_calls": 0,
                    "generation_time": 0.0,
                })

            # Load evaluation
            if evaluation_path.exists():
                with open(evaluation_path, "r") as f:
                    evaluation = json.load(f)
                instance_result.update({
                    "evaluation_status": evaluation.get("status"),
                    "resolved": evaluation.get("resolved", False),
                })
            else:
                instance_result.update({
                    "evaluation_status": "not_evaluated",
                    "resolved": False,
                })

            results.append(instance_result)

        return results

    def create_run_summary(self) -> Dict:
        """Create comprehensive run summary."""
        logger.info(f"🚀 Creating run summary for {self.run_dir}")

        # Load stage summaries
        stage1 = self._load_stage_summary("1")
        stage2 = self._load_stage_summary("2")
        stage3 = self._load_stage_summary("3")

        # Aggregate instance results
        instance_results = self.aggregate_instance_results()

        # Calculate aggregate statistics
        total_instances = len(instance_results)
        resolved_count = sum(1 for r in instance_results if r.get("resolved", False))
        unresolved_count = sum(1 for r in instance_results if not r.get("resolved", False) and r.get("evaluation_status") != "not_evaluated")
        not_evaluated = sum(1 for r in instance_results if r.get("evaluation_status") == "not_evaluated")

        total_cost = sum(r.get("cost", 0.0) for r in instance_results)
        total_time = sum(r.get("generation_time", 0.0) for r in instance_results)
        total_api_calls = sum(r.get("api_calls", 0) for r in instance_results)

        resolution_rate = (resolved_count / total_instances * 100) if total_instances > 0 else 0
        avg_cost = total_cost / total_instances if total_instances > 0 else 0
        avg_time = total_time / total_instances if total_instances > 0 else 0

        # Create summary
        summary = {
            "run_info": {
                "model_name": self.config["model_name"],
                "timestamp": self.config["timestamp"],
                "output_dir": str(self.run_output_dir),
                "swebench_dataset": self.config["swebench_dataset"],
                "swebench_split": self.config["swebench_split"],
            },
            "overall_metrics": {
                "total_instances": total_instances,
                "resolved": resolved_count,
                "unresolved": unresolved_count,
                "not_evaluated": not_evaluated,
                "resolution_rate_percent": round(resolution_rate, 2),
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_instance_usd": round(avg_cost, 4),
                "total_time_seconds": round(total_time, 1),
                "avg_time_per_instance_seconds": round(avg_time, 1),
                "total_api_calls": total_api_calls,
            },
            "stage_summaries": {
                "stage1_generate_patches": stage1,
                "stage2_create_predictions": stage2,
                "stage3_run_evaluation": stage3,
            },
            "instance_results": instance_results,
            "resolved_instances": [r["instance_id"] for r in instance_results if r.get("resolved", False)],
            "unresolved_instances": [r["instance_id"] for r in instance_results if not r.get("resolved", False) and r.get("evaluation_status") != "not_evaluated"],
            "failed_generation": [r["instance_id"] for r in instance_results if r.get("generation_status") in ["error", "timeout"]],
        }

        return summary

    def export_to_csv(self, instance_results: List[Dict], csv_path: Path):
        """Export instance results to CSV."""
        logger.info(f"Exporting results to CSV: {csv_path}")

        if not instance_results:
            logger.warning("No results to export")
            return

        # Define CSV columns
        fieldnames = [
            "instance_id",
            "resolved",
            "generation_status",
            "evaluation_status",
            "cost",
            "api_calls",
            "generation_time",
            "patch_size",
            "exit_status",
            "error",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in instance_results:
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)

        logger.info(f"✅ CSV exported: {csv_path}")

    def aggregate(self) -> Dict:
        """Run full aggregation."""
        # Create run summary
        summary = self.create_run_summary()

        # Save run summary
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Run summary saved: {summary_path}")

        # Export to CSV
        csv_path = self.run_dir / "results.csv"
        self.export_to_csv(summary["instance_results"], csv_path)

        # Print summary
        metrics = summary["overall_metrics"]
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 FINAL RUN SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {summary['run_info']['model_name']}")
        logger.info(f"Timestamp: {summary['run_info']['timestamp']}")
        logger.info(f"\nResults:")
        logger.info(f"  Total instances: {metrics['total_instances']}")
        logger.info(f"  ✅ Resolved: {metrics['resolved']} ({metrics['resolution_rate_percent']}%)")
        logger.info(f"  ❌ Unresolved: {metrics['unresolved']}")
        logger.info(f"  ⏭️ Not evaluated: {metrics['not_evaluated']}")
        logger.info(f"\nCosts:")
        logger.info(f"  💰 Total cost: ${metrics['total_cost_usd']:.4f}")
        logger.info(f"  💰 Avg per instance: ${metrics['avg_cost_per_instance_usd']:.4f}")
        logger.info(f"\nPerformance:")
        logger.info(f"  ⏱️ Total time: {metrics['total_time_seconds']:.1f}s")
        logger.info(f"  ⏱️ Avg per instance: {metrics['avg_time_per_instance_seconds']:.1f}s")
        logger.info(f"  📞 Total API calls: {metrics['total_api_calls']}")

        return summary

    @property
    def run_output_dir(self) -> Path:
        """Get run output directory from config."""
        return Path(self.config["output_dir"])


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Aggregate and analyze results"
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

    # Aggregate results
    aggregator = ResultsAggregator(run_dir)
    summary = aggregator.aggregate()

    logger.info(f"\n✅ Stage 4 complete!")
    logger.info(f"📄 Summary: {run_dir / 'run_summary.json'}")
    logger.info(f"📊 CSV: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()