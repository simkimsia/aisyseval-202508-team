#!/usr/bin/env python3
"""
Master pipeline runner for SWE-bench evaluation.

This script runs all pipeline stages sequentially:
1. Generate patches using mini-swe-agent
2. Create SWE-bench predictions
3. Run evaluation
4. Run security scans
5. Calculate consistency
6. Aggregate results

Usage:
    python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 django__django-11001
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from pipeline_minisweagent_config import create_config_from_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_stage(script: str, args: list) -> bool:
    """Run a pipeline stage script."""
    cmd = [sys.executable, script] + args
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"{'='*80}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run complete SWE-bench evaluation pipeline"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="Instance IDs to process",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64000,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default=["all"],
        help="Which stages to run (default: all)",
    )

    args = parser.parse_args()

    # Determine which stages to run
    if "all" in args.stages:
        stages_to_run = ["1", "2", "3", "4", "5", "6"]
    else:
        stages_to_run = sorted(set(args.stages))

    logger.info(f"🚀 Starting pipeline for {len(args.instances)} instances")
    logger.info(f"Model: {args.model}")
    logger.info(f"Instances: {', '.join(args.instances)}")
    logger.info(f"Stages to run: {', '.join(stages_to_run)}")

    # Create config to determine run directory
    config = create_config_from_args(args)
    run_dir = config.run_output_dir

    # Stage 1: Generate patches
    if "1" in stages_to_run:
        success = run_stage(
            "pipeline_1_generate_patches.py",
            [
                "--model", args.model,
                "--instances"] + args.instances + [
                "--dataset", args.dataset,
                "--split", args.split,
                "--output-dir", args.output_dir,
                "--max-tokens", str(args.max_tokens),
            ]
        )

        if not success:
            logger.error("❌ Stage 1 failed. Stopping pipeline.")
            sys.exit(1)

    # Stage 2: Create predictions
    if "2" in stages_to_run:
        success = run_stage(
            "pipeline_2_create_predictions.py",
            [str(run_dir)]
        )

        if not success:
            logger.error("❌ Stage 2 failed. Stopping pipeline.")
            sys.exit(1)

    # Stage 3: Run evaluation
    if "3" in stages_to_run:
        success = run_stage(
            "pipeline_3_run_evaluation.py",
            [str(run_dir)]
        )

        if not success:
            logger.error("❌ Stage 3 failed. Stopping pipeline.")
            sys.exit(1)

    # Stage 4: Security scan
    if "4" in stages_to_run:
        success = run_stage(
            "pipeline_4_security_scan.py",
            [str(run_dir)]
        )

        if not success:
            logger.error("❌ Stage 4 failed. Stopping pipeline.")
            sys.exit(1)

    # Stage 5: Calculate consistency
    if "5" in stages_to_run:
        success = run_stage(
            "pipeline_5_consistency_check.py",
            [str(run_dir)]
        )

        if not success:
            logger.error("❌ Stage 5 failed. Stopping pipeline.")
            sys.exit(1)

    # Stage 6: Aggregate results
    if "6" in stages_to_run:
        success = run_stage(
            "pipeline_6_aggregate_results.py",
            [str(run_dir)]
        )

        if not success:
            logger.error("❌ Stage 6 failed. Stopping pipeline.")
            sys.exit(1)

    logger.info(f"\n{'='*80}")
    logger.info(f"🎉 PIPELINE COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Results directory: {run_dir}")
    logger.info(f"Summary: {run_dir / 'run_summary.json'}")
    logger.info(f"CSV: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()