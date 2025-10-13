#!/usr/bin/env python3
"""
Stage 5: Consistency Check

This script analyzes consistency across multiple runs of the same instance.
It compares patches, evaluation results, and security scores to measure reproducibility.

Only runs when num_runs > 1.
"""

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# Import shared similarity utilities for better patch comparison
from consistency_evaluator.similarity_utils import hybrid_similarity, consistency_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Analyzes consistency across multiple runs of SWE-bench instances."""

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
        """Discover all run_N directories for an instance."""
        runs = []
        if not instance_dir.exists():
            return runs

        for item in instance_dir.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    run_num = int(item.name.split("_")[1])
                    runs.append(run_num)
                except (IndexError, ValueError):
                    continue
        return sorted(runs)

    def _load_run_data(self, instance_id: str, run_num: int) -> Dict:
        """
        Load all data for a specific run.

        Args:
            instance_id: SWE-bench instance identifier
            run_num: Run number

        Returns:
            Dictionary containing patch, metadata, evaluation, and security data
        """
        run_dir = self.run_dir / instance_id / f"run_{run_num}"

        data = {
            "run_number": run_num,
            "patch_content": None,
            "metadata": None,
            "evaluation": None,
            "security_risk_score": None,
        }

        # Load patch.diff
        patch_path = run_dir / "patch.diff"
        if patch_path.exists():
            with open(patch_path, "r") as f:
                data["patch_content"] = f.read()

        # Load metadata.json
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data["metadata"] = json.load(f)

        # Load evaluation.json
        evaluation_path = run_dir / "evaluation.json"
        if evaluation_path.exists():
            with open(evaluation_path, "r") as f:
                data["evaluation"] = json.load(f)

        # Load security_risk_score.json
        security_path = run_dir / "security_risk_score.json"
        if security_path.exists():
            with open(security_path, "r") as f:
                data["security_risk_score"] = json.load(f)

        return data

    def calculate_patch_consistency(self, runs_data: List[Dict], threshold: float = 0.85) -> Dict:
        """
        Calculate comprehensive patch consistency metrics using combined approach.

        This combines:
        - Pipeline-specific metrics (exact match, line variance, unique count)
        - consistency_metrics() (agreement%, confidence%, normalized confidence%)
        - Detailed similarity breakdowns (AST, text, hybrid)

        Args:
            runs_data: List of run data dictionaries
            threshold: Similarity threshold for agreement calculation (default 0.85)

        Returns:
            Dictionary with comprehensive metrics organized in three tiers:

            Tier 1 - Summary:
                - confidence_percent: Main metric (0-100)
                - exact_match_rate: Perfect byte-identical matches (0.0-1.0)
                - num_unique_patches: Count of distinct patches

            Tier 2 - Detailed:
                - agreement_percent: % above threshold (0-100)
                - normalized_confidence_percent: Rescaled confidence (0-100)
                - avg_ast_similarity: Structural similarity (0.0-1.0)
                - avg_text_similarity: Surface similarity (0.0-1.0)
                - avg_hybrid_similarity: Weighted combo (0.0-1.0)
                - line_count_variance: Size variation

            Tier 3 - Raw Data:
                - pairwise_comparisons: Full comparison details
        """
        patches = [run["patch_content"] for run in runs_data if run["patch_content"]]

        # Handle edge case: < 2 patches
        if len(patches) < 2:
            return {
                # Tier 1: Summary
                "confidence_percent": 100.0,
                "exact_match_rate": 1.0,
                "num_unique_patches": len(patches),
                # Tier 2: Detailed
                "agreement_percent": 100.0,
                "normalized_confidence_percent": 100.0,
                "avg_ast_similarity": 1.0,
                "avg_text_similarity": 1.0,
                "avg_hybrid_similarity": 1.0,
                "line_count_variance": 0.0,
                # Tier 3: Raw data
                "pairwise_comparisons": []
            }

        # Calculate exact match rate (Pipeline-specific)
        patch_counter = Counter(patches)
        most_common_count = patch_counter.most_common(1)[0][1] if patch_counter else 0
        exact_match_rate = most_common_count / len(patches) if patches else 0.0

        # Calculate line count variance (Pipeline-specific)
        line_counts = [len(p.splitlines()) for p in patches]
        line_count_variance = statistics.variance(line_counts) if len(line_counts) > 1 else 0.0

        # Use consistency_metrics() for comprehensive analysis
        agreement_percent, confidence_percent, normalized_confidence, pairwise = consistency_metrics(
            outputs=patches,
            language="python",
            threshold=threshold
        )

        # Calculate detailed similarity averages from pairwise data
        avg_ast_similarity = statistics.mean([p["ast_similarity"] for p in pairwise]) if pairwise else 1.0
        avg_text_similarity = statistics.mean([p["text_similarity"] for p in pairwise]) if pairwise else 1.0
        avg_hybrid_similarity = statistics.mean([p["hybrid_similarity"] for p in pairwise]) if pairwise else 1.0

        # Return combined metrics in three-tier structure
        return {
            # === Tier 1: Summary Metrics ===
            "confidence_percent": round(confidence_percent, 2),
            "exact_match_rate": round(exact_match_rate, 4),
            "num_unique_patches": len(patch_counter),

            # === Tier 2: Detailed Metrics ===
            "agreement_percent": round(agreement_percent, 2),
            "normalized_confidence_percent": round(normalized_confidence, 2),
            "avg_ast_similarity": round(avg_ast_similarity, 4),
            "avg_text_similarity": round(avg_text_similarity, 4),
            "avg_hybrid_similarity": round(avg_hybrid_similarity, 4),
            "line_count_variance": round(line_count_variance, 2),

            # === Tier 3: Raw Data ===
            "pairwise_comparisons": pairwise
        }

    def calculate_evaluation_consistency(self, runs_data: List[Dict]) -> Dict:
        """
        Calculate evaluation consistency metrics.

        Args:
            runs_data: List of run data dictionaries

        Returns:
            Dictionary with evaluation consistency metrics
        """
        evaluations = [run["evaluation"] for run in runs_data if run["evaluation"]]

        if not evaluations:
            return {
                "all_resolved": False,
                "all_failed": False,
                "resolution_rate": 0.0,
                "num_evaluations": 0,
            }

        # Extract resolution status from each evaluation
        resolutions = []
        for eval_data in evaluations:
            # Handle different evaluation formats
            resolved = False
            if isinstance(eval_data, dict):
                # Try different keys that might indicate resolution
                resolved = (
                    eval_data.get("resolved", False) or
                    eval_data.get("passed", False) or
                    eval_data.get("status") == "success"
                )
            resolutions.append(resolved)

        resolution_rate = sum(resolutions) / len(resolutions) if resolutions else 0.0
        all_resolved = all(resolutions)
        all_failed = not any(resolutions)

        return {
            "all_resolved": all_resolved,
            "all_failed": all_failed,
            "resolution_rate": resolution_rate,
            "num_evaluations": len(evaluations),
        }

    def calculate_security_consistency(self, runs_data: List[Dict]) -> Dict:
        """
        Calculate security consistency metrics.

        Args:
            runs_data: List of run data dictionaries

        Returns:
            Dictionary with security consistency metrics
        """
        security_scores = [
            run["security_risk_score"]
            for run in runs_data
            if run["security_risk_score"] and run["security_risk_score"].get("status") == "success"
        ]

        if not security_scores:
            return {
                "risk_level_mode": "UNKNOWN",
                "score_variance": 0.0,
                "num_scans": 0,
            }

        # Extract risk levels and scores
        risk_levels = []
        scores = []

        for sec_data in security_scores:
            result = sec_data.get("security_risk_score_result", {})
            if result:
                risk_levels.append(result.get("risk_level", "UNKNOWN"))
                scores.append(result.get("security_risk_score", 0.0))

        # Find most common risk level
        if risk_levels:
            risk_level_counter = Counter(risk_levels)
            risk_level_mode = risk_level_counter.most_common(1)[0][0]
        else:
            risk_level_mode = "UNKNOWN"

        # Calculate score variance
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0

        return {
            "risk_level_mode": risk_level_mode,
            "score_variance": score_variance,
            "num_scans": len(security_scores),
            "avg_score": statistics.mean(scores) if scores else 0.0,
        }

    def calculate_overall_consistency_score(
        self,
        patch_metrics: Dict,
        eval_metrics: Dict,
        security_metrics: Dict
    ) -> Dict:
        """
        Calculate overall consistency score with weighted metrics.

        Weights:
        - Patch consistency: 40%
        - Evaluation consistency: 40%
        - Security consistency: 20%

        Args:
            patch_metrics: Patch consistency metrics
            eval_metrics: Evaluation consistency metrics
            security_metrics: Security consistency metrics

        Returns:
            Dictionary with overall score and grade
        """
        # Calculate patch consistency score (0-100)
        # Use confidence_percent (already 0-100) weighted with exact_match_rate
        patch_score = (
            patch_metrics["exact_match_rate"] * 50 * 100 +  # Convert 0-1 to 0-100
            patch_metrics["confidence_percent"] * 0.5        # Already 0-100
        )

        # Calculate evaluation consistency score (0-100)
        # High score if all resolved or all failed (consistent), low if mixed
        if eval_metrics["all_resolved"] or eval_metrics["all_failed"]:
            eval_score = 100.0
        else:
            # Penalize for inconsistency
            # Resolution rate of 0.5 (50/50 split) is worst case = 0 score
            # Resolution rate of 0.0 or 1.0 is best case = 100 score
            deviation_from_consensus = abs(eval_metrics["resolution_rate"] - 0.5)
            eval_score = deviation_from_consensus * 200  # Scale to 0-100

        # Calculate security consistency score (0-100)
        # Lower variance is better
        if security_metrics["num_scans"] > 1:
            # Normalize variance to 0-100 scale (lower variance = higher score)
            # Assume variance > 1.0 is very high inconsistency
            max_variance = 1.0
            normalized_variance = min(security_metrics["score_variance"], max_variance) / max_variance
            security_score = (1 - normalized_variance) * 100
        else:
            security_score = 100.0  # Perfect if only one scan

        # Weighted average
        overall_score = (
            patch_score * 0.4 +
            eval_score * 0.4 +
            security_score * 0.2
        )

        # Determine grade
        if overall_score >= 90:
            grade = "EXCELLENT"
        elif overall_score >= 70:
            grade = "GOOD"
        elif overall_score >= 50:
            grade = "FAIR"
        else:
            grade = "POOR"

        return {
            "overall_consistency_score": round(overall_score, 2),
            "consistency_grade": grade,
            "component_scores": {
                "patch_score": round(patch_score, 2),
                "evaluation_score": round(eval_score, 2),
                "security_score": round(security_score, 2),
            }
        }

    def check_instance_consistency(self, instance_id: str) -> Optional[Dict]:
        """
        Check consistency for a single instance across all runs.

        Args:
            instance_id: SWE-bench instance identifier

        Returns:
            Consistency report dictionary or None if insufficient data
        """
        instance_dir = self.run_dir / instance_id
        runs = self._discover_runs(instance_dir)

        if len(runs) < 2:
            logger.warning(f"Instance {instance_id} has fewer than 2 runs, skipping consistency check")
            return None

        logger.info(f"Checking consistency for {instance_id} across {len(runs)} runs")

        # Load data for all runs
        runs_data = []
        for run_num in runs:
            try:
                data = self._load_run_data(instance_id, run_num)
                runs_data.append(data)
            except Exception as e:
                logger.error(f"Error loading data for {instance_id} run {run_num}: {e}")
                continue

        if len(runs_data) < 2:
            logger.warning(f"Insufficient valid data for {instance_id}")
            return None

        # Calculate consistency metrics
        patch_metrics = self.calculate_patch_consistency(runs_data)
        eval_metrics = self.calculate_evaluation_consistency(runs_data)
        security_metrics = self.calculate_security_consistency(runs_data)
        overall_metrics = self.calculate_overall_consistency_score(
            patch_metrics, eval_metrics, security_metrics
        )

        # Create report
        report = {
            "instance_id": instance_id,
            "num_runs": len(runs),
            "patch_consistency": patch_metrics,
            "evaluation_consistency": eval_metrics,
            "security_consistency": security_metrics,
            **overall_metrics,
        }

        return report

    def check_all_instances(self) -> Dict:
        """
        Check consistency for all instances in the run directory.

        Returns:
            Summary dictionary with all instance reports
        """
        instance_ids = self.config.get("instance_ids", [])
        num_runs = self.config.get("num_runs", 1)

        # Skip if only single run
        if num_runs == 1:
            logger.warning("Only 1 run configured, skipping consistency check")
            return {
                "stage": "5_consistency_check",
                "status": "skipped",
                "reason": "num_runs == 1, consistency check requires multiple runs",
                "num_instances": len(instance_ids),
            }

        logger.info(f"Checking consistency for {len(instance_ids)} instance(s)")

        instance_reports = []
        skipped = 0
        checked = 0

        for instance_id in instance_ids:
            try:
                report = self.check_instance_consistency(instance_id)
                if report:
                    # Save per-instance report
                    report_path = self.run_dir / instance_id / "consistency_report.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(report_path, "w") as f:
                        json.dump(report, f, indent=2)

                    instance_reports.append(report)
                    checked += 1
                    logger.info(f"  {instance_id}: {report['consistency_grade']} ({report['overall_consistency_score']:.1f})")
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Error checking consistency for {instance_id}: {e}")
                skipped += 1

        # Calculate aggregated metrics
        if instance_reports:
            avg_overall_score = statistics.mean([r["overall_consistency_score"] for r in instance_reports])
            avg_patch_score = statistics.mean([r["component_scores"]["patch_score"] for r in instance_reports])
            avg_eval_score = statistics.mean([r["component_scores"]["evaluation_score"] for r in instance_reports])
            avg_security_score = statistics.mean([r["component_scores"]["security_score"] for r in instance_reports])

            # Grade distribution
            grades = [r["consistency_grade"] for r in instance_reports]
            grade_distribution = dict(Counter(grades))
        else:
            avg_overall_score = 0.0
            avg_patch_score = 0.0
            avg_eval_score = 0.0
            avg_security_score = 0.0
            grade_distribution = {}

        summary = {
            "stage": "5_consistency_check",
            "status": "completed",
            "num_instances": len(instance_ids),
            "instances_checked": checked,
            "instances_skipped": skipped,
            "aggregated_metrics": {
                "avg_overall_consistency_score": round(avg_overall_score, 2),
                "avg_patch_score": round(avg_patch_score, 2),
                "avg_evaluation_score": round(avg_eval_score, 2),
                "avg_security_score": round(avg_security_score, 2),
                "grade_distribution": grade_distribution,
            },
            "instance_reports": instance_reports,
        }

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Consistency Check - Analyze consistency across multiple runs"
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

    # Run consistency checker
    checker = ConsistencyChecker(run_dir)
    summary = checker.check_all_instances()

    # Save summary
    summary_path = run_dir / "consistency_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nStage 5 complete! Summary saved to {summary_path}")

    if summary["status"] == "skipped":
        logger.info(f"Reason: {summary['reason']}")
    else:
        logger.info(f"Instances checked: {summary['instances_checked']}/{summary['num_instances']}")
        logger.info(f"Average consistency score: {summary['aggregated_metrics']['avg_overall_consistency_score']:.1f}")
        logger.info(f"Grade distribution: {summary['aggregated_metrics']['grade_distribution']}")


if __name__ == "__main__":
    main()
