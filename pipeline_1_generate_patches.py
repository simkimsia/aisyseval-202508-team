#!/usr/bin/env python3
"""
Stage 1: Generate patches using mini-swe-agent

This script runs mini-swe-agent on each SWE-bench instance to generate:
- patch.diff: The code fix
- trajectory.json: Full agent execution trace
- metadata.json: Cost, time, and execution info
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pipeline_minisweagent_config import PipelineConfig, create_config_from_args

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PatchGenerator:
    """Generates patches using mini-swe-agent."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = []

    def generate_patch_for_instance(self, instance_id: str) -> Dict:
        """Generate patch for a single instance using mini-swe-agent."""
        logger.info(f"{'='*80}")
        logger.info(f"Generating patch for {instance_id}")
        logger.info(f"{'='*80}")

        # Use run-specific paths if current_run is set
        output_paths = self.config.get_output_paths(instance_id, self.config.current_run)
        start_time = time.time()

        metadata = {
            "instance_id": instance_id,
            "model_name": self.config.model_name,
            "status": "unknown",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cost": 0.0,
            "api_calls": 0,
            "elapsed_time": 0.0,
            "error": None,
        }

        try:
            # Build mini-swe-agent command
            # Note: mini-extra swebench-single uses --subset for lite/verified, not dataset name
            subset = "lite" if "Lite" in self.config.swebench_dataset else "verified"
            cmd = [
                "mini-extra",
                "swebench-single",
                "--subset", subset,
                "--split", self.config.swebench_split,
                "--model", self.config.model_name,
                "-i", instance_id,
                "-o", str(output_paths["trajectory"]),
            ]

            # Add optional parameters
            if self.config.temperature is not None:
                cmd.extend(["--temperature", str(self.config.temperature)])

            if self.config.custom_config_path:
                cmd.extend(["--config", str(self.config.custom_config_path)])

            if self.config.exit_immediately:
                cmd.append("--exit-immediately")

            logger.info(f"Running command: {' '.join(cmd)}")

            # Run mini-swe-agent with activity-based timeout
            # This monitors for INACTIVITY rather than total runtime
            # Allows long-running processes that are actively working
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            stdout_lines = []
            stderr_lines = []
            last_activity = time.time()
            inactivity_timeout = self.config.timeout  # Timeout for INACTIVITY (not total time)

            logger.info(f"⏱️ Monitoring process with {inactivity_timeout}s inactivity timeout...")

            try:
                # Monitor process with non-blocking reads
                import select
                import sys as sys_module

                # Check if select.select is available (Unix-like systems)
                use_select = hasattr(select, 'select')

                while process.poll() is None:  # While process is running
                    if use_select:
                        # Unix-like: use select for non-blocking I/O
                        readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.5)

                        for stream in readable:
                            line = stream.readline()
                            if line:
                                if stream == process.stdout:
                                    stdout_lines.append(line)
                                else:
                                    stderr_lines.append(line)
                                last_activity = time.time()
                                # Log progress indicator
                                if len(stdout_lines) % 10 == 0:
                                    logger.info(f"  📝 Activity detected (total lines: {len(stdout_lines)})")
                    else:
                        # Windows or systems without select: simpler approach
                        # Try to read from stdout (may block briefly)
                        import threading
                        import queue

                        def enqueue_output(stream, q, stream_type):
                            for line in iter(stream.readline, ''):
                                q.put((stream_type, line))
                            stream.close()

                        # Only set up threads once (check if not already done)
                        if not hasattr(process, '_monitor_threads_started'):
                            out_queue = queue.Queue()
                            stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, out_queue, 'stdout'))
                            stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, out_queue, 'stderr'))
                            stdout_thread.daemon = True
                            stderr_thread.daemon = True
                            stdout_thread.start()
                            stderr_thread.start()
                            process._monitor_threads_started = True
                            process._out_queue = out_queue

                        # Try to get output without blocking
                        try:
                            stream_type, line = process._out_queue.get_nowait()
                            if stream_type == 'stdout':
                                stdout_lines.append(line)
                            else:
                                stderr_lines.append(line)
                            last_activity = time.time()
                            if len(stdout_lines) % 10 == 0:
                                logger.info(f"  📝 Activity detected (total lines: {len(stdout_lines)})")
                        except:
                            pass

                        time.sleep(0.5)

                    # Check for inactivity timeout
                    inactive_duration = time.time() - last_activity
                    if inactive_duration > inactivity_timeout:
                        logger.error(f"⏱️ No activity for {inactive_duration:.1f}s (threshold: {inactivity_timeout}s)")
                        process.kill()
                        process.wait(timeout=5)
                        raise subprocess.TimeoutExpired(cmd, inactivity_timeout)

                # Process finished, get remaining output
                remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                if remaining_stdout:
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)

                returncode = process.returncode

            except subprocess.TimeoutExpired:
                # Clean up
                process.kill()
                try:
                    process.wait(timeout=5)
                except:
                    pass
                raise

            # Combine output
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            elapsed_time = time.time() - start_time
            metadata["elapsed_time"] = elapsed_time

            logger.info(f"✅ Process completed in {elapsed_time:.1f}s")

            if returncode == 0:
                logger.info(f"✅ Mini-swe-agent completed successfully for {instance_id}")
                metadata["status"] = "completed"

                # Extract patch from trajectory
                if output_paths["trajectory"].exists():
                    self._extract_patch_from_trajectory(
                        output_paths["trajectory"],
                        output_paths["patch"],
                        metadata
                    )
                else:
                    logger.error(f"Trajectory file not found: {output_paths['trajectory']}")
                    metadata["status"] = "error"
                    metadata["error"] = "Trajectory file not found"
            else:
                logger.error(f"❌ Mini-swe-agent failed for {instance_id}")
                logger.error(f"Return code: {returncode}")
                logger.error(f"STDERR: {stderr[:500]}")  # Show first 500 chars of stderr
                metadata["status"] = "error"
                metadata["error"] = f"Process failed with return code {returncode}"

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            metadata["elapsed_time"] = elapsed_time
            metadata["status"] = "timeout"
            metadata["error"] = f"Timeout after {self.config.timeout} seconds"
            logger.error(f"⏱️ Timeout for {instance_id}")

        except Exception as e:
            elapsed_time = time.time() - start_time
            metadata["elapsed_time"] = elapsed_time
            metadata["status"] = "error"
            metadata["error"] = str(e)
            logger.error(f"❌ Error for {instance_id}: {e}")

        # Save metadata
        with open(output_paths["metadata"], "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"📊 {instance_id}: {metadata['status']} in {metadata['elapsed_time']:.1f}s")

        return metadata

    def _extract_patch_from_trajectory(
        self, trajectory_path: Path, patch_path: Path, metadata: Dict
    ):
        """Extract patch from trajectory JSON and update metadata."""
        try:
            with open(trajectory_path, "r") as f:
                trajectory = json.load(f)

            # Extract submission (patch) from trajectory
            submission = trajectory.get("info", {}).get("submission", "")

            if not submission:
                logger.warning(f"No submission found in trajectory")
                metadata["status"] = "error"
                metadata["error"] = "No submission in trajectory"
                return

            # Extract just the main code change (first diff block)
            # Remove test/reproduction scripts from patch
            main_patch = submission.split("diff --git a/reproduce_")[0]
            main_patch = main_patch.split("diff --git a/test_")[0]

            # Trim leading whitespace, but preserve trailing whitespace and blank lines
            diff_start = main_patch.find("diff --git")
            if diff_start != -1:
                main_patch = main_patch[diff_start:]

            # Save patch
            with open(patch_path, "w") as f:
                f.write(main_patch)
                if not main_patch.endswith("\n"):
                    f.write("\n")

            # Extract metadata from trajectory
            info = trajectory.get("info", {})
            model_stats = info.get("model_stats", {})

            metadata["cost"] = model_stats.get("instance_cost", 0.0)
            metadata["api_calls"] = model_stats.get("api_calls", 0)
            metadata["exit_status"] = info.get("exit_status", "unknown")

            logger.info(f"✅ Patch extracted: {patch_path}")
            logger.info(f"💰 Cost: ${metadata['cost']:.4f}, API calls: {metadata['api_calls']}")

        except Exception as e:
            logger.error(f"Failed to extract patch: {e}")
            metadata["status"] = "error"
            metadata["error"] = f"Failed to extract patch: {e}"

    def generate_all_patches(self) -> Dict:
        """Generate patches for all instances."""
        logger.info(f"🚀 Starting patch generation for {len(self.config.instance_ids)} instances")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Output: {self.config.run_output_dir}")
        logger.info(f"Running {self.config.num_runs} run(s) per instance")

        # Ensure run-level output directory exists before writing config files
        self.config.create_output_dirs()

        # Save configuration
        config_path = self.config.run_output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Generate patches - loop over runs first, then instances
        results = []
        for run_num in range(1, self.config.num_runs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"=== RUN {run_num} of {self.config.num_runs} ===")
            logger.info(f"{'='*80}\n")

            # Set current run in config
            self.config.current_run = run_num

            # Create output directories for this run
            self.config.create_output_dirs(run_number=run_num)

            for instance_id in self.config.instance_ids:
                logger.info(f"Run {run_num}/{self.config.num_runs} - Instance: {instance_id}")
                result = self.generate_patch_for_instance(instance_id)
                result["run_number"] = run_num  # Add run number to result
                results.append(result)

        # Summary
        completed = sum(1 for r in results if r["status"] == "completed")
        errored = sum(1 for r in results if r["status"] == "error")
        timeout = sum(1 for r in results if r["status"] == "timeout")
        total_cost = sum(r.get("cost", 0.0) for r in results)
        total_time = sum(r.get("elapsed_time", 0.0) for r in results)

        # Group results by run number
        per_run_results = {}
        for run_num in range(1, self.config.num_runs + 1):
            run_results = [r for r in results if r.get("run_number") == run_num]
            per_run_results[f"run_{run_num}"] = {
                "completed": sum(1 for r in run_results if r["status"] == "completed"),
                "errored": sum(1 for r in run_results if r["status"] == "error"),
                "timeout": sum(1 for r in run_results if r["status"] == "timeout"),
                "cost": sum(r.get("cost", 0.0) for r in run_results),
                "time": sum(r.get("elapsed_time", 0.0) for r in run_results),
                "results": run_results,
            }

        summary = {
            "stage": "1_generate_patches",
            "num_runs": self.config.num_runs,
            "total_instances": len(self.config.instance_ids),
            "completed": completed,
            "errored": errored,
            "timeout": timeout,
            "total_cost": total_cost,
            "total_time": total_time,
            "avg_time_per_instance": total_time / len(results) if results else 0,
            "per_run_results": per_run_results,
            "results": results,
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STAGE 1 SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Number of runs: {self.config.num_runs}")
        logger.info(f"Total instances: {summary['total_instances']}")
        logger.info(f"✅ Completed: {completed}")
        logger.info(f"❌ Errored: {errored}")
        logger.info(f"⏱️ Timeout: {timeout}")
        logger.info(f"💰 Total cost: ${total_cost:.4f}")
        logger.info(f"⏱️ Total time: {total_time:.1f}s")
        logger.info(f"⏱️ Avg time: {summary['avg_time_per_instance']:.1f}s")

        # Show per-run summary if multiple runs
        if self.config.num_runs > 1:
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 PER-RUN BREAKDOWN")
            logger.info(f"{'='*80}")
            for run_num in range(1, self.config.num_runs + 1):
                run_key = f"run_{run_num}"
                run_data = per_run_results[run_key]
                logger.info(f"\nRun {run_num}:")
                logger.info(f"  ✅ Completed: {run_data['completed']}")
                logger.info(f"  ❌ Errored: {run_data['errored']}")
                logger.info(f"  ⏱️ Timeout: {run_data['timeout']}")
                logger.info(f"  💰 Cost: ${run_data['cost']:.4f}")
                logger.info(f"  ⏱️ Time: {run_data['time']:.1f}s")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate patches using mini-swe-agent"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["django__django-10914"],
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
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per instance (default: 1)",
    )

    args = parser.parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Check for API key
    if not config.get_api_key():
        required_key = config.get_required_api_key_name()
        logger.error(f"❌ {required_key} not set")
        logger.error(f"Required for model: {config.model_name}")
        logger.error("Set it in .env file or environment:")
        logger.error(f"  export {required_key}=your-api-key-here")
        sys.exit(1)

    # Generate patches
    generator = PatchGenerator(config)
    summary = generator.generate_all_patches()

    # Save stage summary
    summary_path = config.run_output_dir / "stage1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Stage 1 complete! Summary saved to {summary_path}")

    # Continue pipeline even if some instances failed/timed out
    # This allows subsequent stages to process successful instances
    # if summary["errored"] > 0 or summary["timeout"] > 0:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
