The run_summary.json file has 5 main sections:

  1. run_info (lines 228-235)

  Basic metadata about the pipeline run:
  - model_name: Which AI model was used (e.g., "claude-sonnet-4-20250514")
  - timestamp: When the run started
  - output_dir: Where results are stored
  - swebench_dataset: Which dataset was used
  - swebench_split: Dataset split (e.g., "test")
  - num_runs_configured: How many times each instance was run

  2. overall_metrics (lines 236-248)

  Aggregate statistics across all instances and runs:
  - total_instances: Total number of instance-run records processed
  - resolved: How many issues were successfully fixed
  - unresolved: How many failed to fix
  - not_evaluated: How many weren't evaluated
  - resolution_rate_percent: Success rate percentage
  - total_cost_usd: Total API costs for all runs
  - avg_cost_per_instance_usd: Average cost per instance
  - total_time_seconds: Total generation time
  - avg_time_per_instance_seconds: Average time per instance
  - total_api_calls: Total number of API calls made
  - not_security_scanned: Number of instances that failed security scanning

  3. stage_summaries (lines 249-254)

  Summaries from each pipeline stage (stages 1-4), containing stage-specific metadata

  4. multi_run_info (lines 255-263)

  Statistics for multiple-run consistency checking:
  - num_runs: Actual number of runs performed
  - average_resolution_rate_percent: Average resolution rate across runs
  - total_cost_all_runs_usd: Total cost for all runs
  - consistency_metrics: Detailed consistency analysis from Stage 5
  - consistency_grade_mode: Most common consistency grade
  - consistency_grade_distribution: Distribution of consistency grades
  - avg_overall_consistency_score: Average consistency score

  5. instance_results (line 264)

  Detailed array of results for each instance-run combination, including:
  - instance_id, run_number
  - resolved: Whether the issue was fixed
  - generation_status, evaluation_status
  - cost, api_calls, generation_time
  - patch_size, exit_status, error
  - security_risk_score, security_risk_level, security_scan_status

  6. Instance Lists (lines 265-268)

  Pre-filtered lists for quick reference:
  - resolved_instances: Instance IDs that were successfully fixed
  - unresolved_instances: Instance IDs that failed
  - failed_generation: Instances where patch generation failed
  - failed_security_scan: Instances where security scanning failed

  Purpose

  The run_summary.json file serves as a single source of truth for the entire pipeline execution, making it easy to:
  - Track success rates and costs
  - Analyze which instances succeeded/failed
  - Compare consistency across multiple runs
  - Debug issues with specific instances
  - Generate reports and visualizations