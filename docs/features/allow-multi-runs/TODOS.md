# Implementation TODOs: Multiple Runs Per SWE-bench Instance

## Overview

This document tracks the step-by-step implementation of the multiple runs feature. Each step is designed to maintain a compile-safe codebase at all times.

**Important**: After completing each step, verify that:

1. All Python files compile: `python -m py_compile pipeline_*.py run_pipeline.py`
2. The help command works: `python run_pipeline.py --help`
3. Existing single-run behavior still works

---

## Step 1: Update Configuration Model ✅

**Status**: COMPLETED

**File**: `pipeline_minisweagent_config.py`

**Tasks**:

- [x] Add `num_runs: int = 1` field to `PipelineConfig` dataclass
- [x] Add `current_run: Optional[int] = None` field to track active run
- [x] Add `instance_run_dir(instance_id: str, run_number: int) -> Path` method
- [x] Update `instance_output_dir(instance_id: str, run_number: Optional[int] = None)` to accept optional run_number
- [x] Update `get_output_paths(instance_id: str, run_number: Optional[int] = None)` to accept optional run_number
- [x] Update `create_output_dirs(run_number: Optional[int] = None)` to create run subdirectories when run_number specified
- [x] Update `create_config_from_args()` to handle `--num-runs` argument

**Verification**:

```bash
python -c "from pipeline_minisweagent_config import PipelineConfig; c = PipelineConfig(num_runs=3); print(c.num_runs)"
# Should print: 3
```

**Notes**:

- Use `Optional[int]` to maintain backward compatibility
- Default behavior (run_number=None) should work like current code
- When run_number is provided, paths should include `/run_{N}/` subdirectory

---

## Step 2: Update Pipeline Runner ✅

**Status**: COMPLETED

**File**: `run_pipeline.py`

**Tasks**:

- [x] Add `--num-runs` argument to argument parser (default=1, type=int, help="Number of runs per instance")
- [x] Pass `num_runs` to `create_config_from_args()` in main()
- [x] Add logging after config creation: "Running {num_runs} run(s) per instance"
- [x] Update docstring to mention --num-runs parameter

**Verification**:

```bash
python run_pipeline.py --help | grep "num-runs"
# Should show: --num-runs NUM_RUNS  Number of runs per instance (default: 1)
```

**Notes**:

- Don't change stage execution logic yet
- Just pass the parameter through to config
- Stage 1 will handle the actual iteration

---

## Step 3: Update Stage 1 (Generate Patches) ✅

**Status**: COMPLETED

**File**: `pipeline_1_generate_patches.py`

**Tasks**:

- [x] Add `--num-runs` argument to argument parser (default=1)
- [x] Pass `num_runs` to config in `create_config_from_args()`
- [x] Update `PatchGenerator.generate_all_patches()` to loop over runs:

  ```python
  for run_num in range(1, self.config.num_runs + 1):
      logger.info(f"=== RUN {run_num} of {self.config.num_runs} ===")
      for instance_id in self.config.instance_ids:
          # Set current run in config
          self.config.current_run = run_num
          # Generate patch (will use run-specific paths)
          result = self.generate_patch_for_instance(instance_id)
  ```

- [x] Update `generate_patch_for_instance()` to use `self.config.get_output_paths(instance_id, self.config.current_run)`
- [x] Update summary to include:

  ```python
  "num_runs": self.config.num_runs,
  "per_run_results": grouped_by_run,  # Group results by run number
  ```

- [x] Save `stage1_summary.json` at run_output_dir level (not inside run_N/)

**Verification**:

```bash
python pipeline_1_generate_patches.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3 --output-dir output_test
# Check directory structure:
find output_test -type d -name "run_*"
# Should show:
# output_test/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_1
# output_test/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_2
# output_test/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_3

# Check that patch exists in each run:
find output_test -name "patch.diff" -type f
# Should show 3 patch files, one in each run_N directory
```

**Notes**:

- Use nested loops: outer loop for runs, inner loop for instances
- Update progress logging to show "Run X of Y - Instance Z"
- Ensure trajectory.json and metadata.json are also in run-specific directories
- Summary should aggregate across all runs

---

## Step 4: Update Stage 2 (Create Predictions) ✅

**Status**: COMPLETED

**File**: `pipeline_2_create_predictions.py`

**Tasks**:

- [x] Update `PredictionCreator.__init__()` to detect run directories:

  ```python
  def _discover_runs(self, instance_dir: Path) -> List[int]:
      """Discover all run_N directories for an instance."""
      runs = []
      for item in instance_dir.iterdir():
          if item.is_dir() and item.name.startswith("run_"):
              try:
                  run_num = int(item.name.split("_")[1])
                  runs.append(run_num)
              except (IndexError, ValueError):
                  continue
      return sorted(runs)
  ```

- [x] Update `create_prediction_for_instance()` to accept `run_number` parameter
- [x] Update `create_all_predictions()` to iterate over runs:

  ```python
  for instance_id in instance_ids:
      instance_dir = self.run_dir / instance_id
      runs = self._discover_runs(instance_dir)

      if not runs:
          # Legacy single-run structure
          result = self.create_prediction_for_instance(instance_id, run_number=None)
      else:
          # Multi-run structure
          for run_num in runs:
              result = self.create_prediction_for_instance(instance_id, run_number=run_num)
  ```

- [x] Create separate `predictions_all.json` per run when in multi-run mode
- [x] Update summary to track per-run statistics

**Verification**:

```bash
python pipeline_2_create_predictions.py output_test/claude-sonnet-4-20250514/{timestamp}
# Check that prediction.json exists in each run directory:
find output_test -name "prediction.json" -type f
# Should show 3 files
```

**Notes**:

- Support both old (no run_N) and new (run_N) directory structures for backward compatibility
- Each run should have its own predictions_all.json for evaluation

---

## Step 5: Update Stage 3 (Evaluation) ✅

**Status**: COMPLETED

**File**: `pipeline_3_run_evaluation.py`

**Tasks**:

- [x] Add `--run-number` argument to optionally evaluate specific run (default: None = all runs)
- [x] Update `SWEBenchEvaluator` to discover runs (similar to Stage 2)
- [x] Update `run_evaluation()` to iterate over runs:

  ```python
  runs = self._discover_runs()
  for run_num in runs:
      predictions_path = self.run_dir / instance_id / f"run_{run_num}" / "predictions_all.json"
      run_id = f"{model}_{timestamp}_run{run_num}"
      # Run evaluation with run-specific ID
      # Save results to run-specific directory
  ```

- [x] Update `_distribute_results()` to work with multi-run structure
- [x] Update summary to include per-run evaluation metrics

**Verification**:

```bash
python pipeline_3_run_evaluation.py output_test/claude-sonnet-4-20250514/{timestamp}
# Check evaluation.json in each run:
find output_test -name "evaluation.json" -type f
# Should show 3 files
```

**Notes**:

- Each run needs unique run_id to avoid conflicts in SWE-bench harness
- Aggregate results at the end to show overall resolution rates

---

## Step 6: Update Stage 4 (Security Scan) ✅

**Status**: COMPLETED

**File**: `pipeline_4_security_scan.py`

**Tasks**:

- [x] Update `main()` to discover runs for each instance
- [x] For each instance and run combination:

  ```python
  for instance_id in instance_ids:
      runs = discover_runs(run_dir / instance_id)
      for run_num in runs:
          patch_path = run_dir / instance_id / f"run_{run_num}" / "patch.diff"
          summary = secscanner.run_security_risk_scorer(dataset_id, instance_id, run_num)
          summary_path = run_dir / instance_id / f"run_{run_num}" / "security_risk_score.json"
  ```

- [x] Update `SecurityScanner.run_security_risk_scorer()` to accept `run_number` parameter
- [x] Update summary to include per-run security metrics

**Verification**:

```bash
python pipeline_4_security_scan.py output_test/claude-sonnet-4-20250514/{timestamp}
# Check security_risk_score.json in each run:
find output_test -name "security_risk_score.json" -type f
# Should show 3 files
```

---

## Step 7: Implement Stage 5 (Consistency Check) 🔄

**Status**: IN PROGRESS

**File**: `pipeline_5_consistency_check.py`

**Tasks**:

- [x] Create `ConsistencyChecker` class
- [x] Implement `_discover_runs(instance_dir: Path) -> List[int]` to find all run_N directories
- [x] Implement `_load_run_data(instance_id: str, run_num: int) -> Dict` to load:
  - patch.diff content
  - metadata.json
  - evaluation.json
  - security_risk_score.json
- [x] Implement patch consistency metrics:
  - [x] `calculate_exact_match_rate()` - percentage of runs with identical patches
  - [x] `calculate_patch_similarity()` - average similarity score using difflib
  - [x] `calculate_line_consistency()` - consistency in number of lines changed
- [x] Implement evaluation consistency metrics:
  - [x] `calculate_resolution_consistency()` - all pass / all fail / mixed
  - [x] `calculate_success_rate()` - percentage of runs that resolved the issue
- [x] Implement security consistency metrics:
  - [x] `calculate_risk_level_consistency()` - consistency in risk levels
  - [x] `calculate_score_variance()` - variance in security scores
- [x] Implement overall consistency score:
  - [x] Weight different metrics (patch: 40%, evaluation: 40%, security: 20%)
  - [x] Calculate composite score 0-100%
- [x] Create per-instance consistency report:

  ```python
  {
      "instance_id": "...",
      "num_runs": 3,
      "patch_consistency": {
          "exact_match_rate": 0.67,
          "avg_similarity": 0.85,
          "line_count_variance": 2.3
      },
      "evaluation_consistency": {
          "all_resolved": false,
          "all_failed": false,
          "resolution_rate": 0.67
      },
      "security_consistency": {
          "risk_level_mode": "LOW",
          "score_variance": 0.12
      },
      "overall_consistency_score": 78.5,
      "consistency_grade": "GOOD"  # EXCELLENT (>90), GOOD (70-90), FAIR (50-70), POOR (<50)
  }
  ```

- [x] Save per-instance reports: `{instance_id}/consistency_report.json`
- [x] Create aggregated summary: `consistency_summary.json` at run_dir level
- [x] Add skip logic: if num_runs == 1, skip consistency check and log warning

**Verification**:

```bash
python pipeline_5_consistency_check.py output_test/claude-sonnet-4-20250514/{timestamp}
# Check consistency reports:
find output_test -name "consistency_report.json" -type f
# Should show 1 file per instance

# Check consistency summary:
cat output_test/claude-sonnet-4-20250514/{timestamp}/consistency_summary.json
# Should show overall consistency metrics
```

**Notes**:

- Only run when multiple runs are detected
- Use difflib.SequenceMatcher for patch similarity
- Consider using semantic diff tools if available
- Handle missing data gracefully (e.g., if one run failed)

---

## Step 8: Update Stage 6 (Aggregate Results) ⏸️

**Status**: PENDING

**File**: `pipeline_6_aggregate_results.py`

**Tasks**:

- [ ] Update `ResultsAggregator` to discover runs
- [ ] Update `aggregate_instance_results()` to iterate over runs:

  ```python
  for instance_dir in self.run_dir.iterdir():
      runs = self._discover_runs(instance_dir)
      for run_num in runs:
          # Load data from run_num directory
          # Create row in results with run_number column
  ```

- [ ] Update `create_run_summary()` to include:

  ```python
  "multi_run_info": {
      "num_runs": max_run_number,
      "consistency_metrics": load_consistency_summary()
  }
  ```

- [ ] **IMPORTANT**: Update CSV export to include `run_number` column as the second column (after instance_id)
  - Add "run_number" to fieldnames list at line 208
  - CSV should have one row per (instance_id, run_number) combination
  - For single-run (backward compatibility), run_number should be 1
- [ ] Add aggregated metrics:
  - Average resolution rate across runs
  - Total cost (sum of all runs)
  - Consistency grade from stage 5
- [ ] Update logging to show multi-run summary

**Verification**:

```bash
python pipeline_6_aggregate_results.py output_test/claude-sonnet-4-20250514/{timestamp}
# Check CSV structure:
head -20 output_test/claude-sonnet-4-20250514/{timestamp}/results.csv
# Should have run_number column

# Check summary:
cat output_test/claude-sonnet-4-20250514/{timestamp}/run_summary.json | jq '.multi_run_info'
```

**Notes**:

- CSV should have one row per (instance, run) combination
- Summary should aggregate across runs
- Include consistency metrics from stage 5

---

## Step 9: Integration Testing ⏸️

**Status**: PENDING

**Tasks**:

- [ ] Test full pipeline with --num-runs=1 (backward compatibility):

  ```bash
  python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 1
  ```

- [ ] Test full pipeline with --num-runs=3 (new feature):

  ```bash
  python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3
  ```

- [ ] Test partial pipeline execution:

  ```bash
  # Run only stage 1
  python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3 --stages 1
  # Then run stages 2-6
  python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3 --stages 2 3 4 5 6
  ```

- [ ] Test with multiple instances:

  ```bash
  python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 django__django-11001 --num-runs 3
  ```

- [ ] Verify consistency check produces meaningful results with varying patches
- [ ] Test error handling (what if run 2 fails but run 1 and 3 succeed?)

**Verification**: All tests should complete without errors, and output structure should match specification

---

## Step 10: Documentation and Cleanup ⏸️

**Status**: PENDING

**Tasks**:

- [ ] Update main README.md to document --num-runs parameter
- [ ] Update HOW_TO_USE_PIPELINE.md with multi-run examples
- [ ] Add docstrings to all new methods
- [ ] Add inline comments for complex logic (e.g., consistency scoring)
- [ ] Create example outputs in docs/features/allow-multi-runs/examples/
- [ ] Update pipeline flowchart (if exists) to show multi-run loops
- [ ] Add troubleshooting section for common multi-run issues

---

## Rollback Plan

If at any point the implementation needs to be rolled back:

1. **Emergency Rollback** (instant):

   ```bash
   # Users can always use --num-runs=1 to get old behavior
   python run_pipeline.py --num-runs 1 ...
   ```

2. **Code Rollback** (if needed):

   ```bash
   git log --oneline  # Find commit before multi-run implementation
   git revert <commit-hash>
   ```

3. **Data Migration** (if old output format needed):

   ```bash
   # Script to flatten run_N directories (to be written if needed)
   python scripts/migrate_multirun_to_singlerun.py output_dir
   ```

---

## Success Metrics

Implementation is complete when:

- [x] All steps 1-8 are marked as COMPLETED
- [ ] All verification commands pass
- [ ] Full pipeline runs successfully with --num-runs=1 and --num-runs=3
- [ ] Consistency check produces meaningful metrics
- [ ] No breaking changes to existing single-run workflows
- [ ] All Python files compile without errors
- [ ] Documentation is updated

---

## Notes

- **Compile-safe principle**: After each step, the codebase should compile and run
- **Backward compatibility**: --num-runs=1 must work like the old code
- **Incremental testing**: Test each step before moving to the next
- **Error handling**: Gracefully handle missing runs, failed runs, and partial data

Last updated: 2025-10-13
