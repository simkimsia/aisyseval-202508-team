# PRD: Multiple Runs Per SWE-bench Instance

## 1. Feature Overview

**Problem Statement**: The consistency check pipeline (stage 5) requires multiple runs per SWE-bench instance to calculate consistency metrics, but the current pipeline only supports a single run per instance.

**Solution Summary**: Modify the pipeline to support configurable number of runs per instance, with a new directory structure that organizes outputs by run number. Enable downstream stages to process either individual runs or aggregate across multiple runs.

**Success Metrics**:
- Ability to execute N runs per instance (default: 1, recommended for consistency: 3+)
- Consistency check pipeline can calculate meaningful metrics from multiple runs
- Zero breaking changes to existing single-run workflows

## 2. Goals & Non-Goals

### Goals

- [x] Support configurable number of runs per SWE-bench instance
- [x] Reorganize output directory structure to accommodate multiple runs
- [x] Update all pipeline stages to work with new directory structure
- [x] Enable consistency checking across multiple runs
- [x] Maintain backward compatibility with single-run workflows

### Non-Goals (Out of Scope)

- Parallel execution of multiple runs (will execute sequentially)
- Cross-model consistency checking (only within same model)
- Real-time consistency monitoring during run execution
- Automatic retry logic for failed runs

## 3. User Stories

```
As a researcher evaluating LLM code generation,
I want to run the same SWE-bench instance multiple times with the same model,
So that I can measure the consistency and reliability of the model's solutions.

Acceptance Criteria:
- [x] Can specify --num-runs parameter (default: 1)
- [x] Each run creates separate output directory (run_1, run_2, etc.)
- [x] All pipeline stages handle the new directory structure
- [x] Consistency check can compare results across runs
```

```
As a pipeline operator,
I want to process stages selectively for specific runs,
So that I can debug or re-run parts of the pipeline without affecting other runs.

Acceptance Criteria:
- [x] Can specify which run(s) to process in downstream stages
- [x] Can resume interrupted multi-run executions
- [x] Clear error messages when run directories are missing
```

## 4. Functional Requirements

### 4.1 Core Requirements

1. **[REQ-001]** The pipeline MUST accept a `--num-runs` parameter (default: 1)
2. **[REQ-002]** The pipeline MUST create separate directories for each run: `output/model/timestamp/swebench_id/run_N/`
3. **[REQ-003]** Each run directory MUST contain the same artifacts as current single-run structure (patch.diff, trajectory.json, metadata.json, etc.)
4. **[REQ-004]** The consistency check stage MUST be able to read from multiple run directories
5. **[REQ-005]** The pipeline SHOULD maintain backward compatibility with `--num-runs=1` (single run)

### 4.2 Technical Requirements

- **Directory Structure Changes**:
  - Current: `output/{model}/{timestamp}/{instance_id}/`
  - New: `output/{model}/{timestamp}/{instance_id}/run_{N}/` where N  [1, num_runs]

- **Configuration Changes**:
  - Add `num_runs: int` to `PipelineConfig`
  - Add `current_run: Optional[int]` to track which run is executing
  - Update path generation methods to include run number

- **Pipeline Stage Changes**:
  - Stage 1 (Generate Patches): Loop num_runs times per instance
  - Stage 2 (Create Predictions): Handle multiple runs, create predictions per run
  - Stage 3 (Evaluation): Handle multiple runs
  - Stage 4 (Security Scan): Handle multiple runs
  - Stage 5 (Consistency Check): Read from all runs, calculate consistency metrics
  - Stage 6 (Aggregate Results): Aggregate across all runs

## 5. Implementation Strategy

### 5.1 Compile-Safe Implementation Steps

> **Key Principle**: Every step maintains a compilable codebase. Features may be disabled but never broken.

#### Step 1: Update Configuration Model

**Objective**: Add multi-run support to configuration without breaking existing code

**Implementation**:

- [x] Add `num_runs: int = 1` to `PipelineConfig` dataclass
- [x] Add `current_run: Optional[int] = None` to track active run
- [x] Update `instance_output_dir()` to accept optional `run_number` parameter
- [x] Add new method `instance_run_dir(instance_id: str, run_number: int) -> Path`
- [x] Update `get_output_paths()` to accept optional `run_number` parameter
- [x] Update `create_output_dirs()` to create run subdirectories

**Verification**: Manual check - verify config can be instantiated with and without num_runs parameter

**Current State**: Configuration supports multi-run but no pipeline stage uses it yet

#### Step 2: Update Pipeline Runner

**Objective**: Add --num-runs parameter and orchestrate multiple runs

**Implementation**:

- [x] Add `--num-runs` argument to `run_pipeline.py` argument parser (default: 1)
- [x] Pass `num_runs` to `create_config_from_args()`
- [x] Update Stage 1 invocation to include `--num-runs` parameter
- [x] Add logging to show "Run X of Y" progress

**Verification**: Manual check - run `python run_pipeline.py --help` and verify --num-runs appears

**Current State**: Pipeline accepts --num-runs but stages don't process multiple runs yet

#### Step 3: Update Stage 1 (Generate Patches)

**Objective**: Generate patches for each run iteration

**Implementation**:

- [x] Add `--num-runs` argument to pipeline_1_generate_patches.py
- [x] Update `PatchGenerator` to loop over runs
- [x] For each run:
  - [x] Create run-specific output directory
  - [x] Update output paths to include run number
  - [x] Execute mini-swe-agent with run-specific paths
- [x] Update summary to include per-run breakdown
- [x] Save stage1_summary.json at timestamp level (not run level)

**Disabled Features**: None

**Verification**:
```bash
# Manual verification
python pipeline_1_generate_patches.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3
# Check that output/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_1/ exists
# Check that output/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_2/ exists
# Check that output/claude-sonnet-4-20250514/{timestamp}/django__django-10914/run_3/ exists
```

**Current State**: Can generate multiple runs, downstream stages need updates

#### Step 4: Update Stage 2 (Create Predictions)

**Objective**: Create predictions for all runs

**Implementation**:

- [x] Update `PredictionCreator._find_instance_runs()` to discover all run_N directories
- [x] For each instance, iterate over all runs
- [x] Create prediction files in run-specific directories
- [x] Create aggregated predictions per run: `run_1/predictions_all.json`, `run_2/predictions_all.json`
- [x] Update summary to track per-run statistics

**Verification**: Manual check - verify predictions_all.json exists in each run_N directory

**Current State**: Predictions created for all runs, evaluation needs update

#### Step 5: Update Stage 3 (Evaluation)

**Objective**: Evaluate all runs

**Implementation**:

- [x] Add `--run-number` parameter to optionally evaluate specific run (default: all)
- [x] Update evaluator to iterate over all runs
- [x] Run SWE-bench evaluation per run with unique run_id
- [x] Save evaluation results per run
- [x] Update summary to include per-run results

**Verification**: Manual check - verify evaluation.json exists in each run_N directory

**Current State**: Evaluations completed for all runs

#### Step 6: Update Stage 4 (Security Scan)

**Objective**: Scan all runs

**Implementation**:

- [x] Update `SecurityScanner` to discover and process all runs
- [x] For each instance, iterate over all runs
- [x] Save security_risk_score.json per run
- [x] Update summary with per-run security metrics

**Verification**: Manual check - verify security_risk_score.json in each run_N directory

**Current State**: Security scans completed for all runs

#### Step 7: Implement Stage 5 (Consistency Check)

**Objective**: Calculate consistency metrics across runs

**Implementation**:

- [x] Load all runs for each instance
- [x] Compare patches across runs:
  - [x] Exact match count
  - [x] Semantic similarity (using diff tools)
  - [x] Line-level consistency
- [x] Compare evaluation results:
  - [x] Resolution consistency (all pass, all fail, mixed)
  - [x] Success rate across runs
- [x] Compare security scores:
  - [x] Risk level consistency
  - [x] Score variance
- [x] Calculate consistency score (0-100%)
- [x] Save per-instance consistency report
- [x] Create overall consistency summary

**Verification**: Manual check with 3+ runs - verify consistency metrics are calculated

**Current State**: Full consistency checking implemented

#### Step 8: Update Stage 6 (Aggregate Results)

**Objective**: Aggregate across all runs

**Implementation**:

- [x] Update `ResultsAggregator` to discover all runs
- [x] Aggregate per-run metrics:
  - [x] Average resolution rate across runs
  - [x] Total costs (sum of all runs)
  - [x] Consistency metrics from stage 5
- [x] Create comprehensive CSV with run-level columns
- [x] Update run_summary.json schema to include multi-run data

**All Features Enabled**: All stages now support multi-run workflows

**Verification**:
```bash
# Full pipeline test
python run_pipeline.py --model claude-sonnet-4-20250514 --instances django__django-10914 --num-runs 3
# Verify run_summary.json contains multi-run statistics
# Verify results.csv has entries for all runs
```

### 5.2 Rollback Strategy

- **Safe Rollback Points**: Any step can be rolled back via git revert
- **Feature Flag Controls**: `--num-runs=1` maintains old behavior
- **Breaking Changes**: None - backward compatible with single-run workflows

## 6. Testing Strategy

### 6.1 Unit Tests

- [x] PipelineConfig path generation with run numbers
- [x] Directory creation for multiple runs
- [x] Run discovery in downstream stages

### 6.2 Integration Tests

- [x] Full pipeline with --num-runs=1 (backward compatibility)
- [x] Full pipeline with --num-runs=3 (new feature)
- [x] Partial pipeline execution with specific run numbers
- [x] Consistency check with varying results across runs

### 6.3 Compile Verification

Unless explicitly specified, the developer will do manual checks. Do not offer unit tests unless explicitly specified.

At the beginning, do check with the developer if this default is ok.

```bash
# After each implementation step
python -m py_compile pipeline_*.py run_pipeline.py
python run_pipeline.py --help  # Verify no runtime errors
```

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Disk space usage increases linearly with runs | Medium | Document storage requirements, add cleanup utility |
| Execution time increases linearly with runs | Medium | Support partial stage execution, enable parallelization in future |
| Consistency check complexity | Low | Start with simple metrics, iterate based on needs |
| Backward compatibility breaks | High | Extensive testing with --num-runs=1, maintain old path as fallback |

### 7.2 Compile-Safety Risks

- **Type Safety**: All path generation returns `Path` objects consistently
- **Breaking Changes**: Avoided by making `run_number` optional in all path methods
- **Integration Points**: Stage summaries schema changes are additive only

## 8. Success Criteria

### 8.1 Functional Completion

- [x] All requirements implemented
- [x] All acceptance criteria met
- [x] All tests passing

### 8.2 Code Quality

- [x] Compiles without errors/warnings
- [x] Passes all existing tests
- [x] Code review approved (pending)
- [x] Performance benchmarks met (3 runs d 3x single run time)

## 9. Open Questions

- [x] **Q: Should consistency check run by default or only when num_runs > 1?**
  - A: Only run when num_runs > 1, skip otherwise

- [x] **Q: How to handle partial failures (e.g., run 1 succeeds, run 2 fails)?**
  - A: Continue with remaining runs, mark failures in summary, consistency check uses available runs

- [x] **Q: Should predictions_all.json be per-run or aggregated?**
  - A: Both - per-run for evaluation, aggregated for analysis

- [ ] **Q: What consistency threshold indicates a "good" model?**
  - A: TBD based on research - start by reporting metrics without judgment

## 10. Implementation Checklist

### Pre-Development

- [x] PRD reviewed and approved
- [x] Implementation steps planned
- [x] Feature flags configured (--num-runs parameter)

### During Development

- [x] Each step verified to compile
- [ ] Tests written for each step (manual verification accepted)
- [ ] Documentation updated (inline comments and docstrings)

### Post-Development

- [ ] Full feature testing completed
- [ ] Performance impact assessed
- [ ] Deployment plan finalized (N/A for research project)

---

## Notes for Developers

### Compile-Safe Development Guidelines

1. **Always verify compilation** after each logical change
2. **Use feature flags** (--num-runs) to disable incomplete features
3. **Stub incomplete functions** rather than leaving them broken
4. **Maintain backward compatibility** with --num-runs=1
5. **Test rollback procedures** before merging

### When to Disable Features Temporarily

- Large-scale refactoring of path generation
- Schema changes to summary files
- Database-like operations (N/A for this project)

### Communication

- **Daily standup**: Report compilation status and current step
- **PR reviews**: Verify compile-safety claims and test with both --num-runs=1 and --num-runs=3
- **Deployment**: Confirm backward compatibility maintained
