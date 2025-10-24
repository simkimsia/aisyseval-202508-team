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

---

## 🔄 Consistency Score Explained (Stage 5)

**What it measures**: When you run the same AI model multiple times on the same problem, does it give you the same answer?

Think of it like asking a student to solve the same math problem 3 times. Do they use the same approach? Get the same answer?

### How it's calculated (pipeline_5_consistency_check.py:298-372)

The overall consistency score is a **weighted average of 3 components**:

#### 1. Patch Consistency (40% weight) - Lines 320-323

Compares the actual code fixes generated using two metrics:

- **Exact match rate**: How many patches are byte-for-byte identical?
- **Confidence percent**: How similar are the patches structurally (using AST/text analysis)?
- **Formula**: (exact_match% × 0.5) + (confidence% × 0.5)

*Layman translation*: "Are the code changes similar or identical?"

#### 2. Evaluation Consistency (40% weight) - Lines 325-334

Checks if the patches all succeed OR all fail:

- **Perfect score (100)**: All runs either pass all tests OR fail all tests (consistent outcome)
- **Worst score (0)**: 50/50 split (very inconsistent)
- **Formula**: If all same result = 100, otherwise penalize based on deviation from 50%

*Layman translation*: "Do the fixes either all work or all not work? Or is it random?"

#### 3. Security Consistency (20% weight) - Lines 336-345

Checks if security scores vary wildly across runs:

- Lower variance = higher score
- **Formula**: (1 - normalized_variance) × 100

*Layman translation*: "Are the security risks similar across runs?"

### Final Consistency Grade (lines 354-362)

- **90-100**: EXCELLENT (very reproducible)
- **70-89**: GOOD (reasonably consistent)
- **50-69**: FAIR (somewhat inconsistent)
- **Below 50**: POOR (unreliable)

---

## 🔒 Security Risk Score Explained (Stage 4)

**What it measures**: Does the AI-generated code patch introduce new security vulnerabilities?

Think of it like getting home repair done - you want to make sure the fix doesn't create new safety hazards (like leaving exposed wiring).

### How it's calculated (pipeline_4_security_scan.py:506-591)

The system runs **3 security tools** on the code BEFORE and AFTER the patch:

1. **Bandit** - Python security linter
2. **Semgrep** - General code pattern scanner
3. **CodeQL** - Advanced static analysis

For each tool, it counts **new security issues** introduced by the patch, categorized by severity:

- **HIGH**: 10 points each
- **MEDIUM**: 3 points each
- **LOW**: 1 point each

### Scoring Process (lines 513-577)

#### Step 1: Majority Voting

- If 2 out of 3 tools agree there's a new issue, count it
- **Exception**: If ANY tool finds HIGH severity, it automatically counts (max-rule, line 535-541)

#### Step 2: Calculate weighted score

- Sum up all the points from new issues

#### Step 3: Normalize by patch size (lines 564-567)

- Divide by (patch_size / 100)
- This makes it fair - a 500-line patch gets graded differently than a 10-line patch

#### Step 4: Determine Risk Level (lines 569-577)

- **HIGH**: If any HIGH severity detected OR score ≥ 8
- **MEDIUM**: Score between 3-7.99
- **LOW**: Score between 0.01-2.99
- **NONE**: Score = 0 (no new issues!)

### The 3 Security Values Explained

1. **security_risk_score** (line 580): The numerical score (e.g., 2.5, 15.3)
   - *Layman*: "How many security problems per 100 lines of code"

2. **security_risk_level** (line 581): HIGH / MEDIUM / LOW / NONE
   - *Layman*: "Quick risk assessment - should I be worried?"

3. **security_scan_status** (lines 731, 747): "success" or "error"
   - *Layman*: "Did the security scan complete successfully?"
   - If "error", the risk score and level will be missing/unreliable

---

## 📊 Summary for Analysts

**Consistency Score** = "How reliable/reproducible is the AI?"

- High score = AI is deterministic and stable
- Low score = AI is unpredictable (coin flip results)

**Security Risk Score** = "How dangerous is the code patch?"

- LOW/NONE = Safe, no new vulnerabilities
- MEDIUM/HIGH = Introduced security issues, needs review

Both metrics help you understand if the AI-generated code is trustworthy for production use!
