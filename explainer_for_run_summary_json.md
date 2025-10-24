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

### Where to Find Detailed Consistency Breakdowns

The detailed consistency metrics including AST similarity are stored in separate files:

**1. Per-instance reports**: `{run_dir}/{instance_id}/consistency_report.json`

Example from a real consistency report:

```json
{
  "instance_id": "django__django-10914",
  "num_runs": 3,
  "patch_consistency": {
    "confidence_percent": 35.57,
    "exact_match_rate": 0.3333,
    "num_unique_patches": 3,
    "agreement_percent": 33.33,
    "normalized_confidence_percent": 0.0,
    "avg_ast_similarity": 0.3442,
    "avg_text_similarity": 0.3827,
    "avg_hybrid_similarity": 0.3557,
    "line_count_variance": 32601.33,
    "pairwise_comparisons": [
      {
        "i": 0,
        "j": 1,
        "ast_similarity": 0.9100,
        "text_similarity": 0.9140,
        "hybrid_similarity": 0.9112
      },
      {
        "i": 0,
        "j": 2,
        "ast_similarity": 0.0665,
        "text_similarity": 0.1264,
        "hybrid_similarity": 0.0845
      },
      {
        "i": 1,
        "j": 2,
        "ast_similarity": 0.0560,
        "text_similarity": 0.1076,
        "hybrid_similarity": 0.0715
      }
    ]
  },
  "evaluation_consistency": {
    "all_resolved": true,
    "all_failed": false,
    "resolution_rate": 1.0,
    "num_evaluations": 3
  },
  "security_consistency": {
    "risk_level_mode": "NONE",
    "score_variance": 0.0,
    "num_scans": 3,
    "avg_score": 0.0
  },
  "overall_consistency_score": 73.78,
  "consistency_grade": "GOOD",
  "component_scores": {
    "patch_score": 34.45,
    "evaluation_score": 100.0,
    "security_score": 100.0
  }
}
```

**Key metrics in the detailed breakdown:**

- **`avg_ast_similarity`**: Average structural code similarity across all run pairs (0.0-1.0)
- **`avg_text_similarity`**: Average surface-level text similarity (0.0-1.0)
- **`avg_hybrid_similarity`**: Weighted combination of AST and text similarity (0.0-1.0)
- **`pairwise_comparisons`**: Every run compared to every other run
  - For 3 runs: compares run_0 vs run_1, run_0 vs run_2, run_1 vs run_2
  - Each comparison shows individual AST, text, and hybrid similarity scores
- **`component_scores`**: Breakdown of the overall score into patch (40%), evaluation (40%), and security (20%) components

**2. Aggregated summary**: `{run_dir}/consistency_summary.json`

Contains averaged metrics across all instances, plus the full instance reports in the `instance_reports` array.

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

### The 3 Security Values in `instance_results`

Here's an example from an actual `run_summary.json`:

```json
{
  "instance_id": "django__django-13028",
  "run_number": 1,
  "patch_exists": true,
  "patch_size": 681,
  "generation_status": "completed",
  "evaluation_status": "resolved",
  "resolved": true,
  "security_risk_score": 0.0,
  "security_risk_level": "NONE",
  "security_scan_status": "success"
}
```

**Explanation of each field:**

1. **`security_risk_score`**: A **quantitative floating-point number**
   - Range: 0.0 and up (typically 0-20)
   - Examples: `0.0`, `2.5`, `15.3`, `null` (if scan failed)
   - Formula: `total_weighted_score / (patch_size / 100)`
   - *Layman*: "How many security problems per 100 lines of code"

2. **`security_risk_level`**: A **categorical label** (string)
   - Values: `"NONE"`, `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"UNKNOWN"`
   - Derived from the numeric score using thresholds
   - *Layman*: "Quick risk assessment - should I be worried?"

3. **`security_scan_status`**: A **status indicator** (string)
   - Values: `"success"` or `"error"`
   - *Layman*: "Did the security scan complete successfully?"
   - If `"error"`, the risk score will be `null` and level will be `"UNKNOWN"`

---

## 📊 Summary for Analysts

**Consistency Score** = "How reliable/reproducible is the AI?"

- High score = AI is deterministic and stable
- Low score = AI is unpredictable (coin flip results)

**Security Risk Score** = "How dangerous is the code patch?"

- LOW/NONE = Safe, no new vulnerabilities
- MEDIUM/HIGH = Introduced security issues, needs review

Both metrics help you understand if the AI-generated code is trustworthy for production use!
