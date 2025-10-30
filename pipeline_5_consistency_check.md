# Pipeline Stage 5: Consistency Check - Similarity Metrics Documentation

## Overview

This document explains the similarity metrics used in `pipeline_5_consistency_check.py` to assess consistency across multiple runs of the same SWE-bench instance.

The consistency checker uses a **three-tier approach** combining:

1. **Pipeline-specific metrics** (exact match, line variance, unique patches)
2. **Ryan's consistency_metrics()** (agreement%, confidence%, normalized confidence%)
3. **Detailed similarity breakdowns** (AST, text, hybrid)

---

## TL;DR - Quick Answer for Ryan

**Hey Ryan! Here's what we use from your `consistency_evaluator`:**

### Functions Used
✅ **`consistency_metrics()`** - Main function we call directly (line 173)
✅ **`hybrid_similarity()`** - Called internally by `consistency_metrics()`
❌ **`ast_similarity()`, `text_similarity()`, `normalize_python_code()`** - Only used internally by your code

### How It Works
1. **We call once**: `consistency_metrics(patches, "python", 0.85)` → pipeline_5_consistency_check.py:173
2. **You return**: `agreement%`, `confidence%`, `normalized_confidence%`, and pairwise details
3. **We compute averages**: From your pairwise data, we extract `avg_ast_similarity`, `avg_text_similarity`, `avg_hybrid_similarity` → pipeline_5_consistency_check.py:180-182
4. **We add our own**: `exact_match_rate`, `num_unique_patches`, `line_count_variance`
5. **Notebook extracts**: All metrics go into CSV via consolidate_similarity_scores.ipynb

### What Ends Up in the CSV

**From your code:**
- `confidence_percent`, `agreement_percent`, `normalized_confidence_percent` (direct returns)
- `avg_ast_similarity`, `avg_text_similarity`, `avg_hybrid_similarity` (we average your pairwise data)

**From our pipeline:**
- `exact_match_rate`, `num_unique_patches`, `line_count_variance`

**Composite scores (our pipeline combines everything):**
- `patch_score` = 50% exact_match + 50% your confidence_percent
- `evaluation_score` = based on test pass/fail consistency
- `security_score` = based on security scan consistency
- `overall_consistency_score` = 40% patch + 40% eval + 20% security
- `consistency_grade` = EXCELLENT/GOOD/FAIR/POOR

**See these sections for details:**
- [Concrete Example](#concrete-example-number-flow-through-the-system) - Actual numbers flowing through
- [Composite Scores](#composite-scores-how-they-add-up) - How the percentages add up

---

## Table of Contents

- [TL;DR - Quick Answer for Ryan](#tldr---quick-answer-for-ryan)
- [Function Integration & Data Flow](#function-integration--data-flow)
- [Quick Reference](#quick-reference)
- [Core Similarity Functions](#core-similarity-functions)
- [Comprehensive Metrics](#comprehensive-metrics)
- [Three-Tier Output Structure](#three-tier-output-structure)
- [Interpreting Results](#interpreting-results)
- [Examples](#examples)
- [Concrete Example: Number Flow Through the System](#concrete-example-number-flow-through-the-system)

---

## Function Integration & Data Flow

### Which Functions Are Used from consistency_evaluator?

**From `consistency_evaluator/similarity_utils.py`**, the pipeline imports and uses:

```python
# Line 21 in pipeline_5_consistency_check.py
from consistency_evaluator.similarity_utils import hybrid_similarity, consistency_metrics
```

**Primary function**: `consistency_metrics()` (line 173)
- This is the **main workhorse** that:
  - Takes a list of patch strings
  - Performs all pairwise comparisons
  - Returns 4 values: agreement%, confidence%, normalized_confidence%, and pairwise details

**Secondary function**: `hybrid_similarity()` (indirectly used)
- Called internally by `consistency_metrics()`
- Not directly called by the pipeline
- Computes AST, text, and hybrid similarity for each pair

**Note**: The following functions are NOT directly used by the pipeline:
- `normalize_python_code()` - Only used internally by `ast_similarity()`
- `ast_similarity()` - Only called internally by `hybrid_similarity()`
- `text_similarity()` - Only called internally by `hybrid_similarity()`

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PIPELINE INPUT (pipeline_5_consistency_check.py:143)        │
│    patches = ["patch1", "patch2", "patch3"]                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. RYAN'S FUNCTION CALL (pipeline_5_consistency_check.py:173)  │
│    consistency_metrics(outputs=patches, language="python",      │
│                        threshold=0.85)                          │
│                                                                 │
│    Internally performs:                                         │
│    ┌───────────────────────────────────────────────────────┐  │
│    │ For each pair (i, j):                                 │  │
│    │   hybrid_similarity(patch_i, patch_j)                 │  │
│    │     ├─> ast_similarity() → AST score                  │  │
│    │     ├─> text_similarity() → text score                │  │
│    │     └─> 0.7*AST + 0.3*text → hybrid score             │  │
│    └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. RYAN'S FUNCTION RETURNS (pipeline_5_consistency_check.py:173)│
│    agreement_percent    : float (0-100)                         │
│    confidence_percent   : float (0-100)                         │
│    normalized_confidence: float (0-100)                         │
│    pairwise            : List[Dict]                             │
│      ├─> [{"i": 0, "j": 1,                                     │
│      │    "ast_similarity": 0.92,                              │
│      │    "text_similarity": 0.85,                             │
│      │    "hybrid_similarity": 0.899}, ...]                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. PIPELINE PROCESSING (pipeline_5_consistency_check.py:180-182)│
│    Extract averages from pairwise data:                         │
│    - avg_ast_similarity = mean([p["ast_similarity"]])          │
│    - avg_text_similarity = mean([p["text_similarity"]])        │
│    - avg_hybrid_similarity = mean([p["hybrid_similarity"]])    │
│                                                                 │
│    Add pipeline-specific metrics (lines 164-170):              │
│    - exact_match_rate (Counter-based)                          │
│    - num_unique_patches (len(Counter))                         │
│    - line_count_variance (statistics.variance)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. PIPELINE OUTPUT (pipeline_5_consistency_check.py:185-201)   │
│    Three-tier dictionary returned:                              │
│                                                                 │
│    Tier 1 (Summary):                                           │
│      - confidence_percent          [Ryan's]                    │
│      - exact_match_rate            [Pipeline]                  │
│      - num_unique_patches          [Pipeline]                  │
│                                                                 │
│    Tier 2 (Detailed):                                          │
│      - agreement_percent           [Ryan's]                    │
│      - normalized_confidence_percent [Ryan's]                  │
│      - avg_ast_similarity          [Computed from Ryan's data] │
│      - avg_text_similarity         [Computed from Ryan's data] │
│      - avg_hybrid_similarity       [Computed from Ryan's data] │
│      - line_count_variance         [Pipeline]                  │
│                                                                 │
│    Tier 3 (Raw):                                               │
│      - pairwise_comparisons        [Ryan's raw output]         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SAVED TO FILE (pipeline_5_consistency_check.py:458-461)     │
│    {instance_id}/consistency_report.json                        │
│    └─> Contains all three tiers                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. NOTEBOOK EXTRACTION (consolidate_similarity_scores.ipynb)   │
│    Cell 7: Reads consistency_report.json from run_summary.json │
│    Extracts for CSV:                                            │
│      - avg_ast_similarity          [from pairwise averages]    │
│      - avg_text_similarity         [from pairwise averages]    │
│      - avg_hybrid_similarity       [from pairwise averages]    │
│      - exact_match_rate            [pipeline metric]           │
│      - num_unique_patches          [pipeline metric]           │
│      - confidence_percent          [Ryan's metric]             │
│      - normalized_confidence_percent [Ryan's metric]           │
│      - agreement_percent           [Ryan's metric]             │
│      - line_count_variance         [pipeline metric]           │
│                                                                 │
│    Output: consolidated_similarity_scores_{model}.csv           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Integration Points

#### 1. **Main Call Site** (pipeline_5_consistency_check.py:173-177)

```python
def calculate_patch_consistency(self, runs_data: List[Dict], threshold: float = 0.85) -> Dict:
    patches = [run["patch_content"] for run in runs_data if run["patch_content"]]

    # Single function call to Ryan's consistency_metrics()
    agreement_percent, confidence_percent, normalized_confidence, pairwise = consistency_metrics(
        outputs=patches,
        language="python",
        threshold=threshold
    )
```

#### 2. **Extracting Additional Metrics** (pipeline_5_consistency_check.py:180-182)

```python
    # Calculate detailed similarity averages from pairwise data
    avg_ast_similarity = statistics.mean([p["ast_similarity"] for p in pairwise]) if pairwise else 1.0
    avg_text_similarity = statistics.mean([p["text_similarity"] for p in pairwise]) if pairwise else 1.0
    avg_hybrid_similarity = statistics.mean([p["hybrid_similarity"] for p in pairwise]) if pairwise else 1.0
```

**Why calculate averages?** Ryan's `consistency_metrics()` returns pairwise comparisons, but for summary reporting, we need single aggregate scores.

#### 3. **Notebook Extraction** (consolidate_similarity_scores.ipynb, Cell 7)

```python
# Extract patch consistency metrics
patch_consistency = report.get('patch_consistency', {})

record = {
    # Ryan's metrics (direct from consistency_metrics)
    'confidence_percent': patch_consistency.get('confidence_percent'),
    'normalized_confidence_percent': patch_consistency.get('normalized_confidence_percent'),
    'agreement_percent': patch_consistency.get('agreement_percent'),

    # Computed from Ryan's pairwise data
    'avg_ast_similarity': patch_consistency.get('avg_ast_similarity'),
    'avg_text_similarity': patch_consistency.get('avg_text_similarity'),
    'avg_hybrid_similarity': patch_consistency.get('avg_hybrid_similarity'),

    # Pipeline-specific metrics
    'exact_match_rate': patch_consistency.get('exact_match_rate'),
    'num_unique_patches': patch_consistency.get('num_unique_patches'),
    'line_count_variance': patch_consistency.get('line_count_variance'),
}
```

### Metric Sources Summary

| Metric | Source | Calculation Location |
|--------|--------|---------------------|
| **confidence_percent** | 🎯 Ryan's `consistency_metrics()` | `consistency_evaluator/similarity_utils.py:238` |
| **agreement_percent** | 🎯 Ryan's `consistency_metrics()` | `consistency_evaluator/similarity_utils.py:234` |
| **normalized_confidence_percent** | 🎯 Ryan's `consistency_metrics()` | `consistency_evaluator/similarity_utils.py:241-246` |
| **avg_ast_similarity** | 🔧 Computed from Ryan's pairwise data | `pipeline_5_consistency_check.py:180` |
| **avg_text_similarity** | 🔧 Computed from Ryan's pairwise data | `pipeline_5_consistency_check.py:181` |
| **avg_hybrid_similarity** | 🔧 Computed from Ryan's pairwise data | `pipeline_5_consistency_check.py:182` |
| **exact_match_rate** | 📊 Pipeline-specific | `pipeline_5_consistency_check.py:164-166` |
| **num_unique_patches** | 📊 Pipeline-specific | `pipeline_5_consistency_check.py:189` |
| **line_count_variance** | 📊 Pipeline-specific | `pipeline_5_consistency_check.py:169-170` |

**Legend:**
- 🎯 **Ryan's** = Directly returned by `consistency_metrics()`
- 🔧 **Computed** = Calculated from Ryan's pairwise data
- 📊 **Pipeline** = Independently calculated by pipeline code

---

## Quick Reference

| Metric | Range | Description | Best For | Source |
|--------|-------|-------------|----------|--------|
| **confidence_percent** | 0-100 | Main metric: avg similarity × 100 | Quick overview | 🎯 Ryan's |
| **exact_match_rate** | 0.0-1.0 | Byte-identical patches | Detecting determinism | Pipeline |
| **num_unique_patches** | Integer | Count of distinct patches | Understanding diversity | Pipeline |
| **agreement_percent** | 0-100 | % pairs above threshold | Binary yes/no check | 🎯 Ryan's |
| **normalized_confidence_percent** | 0-100 | Rescaled [0.5-1.0]→[0-100] | Human-friendly | 🎯 Ryan's |
| **avg_ast_similarity** | 0.0-1.0 | Structural equivalence | Logic comparison | 🔧 Shared |
| **avg_text_similarity** | 0.0-1.0 | Surface similarity | Style comparison | 🔧 Shared |
| **avg_hybrid_similarity** | 0.0-1.0 | Weighted combo (70%+30%) | Balanced view | 🔧 Shared |
| **line_count_variance** | Float | Size variation | Different approaches | Pipeline |

**Legend**:

- 🎯 **Ryan's** = From `consistency_evaluator/consistency_metrics()`
- 🔧 **Shared** = From `consistency_evaluator/similarity_utils.py` (used by both)
- **Pipeline** = Pipeline-specific metrics

---

## Core Similarity Functions

These functions are imported from `consistency_evaluator/similarity_utils.py`.

### 1. AST Similarity

**Purpose**: Compares code at the **structural level**, ignoring formatting and style.

**How it works**:

1. Parses Python code into Abstract Syntax Tree (AST)
2. Converts AST to normalized string representation
3. Compares using `difflib.SequenceMatcher`

**Use case**: Detecting functionally equivalent code with different formatting.

```python
from consistency_evaluator.similarity_utils import ast_similarity

patch_a = "def foo(x):\n    return x+1"
patch_b = "def foo(x):\n    return x + 1"  # Different spacing

similarity = ast_similarity(patch_a, patch_b)
# Returns: 1.0 (identical structure)
```

**Advantages**:

- ✅ Ignores whitespace, indentation, comments
- ✅ Recognizes semantically equivalent code
- ✅ Focuses on logic, not style

**Limitations**:

- ❌ Only works for Python code
- ❌ May not catch renamed variables (depending on AST dump mode)

---

### 2. Text Similarity

**Purpose**: Compares code at the **surface level**, sensitive to all differences.

**How it works**:

1. Treats code as plain text strings
2. Compares directly using `difflib.SequenceMatcher`

**Use case**: Detecting exact or near-exact matches including formatting.

```python
from consistency_evaluator.similarity_utils import text_similarity

patch_a = "def foo(x):\n    return x+1"
patch_b = "def foo(x):\n    return x + 1"  # Different spacing

similarity = text_similarity(patch_a, patch_b)
# Returns: ~0.975 (high but not perfect)
```

**Advantages**:

- ✅ Works for any language
- ✅ Captures style and formatting differences
- ✅ Simple and fast

**Limitations**:

- ❌ Treats `x+1` and `x + 1` as different
- ❌ Sensitive to trivial changes

---

### 3. Hybrid Similarity

**Purpose**: Combines AST and text similarity for **balanced comparison**.

**Formula**:

```
hybrid = 0.7 × AST_similarity + 0.3 × text_similarity
```

**Why 70/30 weighting?**

- **70% AST**: Prioritizes structural/logical equivalence
- **30% Text**: Accounts for style and formatting

**Use case**: Best general-purpose metric for code comparison.

```python
from consistency_evaluator.similarity_utils import hybrid_similarity

patch_a = "def foo(x):\n    return x+1"
patch_b = "def foo(x):\n    return x + 1"

result = hybrid_similarity(patch_a, patch_b)
# Returns: {'ast': 1.0, 'text': 0.975, 'hybrid': 0.9925}
```

**Interpretation**:

- `hybrid` ≥ 0.95: **Highly similar** (likely same fix)
- `hybrid` 0.80-0.95: **Similar** (related approaches)
- `hybrid` 0.60-0.80: **Somewhat similar** (some overlap)
- `hybrid` < 0.60: **Different** (distinct approaches)

---

## Comprehensive Metrics

### consistency_metrics()

This function (from Ryan's consistency evaluator) provides three key metrics:

#### 1. Agreement Percent (Threshold-Based)

**Formula**:

```
agreement_percent = (pairs_above_threshold / total_pairs) × 100
```

**Default threshold**: 0.85 (85% similarity)

**Interpretation**:

- **100%**: All pairs are highly similar
- **80%**: Most pairs are similar, some outliers
- **50%**: Half similar, half different
- **0%**: No pairs meet the threshold

**Use case**: Binary "consistent or not" assessment.

---

#### 2. Confidence Percent (Raw Average)

**Formula**:

```
confidence_percent = avg(hybrid_similarity) × 100
```

**Range**: 0-100 (but typically 50-100 for Python code)

**Interpretation**:

- **95-100%**: Extremely consistent
- **85-95%**: Highly consistent
- **70-85%**: Moderately consistent
- **50-70%**: Low consistency
- **< 50%**: Very inconsistent

**Use case**: Main metric for consistency reporting.

---

#### 3. Normalized Confidence Percent (Rescaled)

**Formula**:

```
if avg_hybrid ≤ 0.5:
    normalized = 0
elif avg_hybrid ≥ 1.0:
    normalized = 100
else:
    normalized = ((avg_hybrid - 0.5) / 0.5) × 100
```

**Why rescale?**

- Python code rarely has hybrid similarity < 0.5 (due to shared structure)
- Mapping [0.5-1.0] → [0-100] gives better resolution

**Interpretation**:

- **90-100%**: Excellent consistency
- **70-90%**: Good consistency
- **50-70%**: Fair consistency
- **< 50%**: Poor consistency

**Use case**: Human-friendly percentage for reports.

---

## Three-Tier Output Structure

The `calculate_patch_consistency()` function returns metrics organized in three tiers:

### Tier 1: Summary Metrics (Quick Overview)

```json
{
  "confidence_percent": 85.3,      // Main metric (0-100)
  "exact_match_rate": 0.67,        // 67% byte-identical
  "num_unique_patches": 2          // 2 distinct patches
}
```

**Use case**: Dashboard display, quick health check.

---

### Tier 2: Detailed Metrics (Analysis)

```json
{
  "agreement_percent": 80.0,              // 80% above threshold
  "normalized_confidence_percent": 70.6,  // Rescaled confidence
  "avg_ast_similarity": 0.89,             // Structural consistency
  "avg_text_similarity": 0.78,            // Style consistency
  "avg_hybrid_similarity": 0.853,         // Balanced consistency
  "line_count_variance": 2.3              // Size variation
}
```

**Use case**: Detailed reports, debugging inconsistencies.

---

### Tier 3: Raw Data (Debugging)

```json
{
  "pairwise_comparisons": [
    {
      "i": 0,
      "j": 1,
      "ast_similarity": 0.92,
      "text_similarity": 0.85,
      "hybrid_similarity": 0.899
    },
    // ... more pairs
  ]
}
```

**Use case**: Investigating specific pairs, manual analysis.

---

## Interpreting Results

### Scenario 1: Perfect Consistency

```json
{
  "confidence_percent": 100.0,
  "exact_match_rate": 1.0,
  "num_unique_patches": 1,
  "agreement_percent": 100.0,
  "avg_ast_similarity": 1.0,
  "avg_text_similarity": 1.0
}
```

**Interpretation**: Model is **deterministic** - produces identical patches.

**Implications**:

- ✅ Highly reliable
- ✅ Reproducible results
- ℹ️ May lack creativity/exploration

---

### Scenario 2: High Consistency (Different Formatting)

```json
{
  "confidence_percent": 92.5,
  "exact_match_rate": 0.0,         // Not byte-identical
  "num_unique_patches": 3,
  "agreement_percent": 100.0,       // All above threshold
  "avg_ast_similarity": 0.98,      // Structurally same
  "avg_text_similarity": 0.85      // Formatting differs
}
```

**Interpretation**: Functionally **equivalent solutions** with style variations.

**Implications**:

- ✅ Consistent logic
- ✅ Good reliability
- ℹ️ Style is non-deterministic

---

### Scenario 3: Moderate Consistency (Multiple Approaches)

```json
{
  "confidence_percent": 75.0,
  "exact_match_rate": 0.0,
  "num_unique_patches": 3,
  "agreement_percent": 60.0,        // Some pairs similar
  "avg_ast_similarity": 0.72,
  "avg_text_similarity": 0.78
}
```

**Interpretation**: Model explores **multiple valid approaches**.

**Implications**:

- ⚠️ Moderate reliability
- ✅ Creative exploration
- ℹ️ May need majority voting

---

### Scenario 4: Low Consistency (Unstable)

```json
{
  "confidence_percent": 55.0,
  "exact_match_rate": 0.0,
  "num_unique_patches": 5,
  "agreement_percent": 20.0,        // Few pairs similar
  "avg_ast_similarity": 0.52,
  "avg_text_similarity": 0.58,
  "line_count_variance": 45.2       // Very different sizes
}
```

**Interpretation**: Model is **unstable** - produces inconsistent solutions.

**Implications**:

- ❌ Low reliability
- ❌ Unpredictable results
- ⚠️ Requires investigation

---

## Examples

### Example 1: Three Similar Patches

```python
patches = [
    "def calculate_sum(a, b):\n    return a+b",
    "def calculate_sum(a, b):\n    return a + b",
    "def calculate_sum(a, b):\n    return a  +  b"
]
```

**Expected Output**:

```json
{
  "confidence_percent": 99.2,
  "exact_match_rate": 0.0,
  "num_unique_patches": 3,
  "agreement_percent": 100.0,
  "avg_ast_similarity": 1.0,
  "avg_text_similarity": 0.976,
  "avg_hybrid_similarity": 0.992
}
```

**Interpretation**: AST recognizes identical logic despite spacing differences.

---

### Example 2: Two Different Approaches

```python
patches = [
    "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)",
    "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
    "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
]
```

**Expected Output**:

```json
{
  "confidence_percent": 75.5,
  "exact_match_rate": 0.67,  // 2 out of 3 identical
  "num_unique_patches": 2,
  "agreement_percent": 66.67,  // 2 out of 3 pairs similar
  "avg_ast_similarity": 0.65,
  "avg_text_similarity": 0.70,
  "avg_hybrid_similarity": 0.755
}
```

**Interpretation**: Model explores two approaches (recursive vs iterative).

---

## Best Practices

### 1. Use confidence_percent as Primary Metric

```python
if patch_metrics["confidence_percent"] >= 90:
    grade = "EXCELLENT"
elif patch_metrics["confidence_percent"] >= 70:
    grade = "GOOD"
else:
    grade = "NEEDS_REVIEW"
```

### 2. Check AST vs Text for Diagnosis

```python
ast_sim = patch_metrics["avg_ast_similarity"]
text_sim = patch_metrics["avg_text_similarity"]

if ast_sim > 0.9 and text_sim < 0.8:
    print("Consistent logic, varying style")
elif ast_sim < 0.7:
    print("Different approaches")
```

### 3. Use Pairwise Data for Deep Dive

```python
for pair in patch_metrics["pairwise_comparisons"]:
    if pair["hybrid_similarity"] < 0.7:
        print(f"Outlier: patch {pair['i']} vs {pair['j']}")
```

---

## Sample output

**Requires more than 1 run** per test instance

Will be generated in `output/{model}/{timestamp}/{instance_id}/consistency_summary.json`

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
        "ast_similarity": 0.9100942266645843,
        "text_similarity": 0.9140007898894155,
        "hybrid_similarity": 0.9112661956320336
      },
      {
        "i": 0,
        "j": 2,
        "ast_similarity": 0.06649729371479068,
        "text_similarity": 0.12635196603659152,
        "hybrid_similarity": 0.08445369541133094
      },
      {
        "i": 1,
        "j": 2,
        "ast_similarity": 0.0559583565718535,
        "text_similarity": 0.10763799190562301,
        "hybrid_similarity": 0.07146224717198435
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

---

## Concrete Example: Number Flow Through the System

Let's trace actual values through the entire pipeline to show how Ryan's functions produce the final CSV numbers.

### Example Input: 3 Patches for django__django-10914

```python
# Input to pipeline (3 runs of same instance)
patches = [
    "patch_run_0.diff",  # 250 lines
    "patch_run_1.diff",  # 245 lines
    "patch_run_2.diff",  # 15 lines (very different!)
]
```

### Step 1: Ryan's consistency_metrics() Is Called

```python
# pipeline_5_consistency_check.py:173
agreement_percent, confidence_percent, normalized_confidence, pairwise = consistency_metrics(
    outputs=patches,
    language="python",
    threshold=0.85
)
```

### Step 2: Ryan's Function Internally Computes Pairwise Comparisons

```python
# Inside consistency_metrics() (similarity_utils.py:213-231)
pairs = [(0,1), (0,2), (1,2)]  # All combinations

# Pair 0-1: Similar patches
sims_01 = hybrid_similarity(patches[0], patches[1])
# Returns: {"ast": 0.9101, "text": 0.9140, "hybrid": 0.9113}

# Pair 0-2: Very different patches
sims_02 = hybrid_similarity(patches[0], patches[2])
# Returns: {"ast": 0.0665, "text": 0.1264, "hybrid": 0.0845}

# Pair 1-2: Very different patches
sims_12 = hybrid_similarity(patches[1], patches[2])
# Returns: {"ast": 0.0560, "text": 0.1076, "hybrid": 0.0715}
```

### Step 3: Ryan's Function Calculates & Returns Metrics

```python
# similarity_utils.py:234-246
hybrids = [0.9113, 0.0845, 0.0715]

# Metric 1: Agreement percent (threshold = 0.85)
consistent = 1  # Only pair 0-1 is >= 0.85
agreement_percent = (1 / 3) × 100 = 33.33

# Metric 2: Confidence percent (raw average)
avg_hybrid = (0.9113 + 0.0845 + 0.0715) / 3 = 0.3558
confidence_percent = 0.3558 × 100 = 35.58

# Metric 3: Normalized confidence
# avg_hybrid (0.3558) < 0.5, so:
normalized_confidence = 0.0

# Metric 4: Pairwise details
pairwise = [
    {"i": 0, "j": 1, "ast_similarity": 0.9101, "text_similarity": 0.9140, "hybrid_similarity": 0.9113},
    {"i": 0, "j": 2, "ast_similarity": 0.0665, "text_similarity": 0.1264, "hybrid_similarity": 0.0845},
    {"i": 1, "j": 2, "ast_similarity": 0.0560, "text_similarity": 0.1076, "hybrid_similarity": 0.0715}
]

return 33.33, 35.58, 0.0, pairwise
```

### Step 4: Pipeline Extracts Averages from Pairwise Data

```python
# pipeline_5_consistency_check.py:180-182
avg_ast_similarity = mean([0.9101, 0.0665, 0.0560]) = 0.3442
avg_text_similarity = mean([0.9140, 0.1264, 0.1076]) = 0.3827
avg_hybrid_similarity = mean([0.9113, 0.0845, 0.0715]) = 0.3558
```

### Step 5: Pipeline Adds Its Own Metrics

```python
# pipeline_5_consistency_check.py:164-170
# Metric: Exact match rate
patch_counter = Counter(patches)
# Result: {"patch_0": 1, "patch_1": 1, "patch_2": 1}
most_common_count = 1
exact_match_rate = 1 / 3 = 0.3333

# Metric: Number of unique patches
num_unique_patches = len(patch_counter) = 3

# Metric: Line count variance
line_counts = [250, 245, 15]
line_count_variance = variance([250, 245, 15]) = 32601.33
```

### Step 6: Pipeline Returns Combined Dictionary

```python
# pipeline_5_consistency_check.py:185-201
return {
    # Tier 1: Summary
    "confidence_percent": 35.57,           # Ryan's (rounded)
    "exact_match_rate": 0.3333,            # Pipeline
    "num_unique_patches": 3,               # Pipeline

    # Tier 2: Detailed
    "agreement_percent": 33.33,            # Ryan's
    "normalized_confidence_percent": 0.0,  # Ryan's
    "avg_ast_similarity": 0.3442,          # From Ryan's pairwise
    "avg_text_similarity": 0.3827,         # From Ryan's pairwise
    "avg_hybrid_similarity": 0.3557,       # From Ryan's pairwise
    "line_count_variance": 32601.33,       # Pipeline

    # Tier 3: Raw
    "pairwise_comparisons": [...]          # Ryan's raw data
}
```

### Step 7: Saved to consistency_report.json

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
    "pairwise_comparisons": [...]
  }
}
```

### Step 8: Notebook Extracts to CSV

```python
# consolidate_similarity_scores.ipynb, Cell 7
record = {
    'instance_id': 'django__django-10914',
    'avg_ast_similarity': 0.3442,           # ← From Ryan's pairwise
    'avg_text_similarity': 0.3827,          # ← From Ryan's pairwise
    'avg_hybrid_similarity': 0.3557,        # ← From Ryan's pairwise
    'confidence_percent': 35.57,            # ← Ryan's metric
    'normalized_confidence_percent': 0.0,   # ← Ryan's metric
    'agreement_percent': 33.33,             # ← Ryan's metric
    'exact_match_rate': 0.3333,             # ← Pipeline metric
    'num_unique_patches': 3,                # ← Pipeline metric
    'line_count_variance': 32601.33,        # ← Pipeline metric
}
```

### Final CSV Row

```csv
instance_id,avg_ast_similarity,avg_text_similarity,avg_hybrid_similarity,confidence_percent,agreement_percent,exact_match_rate,num_unique_patches,line_count_variance
django__django-10914,0.3442,0.3827,0.3557,35.57,33.33,0.3333,3,32601.33
```

### Summary: Which Numbers Come From Where?

**From Ryan's `consistency_metrics()` directly:**
- `confidence_percent` = 35.57 (avg hybrid × 100)
- `agreement_percent` = 33.33 (% pairs ≥ threshold)
- `normalized_confidence_percent` = 0.0 (rescaled)

**From Ryan's pairwise data (averaged in pipeline):**
- `avg_ast_similarity` = 0.3442
- `avg_text_similarity` = 0.3827
- `avg_hybrid_similarity` = 0.3557

**From pipeline calculations:**
- `exact_match_rate` = 0.3333
- `num_unique_patches` = 3
- `line_count_variance` = 32601.33

**Key Insight**: The three "avg_*_similarity" metrics are **NOT** directly returned by `consistency_metrics()`. Instead, the pipeline:
1. Receives the pairwise comparisons from Ryan's function
2. Extracts the individual similarity scores
3. Computes the averages

This is why you see the averaging logic at pipeline_5_consistency_check.py:180-182.

---

## Composite Scores: How They Add Up

The notebook extracts **composite scores** that combine multiple metrics into overall assessments. These are calculated in `pipeline_5_consistency_check.py` lines 298-372.

### Overview: Three Component Scores → One Overall Score

```
overall_consistency_score = (patch_score × 40%) + (evaluation_score × 40%) + (security_score × 20%)
```

**Weights:**
- **Patch consistency**: 40% (most important - measures solution quality)
- **Evaluation consistency**: 40% (important - measures test pass/fail consistency)
- **Security consistency**: 20% (less critical - measures security risk consistency)

### Component Score 1: Patch Score (40% weight)

**Purpose**: Measures consistency of generated patch content

**Calculation** (pipeline_5_consistency_check.py:320-323):

```python
exact_match_percent = exact_match_rate × 100
patch_score = (exact_match_percent × 50%) + (confidence_percent × 50%)
```

**Components:**
- `exact_match_rate` (0.0-1.0) → converted to 0-100%, weighted 50%
- `confidence_percent` (0-100) → weighted 50%

**Example:**
```python
exact_match_rate = 0.3333  # 33.33% byte-identical matches
confidence_percent = 35.57  # 35.57% average hybrid similarity

patch_score = (33.33 × 0.5) + (35.57 × 0.5)
            = 16.665 + 17.785
            = 34.45
```

**Why blend both?**
- `exact_match_rate`: Rewards determinism (identical patches)
- `confidence_percent`: Rewards semantic similarity (similar but not identical)
- 50/50 balance: Neither perfect determinism nor pure similarity dominates

### Component Score 2: Evaluation Score (40% weight)

**Purpose**: Measures consistency of test evaluation results (pass/fail)

**Calculation** (pipeline_5_consistency_check.py:325-334):

```python
if all_resolved OR all_failed:
    eval_score = 100.0  # Perfect consistency
else:
    # Penalize inconsistency
    deviation_from_consensus = abs(resolution_rate - 0.5)
    eval_score = deviation_from_consensus × 200
```

**Logic:**
- **All resolved (100% pass)**: `eval_score = 100` ✅ Consistent success
- **All failed (0% pass)**: `eval_score = 100` ✅ Consistent failure
- **Mixed results**: Score based on how far from 50/50 split
  - `resolution_rate = 0.5` (50/50): `eval_score = 0` (worst case - unpredictable)
  - `resolution_rate = 0.75` (75/25): `eval_score = 50` (moderate consistency)
  - `resolution_rate = 1.0` (100/0): `eval_score = 100` (perfect consistency)

**Example 1: All tests passed**
```python
all_resolved = True
all_failed = False
resolution_rate = 1.0  # 3/3 runs passed

eval_score = 100.0  # Perfect consistency
```

**Example 2: Mixed results**
```python
all_resolved = False
all_failed = False
resolution_rate = 0.67  # 2/3 runs passed

deviation_from_consensus = abs(0.67 - 0.5) = 0.17
eval_score = 0.17 × 200 = 34.0  # Low consistency
```

### Component Score 3: Security Score (20% weight)

**Purpose**: Measures consistency of security risk assessments

**Calculation** (pipeline_5_consistency_check.py:336-345):

```python
if num_scans > 1:
    max_variance = 1.0
    normalized_variance = min(score_variance, max_variance) / max_variance
    security_score = (1 - normalized_variance) × 100
else:
    security_score = 100.0  # Perfect if only one scan
```

**Logic:**
- Lower variance = higher consistency = higher score
- Variance capped at 1.0 (scores > 1.0 treated as maximally inconsistent)

**Example 1: Identical security scores**
```python
security_scores = [0.0, 0.0, 0.0]  # All "NONE" risk
score_variance = 0.0

normalized_variance = min(0.0, 1.0) / 1.0 = 0.0
security_score = (1 - 0.0) × 100 = 100.0  # Perfect consistency
```

**Example 2: Varying security scores**
```python
security_scores = [0.2, 0.5, 0.8]  # LOW, MEDIUM, HIGH risk
score_variance = 0.09

normalized_variance = min(0.09, 1.0) / 1.0 = 0.09
security_score = (1 - 0.09) × 100 = 91.0  # High consistency
```

### Overall Consistency Score Calculation

**Formula** (pipeline_5_consistency_check.py:348-352):

```python
overall_score = (patch_score × 0.4) + (eval_score × 0.4) + (security_score × 0.2)
```

**Concrete Example: django__django-10914**

```python
# Component scores
patch_score = 34.45      # Low patch consistency (many different solutions)
eval_score = 100.0       # Perfect eval consistency (all tests passed)
security_score = 100.0   # Perfect security consistency (all "NONE" risk)

# Weighted calculation
overall_score = (34.45 × 0.4) + (100.0 × 0.4) + (100.0 × 0.2)
              = 13.78 + 40.0 + 20.0
              = 73.78
```

**Grade Assignment** (pipeline_5_consistency_check.py:355-362):

```python
if overall_score >= 90:  grade = "EXCELLENT"
elif overall_score >= 70: grade = "GOOD"      # ← 73.78 falls here
elif overall_score >= 50: grade = "FAIR"
else:                     grade = "POOR"
```

**Result**: `overall_consistency_score = 73.78`, `consistency_grade = "GOOD"`

### Complete Breakdown in JSON

```json
{
  "instance_id": "django__django-10914",
  "overall_consistency_score": 73.78,
  "consistency_grade": "GOOD",
  "component_scores": {
    "patch_score": 34.45,        // 40% weight → contributes 13.78
    "evaluation_score": 100.0,   // 40% weight → contributes 40.0
    "security_score": 100.0      // 20% weight → contributes 20.0
  }
}
```

### Complete Breakdown in CSV

```csv
instance_id,overall_consistency_score,consistency_grade,patch_score,evaluation_score,security_score
django__django-10914,73.78,GOOD,34.45,100.0,100.0
```

### How Notebook Extracts Composite Scores

**Cell 7 of consolidate_similarity_scores.ipynb:**

```python
# Extract from consistency report
record = {
    # Composite scores
    'overall_consistency_score': report.get('overall_consistency_score'),
    'consistency_grade': report.get('consistency_grade'),

    # Component scores
    'patch_score': report.get('component_scores', {}).get('patch_score'),
    'evaluation_score': report.get('component_scores', {}).get('evaluation_score'),
    'security_score': report.get('component_scores', {}).get('security_score'),
}
```

### Interpretation Guide

| Overall Score | Grade | Meaning |
|--------------|-------|---------|
| **90-100** | EXCELLENT | Highly consistent across all dimensions |
| **70-89** | GOOD | Generally consistent, some variation acceptable |
| **50-69** | FAIR | Moderate consistency issues, investigate further |
| **0-49** | POOR | High inconsistency, model reliability concerns |

**Key Insight**: The overall score tells you if the model is **reliably producing consistent behavior**, even if the patches differ:
- High eval score (all pass or all fail) = consistent correctness/incorrectness
- High patch score + high eval score = model produces similar solutions that consistently work
- Low patch score + high eval score = model explores different solutions, but they all work (creative but reliable)
- Low eval score = unpredictable (sometimes works, sometimes doesn't) ⚠️

### Where Each Composite Score Is Calculated

| Score | Calculation Location | Method |
|-------|---------------------|---------|
| **patch_score** | pipeline_5_consistency_check.py:320-323 | Blend of exact_match_rate and confidence_percent |
| **evaluation_score** | pipeline_5_consistency_check.py:325-334 | Based on resolution_rate consistency |
| **security_score** | pipeline_5_consistency_check.py:336-345 | Inverse of normalized variance |
| **overall_consistency_score** | pipeline_5_consistency_check.py:348-352 | Weighted average (40% + 40% + 20%) |
| **consistency_grade** | pipeline_5_consistency_check.py:355-362 | Threshold-based grading |

### How Ryan's Similarity Metrics Flow Into Overall Score

**Important**: The `avg_ast_similarity`, `avg_text_similarity`, and `avg_hybrid_similarity` metrics are extracted to the CSV for **analysis purposes**, but only the hybrid similarity (via `confidence_percent`) is used in the overall score calculation.

**The calculation chain:**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Pairwise Similarity (Ryan's code)                      │
│ For each pair of patches:                                       │
│   - AST similarity: 0.9101, 0.0665, 0.0560                     │
│   - Text similarity: 0.9140, 0.1264, 0.1076                    │
│   - Hybrid = 70% AST + 30% text                                │
│   - Hybrid similarity: 0.9113, 0.0845, 0.0715                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Average Hybrid (Ryan's consistency_metrics)            │
│ avg_hybrid = mean([0.9113, 0.0845, 0.0715]) = 0.3558          │
│ confidence_percent = 0.3558 × 100 = 35.58                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Patch Score (Pipeline)                                 │
│ patch_score = (exact_match × 50%) + (confidence_percent × 50%) │
│ patch_score = (33.33 × 0.5) + (35.58 × 0.5) = 34.45           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Overall Score (Pipeline)                               │
│ overall = (patch × 40%) + (eval × 40%) + (security × 20%)     │
│ overall = (34.45 × 0.4) + (100 × 0.4) + (100 × 0.2) = 73.78  │
└─────────────────────────────────────────────────────────────────┘
```

**Contribution breakdown to overall_consistency_score:**

| Metric | Weight Chain | Total Contribution |
|--------|--------------|-------------------|
| **AST similarity** | 70% (hybrid) × 50% (patch) × 40% (overall) | **14%** |
| **Text similarity** | 30% (hybrid) × 50% (patch) × 40% (overall) | **6%** |
| **Exact match** | 50% (patch) × 40% (overall) | **20%** |
| **Evaluation consistency** | 40% (overall) - direct | **40%** |
| **Security consistency** | 20% (overall) - direct | **20%** |
| **TOTAL** | | **100%** ✅ |

**Key Insight**: While `avg_ast_similarity`, `avg_text_similarity`, and `avg_hybrid_similarity` are all extracted to the CSV, only the **hybrid similarity** (through `confidence_percent`) actually contributes to the `overall_consistency_score`.

### Why Extract Three Separate Metrics?

The individual AST and text metrics are provided for **diagnostic analysis** even though they don't directly affect scoring. Here's why:

**Purpose of each metric:**

1. **`avg_ast_similarity`** - Diagnostic only
   - Shows **structural/logic consistency**
   - Reveals if the model produces functionally equivalent code
   - Helps identify: "Same algorithm, different style"

2. **`avg_text_similarity`** - Diagnostic only
   - Shows **formatting/style consistency**
   - Reveals surface-level differences
   - Helps identify: "Similar-looking code, different logic"

3. **`avg_hybrid_similarity`** - Used in scoring (via confidence_percent)
   - **Actually contributes to overall_consistency_score**
   - Balanced view: 70% structure + 30% style
   - This is what Ryan's `consistency_metrics()` calculates

**Why not just use hybrid?** Because the breakdown helps you understand **what type of inconsistency** you're seeing:

| Scenario | AST | Text | Hybrid | Interpretation |
|----------|-----|------|--------|----------------|
| **Deterministic** | 1.0 | 1.0 | 1.0 | Identical patches (perfect) |
| **Same logic, different style** | 0.95 | 0.75 | 0.89 | Functionally equivalent, formatting varies |
| **Different approaches** | 0.60 | 0.55 | 0.585 | Multiple valid solutions explored |
| **Unpredictable** | 0.30 | 0.35 | 0.315 | Inconsistent behavior (concerning) |

**Example interpretation:**
```python
# Instance with high AST, low text
avg_ast_similarity = 0.95    # Not directly in overall score - diagnostic
avg_text_similarity = 0.75   # Not directly in overall score - diagnostic
avg_hybrid_similarity = 0.89 # Used in scoring: 0.7×0.95 + 0.3×0.75 = 0.89
confidence_percent = 89.0    # This feeds into patch_score → overall_score

# Diagnosis: Model is consistent in logic (high AST) but varies in formatting (lower text)
# This is GOOD - shows reliable algorithm with minor stylistic differences
```

**Practical use case:**
```python
# Load CSV
df = pd.read_csv('consolidated_similarity_scores.csv')

# Find instances with inconsistent logic (low AST)
logic_issues = df[df['avg_ast_similarity'] < 0.7]
print(f"Found {len(logic_issues)} instances with logic inconsistency")

# Find instances with style variations but consistent logic
style_variations = df[(df['avg_ast_similarity'] > 0.9) & (df['avg_text_similarity'] < 0.8)]
print(f"Found {len(style_variations)} instances with style variations only")
```

**Bottom line**: The three metrics tell a complete story, even though only hybrid affects the final grade.

---

## Related Files

- `consistency_evaluator/similarity_utils.py` - Core similarity functions (Ryan's code)
- `consistency_evaluator/main.py` - API service using these metrics
- `pipeline_5_consistency_check.py` - Pipeline implementation (your code)
- `consolidate_similarity_scores.ipynb` - CSV extraction notebook (your code)

---

## References

- [difflib documentation](https://docs.python.org/3/library/difflib.html)
- [AST module documentation](https://docs.python.org/3/library/ast.html)
- Ryan's consistency evaluator: `consistency_evaluator/README.md`

---

Last updated: 2025-10-30
