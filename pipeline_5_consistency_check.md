# Pipeline Stage 5: Consistency Check - Similarity Metrics Documentation

## Overview

This document explains the similarity metrics used in `pipeline_5_consistency_check.py` to assess consistency across multiple runs of the same SWE-bench instance.

The consistency checker uses a **three-tier approach** combining:

1. **Pipeline-specific metrics** (exact match, line variance, unique patches)
2. **Ryan's consistency_metrics()** (agreement%, confidence%, normalized confidence%)
3. **Detailed similarity breakdowns** (AST, text, hybrid)

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Core Similarity Functions](#core-similarity-functions)
- [Comprehensive Metrics](#comprehensive-metrics)
- [Three-Tier Output Structure](#three-tier-output-structure)
- [Interpreting Results](#interpreting-results)
- [Examples](#examples)

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

## Related Files

- `consistency_evaluator/similarity_utils.py` - Core similarity functions
- `consistency_evaluator/main.py` - API service using these metrics
- `pipeline_5_consistency_check.py` - Pipeline implementation

---

## References

- [difflib documentation](https://docs.python.org/3/library/difflib.html)
- [AST module documentation](https://docs.python.org/3/library/ast.html)
- Ryan's consistency evaluator: `consistency_evaluator/README.md`

---

Last updated: 2025-10-13
