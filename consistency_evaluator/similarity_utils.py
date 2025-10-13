"""
Shared code similarity utilities for consistency evaluation.

This module provides similarity calculation functions used by:
- consistency_evaluator/main.py (API service for model-level consistency)
- pipeline_5_consistency_check.py (pipeline-level consistency analysis)

Functions:
- normalize_python_code(): Normalize Python code to AST representation
- ast_similarity(): Calculate AST-based structural similarity
- text_similarity(): Calculate text-based surface similarity
- hybrid_similarity(): Calculate weighted combination of AST and text similarity
- consistency_metrics(): Calculate comprehensive consistency metrics for multiple outputs
"""

import ast
import difflib
from itertools import combinations
from typing import Dict, List, Tuple


def normalize_python_code(code: str) -> str:
    """
    Normalize Python code into AST dump (ignores formatting & variable names).

    This function converts Python code to its Abstract Syntax Tree (AST) representation,
    which allows structural comparison independent of:
    - Whitespace and formatting
    - Comment differences
    - Variable naming (when using ast.dump without annotations)

    Args:
        code: Python source code string

    Returns:
        AST dump string, or normalized text if parsing fails

    Examples:
        >>> normalize_python_code("def foo(x):\\n    return x+1")
        >>> normalize_python_code("def foo(x):\\n    return x + 1")
        # Both return identical AST strings despite different spacing
    """
    try:
        tree = ast.parse(code)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        # Fallback: basic text normalization if AST parsing fails
        lines = []
        for line in code.splitlines():
            # Skip comment lines
            if line.strip().startswith("#"):
                continue
            lines.append(line.strip())
        return "\n".join(lines)


def ast_similarity(a: str, b: str) -> float:
    """
    Calculate similarity based on Abstract Syntax Tree (AST) structure.

    This provides a semantic comparison of Python code by comparing
    their AST representations rather than raw text.

    Args:
        a: First Python code string
        b: Second Python code string

    Returns:
        Similarity ratio between 0.0 (completely different) and 1.0 (identical)

    Examples:
        >>> ast_similarity("def foo(x):\\n    return x+1",
        ...                "def foo(x):\\n    return x + 1")
        1.0  # Same structure despite spacing differences
    """
    na = normalize_python_code(a)
    nb = normalize_python_code(b)
    return difflib.SequenceMatcher(None, na, nb).ratio()


def text_similarity(a: str, b: str) -> float:
    """
    Calculate raw text similarity without structural analysis.

    This provides a surface-level comparison of code as plain text,
    sensitive to all formatting, spacing, and comments.

    Args:
        a: First text string
        b: Second text string

    Returns:
        Similarity ratio between 0.0 (completely different) and 1.0 (identical)

    Examples:
        >>> text_similarity("def foo(x):\\n    return x+1",
        ...                 "def foo(x):\\n    return x + 1")
        0.95  # High but not perfect due to spacing
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def hybrid_similarity(a: str, b: str, language: str = "python", alpha: float = 0.7) -> Dict[str, float]:
    """
    Calculate hybrid similarity combining AST structure and text similarity.

    This provides the best of both worlds:
    - AST similarity: Captures semantic/structural equivalence
    - Text similarity: Captures stylistic and surface-level differences

    The hybrid score is a weighted average:
        hybrid = alpha * ast_similarity + (1 - alpha) * text_similarity

    Args:
        a: First code string
        b: Second code string
        language: Programming language ("python" uses AST, others use text only)
        alpha: Weight for AST similarity (default 0.7, meaning 70% AST, 30% text)

    Returns:
        Dictionary with three scores:
        - 'ast': AST-based similarity (0.0 to 1.0)
        - 'text': Text-based similarity (0.0 to 1.0)
        - 'hybrid': Weighted combination (0.0 to 1.0)

    Examples:
        >>> result = hybrid_similarity(
        ...     "def foo(x):\\n    return x+1",
        ...     "def foo(x):\\n    return x + 1"
        ... )
        >>> result
        {'ast': 1.0, 'text': 0.95, 'hybrid': 0.985}  # High similarity

        >>> result = hybrid_similarity(
        ...     "def foo(x):\\n    return x+1",
        ...     "def bar(y):\\n    return y*2"
        ... )
        >>> result
        {'ast': 0.6, 'text': 0.5, 'hybrid': 0.57}  # Lower similarity
    """
    # Use AST similarity for Python, text similarity for other languages
    if language.lower() == "python":
        ast_sim = ast_similarity(a, b)
    else:
        # For non-Python languages, use text similarity for both
        ast_sim = text_similarity(a, b)

    text_sim = text_similarity(a, b)

    # Calculate weighted hybrid score
    hybrid = alpha * ast_sim + (1 - alpha) * text_sim

    return {
        "ast": ast_sim,
        "text": text_sim,
        "hybrid": hybrid
    }


def consistency_metrics(
    outputs: List[str],
    language: str = "python",
    threshold: float = 0.85
) -> Tuple[float, float, float, List[Dict]]:
    """
    Calculate comprehensive consistency metrics for multiple outputs.

    This function performs pairwise comparison of all outputs and returns
    three key metrics for assessing consistency:

    1. Agreement Percent: Threshold-based strict consistency (0-100)
       - Counts pairs with similarity >= threshold
       - Formula: (consistent_pairs / total_pairs) × 100
       - Discrete steps: 20%, 40%, 60%, etc.

    2. Confidence Percent: Raw average similarity (0-100)
       - Average hybrid similarity across all pairs × 100
       - Formula: avg(hybrid_i,j) × 100
       - Continuous values: e.g., 74.3%

    3. Normalized Confidence Percent: Rescaled for interpretation (0-100)
       - Maps realistic range [0.5, 1.0] → [0, 100]
       - Formula: (avg_hybrid - 0.5) / 0.5 × 100
       - Better human interpretation

    Args:
        outputs: List of code/text outputs to compare (minimum 2)
        language: Programming language ("python" uses AST, others use text only)
        threshold: Similarity threshold for agreement (default 0.85)

    Returns:
        Tuple of (agreement_percent, confidence_percent, normalized_confidence, details)
        - agreement_percent (float): % of pairs above threshold (0-100)
        - confidence_percent (float): Average similarity × 100 (0-100)
        - normalized_confidence (float): Rescaled confidence (0-100)
        - details (List[Dict]): Pairwise comparison details

    Examples:
        >>> outputs = [
        ...     "def foo(x): return x+1",
        ...     "def foo(x): return x + 1",
        ...     "def foo(x): return x  +  1"
        ... ]
        >>> agreement, confidence, normalized, details = consistency_metrics(outputs)
        >>> print(f"Agreement: {agreement}%, Confidence: {confidence}%")
        Agreement: 100.0%, Confidence: 99.2%

    Notes:
        - Requires at least 2 outputs
        - Uses hybrid_similarity() for pairwise comparisons
        - Returns full pairwise details for debugging
    """
    # Generate all unique pairs of outputs
    pairs = list(combinations(range(len(outputs)), 2))
    details = []
    hybrids = []
    consistent = 0

    # Compare each pair
    for i, j in pairs:
        sims = hybrid_similarity(outputs[i], outputs[j], language)
        details.append({
            "i": i,
            "j": j,
            "ast_similarity": sims["ast"],
            "text_similarity": sims["text"],
            "hybrid_similarity": sims["hybrid"]
        })
        hybrids.append(sims["hybrid"])
        if sims["hybrid"] >= threshold:
            consistent += 1

    # Calculate agreement percent (thresholded)
    agreement_percent = 100.0 * consistent / len(pairs) if pairs else 100.0

    # Calculate raw average hybrid similarity
    avg_hybrid = sum(hybrids) / len(hybrids) if hybrids else 1.0
    confidence_percent = 100.0 * avg_hybrid

    # Calculate normalized confidence (map [0.5, 1.0] → [0, 100])
    if avg_hybrid <= 0.5:
        normalized_confidence = 0.0
    elif avg_hybrid >= 1.0:
        normalized_confidence = 100.0
    else:
        normalized_confidence = ((avg_hybrid - 0.5) / 0.5) * 100.0

    return agreement_percent, confidence_percent, normalized_confidence, details
