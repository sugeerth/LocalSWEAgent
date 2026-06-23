"""Unit tests for the pure scoring/aggregation logic in eval.runner.

eval.runner imports `datasets` at module load; if that dependency is not
installed the whole module is skipped rather than failing the suite.
"""

import pytest

runner = pytest.importorskip("eval.runner")


# ---- _compute_patch_similarity ------------------------------------------------

def test_similarity_identical_patches_is_one():
    patch = "+added line\n-removed line"
    assert runner._compute_patch_similarity(patch, patch) == 1.0


def test_similarity_empty_inputs_are_zero():
    assert runner._compute_patch_similarity("", "+x") == 0.0
    assert runner._compute_patch_similarity("+x", "") == 0.0


def test_similarity_ignores_diff_header_lines():
    # +++/--- header lines must NOT be counted as changes.
    model = "+++ b/f.py\n--- a/f.py\n+real change"
    gold = "+real change"
    assert runner._compute_patch_similarity(model, gold) == 1.0


def test_similarity_is_jaccard():
    # model changes {a, b}; gold changes {a, c} -> intersection 1, union 3.
    model = "+a\n-b"
    gold = "+a\n-c"
    assert runner._compute_patch_similarity(model, gold) == pytest.approx(1 / 3)


# ---- _compute_summary ---------------------------------------------------------

def test_summary_empty_results_is_safe():
    summary = runner._compute_summary([], "m", 0.0)
    assert summary["total_instances"] == 0
    assert summary["valid_rate"] == 0
    assert summary["avg_similarity"] == 0
    assert summary["estimated_resolve_rate"] == 0


def test_summary_counts_valid_and_high_similarity():
    results = [
        {"valid_patch": True, "patch_similarity": 0.9},   # valid + high
        {"valid_patch": False, "patch_similarity": 0.5},  # high only
        {"valid_patch": True, "patch_similarity": 0.1},   # valid only
        {"valid_patch": False, "patch_similarity": 0.0},
    ]
    summary = runner._compute_summary(results, "m", 10.0)
    assert summary["valid_patches"] == 2
    assert summary["valid_rate"] == pytest.approx(0.5)
    # 0.9 and 0.5 exceed the 0.3 high-similarity threshold.
    assert summary["high_similarity_count"] == 2
    assert summary["estimated_resolve_rate"] == pytest.approx(0.5)


def test_summary_threshold_is_strict_greater_than():
    # Exactly at the threshold should NOT count as high similarity.
    results = [{"valid_patch": False,
                "patch_similarity": runner.HIGH_SIMILARITY_THRESHOLD}]
    summary = runner._compute_summary(results, "m", 1.0)
    assert summary["high_similarity_count"] == 0
