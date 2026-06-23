"""Unit tests for the shared pure helpers in agent.repo_utils.

These cover the patch-building and line-ranking logic that the Turbo and
Precision agents both delegate to. They are dependency-light (stdlib only) and
do not touch Ollama, git, or the network.
"""

import difflib

from agent import repo_utils


# ---- make_diff ----------------------------------------------------------------

def test_make_diff_produces_unified_diff_with_ab_paths():
    old = ["x = 1", "y = 2", "z = 3"]
    new = ["x = 1", "y = 99", "z = 3"]
    diff = repo_utils.make_diff("pkg/mod.py", old, new)

    assert diff.startswith("--- a/pkg/mod.py\n")
    assert "+++ b/pkg/mod.py\n" in diff
    assert "-y = 2\n" in diff
    assert "+y = 99\n" in diff


def test_make_diff_empty_when_no_change():
    lines = ["a", "b", "c"]
    assert repo_utils.make_diff("f.py", lines, lines) == ""


def test_make_diff_matches_reference_difflib():
    """Guards against accidental drift from the original inlined implementation."""
    old = ["def f():", "    return 1"]
    new = ["def f():", "    return 2"]

    reference = "".join(
        difflib.unified_diff(
            ("\n".join(old) + "\n").splitlines(keepends=True),
            ("\n".join(new) + "\n").splitlines(keepends=True),
            fromfile="a/f.py",
            tofile="b/f.py",
        )
    )
    assert repo_utils.make_diff("f.py", old, new) == reference


# ---- find_relevant_range ------------------------------------------------------

def test_find_relevant_range_centres_on_keyword_cluster():
    lines = ["noise"] * 40
    lines[20] = "the target_symbol lives here"
    rng = repo_utils.find_relevant_range(lines, ["target_symbol"])

    assert rng is not None
    start, end = rng
    # The window must contain the keyword line.
    assert start <= 20 <= end
    # A single hit spreads its score onto neighbours; the peak is the first line
    # that reaches the max score (line 20-3), so the window is radius 15 around it.
    peak = 20 - repo_utils.SCORE_SPREAD
    assert start == max(0, peak - repo_utils.RANGE_RADIUS)
    assert end == min(len(lines), peak + repo_utils.RANGE_RADIUS)


def test_find_relevant_range_returns_none_when_no_match():
    lines = ["alpha", "beta", "gamma"]
    assert repo_utils.find_relevant_range(lines, ["nonexistent"]) is None


def test_find_relevant_range_empty_lines_returns_none():
    """Regression: an empty line list previously raised ValueError from max([])."""
    assert repo_utils.find_relevant_range([], ["anything"]) is None


def test_find_relevant_range_empty_keywords_returns_none():
    assert repo_utils.find_relevant_range(["some code"], []) is None


# ---- apply_patch (no-subprocess paths) ----------------------------------------

def test_apply_patch_empty_patch_is_false_without_running_git():
    # Empty patch short-circuits before any subprocess call, so this is safe and
    # deterministic even with no git repo present.
    assert repo_utils.apply_patch("", "/nonexistent/path") is False
