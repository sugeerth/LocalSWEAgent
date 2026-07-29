"""Verify the Turbo/Precision agents delegate their shared helpers to
agent.repo_utils (i.e. the de-duplication preserved behaviour).

Agents are constructed via __new__ to skip __init__, which would otherwise try
to reach a running Ollama server. Only the pure helper methods are exercised.
"""

from agent import repo_utils
from agent.turbo_agent import TurboSWEAgent
from agent.precision_agent import PrecisionSWEAgent


def _bare(cls):
    return cls.__new__(cls)


def test_turbo_make_diff_delegates():
    agent = _bare(TurboSWEAgent)
    old = ["a", "b"]
    new = ["a", "c"]
    assert agent._make_diff("f.py", old, new) == repo_utils.make_diff("f.py", old, new)


def test_precision_make_diff_delegates():
    agent = _bare(PrecisionSWEAgent)
    old = ["a", "b"]
    new = ["a", "c"]
    assert agent._make_diff("f.py", old, new) == repo_utils.make_diff("f.py", old, new)


def test_both_agents_find_relevant_range_delegate():
    lines = ["noise"] * 30
    lines[10] = "needle here"
    turbo = _bare(TurboSWEAgent)
    precision = _bare(PrecisionSWEAgent)
    expected = repo_utils.find_relevant_range(lines, ["needle"])

    assert turbo._find_relevant_range(lines, ["needle"]) == expected
    assert precision._find_relevant_range(lines, ["needle"]) == expected


def test_apply_patch_empty_is_false_for_both_agents():
    assert _bare(TurboSWEAgent)._apply_patch("", "/tmp") is False
    assert _bare(PrecisionSWEAgent)._apply_patch("", "/tmp") is False
