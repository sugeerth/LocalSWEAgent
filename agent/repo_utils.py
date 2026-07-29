"""Shared repo/patch helpers used by the agent variants.

These were previously copy-pasted (byte-identical) across the Turbo and
Precision agents. Centralising them keeps a single source of truth for the
patch-building and git-apply behaviour that the whole project relies on.

Only pure or thin-subprocess helpers live here; agent-specific prompting and
strategy logic stays in the individual agent modules.
"""

import os
import subprocess
import tempfile
import difflib
from typing import Optional


# How far (in lines) to expand around the highest-scoring line when picking the
# slice of a file to show the LLM. Kept as a constant so the window is obvious.
RANGE_RADIUS = 15
# Window used to spread a keyword hit's score onto neighbouring lines.
SCORE_SPREAD = 3


def make_diff(filepath: str, old_lines: list[str], new_lines: list[str]) -> str:
    """Create a unified diff (a/<path> -> b/<path>) from old and new line lists."""
    old_text = '\n'.join(old_lines) + '\n'
    new_text = '\n'.join(new_lines) + '\n'
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return ''.join(diff)


def apply_patch(patch: str, repo_dir: str) -> bool:
    """Apply ``patch`` inside ``repo_dir`` (not a dry run). Returns True on success.

    Falls back to lenient git-apply flags if the strict apply fails.
    """
    if not patch:
        return False
    try:
        r = subprocess.run(
            ["git", "apply", "--whitespace=fix", "-"],
            input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10
        )
        if r.returncode == 0:
            return True
        r2 = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "--unidiff-zero", "-"],
            input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10
        )
        return r2.returncode == 0
    except Exception:
        return False


def find_relevant_range(lines: list[str], keywords: list[str]) -> Optional[tuple]:
    """Return the (start, end) line range most relevant to ``keywords``.

    Scores each line by keyword hits (spreading the score onto neighbours) and
    returns a window around the peak. Returns ``None`` when nothing matches or
    when there are no lines to score.
    """
    if not lines:
        return None
    scores = [0] * len(lines)
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                # Score nearby lines too so the window centres on a cluster.
                for j in range(max(0, i - SCORE_SPREAD), min(len(lines), i + SCORE_SPREAD + 1)):
                    scores[j] += 1

    if max(scores) == 0:
        return None

    peak = scores.index(max(scores))
    start = max(0, peak - RANGE_RADIUS)
    end = min(len(lines), peak + RANGE_RADIUS)
    return (start, end)


def clone_repo(repo: str, commit: str, prefix: str) -> Optional[str]:
    """Shallow-clone ``repo`` into a temp dir and check out ``commit``.

    ``prefix`` namespaces the temp directory per agent (e.g. "turbo_").
    Unshallows on demand if the requested commit is not in the shallow clone.
    Returns the checkout path, or ``None`` if cloning/checkout fails.
    """
    repo_dir = os.path.join(tempfile.mkdtemp(prefix=prefix), repo.replace("/", "_"))
    try:
        print(f"  Cloning {repo}...")
        subprocess.run(
            ["git", "clone", "--depth", "100", f"https://github.com/{repo}.git", repo_dir],
            capture_output=True, timeout=120, check=True
        )
        try:
            subprocess.run(["git", "checkout", commit], cwd=repo_dir,
                           capture_output=True, timeout=30, check=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "fetch", "--unshallow"], cwd=repo_dir,
                           capture_output=True, timeout=180)
            subprocess.run(["git", "checkout", commit], cwd=repo_dir,
                           capture_output=True, timeout=30, check=True)
        return repo_dir
    except Exception as e:
        print(f"  Clone failed: {e}")
        return None
