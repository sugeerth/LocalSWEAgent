"""
PrecisionSWEAgent - Maximum precision approach for small LLMs.

Key insight: Small models fail because they get the WRONG file or WRONG location.
Solution: Use deterministic code analysis (AST, grep) to narrow down to the EXACT
function, then ask the LLM a very simple question about that one function.
"""

import os
import re
import ast
import subprocess
import tempfile
import shutil
import difflib
from pathlib import Path
from typing import Optional

from .ollama_client import OllamaClient

SYSTEM = """You are a code bug fixer. You respond with ONLY code, no explanations.
When asked to fix a line, output ONLY the fixed line, nothing else."""


class PrecisionSWEAgent:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.llm = OllamaClient(model=model)

    def solve(self, instance: dict) -> dict:
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem = instance["problem_statement"]
        iid = instance["instance_id"]

        print(f"\n{'='*60}")
        print(f"[PRECISION] {iid}")
        print(f"{'='*60}")

        repo_dir = self._clone(repo, base_commit)
        if not repo_dir:
            return {"instance_id": iid, "model_patch": "", "valid_patch": False}

        try:
            # Phase 1: Deterministic file finding
            target_files = self._find_target_files(repo_dir, problem)
            print(f"  Target files: {target_files[:3]}")

            if not target_files:
                return {"instance_id": iid, "model_patch": "", "valid_patch": False}

            # Phase 2: Find the exact function/method
            for fpath in target_files[:3]:
                result = self._try_fix_file(fpath, repo_dir, problem)
                if result:
                    patch, applied = result
                    if applied:
                        print(f"  PATCH APPLIES to {fpath}")
                        # Verify it's not a nonsense change
                        return {"instance_id": iid, "model_patch": patch, "valid_patch": True}
                    else:
                        # Keep trying other strategies
                        subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            # Phase 3: Fallback - add-line strategy for "add" type issues
            for fpath in target_files[:2]:
                result = self._try_add_line(fpath, repo_dir, problem)
                if result:
                    patch, applied = result
                    if applied:
                        print(f"  ADD-LINE PATCH APPLIES to {fpath}")
                        return {"instance_id": iid, "model_patch": patch, "valid_patch": True}
                    subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            return {"instance_id": iid, "model_patch": "", "valid_patch": False}
        finally:
            shutil.rmtree(repo_dir, ignore_errors=True)

    def _find_target_files(self, repo_dir: str, problem: str) -> list[str]:
        """Deterministic file finding using multiple signals."""
        file_scores: dict[str, int] = {}

        # 1. Extract explicit file paths from the issue
        paths = re.findall(r'[\w/]+\.py', problem)
        for p in paths:
            # Search for this file in repo
            for root, dirs, files in os.walk(repo_dir):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.tox', 'node_modules'}]
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), repo_dir)
                    if rel.endswith(p) and self._is_source(rel):
                        file_scores[rel] = file_scores.get(rel, 0) + 10

        # 2. Extract identifiers and grep
        identifiers = set()
        identifiers.update(self._extract_keywords(problem))
        # Also get imports from code in the issue
        for m in re.findall(r'from\s+([\w.]+)\s+import\s+(\w+)', problem):
            mod_parts = m[0].split('.')
            identifiers.add(m[1])  # imported name
            if len(mod_parts) > 1:
                identifiers.add(mod_parts[-1])  # last module part

        for ident in list(identifiers)[:12]:
            try:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", ident, repo_dir],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.strip().split("\n")[:20]:
                    if line:
                        rel = os.path.relpath(line, repo_dir)
                        if self._is_source(rel):
                            file_scores[rel] = file_scores.get(rel, 0) + 1
            except Exception:
                pass

        # 3. Check for code blocks in the issue that reference specific modules
        code_blocks = re.findall(r'```\w*\n(.+?)```', problem, re.DOTALL)
        for block in code_blocks:
            imports = re.findall(r'from\s+([\w.]+)\s+import', block)
            for imp in imports:
                modpath = imp.replace('.', '/') + '.py'
                for root, dirs, files in os.walk(repo_dir):
                    dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__'}]
                    for f in files:
                        rel = os.path.relpath(os.path.join(root, f), repo_dir)
                        if rel.endswith(modpath) and self._is_source(rel):
                            file_scores[rel] = file_scores.get(rel, 0) + 15

        ranked = sorted(file_scores.items(), key=lambda x: -x[1])
        return [f for f, s in ranked if s >= 2][:8]

    def _try_fix_file(self, fpath: str, repo_dir: str, problem: str) -> Optional[tuple[str, bool]]:
        """Try to fix a specific file."""
        full_path = os.path.join(repo_dir, fpath)
        content = Path(full_path).read_text(errors='replace')
        lines = content.split('\n')

        # Find relevant function/section
        keywords = self._extract_keywords(problem)
        rng = self._find_relevant_range(lines, keywords)
        if not rng:
            return None

        start, end = rng
        # Show generous context with line numbers
        ctx_start = max(0, start - 3)
        ctx_end = min(len(lines), end + 5)
        numbered = '\n'.join(f'{i+1:4d}: {lines[i]}' for i in range(ctx_start, ctx_end))

        # Strategy A: Ask which line to change
        resp = self.llm.generate(f"""Look at this code from {fpath} and the bug report below.

BUG: {problem[:1200]}

CODE:
{numbered}

Which SINGLE line number has the bug? And what should it be changed to?

Reply EXACTLY like this (two lines only):
LINE: <number>
FIX: <the corrected line of code>""", system=SYSTEM, temperature=0.05)

        line_match = re.search(r'LINE:\s*(\d+)', resp)
        fix_match = re.search(r'FIX:\s*(.+)', resp)

        if line_match and fix_match:
            line_num = int(line_match.group(1)) - 1
            fix_line = fix_match.group(1).rstrip()

            if 0 <= line_num < len(lines):
                old_lines = lines[:]
                indent = re.match(r'^(\s*)', lines[line_num]).group(1)
                lines[line_num] = indent + fix_line.lstrip()

                if old_lines[line_num] != lines[line_num]:
                    patch = self._make_diff(fpath, old_lines, lines)
                    if patch:
                        applied = self._apply_patch(patch, repo_dir)
                        return (patch, applied)

        return None

    def _try_add_line(self, fpath: str, repo_dir: str, problem: str) -> Optional[tuple[str, bool]]:
        """Try adding a new line (for 'add feature' type issues)."""
        full_path = os.path.join(repo_dir, fpath)
        content = Path(full_path).read_text(errors='replace')
        lines = content.split('\n')

        keywords = self._extract_keywords(problem)
        rng = self._find_relevant_range(lines, keywords)
        if not rng:
            return None

        start, end = rng
        ctx_start = max(0, start - 3)
        ctx_end = min(len(lines), end + 5)
        numbered = '\n'.join(f'{i+1:4d}: {lines[i]}' for i in range(ctx_start, ctx_end))

        resp = self.llm.generate(f"""This code in {fpath} needs a new line added to fix this issue:

ISSUE: {problem[:1200]}

CODE:
{numbered}

Where should the new line be inserted, and what should it contain?

Reply EXACTLY like this:
AFTER_LINE: <line number to insert after>
NEW_LINE: <the new line of code to add>""", system=SYSTEM, temperature=0.05)

        after_match = re.search(r'AFTER_LINE:\s*(\d+)', resp)
        new_match = re.search(r'NEW_LINE:\s*(.+)', resp)

        if after_match and new_match:
            after_num = int(after_match.group(1))
            new_line = new_match.group(1).rstrip()

            if 0 < after_num <= len(lines):
                old_lines = lines[:]
                # Match indentation of the line we're inserting after
                indent = re.match(r'^(\s*)', lines[after_num - 1]).group(1)
                lines.insert(after_num, indent + new_line.lstrip())

                patch = self._make_diff(fpath, old_lines, lines)
                if patch:
                    applied = self._apply_patch(patch, repo_dir)
                    return (patch, applied)

        return None

    def _is_source(self, path: str) -> bool:
        """Check if a path is a source file (not test, docs, etc)."""
        lower = path.lower()
        skip = ['test', 'example', 'docs/', 'doc/', 'migration', '.git/',
                'conftest', 'setup.py', 'setup.cfg', 'manage.py']
        return not any(s in lower for s in skip) and path.endswith('.py')

    def _extract_keywords(self, text: str) -> list[str]:
        kw = []
        # Backtick references (both regular and smart quotes)
        kw += re.findall(r'[`\u2018\u2019](\w+)[`\u2018\u2019]', text)
        # Method/function calls
        kw += re.findall(r'\.(\w{3,})\(', text)
        # Error types
        kw += re.findall(r'(\w+Error)\b', text)
        # Function/class defs
        kw += re.findall(r'def\s+(\w+)', text)
        kw += re.findall(r'class\s+(\w+)', text)
        # CamelCase identifiers (class names, etc)
        kw += re.findall(r'\b([A-Z][a-z]+[A-Z]\w*)\b', text)
        kw += re.findall(r'\b([A-Z][a-z]{2,}\w*)\b', text)
        # Import targets
        for m in re.findall(r'import\s+(\w+)', text):
            kw.append(m)
        # Technical terms (longer words that appear multiple times or look code-like)
        words = re.findall(r'\b(\w{5,})\b', text)
        from collections import Counter
        for w, count in Counter(words).most_common(20):
            if count >= 2 or '_' in w or (w[0].isupper() and not w.isupper()):
                kw.append(w)
        # Stop words
        stop = {'self', 'none', 'true', 'false', 'return', 'import', 'class',
                'should', 'would', 'could', 'django', 'python', 'that', 'this',
                'with', 'have', 'from', 'model', 'field', 'value', 'name',
                'Description', 'which', 'there', 'their', 'about', 'after',
                'before', 'doesn', 'https', 'currently', 'still'}
        kw = [k for k in kw if len(k) > 3 and k.lower() not in stop and k not in stop]
        return list(dict.fromkeys(kw))[:20]

    def _find_relevant_range(self, lines: list[str], keywords: list[str]) -> Optional[tuple]:
        scores = [0] * len(lines)
        for i, line in enumerate(lines):
            for kw in keywords:
                if kw in line:
                    for j in range(max(0, i-3), min(len(lines), i+4)):
                        scores[j] += 1
        if max(scores) == 0:
            return None
        peak = scores.index(max(scores))
        return (max(0, peak - 15), min(len(lines), peak + 15))

    def _make_diff(self, filepath: str, old_lines: list[str], new_lines: list[str]) -> str:
        old_text = '\n'.join(old_lines) + '\n'
        new_text = '\n'.join(new_lines) + '\n'
        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        )
        return ''.join(diff)

    def _apply_patch(self, patch: str, repo_dir: str) -> bool:
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
        except:
            return False

    def _clone(self, repo: str, commit: str) -> Optional[str]:
        repo_dir = os.path.join(tempfile.mkdtemp(prefix="precision_"), repo.replace("/", "_"))
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
