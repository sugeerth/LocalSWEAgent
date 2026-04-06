"""
TurboSWEAgent - Aggressively optimized to solve at least one SWE-bench instance.

Strategy: Multiple approaches combined:
1. Laser-focused prompting with exact file path from issue analysis
2. Multi-turn self-correction loop (generate -> apply -> fix errors -> retry)
3. Show the LLM the exact function/class where the bug lives
4. Ask for ONLY the specific line to change
5. Ensemble: try 3 different prompt strategies, pick the one that applies
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
import difflib
from pathlib import Path
from typing import Optional

from .ollama_client import OllamaClient


SYSTEM = """You are a surgical code fixer. You make MINIMAL one-line changes.
You NEVER add new code. You ONLY modify existing lines.
You respond with ONLY the fix, nothing else."""


class TurboSWEAgent:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.llm = OllamaClient(model=model)
        self.max_retries = 3

    def solve(self, instance: dict) -> dict:
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem = instance["problem_statement"]
        hints = instance.get("hints_text", "")
        iid = instance["instance_id"]

        print(f"\n{'='*60}")
        print(f"[TURBO] Solving: {iid}")
        print(f"{'='*60}")

        repo_dir = self._clone(repo, base_commit)
        if not repo_dir:
            return {"instance_id": iid, "model_patch": "", "valid_patch": False}

        try:
            # Strategy 1: Direct line-fix approach
            patch = self._strategy_direct_fix(problem, hints, repo_dir)
            if patch and self._apply_patch(patch, repo_dir):
                print("  [OK] Strategy 1 (direct fix) succeeded!")
                return {"instance_id": iid, "model_patch": patch, "valid_patch": True}

            # Reset repo
            subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            # Strategy 2: Show exact code context, ask for replacement
            patch = self._strategy_contextual_replace(problem, hints, repo_dir)
            if patch and self._apply_patch(patch, repo_dir):
                print("  [OK] Strategy 2 (contextual replace) succeeded!")
                return {"instance_id": iid, "model_patch": patch, "valid_patch": True}

            subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            # Strategy 3: Multi-turn refinement
            patch = self._strategy_iterative(problem, hints, repo_dir)
            if patch and self._apply_patch(patch, repo_dir):
                print("  [OK] Strategy 3 (iterative) succeeded!")
                return {"instance_id": iid, "model_patch": patch, "valid_patch": True}

            # Return best attempt even if it doesn't apply cleanly
            return {"instance_id": iid, "model_patch": patch or "", "valid_patch": False}

        finally:
            shutil.rmtree(repo_dir, ignore_errors=True)

    def _strategy_direct_fix(self, problem: str, hints: str, repo_dir: str) -> str:
        """Find files via grep FIRST, then ask LLM to make a surgical fix."""
        print("  Strategy 1: Direct fix...")

        # Step 1: Find relevant files using grep (not LLM)
        files = self._grep_for_relevant_files(repo_dir, problem)
        if not files:
            print("    No relevant files found via grep")
            return ""

        # Try each candidate file
        for real_path in files[:3]:
            content = Path(os.path.join(repo_dir, real_path)).read_text(errors='replace')
            lines = content.split('\n')

            # Find the relevant section
            keywords = self._extract_keywords(problem)
            relevant_range = self._find_relevant_range(lines, keywords)

            if relevant_range:
                start, end = relevant_range
                snippet = '\n'.join(f'{i+1}: {lines[i]}' for i in range(max(0, start-5), min(len(lines), end+10)))
            else:
                continue

            print(f"    Trying file: {real_path} (lines {start}-{end})")

            # Ask for the exact fix with very constrained prompt
            resp = self.llm.generate(f"""You must fix this bug by modifying exactly ONE line in {real_path}.

BUG REPORT: {problem[:1500]}

Here is the relevant code with line numbers:
{snippet}

Instructions:
- Find the EXACT line that needs to change
- Respond with ONLY these two lines, nothing else:

LINE: <the line number>
NEW: <the complete replacement line>""", system=SYSTEM, temperature=0.05)

            # Parse response
            line_match = re.search(r'LINE:\s*(\d+)', resp)
            new_match = re.search(r'NEW:\s*(.+)', resp)

            if not line_match or not new_match:
                print(f"    Could not parse: {resp[:150]}")
                continue

            line_num = int(line_match.group(1)) - 1
            new_line = new_match.group(1)

            if line_num < 0 or line_num >= len(lines):
                print(f"    Invalid line {line_num+1}")
                continue

            # Build the patch
            old_lines = lines[:]
            orig_indent = re.match(r'^(\s*)', lines[line_num]).group(1)
            new_stripped = new_line.strip()
            lines[line_num] = orig_indent + new_stripped

            patch = self._make_diff(real_path, old_lines, lines)
            if patch:
                return patch

        return ""

    def _strategy_contextual_replace(self, problem: str, hints: str, repo_dir: str) -> str:
        """Show LLM the exact code context and ask for search/replace."""
        print("  Strategy 2: Contextual replace...")

        # Find relevant files
        files = self._grep_for_relevant_files(repo_dir, problem)
        if not files:
            print("    No relevant files found")
            return ""

        best_patch = ""
        for fpath in files[:3]:
            content = Path(os.path.join(repo_dir, fpath)).read_text(errors='replace')
            lines = content.split('\n')

            # Find relevant section
            keywords = self._extract_keywords(problem)
            rng = self._find_relevant_range(lines, keywords)
            if not rng:
                continue

            start, end = rng
            # Show generous context
            ctx_start = max(0, start - 10)
            ctx_end = min(len(lines), end + 10)
            snippet = '\n'.join(lines[ctx_start:ctx_end])

            resp = self.llm.generate(f"""Fix the bug described below. Show ONLY the old line and new line.

BUG: {problem[:1500]}

File: {fpath}, lines {ctx_start+1}-{ctx_end}:
```
{snippet}
```

Respond EXACTLY like this:
OLD: <the exact existing line to change>
NEW: <the replacement line>

Change ONLY ONE line. Be exact with the OLD line - copy it character for character.""",
                system=SYSTEM, temperature=0.05)

            old_match = re.search(r'OLD:\s*(.+)', resp)
            new_match = re.search(r'NEW:\s*(.+)', resp)

            if old_match and new_match:
                old_line = old_match.group(1).strip()
                new_line = new_match.group(1).strip()

                # Find and replace
                for i in range(ctx_start, ctx_end):
                    if old_line in lines[i] or lines[i].strip() == old_line:
                        indent = re.match(r'^(\s*)', lines[i]).group(1)
                        old_lines = lines[:]
                        lines[i] = indent + new_line
                        patch = self._make_diff(fpath, old_lines, lines)
                        if patch:
                            return patch
                        break

        return best_patch

    def _strategy_iterative(self, problem: str, hints: str, repo_dir: str) -> str:
        """Multi-turn: generate, try to apply, fix errors, retry."""
        print("  Strategy 3: Iterative refinement...")

        files = self._grep_for_relevant_files(repo_dir, problem)
        if not files:
            return ""

        # Read all relevant code
        code_ctx = ""
        for f in files[:2]:
            content = Path(os.path.join(repo_dir, f)).read_text(errors='replace')
            # Truncate
            if len(content) > 6000:
                lines = content.split('\n')
                keywords = self._extract_keywords(problem)
                rng = self._find_relevant_range(lines, keywords)
                if rng:
                    s, e = rng
                    content = '\n'.join(lines[max(0,s-20):min(len(lines),e+20)])
            code_ctx += f"\n=== {f} ===\n{content}\n"

        for attempt in range(self.max_retries):
            resp = self.llm.generate(f"""{'[RETRY] Previous patch had errors. Try a different approach. ' if attempt > 0 else ''}
Fix this bug with a SEARCH/REPLACE block.

BUG: {problem[:1500]}

CODE:
{code_ctx[:8000]}

Respond with:
FILE: <path>
<<<<<<< SEARCH
<exact existing code - 3-5 lines>
=======
<replacement code>
>>>>>>> REPLACE""", system=SYSTEM, temperature=0.1 + attempt * 0.15)

            patch = self._parse_search_replace(resp, repo_dir, files)
            if patch:
                if self._validate_patch(patch, repo_dir):
                    return patch
                print(f"    Attempt {attempt+1}: patch generated but invalid, retrying...")
            else:
                print(f"    Attempt {attempt+1}: no patch parsed, retrying...")

        return patch or ""

    # ---- Helpers ----

    def _clone(self, repo: str, commit: str) -> Optional[str]:
        repo_dir = os.path.join(tempfile.mkdtemp(prefix="turbo_"), repo.replace("/", "_"))
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

    def _find_file(self, repo_dir: str, target: str) -> Optional[str]:
        """Find a file in the repo matching the target path. Never return test files."""
        # Direct match
        if os.path.exists(os.path.join(repo_dir, target)):
            if 'test' not in target.lower():
                return target
        # Search - prefer source files over test files
        basename = os.path.basename(target)
        candidates = []
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.tox'}]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                if rel == target or rel.endswith(target):
                    candidates.append(rel)
                elif f == basename:
                    candidates.append(rel)
        # Filter: prefer non-test source files
        source = [c for c in candidates if 'test' not in c.lower() and 'example' not in c.lower()]
        if source:
            return source[0]
        return candidates[0] if candidates else None

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from issue text."""
        # Get function/class names
        kw = re.findall(r'`(\w+)`', text)
        kw += re.findall(r'(\w+Error|\w+Exception)', text)
        kw += re.findall(r'\.(\w+)\(', text)
        # Get identifiers from code blocks
        code_blocks = re.findall(r'```\w*\n(.+?)```', text, re.DOTALL)
        for block in code_blocks:
            kw += re.findall(r'(\w+)', block)
        # Filter
        kw = [k for k in kw if len(k) > 3 and k.lower() not in
              {'self', 'none', 'true', 'false', 'return', 'import', 'from', 'class',
               'this', 'that', 'with', 'should', 'would', 'could', 'django', 'python'}]
        return list(dict.fromkeys(kw))[:15]

    def _find_relevant_range(self, lines: list[str], keywords: list[str]) -> Optional[tuple]:
        """Find the range of lines most relevant to keywords."""
        scores = [0] * len(lines)
        for i, line in enumerate(lines):
            for kw in keywords:
                if kw in line:
                    # Score nearby lines too
                    for j in range(max(0, i-3), min(len(lines), i+4)):
                        scores[j] += 1

        if max(scores) == 0:
            return None

        # Find the peak
        peak = scores.index(max(scores))
        start = max(0, peak - 15)
        end = min(len(lines), peak + 15)
        return (start, end)

    def _grep_for_relevant_files(self, repo_dir: str, problem: str) -> list[str]:
        """Use grep to find relevant files."""
        keywords = self._extract_keywords(problem)
        file_scores: dict[str, int] = {}

        for kw in keywords[:8]:
            try:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", kw, repo_dir],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.strip().split("\n"):
                    if line and '.git/' not in line:
                        rel = os.path.relpath(line, repo_dir)
                        if ('test' not in rel.lower() and 'migration' not in rel.lower()
                            and 'example' not in rel.lower() and 'docs/' not in rel
                            and 'setup.py' not in rel and 'conftest' not in rel):
                            file_scores[rel] = file_scores.get(rel, 0) + 1
            except Exception:
                pass

        # Sort by score
        ranked = sorted(file_scores.items(), key=lambda x: -x[1])
        result = [f for f, _ in ranked[:5]]
        if result:
            print(f"    Top files: {result[:3]}")
        return result

    def _make_diff(self, filepath: str, old_lines: list[str], new_lines: list[str]) -> str:
        """Create unified diff from old and new line lists."""
        old_text = '\n'.join(old_lines) + '\n'
        new_text = '\n'.join(new_lines) + '\n'
        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        )
        return ''.join(diff)

    def _parse_search_replace(self, resp: str, repo_dir: str, files: list[str]) -> str:
        """Parse SEARCH/REPLACE block from LLM response."""
        file_match = re.search(r'FILE:\s*(.+\.py)', resp)
        search_match = re.search(r'<<<<<<< SEARCH\n(.+?)\n=======', resp, re.DOTALL)
        replace_match = re.search(r'=======\n(.+?)\n>>>>>>>', resp, re.DOTALL)

        if not search_match or not replace_match:
            return ""

        search_text = search_match.group(1)
        replace_text = replace_match.group(1)
        target_file = file_match.group(1).strip() if file_match else files[0]

        real_path = self._find_file(repo_dir, target_file)
        if not real_path:
            # Try all candidate files
            for f in files:
                fp = os.path.join(repo_dir, f)
                if os.path.exists(fp):
                    content = Path(fp).read_text(errors='replace')
                    if search_text.strip() in content:
                        real_path = f
                        break
        if not real_path:
            return ""

        content = Path(os.path.join(repo_dir, real_path)).read_text(errors='replace')

        # Try exact replacement
        if search_text in content:
            new_content = content.replace(search_text, replace_text, 1)
        elif search_text.strip() in content:
            # Try with stripped search
            new_content = content.replace(search_text.strip(), replace_text.strip(), 1)
        else:
            return ""

        if new_content == content:
            return ""

        return self._make_diff(
            real_path,
            content.split('\n'),
            new_content.split('\n')
        )

    def _validate_patch(self, patch: str, repo_dir: str) -> bool:
        if not patch or '---' not in patch:
            return False
        try:
            r = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "-"],
                input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10
            )
            return r.returncode == 0
        except:
            return False

    def _apply_patch(self, patch: str, repo_dir: str) -> bool:
        """Actually apply the patch (not just check)."""
        if not patch:
            return False
        try:
            r = subprocess.run(
                ["git", "apply", "--whitespace=fix", "-"],
                input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10
            )
            if r.returncode == 0:
                return True
            # Try lenient
            r2 = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--unidiff-zero", "-"],
                input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10
            )
            return r2.returncode == 0
        except:
            return False
