"""
LocalSWEAgent - A lightweight SWE-bench agent powered by local LLMs via Ollama.

Strategy: Multi-step reasoning with focused code retrieval.
1. Analyze the issue to understand the bug/feature
2. Locate relevant files using repo structure + search
3. Read and understand the relevant code
4. Generate a minimal, targeted patch
"""

import os
import re
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from .ollama_client import OllamaClient


SYSTEM_PROMPT = """You are an expert software engineer solving GitHub issues.
You analyze bug reports and feature requests, locate the relevant code, and produce minimal patches.
Always respond with concrete code changes. Be precise and surgical in your fixes.
When generating patches, use unified diff format."""


class LocalSWEAgent:
    def __init__(self, model: str = "qwen2.5:3b", workspace: Optional[str] = None):
        self.llm = OllamaClient(model=model)
        self.workspace = workspace or tempfile.mkdtemp(prefix="swe_agent_")
        self.history: list[dict] = []
        self.stats = {"llm_calls": 0, "tokens_est": 0}

    def solve(self, instance: dict) -> dict:
        """
        Solve a SWE-bench instance.

        Args:
            instance: Dict with keys: instance_id, repo, base_commit, problem_statement,
                     hints_text, patch (gold), test_patch, version

        Returns:
            Dict with instance_id, model_patch, success, reasoning
        """
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem = instance["problem_statement"]
        hints = instance.get("hints_text", "")
        instance_id = instance["instance_id"]

        print(f"\n{'='*60}")
        print(f"Solving: {instance_id}")
        print(f"Repo: {repo}")
        print(f"{'='*60}")

        # Step 1: Clone and checkout the repo
        repo_dir = self._setup_repo(repo, base_commit)
        if not repo_dir:
            return self._make_result(instance_id, "", False, "Failed to clone repo")

        try:
            # Step 2: Analyze the issue
            analysis = self._analyze_issue(problem, hints, repo)

            # Step 3: Find relevant files
            relevant_files = self._find_relevant_files(repo_dir, analysis, problem)

            # Step 4: Read the code
            code_context = self._read_files(repo_dir, relevant_files)

            # Step 5: Generate the patch
            patch = self._generate_patch(problem, hints, code_context, relevant_files, repo_dir)

            # Step 6: Validate patch syntax
            is_valid = self._validate_patch(patch, repo_dir)

            return self._make_result(instance_id, patch, is_valid,
                                    f"Analysis: {analysis[:200]}")

        except Exception as e:
            return self._make_result(instance_id, "", False, f"Error: {str(e)}")
        finally:
            # Cleanup
            if repo_dir and os.path.exists(repo_dir):
                shutil.rmtree(repo_dir, ignore_errors=True)

    def _setup_repo(self, repo: str, base_commit: str) -> Optional[str]:
        """Clone repo and checkout the base commit."""
        repo_dir = os.path.join(self.workspace, repo.replace("/", "_"))
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

        try:
            print(f"  Cloning {repo}...")
            subprocess.run(
                ["git", "clone", "--depth", "50", f"https://github.com/{repo}.git", repo_dir],
                capture_output=True, timeout=120, check=True
            )
            # Fetch the specific commit if shallow clone doesn't have it
            try:
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=30, check=True
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["git", "fetch", "--unshallow"],
                    cwd=repo_dir, capture_output=True, timeout=180
                )
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=30, check=True
                )
            print(f"  Checked out {base_commit[:8]}")
            return repo_dir
        except Exception as e:
            print(f"  Failed to setup repo: {e}")
            return None

    def _analyze_issue(self, problem: str, hints: str, repo: str) -> str:
        """Use LLM to analyze the issue and identify what needs to change."""
        self.stats["llm_calls"] += 1
        prompt = f"""Analyze this GitHub issue for the {repo} repository.

ISSUE:
{problem[:3000]}

{f'HINTS: {hints[:1000]}' if hints else ''}

Provide a concise analysis:
1. What is the bug or requested feature? (1-2 sentences)
2. What files/modules likely need to change? (list file paths)
3. What is the likely root cause? (1-2 sentences)
4. What is the minimal fix? (1-2 sentences)

Be specific about file paths based on standard Python project layouts."""

        analysis = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        print(f"  Analysis complete ({len(analysis)} chars)")
        return analysis

    def _find_relevant_files(self, repo_dir: str, analysis: str,
                             problem: str) -> list[str]:
        """Find files relevant to the issue using search + LLM guidance."""
        # Extract file paths mentioned in the issue and analysis
        all_text = problem + "\n" + analysis
        mentioned_files = re.findall(r'[\w/]+\.py', all_text)

        # Also extract class/function names to search for
        identifiers = re.findall(r'`(\w+(?:\.\w+)*)`', problem)
        identifiers += re.findall(r'class\s+(\w+)', all_text)
        identifiers += re.findall(r'def\s+(\w+)', all_text)
        identifiers += re.findall(r'(\w+Error|\w+Exception)', problem)

        found_files = set()

        # Check mentioned files exist
        for f in mentioned_files:
            candidates = []
            for root, dirs, files in os.walk(repo_dir):
                # Skip hidden dirs and common non-source dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                          {'node_modules', '__pycache__', '.git', 'venv', '.tox'}]
                for fname in files:
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, repo_dir)
                    if rel.endswith(f) or f in rel:
                        candidates.append(rel)
            found_files.update(candidates[:3])

        # Search for identifiers in code
        for ident in identifiers[:10]:  # Limit searches
            if len(ident) < 3:
                continue
            try:
                result = subprocess.run(
                    ["grep", "-rl", "--include=*.py", ident, repo_dir],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.strip().split("\n")[:3]:
                    if line:
                        found_files.add(os.path.relpath(line, repo_dir))
            except Exception:
                pass

        # Filter to likely relevant files (skip tests, docs, configs)
        relevant = []
        for f in found_files:
            if any(skip in f for skip in ['test_', 'tests/', 'docs/', 'example',
                                           '.git/', 'conftest']):
                continue
            relevant.append(f)

        # If we found too many, ask LLM to narrow down
        if len(relevant) > 8:
            self.stats["llm_calls"] += 1
            file_list = "\n".join(relevant[:30])
            prompt = f"""Given this issue:
{problem[:1000]}

Which of these files are MOST likely to need changes? Pick the top 5 most relevant:
{file_list}

Return ONLY file paths, one per line."""
            response = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            narrowed = [f.strip() for f in response.split("\n")
                       if f.strip() in relevant]
            if narrowed:
                relevant = narrowed[:5]

        # Also limit to reasonable number
        relevant = relevant[:8]
        print(f"  Found {len(relevant)} relevant files: {relevant[:3]}...")
        return relevant

    def _read_files(self, repo_dir: str, files: list[str]) -> str:
        """Read the content of relevant files."""
        context = ""
        for f in files:
            fpath = os.path.join(repo_dir, f)
            if os.path.exists(fpath):
                try:
                    content = Path(fpath).read_text(errors='replace')
                    # Truncate very long files
                    if len(content) > 8000:
                        # Try to find the most relevant section
                        lines = content.split('\n')
                        content = '\n'.join(lines[:300])
                        content += f"\n... (truncated, {len(lines)} total lines)"
                    context += f"\n{'='*40}\nFILE: {f}\n{'='*40}\n{content}\n"
                except Exception:
                    pass
        return context

    def _generate_patch(self, problem: str, hints: str, code_context: str,
                        files: list[str], repo_dir: str) -> str:
        """Generate a patch using search-replace blocks, then convert to unified diff."""
        self.stats["llm_calls"] += 1

        prompt = f"""You must fix this GitHub issue.

ISSUE:
{problem[:2000]}

{f'HINTS: {hints[:500]}' if hints else ''}

RELEVANT CODE:
{code_context[:12000]}

Describe your fix using SEARCH/REPLACE blocks. For each change, specify:
1. The file path
2. The exact existing code to find (SEARCH)
3. The replacement code (REPLACE)

Format each change EXACTLY like this:

FILE: path/to/file.py
<<<<<<< SEARCH
exact existing code lines
=======
replacement code lines
>>>>>>> REPLACE

Be precise - the SEARCH block must match the existing code exactly.
Only show the lines that need to change plus 1-2 context lines.
You can have multiple FILE/SEARCH/REPLACE blocks."""

        response = self.llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=4096)
        print(f"  LLM response ({len(response)} chars)")

        # Parse search/replace blocks and build patch
        patch = self._build_patch_from_blocks(response, repo_dir, files)

        if not patch:
            # Fallback: try asking for unified diff directly
            self.stats["llm_calls"] += 1
            prompt2 = f"""Fix this issue. Output ONLY a unified diff patch.

ISSUE: {problem[:1000]}

CODE:
{code_context[:8000]}

Output format - ONLY this, nothing else:
--- a/file.py
+++ b/file.py
@@ -N,M +N,M @@
 context
-old
+new
 context"""
            patch = self.llm.generate(prompt2, system=SYSTEM_PROMPT, max_tokens=4096)
            patch = self._extract_diff(patch)

        print(f"  Generated patch ({len(patch)} chars)")
        return patch

    def _build_patch_from_blocks(self, response: str, repo_dir: str,
                                  candidate_files: list[str]) -> str:
        """Parse SEARCH/REPLACE blocks and build a unified diff."""
        import difflib

        # Parse blocks
        blocks = []
        current_file = None
        in_search = False
        in_replace = False
        search_lines = []
        replace_lines = []

        for line in response.split('\n'):
            if line.startswith('FILE:'):
                current_file = line[5:].strip()
            elif '<<<<<<< SEARCH' in line:
                in_search = True
                search_lines = []
            elif '=======' in line and in_search:
                in_search = False
                in_replace = True
                replace_lines = []
            elif '>>>>>>> REPLACE' in line and in_replace:
                in_replace = False
                if current_file and search_lines:
                    blocks.append({
                        'file': current_file,
                        'search': '\n'.join(search_lines),
                        'replace': '\n'.join(replace_lines),
                    })
            elif in_search:
                search_lines.append(line)
            elif in_replace:
                replace_lines.append(line)

        if not blocks:
            return ""

        # Apply blocks and generate unified diff
        full_patch = ""
        for block in blocks:
            file_path = block['file']
            # Resolve file path
            real_path = None
            for candidate in [file_path] + candidate_files:
                fp = os.path.join(repo_dir, candidate)
                if os.path.exists(fp):
                    # Check if this file contains the search text
                    content = Path(fp).read_text(errors='replace')
                    if block['search'].strip() in content:
                        real_path = candidate
                        break
                    elif not real_path and candidate == file_path:
                        real_path = candidate

            if not real_path:
                # Try fuzzy file match
                for candidate in candidate_files:
                    fp = os.path.join(repo_dir, candidate)
                    if os.path.exists(fp):
                        content = Path(fp).read_text(errors='replace')
                        if block['search'].strip()[:50] in content:
                            real_path = candidate
                            break

            if not real_path:
                continue

            fp = os.path.join(repo_dir, real_path)
            original = Path(fp).read_text(errors='replace')

            # Apply the replacement
            modified = original.replace(block['search'], block['replace'], 1)

            if modified == original:
                # Try with stripped whitespace matching
                search_stripped = block['search'].strip()
                for i, line in enumerate(original.split('\n')):
                    if search_stripped.split('\n')[0].strip() in line:
                        # Found approximate location, try line-by-line replace
                        orig_lines = original.split('\n')
                        search_lines = block['search'].split('\n')
                        replace_lines = block['replace'].split('\n')

                        # Find best matching position
                        for j in range(max(0, i-5), min(len(orig_lines), i+5)):
                            window = '\n'.join(orig_lines[j:j+len(search_lines)])
                            if search_stripped[:30] in window:
                                new_lines = orig_lines[:j] + replace_lines + orig_lines[j+len(search_lines):]
                                modified = '\n'.join(new_lines)
                                break
                        break

            if modified != original:
                # Generate unified diff
                diff = difflib.unified_diff(
                    original.splitlines(keepends=True),
                    modified.splitlines(keepends=True),
                    fromfile=f"a/{real_path}",
                    tofile=f"b/{real_path}",
                )
                full_patch += ''.join(diff)

        return full_patch

    def _extract_diff(self, text: str) -> str:
        """Extract unified diff from LLM output."""
        lines = text.split('\n')
        diff_lines = []
        in_diff = False

        for line in lines:
            if line.startswith('---') and not line.startswith('----'):
                in_diff = True
                diff_lines = [line]  # Reset - start fresh from this ---
            elif in_diff:
                diff_lines.append(line)
            elif line.startswith('diff --git'):
                in_diff = True
                diff_lines.append(line)

        if diff_lines:
            # Clean up trailing non-diff content
            while diff_lines and not any(
                diff_lines[-1].startswith(p) for p in ['+', '-', ' ', '@', 'diff', '---', '+++']
            ):
                diff_lines.pop()
            return '\n'.join(diff_lines) + '\n'

        # Fallback: return the whole text if it looks like a diff
        if '---' in text and '+++' in text and '@@' in text:
            return text

        return text

    def _validate_patch(self, patch: str, repo_dir: str) -> bool:
        """Check if the patch has valid syntax and could potentially apply."""
        if not patch or '---' not in patch or '+++' not in patch:
            return False
        if '@@' not in patch:
            return False

        # Try to apply the patch (dry run)
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "--whitespace=fix", "-"],
                input=patch, capture_output=True, text=True,
                cwd=repo_dir, timeout=10
            )
            if result.returncode == 0:
                print("  Patch validates OK")
                return True
            # Try with more lenient options
            result2 = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "--unidiff-zero", "-"],
                input=patch, capture_output=True, text=True,
                cwd=repo_dir, timeout=10
            )
            if result2.returncode == 0:
                print("  Patch validates OK (lenient)")
                return True
            else:
                print(f"  Patch validation failed: {result.stderr[:200]}")
                return False
        except Exception as e:
            print(f"  Patch validation error: {e}")
            return False

    def _make_result(self, instance_id: str, patch: str,
                     success: bool, reasoning: str) -> dict:
        return {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": self.llm.model,
            "valid_patch": success,
            "reasoning": reasoning,
            "stats": dict(self.stats)
        }
