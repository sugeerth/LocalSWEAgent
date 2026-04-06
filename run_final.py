#!/usr/bin/env python3
"""
Final approach: Use the import paths from issue text to find exact files,
then show the LLM only the 20-line function and ask the simplest possible question.
"""

import sys, os, json, time, re, subprocess, tempfile, shutil, difflib
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from agent.ollama_client import OllamaClient
from eval.runner import _compute_patch_similarity

SYSTEM = "You are a precise code fixer. Output ONLY code. No explanations."

TARGETS = [
    "django__django-10914",
    "django__django-11133",
    "django__django-11179",
    "django__django-12908",
    "django__django-13230",
    "django__django-13447",
]


def clone(repo, commit):
    d = os.path.join(tempfile.mkdtemp(prefix="final_"), repo.replace("/", "_"))
    subprocess.run(["git", "clone", "--depth", "100",
                    f"https://github.com/{repo}.git", d],
                   capture_output=True, timeout=120, check=True)
    try:
        subprocess.run(["git", "checkout", commit], cwd=d,
                       capture_output=True, timeout=30, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "fetch", "--unshallow"], cwd=d,
                       capture_output=True, timeout=180)
        subprocess.run(["git", "checkout", commit], cwd=d,
                       capture_output=True, timeout=30, check=True)
    return d


def find_files(repo_dir, problem):
    """Use ALL signals to find the right file."""
    scores = {}

    # 1. Import paths -> file paths (check module.py AND module/*.py)
    for mod in re.findall(r'from\s+([\w.]+)\s+import', problem):
        parts = mod.split('.')
        # Try: django/http/response.py, django/http.py, etc.
        possible_paths = []
        possible_paths.append('/'.join(parts) + '.py')         # django/http.py
        possible_paths.append('/'.join(parts) + '/__init__.py') # django/http/__init__.py
        # Also try all .py files in the module directory
        mod_dir = '/'.join(parts)

        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.tox'}]
            for f in files:
                if not f.endswith('.py'):
                    continue
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                if 'test' in rel.lower():
                    continue
                # Check if file matches any possible path
                for pp in possible_paths:
                    if rel.endswith(pp):
                        scores[rel] = scores.get(rel, 0) + 20
                # Check if file is in the module directory
                if mod_dir in rel and rel.endswith('.py'):
                    scores[rel] = scores.get(rel, 0) + 10

    # 1b. Module references like Django.db.models.deletion
    for mod_ref in re.findall(r'[Dd]jango\.([\w.]+)', problem):
        mod_path = 'django/' + mod_ref.replace('.', '/') + '.py'
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.tox'}]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                if rel.endswith(mod_path) and 'test' not in rel.lower():
                    scores[rel] = scores.get(rel, 0) + 25

    # 2. Explicit file paths mentioned
    for p in re.findall(r'[\w./]+\.py', problem):
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__'}]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                if (rel.endswith(p) or p in rel) and 'test' not in rel.lower():
                    scores[rel] = scores.get(rel, 0) + 15

    # 3. Grep for distinctive terms
    terms = set()
    terms.update(re.findall(r'\b([A-Z][a-z]+[A-Z]\w+)\b', problem))  # CamelCase
    terms.update(re.findall(r'\b([A-Z][a-z]{2,}\w+)\b', problem))
    terms.update(re.findall(r'`(\w{3,})`', problem))
    terms.update(re.findall(r'[\u2018\u201c](\w{3,})[\u2019\u201d]', problem))
    terms.update(re.findall(r'\.(\w{4,})\(', problem))
    # Filter
    stop = {'Description', 'When', 'This', 'None', 'True', 'False', 'Django',
            'Python', 'That', 'Also', 'Should', 'Would', 'Could', 'Content',
            'Sqlite', 'Postgresql', 'The', 'For', 'After', 'Before'}
    terms = {t for t in terms if t not in stop and len(t) > 3}

    for term in list(terms)[:15]:
        try:
            r = subprocess.run(["grep", "-rl", "--include=*.py", term, repo_dir],
                             capture_output=True, text=True, timeout=10)
            for line in r.stdout.strip().split("\n")[:10]:
                if line:
                    rel = os.path.relpath(line, repo_dir)
                    if 'test' not in rel.lower() and 'migration' not in rel.lower() and '.git' not in rel:
                        scores[rel] = scores.get(rel, 0) + 1
        except:
            pass

    # 4. Re-rank: files containing MORE of the terms get big bonus
    for fpath, base_score in list(scores.items()):
        fp = os.path.join(repo_dir, fpath)
        if os.path.exists(fp):
            try:
                content = Path(fp).read_text(errors='replace')
                for term in terms:
                    if term in content:
                        scores[fpath] = scores.get(fpath, 0) + 3
                # Bonus for files with class/function definitions matching keywords
                for term in terms:
                    if f'def {term.lower()}' in content.lower() or f'class {term}' in content:
                        scores[fpath] = scores.get(fpath, 0) + 8
            except:
                pass

    # 5. Prefer non-__init__ files (actual implementation)
    for fpath in list(scores.keys()):
        if '__init__' in fpath:
            scores[fpath] = scores.get(fpath, 0) - 5

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [f for f, s in ranked[:8]]


def find_function(lines, keywords, problem=""):
    """Find the function/method most relevant to the keywords."""
    # Check if issue mentions specific line numbers
    line_refs = re.findall(r':(\d{2,4})', problem)
    line_refs += re.findall(r'line\s+(\d{2,4})', problem, re.IGNORECASE)

    # Score each line
    line_scores = [0] * len(lines)
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                for j in range(max(0, i-5), min(len(lines), i+6)):
                    line_scores[j] += 2 if j == i else 1

    # Boost lines referenced in the issue
    for ref in line_refs:
        ln = int(ref) - 1
        if 0 <= ln < len(lines):
            for j in range(max(0, ln-5), min(len(lines), ln+6)):
                line_scores[j] += 10

    if max(line_scores) == 0:
        return None

    peak = line_scores.index(max(line_scores))

    # Find the enclosing function/class
    func_start = peak
    for i in range(peak, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            func_start = i
            break

    # Find end of function (next def/class at same or lower indent)
    indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    func_end = min(len(lines), func_start + 50)
    for i in range(func_start + 1, min(len(lines), func_start + 100)):
        stripped = lines[i].lstrip()
        if stripped and (stripped.startswith('def ') or stripped.startswith('class ')):
            curr_indent = len(lines[i]) - len(lines[i].lstrip())
            if curr_indent <= indent:
                func_end = i
                break

    return (max(0, func_start - 2), min(len(lines), func_end + 2))


def make_diff(filepath, old_lines, new_lines):
    old = '\n'.join(old_lines) + '\n'
    new = '\n'.join(new_lines) + '\n'
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{filepath}", tofile=f"b/{filepath}"))


def apply_patch(patch, repo_dir):
    if not patch:
        return False
    for flags in [["--whitespace=fix"], ["--whitespace=nowarn", "--unidiff-zero"]]:
        r = subprocess.run(["git", "apply"] + flags + ["-"],
                          input=patch, capture_output=True, text=True, cwd=repo_dir, timeout=10)
        if r.returncode == 0:
            return True
    return False


def solve_instance(llm, inst):
    iid = inst["instance_id"]
    problem = inst["problem_statement"]
    print(f"\n{'='*60}")
    print(f"Solving: {iid}")

    repo_dir = clone(inst["repo"], inst["base_commit"])

    try:
        files = find_files(repo_dir, problem)
        print(f"  Files: {files[:4]}")

        if not files:
            return {"instance_id": iid, "model_patch": "", "valid_patch": False}

        # Extract keywords for function finding
        keywords = set()
        keywords.update(re.findall(r'\b([A-Z][a-z]+\w+)\b', problem))
        keywords.update(re.findall(r'\.(\w{4,})\(', problem))
        keywords.update(re.findall(r'`(\w+)`', problem))
        keywords = [k for k in keywords if len(k) > 3]

        for fpath in files[:5]:
            content = Path(os.path.join(repo_dir, fpath)).read_text(errors='replace')
            lines = content.split('\n')

            rng = find_function(lines, keywords, problem)
            if not rng:
                continue

            start, end = rng
            numbered = '\n'.join(f'{i+1}: {lines[i]}' for i in range(start, end))
            print(f"  Trying {fpath} lines {start+1}-{end}")

            # Try ALL 3 strategies for each file, collect candidates, pick best

            candidates = []

            # Strategy A: Modify one line
            subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)
            resp = llm.generate(f"""Fix this bug by changing ONE line in {fpath}.

BUG: {problem[:1500]}

{numbered}

Reply with EXACTLY:
LINE: <number>
NEW: <replacement line>""", system=SYSTEM, temperature=0.05)

            line_m = re.search(r'LINE:\s*(\d+)', resp)
            new_m = re.search(r'NEW:\s*(.+)', resp)

            if line_m and new_m:
                ln = int(line_m.group(1)) - 1
                if 0 <= ln < len(lines):
                    old = lines[:]
                    indent = re.match(r'^(\s*)', lines[ln]).group(1)
                    lines_copy = old[:]
                    lines_copy[ln] = indent + new_m.group(1).strip()
                    if lines_copy[ln] != old[ln]:
                        patch = make_diff(fpath, old, lines_copy)
                        if patch and apply_patch(patch, repo_dir):
                            candidates.append(('A-fix', patch))
                        subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            # Strategy B: Add one line
            resp2 = llm.generate(f"""Fix this bug by ADDING one new line of code to {fpath}.
Do NOT modify existing lines. INSERT a new line.

BUG: {problem[:1500]}

{numbered}

Reply with EXACTLY:
AFTER: <line number to insert after>
CODE: <the new line to add, with proper indentation>""", system=SYSTEM, temperature=0.05)

            after_m = re.search(r'AFTER:\s*(\d+)', resp2)
            code_m = re.search(r'CODE:\s*(.+)', resp2)

            if after_m and code_m:
                after = int(after_m.group(1))
                cur_lines = content.split('\n')
                if 0 < after <= len(cur_lines):
                    old = cur_lines[:]
                    indent = re.match(r'^(\s*)', cur_lines[after-1]).group(1)
                    new_lines = old[:after] + [indent + code_m.group(1).strip()] + old[after:]
                    patch = make_diff(fpath, old, new_lines)
                    if patch and apply_patch(patch, repo_dir):
                        candidates.append(('B-add', patch))
                    subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            # Strategy C: Search/Replace
            resp3 = llm.generate(f"""Fix this bug. Show the exact old code and new code from {fpath}.

BUG: {problem[:1500]}

{numbered}

OLD_CODE: <paste the exact broken line(s)>
NEW_CODE: <the fixed version>""", system=SYSTEM, temperature=0.1)

            old_m = re.search(r'OLD_CODE:\s*(.+)', resp3)
            new_m2 = re.search(r'NEW_CODE:\s*(.+)', resp3)

            if old_m and new_m2:
                old_code = old_m.group(1).strip()
                new_code = new_m2.group(1).strip()
                if old_code in content:
                    new_content = content.replace(old_code, new_code, 1)
                    patch = make_diff(fpath, content.split('\n'), new_content.split('\n'))
                    if patch and apply_patch(patch, repo_dir):
                        candidates.append(('C-replace', patch))
                    subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)

            if candidates:
                # Score candidates: prefer smaller, more surgical patches
                def patch_score(item):
                    name, p = item
                    lines_changed = p.count('\n+') + p.count('\n-') - 2  # minus headers
                    # Prefer add-line for "add" type issues
                    add_bonus = 5 if 'add' in problem.lower() and 'B-add' in name else 0
                    # Prefer smaller patches (more surgical)
                    return -lines_changed + add_bonus

                candidates.sort(key=patch_score, reverse=True)
                best = candidates[0]
                print(f"  [{best[0]}] selected! ({len(candidates)} candidates)")
                subprocess.run(["git", "checkout", "."], cwd=repo_dir, capture_output=True)
                apply_patch(best[1], repo_dir)
                return {"instance_id": iid, "model_patch": best[1], "valid_patch": True}

        return {"instance_id": iid, "model_patch": "", "valid_patch": False}
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)


def main():
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    imap = {i["instance_id"]: i for i in ds}

    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    llm = OllamaClient(model=model)
    results = []
    solved = 0

    for iid in TARGETS:
        inst = imap.get(iid)
        if not inst:
            continue
        t0 = time.time()
        r = solve_instance(llm, inst)
        r["time_seconds"] = round(time.time() - t0, 1)
        r["gold_patch"] = inst["patch"]
        r["patch_similarity"] = _compute_patch_similarity(r.get("model_patch", ""), inst["patch"])
        if r.get("valid_patch"):
            solved += 1
        results.append(r)
        print(f"  Valid={r['valid_patch']} Sim={r['patch_similarity']:.1%} Time={r['time_seconds']}s")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/final_results.json", "w") as f:
        json.dump({"model": model, "solved": solved, "total": len(results), "results": results},
                  f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"FINAL: {solved}/{len(results)} patches apply correctly")
    for r in results:
        tag = "PASS" if r["valid_patch"] else "FAIL"
        print(f"  [{tag}] {r['instance_id']:45s} sim={r['patch_similarity']:.1%}")
        if r["valid_patch"]:
            print(f"        PATCH: {r['model_patch'][:200]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
