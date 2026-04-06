#!/usr/bin/env python3
"""Precision mode - find exact file, exact function, make exact fix."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from agent.precision_agent import PrecisionSWEAgent
from eval.runner import _compute_patch_similarity

EASY_TARGETS = [
    "django__django-10914",   # FILE_UPLOAD_PERMISSIONS = None -> 0o644
    "django__django-11133",   # isinstance(value, bytes) -> isinstance(value, (bytes, memoryview))
    "django__django-11179",   # Add setattr(instance, model._meta.pk.attname, None)
    "django__django-12908",   # Add self._not_support_combined_queries('distinct')
    "django__django-13230",   # Add comments= kwarg to add_item()
    "django__django-13447",   # Add 'model': model to dict
]

def main():
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instance_map = {i["instance_id"]: i for i in ds}

    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
    agent = PrecisionSWEAgent(model=model)
    results = []
    solved = 0

    for iid in EASY_TARGETS:
        inst = instance_map.get(iid)
        if not inst:
            continue

        t0 = time.time()
        result = agent.solve(inst)
        result["time_seconds"] = round(time.time() - t0, 1)
        result["gold_patch"] = inst["patch"]
        result["patch_similarity"] = _compute_patch_similarity(
            result.get("model_patch", ""), inst["patch"]
        )

        if result.get("valid_patch"):
            solved += 1
            print(f"  >>> PATCH APPLIES! Sim={result['patch_similarity']:.1%}")
        else:
            print(f"  No valid patch. Sim={result['patch_similarity']:.1%}")

        results.append(result)
        os.makedirs("results", exist_ok=True)
        with open("results/precision_results.json", "w") as f:
            json.dump({"model": model, "results": results, "solved": solved}, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"PRECISION RESULTS: {solved}/{len(results)} patches apply")
    for r in results:
        s = "APPLY" if r.get("valid_patch") else "FAIL"
        sim = r.get("patch_similarity", 0)
        print(f"  [{s:5s}] {r['instance_id']:45s} sim={sim:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
