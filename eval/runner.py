"""
Evaluation runner for the LocalSWEAgent.
Downloads SWE-bench Lite and runs the agent on a subset of instances.
"""

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

from agent.swe_agent import LocalSWEAgent


# Curated subset of SWE-bench Lite instances that are good for local LLM testing
# These are chosen for: smaller repos, clearer issues, Python-focused fixes
CURATED_INSTANCES = [
    "astropy__astropy-12907",
    "django__django-11039",
    "django__django-11099",
    "django__django-11179",
    "django__django-11283",
    "django__django-11620",
    "django__django-11742",
    "django__django-11815",
    "django__django-11848",
    "django__django-11905",
    "django__django-11964",
    "django__django-12184",
    "django__django-12286",
    "django__django-12453",
    "django__django-12708",
    "django__django-12983",
    "django__django-13028",
    "django__django-13230",
    "django__django-13315",
    "django__django-13401",
    "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23476",
    "requests__requests-3362",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13496",
    "sympy__sympy-13146",
    "sympy__sympy-13437",
    "sympy__sympy-13480",
    "sympy__sympy-14396",
    "sympy__sympy-15011",
]


def load_swebench_lite(split: str = "test") -> list[dict]:
    """Load SWE-bench Lite dataset."""
    print("Loading SWE-bench Lite dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    return list(ds)


def run_evaluation(
    model: str = "qwen2.5:3b",
    num_instances: int = 10,
    curated: bool = True,
    output_dir: str = "results",
) -> dict:
    """
    Run the agent on SWE-bench Lite instances.

    Args:
        model: Ollama model name
        num_instances: Number of instances to evaluate
        curated: Whether to use curated subset
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    instances = load_swebench_lite()
    instance_map = {inst["instance_id"]: inst for inst in instances}

    # Select instances
    if curated:
        selected_ids = CURATED_INSTANCES[:num_instances]
        selected = [instance_map[iid] for iid in selected_ids if iid in instance_map]
    else:
        selected = instances[:num_instances]

    print(f"\nRunning evaluation on {len(selected)} instances with model: {model}")
    print(f"Output directory: {output_dir}\n")

    # Run agent
    agent = LocalSWEAgent(model=model)
    results = []
    start_time = time.time()

    for i, instance in enumerate(selected):
        instance_start = time.time()
        print(f"\n[{i+1}/{len(selected)}] ", end="")

        try:
            result = agent.solve(instance)
            result["time_seconds"] = round(time.time() - instance_start, 1)

            # Check similarity with gold patch (rough heuristic)
            gold_patch = instance.get("patch", "")
            result["gold_patch"] = gold_patch
            result["patch_similarity"] = _compute_patch_similarity(
                result["model_patch"], gold_patch
            )

        except Exception as e:
            result = {
                "instance_id": instance["instance_id"],
                "model_patch": "",
                "valid_patch": False,
                "reasoning": f"Exception: {traceback.format_exc()}",
                "time_seconds": round(time.time() - instance_start, 1),
                "patch_similarity": 0.0,
                "gold_patch": instance.get("patch", ""),
            }

        results.append(result)

        # Save intermediate results
        _save_results(results, model, output_dir, start_time)

        print(f"  Time: {result['time_seconds']}s | "
              f"Valid: {result.get('valid_patch', False)} | "
              f"Similarity: {result.get('patch_similarity', 0):.1%}")

    total_time = time.time() - start_time
    summary = _compute_summary(results, model, total_time)
    _save_results(results, model, output_dir, start_time, summary)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Instances: {len(results)}")
    print(f"Valid patches: {summary['valid_patches']}/{len(results)}")
    print(f"Avg similarity: {summary['avg_similarity']:.1%}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Results saved to: {output_dir}/")

    return summary


def _compute_patch_similarity(model_patch: str, gold_patch: str) -> float:
    """
    Rough heuristic for patch similarity.
    Compares the changed lines between model and gold patches.
    """
    if not model_patch or not gold_patch:
        return 0.0

    def extract_changes(patch: str) -> set[str]:
        changes = set()
        for line in patch.split('\n'):
            line = line.strip()
            if line.startswith('+') and not line.startswith('+++'):
                changes.add(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                changes.add(line[1:].strip())
        return changes

    model_changes = extract_changes(model_patch)
    gold_changes = extract_changes(gold_patch)

    if not gold_changes:
        return 0.0

    intersection = model_changes & gold_changes
    union = model_changes | gold_changes

    if not union:
        return 0.0

    # Jaccard similarity
    return len(intersection) / len(union)


def _compute_summary(results: list[dict], model: str, total_time: float) -> dict:
    valid = sum(1 for r in results if r.get("valid_patch"))
    similarities = [r.get("patch_similarity", 0) for r in results]
    high_sim = sum(1 for s in similarities if s > 0.3)

    return {
        "model": model,
        "total_instances": len(results),
        "valid_patches": valid,
        "valid_rate": valid / len(results) if results else 0,
        "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
        "high_similarity_count": high_sim,
        "estimated_resolve_rate": high_sim / len(results) if results else 0,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_instance": round(total_time / len(results), 1) if results else 0,
        "timestamp": datetime.now().isoformat(),
    }


def _save_results(results: list[dict], model: str, output_dir: str,
                   start_time: float, summary: dict = None):
    """Save results to JSON files."""
    # Save detailed results
    out_path = os.path.join(output_dir, "results.json")
    output = {
        "model": model,
        "results": results,
        "summary": summary or _compute_summary(results, model, time.time() - start_time),
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save in SWE-bench submission format
    predictions = []
    for r in results:
        predictions.append({
            "instance_id": r["instance_id"],
            "model_patch": r.get("model_patch", ""),
            "model_name_or_path": model,
        })
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    run_evaluation(model=args.model, num_instances=args.num, output_dir=args.output)
