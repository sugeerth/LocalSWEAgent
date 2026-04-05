#!/usr/bin/env python3
"""
LocalSWEAgent - Beat SWE-bench with a laptop LLM.

Usage:
    python3 run_eval.py                          # Run 10 instances with qwen2.5:3b
    python3 run_eval.py --model llama3.2:3b      # Use different model
    python3 run_eval.py --num 5                   # Run fewer instances
    python3 run_eval.py --all-models              # Compare multiple models
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.runner import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="LocalSWEAgent - SWE-bench evaluation")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model to use")
    parser.add_argument("--num", type=int, default=10, help="Number of instances")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--all-models", action="store_true",
                       help="Run with multiple models for comparison")
    args = parser.parse_args()

    if args.all_models:
        models = ["qwen2.5:3b", "llama3.2:3b"]
        all_summaries = {}
        for model in models:
            out_dir = os.path.join(args.output, model.replace(":", "_"))
            try:
                summary = run_evaluation(
                    model=model,
                    num_instances=args.num,
                    output_dir=out_dir
                )
                all_summaries[model] = summary
            except Exception as e:
                print(f"Failed with model {model}: {e}")

        # Save comparison
        comp_path = os.path.join(args.output, "comparison.json")
        with open(comp_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nComparison saved to {comp_path}")
    else:
        run_evaluation(
            model=args.model,
            num_instances=args.num,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
