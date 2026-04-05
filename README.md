# LocalSWEAgent

**Can a 3B parameter LLM running on a laptop solve real GitHub issues?**

A lightweight SWE-bench agent powered by local LLMs via [Ollama](https://ollama.com). Zero cloud API costs, full privacy, runs entirely on your machine.

## Results

| Metric | Value |
|--------|-------|
| Model | Qwen 2.5 3B (1.9GB) |
| Hardware | Apple M3 Pro, 36GB RAM |
| Instances attempted | 14 |
| Repos successfully cloned | 7 |
| Patches generated | 6/7 (85.7%) |
| Correct fixes (resolve rate) | 0% |
| Avg time per instance | ~144s |
| Total API cost | **$0.00** |

### Key Finding

The 3B model generates plausible-looking patches for 85% of issues where it can access the repo. It correctly identifies the right file to modify in most cases and produces syntactically valid Python. However, it targets the **wrong root cause** every time - applying superficial fixes instead of addressing the real bug. This establishes a zero-cost baseline and shows exactly where small models break down on SWE tasks.

## Architecture

```
Issue -> [1. Analyze Issue] -> [2. Find Files] -> [3. Read Code] -> [4. Generate Fix] -> [5. Validate Patch]
              LLM                grep + LLM          read              SEARCH/REPLACE        git apply --check
```

### Key Techniques

- **Search/Replace Blocks**: Instead of asking a 3B model to produce perfect unified diffs, we ask for structured SEARCH/REPLACE blocks and programmatically build the patch via `difflib`
- **Smart File Discovery**: Two-stage retrieval - regex extraction of file paths and identifiers from the issue, followed by grep-based search
- **Minimal Context**: Works within 4k-8k context instead of 100k+ tokens. Forces surgical precision
- **Zero-Shot**: No fine-tuning, no few-shot examples. System prompt is under 200 tokens

## Quick Start

```bash
# Install Ollama (https://ollama.com)
ollama pull qwen2.5:3b

# Install dependencies
pip install -r requirements.txt

# Run evaluation on 5 instances
python3 run_eval.py --num 5

# Compare multiple models
python3 run_eval.py --all-models --num 5

# View results dashboard
open web/index.html
```

## Project Structure

```
.
├── agent/
│   ├── ollama_client.py    # Ollama API client
│   └── swe_agent.py        # Main agent (analyze, find, read, patch, validate)
├── eval/
│   └── runner.py           # SWE-bench Lite evaluation harness
├── web/
│   └── index.html          # Interactive results dashboard
├── results/
│   ├── results.json        # Detailed results with patches
│   └── predictions.json    # SWE-bench submission format
├── run_eval.py             # CLI entry point
└── requirements.txt
```

## Comparison with Cloud Agents

| Agent | Resolve Rate | Cost/Instance | Privacy |
|-------|-------------|---------------|---------|
| Claude 3.5 + Agentless | 33.0% | $0.34 | Cloud |
| GPT-4o + SWE-agent | 23.3% | $1.50 | Cloud |
| GPT-4 + SWE-agent | 18.0% | $2.00 | Cloud |
| Claude 3 Haiku RAG | 10.7% | $0.05 | Cloud |
| RAG baseline | 2.7% | $0.10 | Cloud |
| **LocalSWEAgent (3B)** | **0%** | **$0.00** | **Local** |

## Roadmap

- [ ] Iterative refinement with test execution feedback
- [ ] Try 7B-14B models (DeepSeek-Coder, CodeLlama)
- [ ] RAG over repository git history
- [ ] Multi-turn agent loop with tool use
- [ ] Fine-tune on SWE-bench training set

## License

MIT
