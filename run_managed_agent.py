"""Solve one SWE-bench task via Managed Agents.

Setup:
  export ANTHROPIC_API_KEY=sk-ant-...         # DO NOT hardcode
  export GITHUB_TOKEN=ghp_...                 # contents:read is enough
  pip install anthropic datasets

Run once to create the agent + environment, then again to solve a task:
  python3 run_managed_agent.py                # creates agent, prints IDs to export
  export SWEBENCH_AGENT_ID=agent_...
  export SWEBENCH_ENV_ID=env_...
  python3 run_managed_agent.py django__django-10914
"""

import os
import sys
import time

import anthropic
from datasets import load_dataset

MODEL = "claude-opus-4-6"
AGENT_NAME = "swebench-solver"
ENV_NAME = "swebench-env"

SYSTEM = """You are a software engineer fixing a bug in an open-source Python repo.

The repository is mounted at /workspace/repo and checked out to the base commit.
Your job:
1. Read the problem statement carefully.
2. Explore the repo with grep/glob/read to find the relevant file(s).
3. Make a minimal edit that fixes the bug (no refactoring, no new features).
4. Print the final diff to stdout with: cd /workspace/repo && git diff

Do not install anything. Do not run the test suite. Just produce the fix and the diff."""


def ensure_setup(client: anthropic.Anthropic) -> tuple[str, str]:
    agent_id = os.environ.get("SWEBENCH_AGENT_ID")
    env_id = os.environ.get("SWEBENCH_ENV_ID")
    if agent_id and env_id:
        return agent_id, env_id

    env = client.beta.environments.create(
        name=ENV_NAME,
        config={"type": "cloud", "networking": {"type": "unrestricted"}},
    )
    agent = client.beta.agents.create(
        name=AGENT_NAME,
        model=MODEL,
        system=SYSTEM,
        tools=[{"type": "agent_toolset_20260401", "default_config": {"enabled": True}}],
    )
    print(f"\n  export SWEBENCH_AGENT_ID={agent.id}")
    print(f"  export SWEBENCH_ENV_ID={env.id}\n")
    print("Setup done. Re-run with a task ID to solve one.")
    sys.exit(0)


def load_task(instance_id: str) -> dict:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    for row in ds:
        if row["instance_id"] == instance_id:
            return row
    raise SystemExit(f"task not found: {instance_id}")


def solve(client, agent_id: str, env_id: str, task: dict) -> None:
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        title=f"solve {task['instance_id']}",
    )
    print(f"session: {session.id}")

    user_msg = (
        f"Fix this bug in https://github.com/{task['repo']}.\n\n"
        f"First, clone the repo and check out the base commit:\n"
        f"  git clone --depth 100 https://github.com/{task['repo']}.git /workspace/repo\n"
        f"  cd /workspace/repo && git fetch --unshallow && git checkout {task['base_commit']}\n\n"
        f"=== PROBLEM STATEMENT ===\n{task['problem_statement']}\n\n"
        f"When done, run `cd /workspace/repo && git diff` and print the output."
    )

    # stream-first, then send
    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(
            session_id=session.id,
            events=[{"type": "user.message",
                     "content": [{"type": "text", "text": user_msg}]}],
        )
        for event in stream:
            t = event.type
            if t == "agent.message":
                for b in event.content:
                    if b.type == "text":
                        print(b.text, end="", flush=True)
            elif t == "agent.tool_use":
                print(f"\n[tool: {event.name}]", flush=True)
            elif t == "session.status_terminated":
                print("\n[terminated]")
                break
            elif t == "session.status_idle":
                if getattr(event.stop_reason, "type", None) == "requires_action":
                    continue
                print("\n[idle — done]")
                break

    # small settle for status-write race before any cleanup
    time.sleep(1)
    final = client.beta.sessions.retrieve(session_id=session.id)
    print(f"\nfinal status: {final.status}")


def main() -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("set ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    agent_id, env_id = ensure_setup(client)

    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 run_managed_agent.py <instance_id>")
    task = load_task(sys.argv[1])
    solve(client, agent_id, env_id, task)


if __name__ == "__main__":
    main()
