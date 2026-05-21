"""SWE-bench Managed Agent dashboard.

  export ANTHROPIC_API_KEY=sk-ant-...
  export SWEBENCH_AGENT_ID=agent_...   # from run_managed_agent.py setup
  export SWEBENCH_ENV_ID=env_...
  python3 web/server.py
  # open http://localhost:5050
"""

import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from datasets import load_dataset
from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 10 Django tasks from SWE-bench_Lite — same repo keeps clone cost consistent.
TASKS = [
    "django__django-10914",
    "django__django-11133",
    "django__django-11179",
    "django__django-11583",
    "django__django-12908",
    "django__django-13230",
    "django__django-13447",
    "django__django-13710",
    "django__django-14016",
    "django__django-14155",
]

MODEL = "claude-opus-4-6"
PARALLELISM = 5  # concurrent sessions

app = Flask(__name__, template_folder=str(ROOT / "web"))
subscribers: list[queue.Queue] = []
sub_lock = threading.Lock()

state = {
    "tasks": {tid: {"id": tid, "status": "pending", "events": [], "diff": None,
                    "error": None, "started_at": None, "ended_at": None,
                    "session_id": None, "tool_calls": 0} for tid in TASKS},
    "order": list(TASKS),
    "running": False,
}
state_lock = threading.Lock()


def publish(event: dict) -> None:
    payload = f"data: {json.dumps(event)}\n\n"
    with sub_lock:
        dead = []
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)


def update_task(tid: str, **kwargs) -> None:
    with state_lock:
        state["tasks"][tid].update(kwargs)
        snapshot = dict(state["tasks"][tid])
    publish({"type": "task_update", "task": snapshot})


def append_event(tid: str, kind: str, text: str) -> None:
    evt = {"t": time.time(), "kind": kind, "text": text}
    with state_lock:
        state["tasks"][tid]["events"].append(evt)
        if kind == "tool":
            state["tasks"][tid]["tool_calls"] += 1
        tool_calls = state["tasks"][tid]["tool_calls"]
    publish({"type": "event", "task_id": tid, "event": evt, "tool_calls": tool_calls})


def solve_task(client: anthropic.Anthropic, agent_id: str, env_id: str, task: dict) -> None:
    tid = task["instance_id"]
    update_task(tid, status="running", started_at=time.time())

    try:
        session = client.beta.sessions.create(
            agent=agent_id,
            environment_id=env_id,
            title=f"swebench {tid}",
        )
        update_task(tid, session_id=session.id)

        user_msg = (
            f"Fix this bug in https://github.com/{task['repo']} at commit "
            f"{task['base_commit']}.\n\n"
            f"Fast single-commit fetch (no full clone):\n"
            f"  mkdir -p /workspace/repo && cd /workspace/repo && git init -q && \\\n"
            f"  git remote add origin https://github.com/{task['repo']}.git && \\\n"
            f"  git fetch --depth 1 origin {task['base_commit']} -q && \\\n"
            f"  git checkout -q FETCH_HEAD\n\n"
            f"Then find and fix the bug with the MINIMAL edit. Do not refactor. "
            f"Do not run tests.\n\n"
            f"=== PROBLEM STATEMENT ===\n{task['problem_statement']}\n\n"
            f"When done, run `cd /workspace/repo && git diff` and print the output."
        )

        diff_text: list[str] = []
        last_text_was_diff = False

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
                        if getattr(b, "type", None) == "text":
                            snippet = b.text.strip()
                            if snippet:
                                append_event(tid, "text", snippet[:2000])
                                if "diff --git" in b.text:
                                    diff_text.append(b.text)
                                    last_text_was_diff = True
                elif t == "agent.tool_use":
                    name = getattr(event, "name", "?")
                    inp = getattr(event, "input", {}) or {}
                    summary = inp.get("command") or inp.get("pattern") or \
                              inp.get("path") or inp.get("file_path") or ""
                    append_event(tid, "tool", f"{name}  {str(summary)[:200]}")
                elif t == "agent.tool_result":
                    content = getattr(event, "content", None)
                    text_out = ""
                    if isinstance(content, list):
                        for c in content:
                            if getattr(c, "type", None) == "text":
                                text_out += getattr(c, "text", "")
                    if text_out and "diff --git" in text_out:
                        diff_text.append(text_out)
                    if text_out:
                        append_event(tid, "tool_result", text_out[:1500])
                elif t == "session.status_terminated":
                    break
                elif t == "session.status_idle":
                    if getattr(getattr(event, "stop_reason", None), "type", None) \
                            == "requires_action":
                        continue
                    break

        diff_combined = "\n".join(diff_text).strip()
        solved = bool(diff_combined)
        update_task(
            tid,
            status="solved" if solved else "no_diff",
            diff=diff_combined or None,
            ended_at=time.time(),
        )
    except Exception as e:
        append_event(tid, "error", f"{type(e).__name__}: {e}")
        update_task(tid, status="failed", error=str(e), ended_at=time.time())


def worker() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        publish({"type": "fatal", "msg": "ANTHROPIC_API_KEY not set"})
        return
    agent_id = os.environ.get("SWEBENCH_AGENT_ID")
    env_id = os.environ.get("SWEBENCH_ENV_ID")
    if not (agent_id and env_id):
        publish({"type": "fatal",
                 "msg": "SWEBENCH_AGENT_ID / SWEBENCH_ENV_ID not set"})
        return

    client = anthropic.Anthropic()
    publish({"type": "log", "msg": "loading SWE-bench_Lite dataset…"})
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_id = {r["instance_id"]: r for r in ds if r["instance_id"] in TASKS}

    runnable = []
    for tid in TASKS:
        if tid not in by_id:
            update_task(tid, status="failed", error="not in dataset")
        else:
            runnable.append(by_id[tid])

    publish({"type": "log",
             "msg": f"running {len(runnable)} tasks with parallelism={PARALLELISM}"})
    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futures = [ex.submit(solve_task, client, agent_id, env_id, t)
                   for t in runnable]
        for f in futures:
            try:
                f.result()
            except Exception as e:
                publish({"type": "log", "msg": f"task error: {e}"})

    with state_lock:
        state["running"] = False
    publish({"type": "run_done"})


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/agent-info")
def api_agent_info():
    agent_id = os.environ.get("SWEBENCH_AGENT_ID")
    env_id = os.environ.get("SWEBENCH_ENV_ID")
    if not (agent_id and env_id):
        return jsonify({"error": "agent/env not configured"}), 400
    client = anthropic.Anthropic()
    try:
        agent = client.beta.agents.retrieve(agent_id=agent_id)
        env = client.beta.environments.retrieve(environment_id=env_id)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    def dump(v):
        if v is None: return None
        if hasattr(v, "model_dump"): return v.model_dump()
        try: return dict(v)
        except Exception: return str(v)

    return jsonify({
        "agent": {
            "id": agent.id,
            "name": agent.name,
            "version": agent.version,
            "model": dump(getattr(agent, "model", None)),
            "system": getattr(agent, "system", None),
            "description": getattr(agent, "description", None),
            "tools": [t.model_dump() if hasattr(t, "model_dump") else dict(t)
                      for t in (getattr(agent, "tools", []) or [])],
            "mcp_servers": [s.model_dump() if hasattr(s, "model_dump") else dict(s)
                           for s in (getattr(agent, "mcp_servers", []) or [])],
            "skills": [s.model_dump() if hasattr(s, "model_dump") else dict(s)
                       for s in (getattr(agent, "skills", []) or [])],
            "created_at": str(getattr(agent, "created_at", "")),
        },
        "environment": {
            "id": env.id,
            "name": env.name,
            "config": env.config.model_dump() if hasattr(env.config, "model_dump")
                      else dict(env.config),
            "created_at": str(getattr(env, "created_at", "")),
        },
        "parallelism": PARALLELISM,
        "beta_header": "managed-agents-2026-04-01",
    })


@app.route("/api/state")
def api_state():
    with state_lock:
        return jsonify({
            "tasks": [state["tasks"][tid] for tid in state["order"]],
            "running": state["running"],
        })


@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        if state["running"]:
            return jsonify({"ok": False, "msg": "already running"}), 409
        state["running"] = True
        # reset state
        for tid in state["order"]:
            state["tasks"][tid] = {
                "id": tid, "status": "pending", "events": [], "diff": None,
                "error": None, "started_at": None, "ended_at": None,
                "session_id": None, "tool_calls": 0,
            }
    threading.Thread(target=worker, daemon=True).start()
    publish({"type": "run_started"})
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue(maxsize=1000)
    with sub_lock:
        subscribers.append(q)

    def gen():
        try:
            yield "data: {\"type\":\"hello\"}\n\n"
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with sub_lock:
                if q in subscribers:
                    subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
