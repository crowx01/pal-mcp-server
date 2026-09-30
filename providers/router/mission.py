"""Writer -> Executor -> Judge mission pipeline.

Divides one mission across several models acting as sub-agents:
  - WRITER   authors the files-to-create and commands-to-run (no tools).
  - EXECUTOR (a different model) actually creates those files and runs those
    commands on Kali through the tool loop, returning real results.
  - JUDGE    checks the executor's results against pinned acceptance criteria
    and returns COMPLETE, or CONTINUE with feedback that goes back to the writer.

Every model call goes through ``dispatch.generate`` (key_pool rotation +
guardrails + Headroom). Anthropic/Claude is never selected. No database; a hard
iteration cap prevents loops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

log = logging.getLogger(__name__)

_WRITER_SYS = (
    "You are the WRITER sub-agent of a mission. Do NOT run tools. Given the "
    "mission, the acceptance criteria and any feedback, output ONLY one JSON "
    "object and nothing else:\n"
    '{"files":[{"path":"...","content":"..."}],"commands":["..."],'
    '"rationale":"...","done_hint":false}\n'
    "File contents must be complete. Commands are plain bash run on Kali."
)
_JUDGE_SYS = (
    "You are the JUDGE sub-agent. Given the mission, the acceptance criteria and "
    "the executor's results, output ONLY one JSON object:\n"
    '{"decision":"COMPLETE","reason":"...","feedback":"..."}\n'
    'decision is "COMPLETE" or "CONTINUE". Say COMPLETE only when every '
    "acceptance criterion is verified by the results; otherwise CONTINUE with "
    "concrete feedback for the writer."
)


def _no_claude(models: list[str]) -> list[str]:
    return [m for m in models if "claude" not in m.lower() and "anthropic" not in m.lower()]


def _env_models(var: str, default: list[str]) -> list[str]:
    got = [m.strip() for m in (os.getenv(var, "") or "").split(",") if m.strip()]
    return _no_claude(got or default)


def _pick(models: list[str]) -> str | None:
    from providers.router import chat_repl

    for m in models:
        try:
            if chat_repl._is_available(m):
                return m
        except Exception:
            pass
    return models[0] if models else None


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def run_mission(
    goal: str,
    *,
    writer: str | None = None,
    executor: str | None = None,
    judge: str | None = None,
    max_iters: int | None = None,
    full: bool = True,
) -> dict:
    """Run the writer->executor->judge loop; return a result dict."""
    from providers.router import chat_repl, dispatch

    writer = writer or _pick(_env_models("PAL_MISSION_WRITER_MODELS", ["gemini-3.5-flash-lite", "qwen3"]))
    executor = executor or _pick(_env_models("PAL_MISSION_EXECUTOR_MODELS", ["qwen3", "openai/gpt-oss-120b"]))
    judge = judge or _pick(_env_models("PAL_MISSION_REVIEWER_MODELS", [writer or "qwen3"]))
    max_iters = int(max_iters or os.getenv("PAL_MISSION_MAX_ITERS", "4"))
    criteria = f"The mission is accomplished and verified on disk: {goal}"

    transcript: list[dict] = []
    feedback = ""
    for iteration in range(1, max_iters + 1):
        # WRITER (sub-agent): author files + commands
        wr = await asyncio.to_thread(
            dispatch.generate,
            writer,
            f"MISSION: {goal}\nACCEPTANCE: {criteria}\nFEEDBACK: {feedback or '(none)'}",
            _WRITER_SYS,
            temperature=0.2,
            category="mission",
            tool="mission",
            max_output_tokens=int(os.getenv("PAL_TOOLS_MAX_TOKENS", "8192")),
        )
        plan = _extract_json(getattr(wr, "content", "") or "") or {}

        # EXECUTOR (sub-agent): really write files and run commands on Kali
        report: dict = {"files_written": [], "results": []}
        for entry in plan.get("files", []) or []:
            path, content = entry.get("path"), entry.get("content", "")
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as fh:
                    fh.write(content)
                report["files_written"].append(path)
        for cmd in plan.get("commands", []) or []:
            answer, _tr = await chat_repl._tools_loop(
                f"Run this exact bash command and report its output: {cmd}",
                executor,
                os.getcwd(),
                max_steps=3,
                full=full,
            )
            report["results"].append({"cmd": cmd, "answer": (answer or "")[:600]})

        # JUDGE (sub-agent): verify against pinned criteria
        jr = await asyncio.to_thread(
            dispatch.generate,
            judge,
            f"MISSION: {goal}\nACCEPTANCE: {criteria}\nRESULTS: {json.dumps(report)[:4000]}",
            _JUDGE_SYS,
            temperature=0.1,
            category="mission",
            tool="mission",
        )
        verdict = _extract_json(getattr(jr, "content", "") or "") or {"decision": "CONTINUE", "feedback": "no verdict"}
        transcript.append({
            "iteration": iteration, "writer": writer, "executor": executor, "judge": judge,
            "plan": plan, "exec": report, "verdict": verdict,
        })
        if str(verdict.get("decision", "")).upper() == "COMPLETE":
            return {"status": "COMPLETE", "iterations": iteration, "writer": writer,
                    "executor": executor, "judge": judge, "transcript": transcript}
        feedback = verdict.get("feedback", "")

    return {"status": "INCOMPLETE", "iterations": max_iters, "writer": writer,
            "executor": executor, "judge": judge, "transcript": transcript}


def run_mission_sync(goal: str, **kwargs) -> dict:
    """Blocking wrapper for CLI use."""
    return asyncio.run(run_mission(goal, **kwargs))


def main(argv: list[str]) -> int:
    """CLI: pal mission "<goal>" [--writer M] [--executor M] [--judge M] [--max-iters N] [--json]"""
    import argparse

    ap = argparse.ArgumentParser(prog="pal mission", description="writer->executor->judge mission")
    ap.add_argument("goal", help="the mission goal")
    ap.add_argument("--writer")
    ap.add_argument("--executor")
    ap.add_argument("--judge")
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--ro", action="store_true", help="read-only executor tools")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    os.environ.setdefault("PAL_TOOLBELT", "1")
    from server import configure_providers

    configure_providers()
    res = run_mission_sync(
        a.goal, writer=a.writer, executor=a.executor, judge=a.judge,
        max_iters=a.max_iters, full=not a.ro,
    )
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"status: {res['status']}  iterations: {res['iterations']}")
        print(f"writer={res['writer']}  executor={res['executor']}  judge={res['judge']}")
        last = (res.get("transcript") or [{}])[-1]
        print(f"files_written: {last.get('exec', {}).get('files_written', [])}")
        print(f"verdict: {last.get('verdict', {}).get('reason', '')[:200]}")
    return 0 if res["status"] == "COMPLETE" else 1
