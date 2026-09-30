"""Tests for the writer->executor->judge mission pipeline."""

import asyncio
import json
from types import SimpleNamespace

from providers.router import mission


def _resp(content: str):
    return SimpleNamespace(content=content, metadata={})


def test_complete_cycle(monkeypatch, tmp_path):
    """Writer authors a file, executor writes it, judge marks COMPLETE."""
    target = tmp_path / "out.txt"
    writer_json = json.dumps({"files": [{"path": str(target), "content": "HELLO"}], "commands": []})

    calls = {"n": 0}

    def fake_generate(model, prompt, system, **kw):
        calls["n"] += 1
        if "WRITER" in system:
            return _resp(writer_json)
        return _resp(json.dumps({"decision": "COMPLETE", "reason": "verified"}))

    async def fake_loop(task, model, cwd, max_steps=3, full=True):
        return ("ran", [])

    monkeypatch.setattr("providers.router.dispatch.generate", fake_generate)
    monkeypatch.setattr("providers.router.chat_repl._tools_loop", fake_loop)
    monkeypatch.setattr(mission, "_pick", lambda models: models[0])

    res = asyncio.run(mission.run_mission("make out.txt", writer="w", executor="e", judge="j", max_iters=3, auto=False))
    assert res["status"] == "COMPLETE"
    assert res["iterations"] == 1
    assert target.read_text() == "HELLO"
    assert str(target) in res["transcript"][0]["exec"]["files_written"]


def test_iteration_cap(monkeypatch):
    """Judge never says COMPLETE -> stops at the cap, status INCOMPLETE."""
    def fake_generate(model, prompt, system, **kw):
        if "WRITER" in system:
            return _resp(json.dumps({"files": [], "commands": []}))
        return _resp(json.dumps({"decision": "CONTINUE", "feedback": "not done"}))

    async def fake_loop(task, model, cwd, max_steps=3, full=True):
        return ("", [])

    monkeypatch.setattr("providers.router.dispatch.generate", fake_generate)
    monkeypatch.setattr("providers.router.chat_repl._tools_loop", fake_loop)
    monkeypatch.setattr(mission, "_pick", lambda models: models[0])

    res = asyncio.run(mission.run_mission("goal", writer="w", executor="e", judge="j", max_iters=2, auto=False))
    assert res["status"] == "INCOMPLETE"
    assert res["iterations"] == 2
    assert len(res["transcript"]) == 2


def test_claude_excluded():
    assert mission._no_claude(["qwen3", "anthropic/claude-opus-4.5", "gemini-x"]) == ["qwen3", "gemini-x"]


def test_extract_json_tolerates_prose():
    assert mission._extract_json('sure! {"decision":"COMPLETE"} done') == {"decision": "COMPLETE"}
    assert mission._extract_json("no json here") is None


def test_heuristic_complexity_scales():
    easy = mission._heuristic_assess("write ok to /tmp/x.txt")
    hard = mission._heuristic_assess(
        "enumerate the subdomains, then scan each with nmap, and analyze the results, "
        "and write a report, and verify it, and summarize the findings"
    )
    assert easy["complexity"] <= 2
    assert hard["complexity"] >= 4


def test_team_for_scales():
    assert mission._team_for(1)["single"] is True
    assert mission._team_for(3) == {"single": False, "reviewers": 0}
    assert mission._team_for(5)["reviewers"] == 2


def test_heuristic_type_detection():
    assert mission._heuristic_assess("nmap scan of the target")["type"] == "recon"
    assert mission._heuristic_assess("implement a python function")["type"] == "code"
