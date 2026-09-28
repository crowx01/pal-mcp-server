"""Unit tests for the unified `pal` CLI dispatcher and diag collector."""

from __future__ import annotations

import json

import pytest

import cli
from providers.router import diag
from providers.router import episode_store as es


@pytest.fixture(autouse=True)
def _tmp_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "EPISODE_PATH", tmp_path / "episodes.jsonl")
    es.clear()
    yield
    es.clear()


def test_diag_collect_has_learning_sections():
    data = diag.collect()
    assert "episodes" in data and "bandit" in data
    assert data["flags"]["PAL_EPISODE_STORE"] in (True, False)
    assert data["flags"]["PAL_BANDIT"] in (True, False)


def test_diag_collect_is_json_serializable():
    # default=str guards against dataclass/Path leaks
    json.dumps(diag.collect(), default=str)


def test_diag_render_smoke():
    out = diag.render(diag.collect())
    assert "PAL smart-router diag" in out
    assert "[episodes]" in out and "[bandit]" in out


def test_cli_diag_json(capsys):
    rc = cli.main(["diag", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "bandit" in payload


def test_cli_distill_passthrough(capsys):
    rc = cli.main(["distill", "--print-only"])
    assert rc == 0
    assert "distillation" in capsys.readouterr().out


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])


def test_diag_survives_toolbelt_failure(monkeypatch):
    # simulate the optional toolbelt/openai import blowing up
    import providers.tooling.toolbelt as tb

    monkeypatch.setattr(tb, "is_enabled", lambda: (_ for _ in ()).throw(ImportError("no openai")))
    data = diag.collect()
    assert "toolbelt" in data  # degraded, not crashed
