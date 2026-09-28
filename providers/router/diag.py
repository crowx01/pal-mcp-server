"""Smart-router diagnostics collector — the data behind `pal diag`.

Importable so both the `pal` CLI and scripts/pal-diag.py share one code path.
Every section is defensive: a missing optional dependency (e.g. toolbelt's
openai import) degrades to a note instead of crashing the whole dump.
"""

from __future__ import annotations

import json


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:  # diagnostics must never crash
        return {"error": f"{type(exc).__name__}: {exc}"} if default is None else default


def collect() -> dict:
    from providers.router import (
        bandit,
        classifier,
        episode_store,
        health_probe,
        rate_limit,
        refusal_memory,
        response_cache,
        self_heal,
    )

    def _toolbelt():
        from providers.tooling import toolbelt as tb_mod

        if not tb_mod.is_enabled():
            return {"note": "PAL_TOOLBELT=0"}
        tb = tb_mod.get_toolbelt()
        return tb.snapshot() if tb else {"note": "toolbelt unavailable"}

    # per-category bandit view: how PAL would currently order each category
    def _bandit_view():
        out = {}
        for cat, cands in classifier.CATEGORY_PREFERENCES.items():
            out[cat] = {
                "enabled": bandit.is_enabled(),
                "ranking": bandit.explain(cat, list(cands)),
            }
        return out

    def _episodes_view():
        recent = episode_store.get_recent(limit=20)
        return {
            "enabled": episode_store.is_enabled(),
            "path": str(episode_store.EPISODE_PATH),
            "in_memory_recent": len(recent),
            "last": [
                {
                    "model": e.model,
                    "category": e.category,
                    "outcome": e.outcome,
                    "err_class": e.err_class,
                    "latency_ms": e.latency_ms,
                }
                for e in recent[:5]
            ],
        }

    return {
        "flags": {
            "PAL_SELF_HEAL": _safe(self_heal.is_enabled, False),
            "PAL_HEALTH_PROBE": _safe(health_probe.is_enabled, False),
            "PAL_RATE_LIMIT": _safe(rate_limit.is_enabled, False),
            "PAL_CACHE": _safe(response_cache.is_enabled, False),
            "PAL_REFUSAL_MEMORY": _safe(refusal_memory.is_enabled, False),
            "PAL_CLASSIFIER": _safe(classifier.is_enabled, False),
            "PAL_EPISODE_STORE": _safe(episode_store.is_enabled, False),
            "PAL_BANDIT": _safe(bandit.is_enabled, False),
        },
        "self_heal_alias_log": str(self_heal.DRIFT_LOG),
        "health": _safe(health_probe.snapshot, {}),
        "rate_limit": _safe(rate_limit.snapshot, {}),
        "cache": _safe(response_cache.snapshot, {}),
        "refusals": _safe(refusal_memory.snapshot, {}),
        "episodes": _safe(_episodes_view, None),
        "bandit": _safe(_bandit_view, None),
        "toolbelt": _safe(_toolbelt, {"note": "toolbelt import failed"}),
    }


def render(data: dict) -> str:
    lines = ["=" * 60, "PAL smart-router diag", "=" * 60]
    for k, v in data["flags"].items():
        lines.append(f"  {k:<22} {'on' if v else 'off'}")
    lines.append("-" * 60)
    for section in ("health", "rate_limit", "cache", "refusals", "episodes", "bandit", "toolbelt"):
        lines.append(f"[{section}]")
        lines.append(json.dumps(data.get(section), indent=2, sort_keys=True, default=str))
    lines.append(f"\ndrift log: {data['self_heal_alias_log']}")
    return "\n".join(lines)
