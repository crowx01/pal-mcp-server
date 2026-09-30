"""Centralized provider dispatch for REPL-side callers (/tools, /debate).

`tools/simple/base.py` already wraps the MCP tool path with size_guard
pre-flight rerouting, fallback_chain retries, and refusal/episode recording
(server.py's ``handle_call_tool``). The chat REPL's ``/tools`` and ``/debate``
commands call ``provider.generate_content`` directly, bypassing all of that.

This module gives both call sites the same treatment through one function:
size-guard reroute -> fallback-chain call -> refusal classification ->
episode_store recording. Recording is prompt-hash only (episode_store.record
hashes internally; the raw prompt is never persisted).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)


def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    *,
    temperature: float = 0.3,
    category: str = "",
    tool: str = "repl",
):
    """Resolve a provider for ``model``, reroute/fall back as needed, and
    record the outcome (success/refusal/error) to episode_store + refusal_memory.

    Returns the ModelResponse on success; raises on total fallback exhaustion.
    """
    from providers.registry import ModelProviderRegistry
    from providers.router import episode_store, refusal_memory
    from providers.router.fallback_chain import call_with_fallback
    from providers.router.size_guard import check_or_reroute

    dispatch_model = model
    ok, hint = check_or_reroute(dispatch_model, prompt)
    if not ok and hint and hint.startswith("route:") and hint != "route:none":
        alt = hint.split(":", 1)[1]
        log.warning("size-guard: %s over cap → pre-routing to %s", dispatch_model, alt)
        dispatch_model = alt

    def _invoke(_model: str):
        prov = ModelProviderRegistry.get_provider_for_model(_model)
        if prov is None:
            raise RuntimeError(f"no provider for {_model}")
        return prov.generate_content(prompt, _model, system, temperature)

    t0 = time.time()
    try:
        resp = call_with_fallback(_invoke, dispatch_model)
    except Exception as exc:
        lat_ms = int((time.time() - t0) * 1000)
        tag = refusal_memory.classify(str(exc))
        if tag:
            refusal_memory.record(model, category, tag)
        episode_store.record(
            model,
            category,
            "refusal" if (tag or "").startswith("refusal:") else "error",
            latency_ms=lat_ms,
            prompt=prompt,
            err_class=refusal_memory.classify_class(tag or str(exc)),
            tool=tool,
            reason=tag or str(exc),
        )
        raise

    lat_ms = int((time.time() - t0) * 1000)
    text = getattr(resp, "content", "") or ""
    tag = refusal_memory.classify(text)
    is_refusal = bool(tag and tag.startswith("refusal:"))
    if is_refusal:
        refusal_memory.record(model, category, tag)
    else:
        refusal_memory.record_success(model, category)
    episode_store.record(
        model,
        category,
        "refusal" if is_refusal else "success",
        latency_ms=lat_ms,
        prompt=prompt,
        err_class=refusal_memory.classify_class(tag) if is_refusal else None,
        tool=tool,
        reason=tag,
    )
    return resp
