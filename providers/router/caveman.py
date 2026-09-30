from __future__ import annotations

import os

def is_enabled() -> bool:
    val = os.getenv("PAL_CAVEMAN", "").strip().lower()
    return val in ("1", "on", "true", "lite", "full", "ultra")

def level() -> str:
    val = os.getenv("PAL_CAVEMAN", "").strip().lower()
    if not val or val in ("0", "false", "off"): 
        return ""
    if val in ("1", "on", "true"):
        return "ultra"
    return val

def system_prefix() -> str:
    if not is_enabled():
        return ""
    lvl = level()
    return (
        f"[CAVEMAN {lvl.upper()} COMPRESSION ACTIVE] Answer in maximally compressed caveman style: "
        "drop articles, filler words, and pleasantries. Keep ALL facts, numbers, code, file paths, "
        "and evidence byte-exact. Prefer fragments over sentences. Cut roughly 60 percent of words."
    )
