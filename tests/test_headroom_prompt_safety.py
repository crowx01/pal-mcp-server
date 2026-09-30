"""Headroom must never rewrite system prompts or prose user prompts."""

from providers.router import headroom_adapter as h

PROSE = "\n".join(f"Rule {i}: keep answer {i} verbatim and concise." for i in range(400))


def test_structured_only_skips_prose():
    assert h.compress_tool_result(PROSE, structured_only=True) == PROSE


def test_payload_skips_system_and_user_prose():
    msgs = [{"role": "system", "content": PROSE}, {"role": "user", "content": PROSE}]
    assert h.compress_payload(msgs) == msgs


def test_log_summary_keeps_real_first_line():
    text = "\n".join(f"conn {i} ok" for i in range(3000))
    out = h._compress_log(text)
    assert "conn 0 ok" in out and "conn #" not in out
