import os
import json
import pytest

from providers.router import lesson_store, distill, ethics_gate


def test_benign_lesson_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(lesson_store, "LESSON_STORE_PATH", str(tmp_path / "lessons.json"))
    
    # Benign lesson
    lesson_text = "Use descriptive variable names."
    lesson_store.save_lesson(lesson_text)
    
    lessons = lesson_store.load_lessons()
    assert any(lesson_text in l.get("content", "") for l in lessons)


def test_sensitive_lesson_diverted_to_ethics_review(tmp_path, monkeypatch):
    ethics_file = tmp_path / "ethics_review.jsonl"
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "ethics_review.jsonl") if ".pal/ethics_review.jsonl" in p else p)
    monkeypatch.setattr(lesson_store, "LESSON_STORE_PATH", str(tmp_path / "lessons.json"))
    
    # Force ethics gate to consider this sensitive
    monkeypatch.setattr(ethics_gate, "is_ethically_sensitive", lambda text: "exploit" in text.lower())
    
    sensitive_text = "How to exploit CVE-2023-XXXX for RCE"
    lesson_store.save_lesson(sensitive_text)
    
    # Verify not persisted in regular lessons
    lessons = lesson_store.load_lessons()
    assert not any(sensitive_text in l.get("content", "") for l in lessons)
    
    # Verify written to ethics_review.jsonl
    assert ethics_file.exists()
    lines = ethics_file.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["lesson"] == sensitive_text


def test_pal_learn_auto_flag(monkeypatch):
    # Test PAL_LEARN_AUTO=0 vs 1 behavior in distill approval gate
    called = []
    
    def mock_apply(proposal):
        called.append(proposal)
        
    monkeypatch.setattr(distill, "_apply_proposal_internal", mock_apply)
    
    proposal = {"id": "test-1"}
    
    # With PAL_LEARN_AUTO=1 (default), should auto-apply
    monkeypatch.setenv("PAL_LEARN_AUTO", "1")
    # If distill.process_proposal or equivalent checks PAL_LEARN_AUTO
    # Let's test the distill apply mechanism directly if applicable
    assert distill.should_auto_apply() is True

    # With PAL_LEARN_AUTO=0, should require manual approval
    monkeypatch.setenv("PAL_LEARN_AUTO", "0")
    assert distill.should_auto_apply() is False
