from __future__ import annotations

from modules.memory.application.turn_mark_extractor_dialog_v1 import (
    apply_turn_marks,
    default_marks_keep_all,
    generate_pin_intents,
    pin_intents_to_facts,
    validate_and_normalize_marks,
)


def test_validate_marks_defaults_and_span() -> None:
    turns = [
        {"turn_id": "t0001", "role": "user", "text": "hello world"},
        {"turn_id": "t0002", "role": "assistant", "text": "ok"},
    ]
    marks = [
        {"turn_id": "t0001", "keep": True, "span": {"start": 0, "end": 5}, "category": "fact"},
        {"turn_id": "t0002", "keep": False},
    ]
    out = validate_and_normalize_marks(turns=turns, marks=marks)
    assert out[0]["turn_id"] == "t0001"
    assert out[0]["span"] == {"start": 0, "end": 5}
    assert out[0]["evidence_level"] == "S0_user_claim"
    assert out[0]["forget_policy"] in {"temporary", "permanent", "until_changed"}

    kept = apply_turn_marks(turns, out)
    assert kept[0]["text"] == "hello"


def test_default_marks_keep_all() -> None:
    turns = [{"turn_id": "t0001", "role": "user", "text": "x"}]
    marks = default_marks_keep_all(turns)
    assert marks[0]["keep"] is True
    assert marks[0]["turn_id"] == "t0001"


def test_generate_pin_intents_and_facts() -> None:
    turns = [
        {"turn_id": "t0001", "role": "user", "text": "please remember this"},
        {"turn_id": "t0002", "role": "assistant", "text": "note a"},
        {"turn_id": "t0003", "role": "assistant", "text": "note b"},
    ]
    marks = [
        {"turn_id": "t0002", "keep": True, "user_triggered_save": True, "category": "note"},
        {"turn_id": "t0003", "keep": True, "user_triggered_save": True, "category": "note"},
    ]
    norm = validate_and_normalize_marks(turns=turns, marks=marks)
    pins = generate_pin_intents(turns, norm, window=4)
    assert pins
    facts = pin_intents_to_facts(pin_intents=pins, turns=turns)
    assert facts
    assert "User pinned note" in facts[0]["statement"]


def test_validate_marks_best_effort_coerces_invalid_fields() -> None:
    turns = [{"turn_id": "t0001", "role": "assistant", "text": "ok"}]
    marks = [
        {
            "turn_id": "t0001",
            "keep": True,
            "category": "invalid_category",
            "subtype": "invalid_subtype",
            "evidence_level": "invalid_evidence",
            "importance": "oops",
            "span": {"start": 0, "end": 999},
            "user_triggered_save": "nope",
        }
    ]
    out = validate_and_normalize_marks(turns=turns, marks=marks, strict=False)
    assert out
    assert out[0]["category"] == "note"
    assert out[0]["subtype"] in {"profile", "constraint", "commitment", "decision", "tool_grounded_fact", "user_pinned_note"}
    assert out[0]["evidence_level"] in {"S0_user_claim", "S1_ai_inference", "S2_tool_grounded", "S3_user_confirmed"}
    assert out[0]["importance"] == 0.5
    assert out[0]["span"] is None
