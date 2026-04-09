from __future__ import annotations

from modules.memory.application.event_extractor_dialog_tkg_v1 import (
    SYSTEM_PROMPT,
    build_event_context,
    parse_event_json,
)
import modules.memory.application.event_extractor_dialog_tkg_v1 as event_extractor


def test_event_extractor_prompt_loaded() -> None:
    assert "Event != turn" in SYSTEM_PROMPT
    assert "strict JSON" in SYSTEM_PROMPT


def test_build_event_context_includes_allowed_turn_ids() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "Hello."},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "Hi."},
    ]
    ctx, ids = build_event_context(session_id="sess-1", turns=turns)
    assert "Allowed turn_ids" in ctx
    assert ids == ["D1:1", "D1:2"]


def test_parse_event_json_reads_events_list() -> None:
    raw = '{"events":[{"summary":"User plans a meeting","event_type":"Atomic","source_turn_ids":["D1:1"],"evidence_status":"mapped"}]}'
    items = parse_event_json(raw)
    assert isinstance(items, list)
    assert items and items[0].get("summary") == "User plans a meeting"


def test_normalize_event_topic_fields() -> None:
    ev = event_extractor._normalize_event(
        {
            "summary": "Trip planning",
            "source_turn_ids": ["D1:1"],
            "topic_id": "日本旅行",
            "topic_path": "travel/japan",
            "tags": ["travel", "planning", "travel"],
            "keywords": "机票",
        },
        allowed_turn_ids={"D1:1"},
    )
    assert ev is not None
    assert ev.get("topic_id") == "日本旅行"
    assert ev.get("topic_path") == "travel/japan"
    assert ev.get("tags") == ["travel", "planning"]
    assert ev.get("keywords") == ["机票"]


def test_segment_turns_respects_session_boundaries() -> None:
    turns = [
        {"session_idx": 1, "text": "t1"},
        {"session_idx": 1, "text": "t2"},
        {"session_idx": 1, "text": "t3"},
        {"session_idx": 2, "text": "t4"},
        {"session_idx": 2, "text": "t5"},
    ]
    segments = event_extractor._segment_turns(turns, max_turns=2)
    assert len(segments) == 3
    assert segments[0][0] == 1
    assert all(t.get("session_idx") == 1 for t in segments[0][1])
    assert segments[1][0] == 1
    assert all(t.get("session_idx") == 1 for t in segments[1][1])
    assert segments[2][0] == 2
    assert all(t.get("session_idx") == 2 for t in segments[2][1])
