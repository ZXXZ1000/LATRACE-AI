from __future__ import annotations

from modules.memory.application.dialog_tkg_unified_extractor_v1 import SYSTEM_PROMPT, parse_unified_json
import modules.memory.application.dialog_tkg_unified_extractor_v1 as unified_extractor


def test_unified_extractor_prompt_loaded() -> None:
    assert "extract BOTH semantic events and long-term knowledge" in SYSTEM_PROMPT


def test_parse_unified_json() -> None:
    raw = """
    {
      "events": [{"summary": "A", "source_turn_ids": ["D1:1"]}],
      "knowledge": [{"statement": "B", "source_turn_ids": ["D1:2"]}],
      "states": [{"subject_ref": "user", "property": "mood", "value": "happy", "source_turn_ids": ["D1:1"]}]
    }
    """
    parsed = parse_unified_json(raw)
    assert isinstance(parsed.get("events"), list)
    assert isinstance(parsed.get("knowledge"), list)
    assert isinstance(parsed.get("states"), list)
    assert parsed["events"][0]["summary"] == "A"
    assert parsed["knowledge"][0]["statement"] == "B"
    assert parsed["states"][0]["property"] == "mood"


def test_normalize_event_preserves_missing_confidence() -> None:
    ev = unified_extractor._normalize_event(
        {"summary": "A", "source_turn_ids": ["D1:1"]},
        allowed_turn_ids={"D1:1"},
    )
    assert ev is not None
    assert ev.get("event_confidence") is None
    assert ev.get("evidence_confidence") is None


def test_normalize_event_topic_fields() -> None:
    ev = unified_extractor._normalize_event(
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


def test_normalize_state_basic() -> None:
    st = unified_extractor._normalize_state(
        {"subject_ref": "user", "property": "job_status", "value": "employed", "source_turn_ids": ["D1:1"]},
        allowed_turn_ids={"D1:1"},
    )
    assert st is not None
    assert st.get("property") == "job_status"
    assert st.get("value") == "employed"


def test_normalize_state_mvp_allowlist(monkeypatch) -> None:
    monkeypatch.delenv("MEMORY_STATE_PROPERTIES", raising=False)
    st = unified_extractor._normalize_state(
        {"subject_ref": "user", "property": "stress_level", "value": "high", "source_turn_ids": ["D1:1"]},
        allowed_turn_ids={"D1:1"},
    )
    assert st is None


def test_language_consistency_validator() -> None:
    lang_map = {"c1": "zh", "e1": "en"}
    ok, errs = unified_extractor._validate_language_consistency(
        events=[
            {"summary": "用户报名课程", "source_turn_ids": ["c1"]},
            {"summary": "user booked a flight", "source_turn_ids": ["e1"]},
        ],
        knowledge=[
            {"statement": "学费是399元", "source_turn_ids": ["c1"]},
            {"statement": "The price was $120.", "source_turn_ids": ["e1"]},
        ],
        turn_lang_map=lang_map,
    )
    assert ok
    assert errs == []

    ok2, errs2 = unified_extractor._validate_language_consistency(
        events=[{"summary": "user booked a flight", "source_turn_ids": ["c1"]}],
        knowledge=[],
        turn_lang_map=lang_map,
    )
    assert not ok2
    assert errs2


def test_segment_turns_preserves_session_boundaries() -> None:
    turns = [
        {"dia_id": "D1:1", "session_idx": 1, "speaker": "User", "text": "Hi"},
        {"dia_id": "D1:2", "session_idx": 1, "speaker": "Assistant", "text": "Hello"},
        {"dia_id": "D2:1", "session_idx": 2, "speaker": "User", "text": "Next"},
    ]
    segments = unified_extractor._segment_turns(turns, max_turns=1)
    assert len(segments) == 3
    assert segments[0][0] == 1
    assert segments[1][0] == 1
    assert segments[2][0] == 2


class _FakeBatchEmbedder:
    def __init__(self) -> None:
        self.batch_calls = []

    def __call__(self, text: str) -> list[float]:
        raise AssertionError("encode_batch should be used for alignment")

    def encode_batch(self, texts: list[str], bsz: int | None = None) -> list[list[float]]:
        self.batch_calls.append((list(texts), bsz))
        out = []
        for t in texts:
            if "alpha" in t:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


def test_alignment_uses_batch_embedder() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "alpha"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "beta"},
    ]
    events = [
        {"summary": "alpha event", "desc": "", "source_turn_ids": [], "evidence_status": "unmapped"},
    ]
    embedder = _FakeBatchEmbedder()
    out = unified_extractor._align_events_with_turns(
        events=events,
        turns=turns,
        turn_ids=["D1:1", "D1:2"],
        min_score=0.0,
        top_k=1,
        embed=embedder,
        batch_size=2,
        embed_concurrency=2,
    )
    assert len(embedder.batch_calls) == 2
    assert out[0]["source_turn_ids"] == ["D1:1"]
    assert out[0]["evidence_status"] == "weak"


def test_unified_extractor_uses_single_fallback_reference_time(monkeypatch) -> None:
    class _FakeDatetime:
        calls = 0

        @classmethod
        def now(cls):  # type: ignore[override]
            cls.calls += 1
            from datetime import datetime as _dt
            if cls.calls == 1:
                return _dt(2025, 1, 1, 9, 0)
            return _dt(2025, 1, 1, 10, 0)

        @classmethod
        def fromisoformat(cls, value: str):  # type: ignore[override]
            from datetime import datetime as _dt
            return _dt.fromisoformat(value)

    class _FakeAdapter:
        def generate(self, *_args, **_kwargs) -> str:
            return '{"events": [], "knowledge": []}'

    monkeypatch.setattr(unified_extractor, "datetime", _FakeDatetime)
    monkeypatch.setattr(
        unified_extractor,
        "get_dialog_event_settings",
        lambda _cfg: {"segment_max_turns": 1, "event_extract_concurrency": 1},
    )

    contexts = []

    def _trace(payload):
        if payload.get("stage") == "build_unified_context" and payload.get("context"):
            contexts.append(str(payload.get("context")))

    extractor = unified_extractor.build_dialog_tkg_unified_extractor_v1_from_env(
        session_id="sess-ref",
        reference_time_iso=None,
        adapter=_FakeAdapter(),
        trace_hook=_trace,
        trace_include_context=True,
    )
    assert extractor is not None

    turns = [
        {"session_idx": 1, "speaker": "User", "text": "A"},
        {"session_idx": 1, "speaker": "Assistant", "text": "B"},
    ]
    extractor(turns)
    assert len(contexts) == 2
    ref_lines = [line for ctx in contexts for line in ctx.splitlines() if line.startswith("Reference Time: ")]
    assert len(ref_lines) == 2
    assert ref_lines[0] == ref_lines[1]
    assert _FakeDatetime.calls == 1


def test_unified_extractor_applies_max_events_per_session(monkeypatch) -> None:
    import ast
    import json

    class _FakeAdapter:
        def generate(self, messages, response_format=None) -> str:  # type: ignore[override]
            ctx = str(messages[-1].get("content") or "")
            lines = ctx.splitlines()
            allowed = []
            for i, line in enumerate(lines):
                if line.startswith("Allowed turn_ids"):
                    if i + 1 < len(lines):
                        allowed = ast.literal_eval(lines[i + 1])
                    break
            tid = allowed[0] if allowed else "t1"
            return json.dumps(
                {
                    "events": [
                        {
                            "summary": f"evt {tid}",
                            "source_turn_ids": [tid],
                            "evidence_status": "mapped",
                        }
                    ],
                    "knowledge": [],
                }
            )

    monkeypatch.setattr(
        unified_extractor,
        "get_dialog_event_settings",
        lambda _cfg: {
            "segment_max_turns": 1,
            "event_extract_concurrency": 1,
            "alignment_min_score": 0.0,
            "alignment_top_k": 1,
            "max_events_per_session": 1,
        },
    )
    monkeypatch.setattr(unified_extractor, "build_embedding_from_settings", lambda _cfg: lambda _t: [0.0])

    extractor = unified_extractor.build_dialog_tkg_unified_extractor_v1_from_env(
        session_id="sess-max",
        adapter=_FakeAdapter(),
    )
    assert extractor is not None

    turns = [
        {"session_idx": 1, "speaker": "User", "text": "A"},
        {"session_idx": 1, "speaker": "Assistant", "text": "B"},
        {"session_idx": 2, "speaker": "User", "text": "C"},
        {"session_idx": 2, "speaker": "Assistant", "text": "D"},
    ]
    out = extractor(turns)
    events = out.get("events") or []
    assert len(events) == 2
    assert events[0]["source_turn_ids"] == ["t1"]
    assert events[1]["source_turn_ids"] == ["t3"]
