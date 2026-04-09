from __future__ import annotations

from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1
from modules.memory.domain.dialog_tkg_vector_index_v1 import (
    TKG_DIALOG_UTTERANCE_INDEX_SOURCE_V1,
    build_dialog_tkg_utterance_index_entries_v1,
)


def test_dialog_tkg_utterance_index_builds_entries_with_mapping_ids() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
    ]
    events = [
        {
            "summary": "User expresses preference for sci-fi",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1", "D1:2"],
            "evidence_status": "mapped",
            "event_confidence": 0.8,
        }
    ]
    g = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-idx",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )
    r = build_dialog_tkg_utterance_index_entries_v1(
        tenant_id="t",
        session_id="sess-idx",
        user_tokens=["u:alice"],
        memory_domain="dialog",
        turns=turns,
        graph_build=g,
    )
    assert len(r.entries) == 2
    assert len(r.index_ids) == 2

    e0 = r.entries[0]
    assert e0.kind == "semantic"
    assert e0.modality == "text"
    assert e0.metadata.get("source") == TKG_DIALOG_UTTERANCE_INDEX_SOURCE_V1
    assert e0.metadata.get("tkg_utterance_id") == g.graph_ids["utterance_ids"][0]
    assert e0.metadata.get("tkg_event_id") == g.graph_ids["event_ids"][0]
    assert e0.metadata.get("tkg_timeslice_id") == g.graph_ids["timeslice_id"]
    assert e0.metadata.get("tkg_segment_id") == g.graph_ids["segment_id"]
    assert e0.metadata.get("dedup_skip") is True


def test_dialog_tkg_utterance_index_supports_multi_event_mapping() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "Hello"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "Hi"},
    ]
    events = [
        {
            "summary": "User greets",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        },
        {
            "summary": "Assistant responds",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        },
    ]
    g = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-multi",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
    )
    r = build_dialog_tkg_utterance_index_entries_v1(
        tenant_id="t",
        session_id="sess-multi",
        user_tokens=["u:alice"],
        memory_domain="dialog",
        turns=turns,
        graph_build=g,
    )
    md = r.entries[0].metadata
    assert md.get("tkg_event_id") is None
    assert set(md.get("tkg_event_ids") or []) == set(g.graph_ids["event_ids"])
