from __future__ import annotations

from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1
from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid, make_event_id


def test_dialog_graph_upsert_v1_builds_minimal_tkg_structure() -> None:
    turns = [
        {"speaker": "User", "text": "I like sci-fi movies.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
        {"speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
    ]
    events = [
        {
            "summary": "User expresses preference for sci-fi movies",
            "event_type": "Atomic",
            "source_turn_ids": ["t1"],
            "evidence_status": "mapped",
            "event_confidence": 0.8,
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-1",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )

    req = build.request
    assert len(req.segments) == 1
    assert len(req.time_slices) == 1
    assert len(req.events) == 1
    assert len(req.utterances) == 2
    assert len(req.entities) == 2

    rels = [e.rel_type for e in req.edges]
    assert "COVERS_EVENT" in rels
    assert "SUPPORTED_BY" in rels
    assert "SPOKEN_BY" in rels
    assert "INVOLVES" in rels
    assert "TEMPORALLY_CONTAINS" in rels

    # All nodes/edges carry tenant_id for in-process validation.
    assert req.segments[0].tenant_id == "t"
    assert req.time_slices[0].tenant_id == "t"
    assert all(e.tenant_id == "t" for e in req.events)
    assert all(u.tenant_id == "t" for u in req.utterances)
    assert all(ent.tenant_id == "t" for ent in req.entities)
    assert all(ed.tenant_id == "t" for ed in req.edges)


def test_dialog_graph_upsert_v1_includes_knowledge_from_facts() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
    ]
    events = [
        {
            "summary": "User likes sci-fi movies",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
            "event_confidence": 0.8,
        }
    ]
    facts = [
        {
            "op": "ADD",
            "type": "preference",
            "statement": "User likes sci-fi movies.",
            "scope": "until_changed",
            "status": "n/a",
            "importance": "medium",
            "source_sample_id": "sess-k1",
            "source_turn_ids": ["D1:1"],
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-k1",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        facts_raw=facts,
        events_raw=events,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )

    req = build.request
    assert len(req.knowledge) == 1
    assert any(e.rel_type == "DERIVED_FROM" and e.src_type == "Knowledge" and e.dst_type == "Event" for e in req.edges)
    assert any(e.rel_type == "STATED_BY" and e.src_type == "Knowledge" and e.dst_type == "Entity" for e in req.edges)
    assert any(e.rel_type == "CONTAINS" and e.src_type == "TimeSlice" and e.dst_type == "Knowledge" for e in req.edges)


def test_dialog_graph_upsert_v1_consumes_llm_participants_and_fact_mentions() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "Alice", "text": "I met Bob and discussed Carol's timeline.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "Understood.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
    ]
    events = [
        {
            "summary": "Alice discusses project timeline with Bob",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
            "event_confidence": 0.9,
            "participants": ["Alice", "Bob", " Bob "],  # duplicate + overlap with speaker
        }
    ]
    facts = [
        {
            "op": "ADD",
            "type": "fact",
            "statement": "Bob and Carol are involved in the project timeline discussion.",
            "source_sample_id": "sess-mentions",
            "source_turn_ids": ["D1:1"],
            "mentions": ["Bob", "Carol", "Carol"],
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-mentions",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        facts_raw=facts,
        events_raw=events,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )

    req = build.request
    entity_by_name = {str(getattr(ent, "name", "")): str(getattr(ent, "id", "")) for ent in req.entities}
    assert "Alice" in entity_by_name
    assert "Bob" in entity_by_name
    assert "Carol" in entity_by_name

    ev_id = str(req.events[0].id)
    fact_id = str(req.knowledge[0].id)
    bob_id = entity_by_name["Bob"]
    carol_id = entity_by_name["Carol"]
    alice_id = entity_by_name["Alice"]

    # Event participants are consumed into Event-[INVOLVES]->Entity.
    assert any(e.rel_type == "INVOLVES" and str(e.src_id) == ev_id and str(e.dst_id) == bob_id for e in req.edges)

    # Speaker overlap ("Alice" in participants) should not create a duplicate INVOLVES edge.
    alice_event_involves = [
        e for e in req.edges if e.rel_type == "INVOLVES" and str(e.src_id) == ev_id and str(e.dst_id) == alice_id
    ]
    assert len(alice_event_involves) == 1

    # Knowledge mentions are consumed into Knowledge-[MENTIONS]->Entity.
    assert any(e.rel_type == "MENTIONS" and str(e.src_type) == "Knowledge" and str(e.src_id) == fact_id and str(e.dst_id) == bob_id for e in req.edges)
    assert any(e.rel_type == "MENTIONS" and str(e.src_type) == "Knowledge" and str(e.src_id) == fact_id and str(e.dst_id) == carol_id for e in req.edges)

    # graph_ids should include non-speaker entities as well (mainline participants/mentions).
    assert bob_id in [str(x) for x in (build.graph_ids.get("entity_ids") or [])]
    assert carol_id in [str(x) for x in (build.graph_ids.get("entity_ids") or [])]


def test_dialog_graph_upsert_v1_entity_id_is_case_insensitive_across_requests() -> None:
    build_speaker = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-a",
        user_tokens=["u:alice"],
        turns=[{"dia_id": "D1:1", "speaker": "Bob", "text": "hello"}],
        memory_domain="dialog",
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )
    bob_speaker = next(ent for ent in build_speaker.request.entities if str(getattr(ent, "name", "")) == "Bob")

    build_participant = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-b",
        user_tokens=["u:alice"],
        turns=[{"dia_id": "D2:1", "speaker": "Alice", "text": "I met bob."}],
        memory_domain="dialog",
        events_raw=[
            {
                "summary": "Alice met bob",
                "source_turn_ids": ["D2:1"],
                "participants": ["bob"],
            }
        ],
        reference_time_iso="2025-01-02T00:00:00+00:00",
        turn_interval_seconds=60,
    )
    bob_participant = next(ent for ent in build_participant.request.entities if str(getattr(ent, "name", "")) == "bob")

    assert str(bob_speaker.id) == str(bob_participant.id)


def test_dialog_graph_upsert_v1_maps_multi_session_turn_ids() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "alpha"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "beta"},
        {"dia_id": "D2:1", "speaker": "User", "text": "gamma"},
        {"dia_id": "D2:2", "speaker": "Assistant", "text": "delta"},
    ]
    events = [
        {
            "summary": "Second session starts",
            "event_type": "Atomic",
            "source_turn_ids": ["D2:1"],
            "evidence_status": "mapped",
        }
    ]
    facts = [
        {
            "op": "ADD",
            "type": "fact",
            "statement": "Gamma was mentioned",
            "source_sample_id": "sess-multi",
            "source_turn_ids": ["D2:1"],
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-multi",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
        facts_raw=facts,
        reference_time_iso="2025-01-01T00:00:00+00:00",
        turn_interval_seconds=60,
    )

    expected_utt_id = generate_uuid("tkg.dialog.utterance", "t|sess-multi|3")
    edges = build.request.edges
    assert any(
        e.rel_type == "SUPPORTED_BY"
        and e.src_type == "Event"
        and e.dst_type == "UtteranceEvidence"
        and e.dst_id == expected_utt_id
        for e in edges
    )
    assert any(
        e.rel_type == "SUPPORTED_BY"
        and e.src_type == "Knowledge"
        and e.dst_type == "UtteranceEvidence"
        and e.dst_id == expected_utt_id
        for e in edges
    )


def test_dialog_graph_upsert_v1_derives_time_bucket() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "alpha", "timestamp_iso": "2026-01-29T10:00:00+00:00"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "beta", "timestamp_iso": "2026-01-29T10:01:00+00:00"},
    ]
    events = [
        {
            "summary": "User mentioned a plan",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-tb",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
        reference_time_iso="2026-01-29T10:00:00+00:00",
        turn_interval_seconds=60,
    )
    ev = build.request.events[0]
    assert ev.time_bucket == ["2026", "2026-01", "2026-01-29"]


def test_dialog_graph_upsert_v1_logical_event_id_single_turn_unique() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "alpha"},
        {"dia_id": "D1:2", "speaker": "Assistant", "text": "beta"},
    ]
    events = [
        {
            "summary": "User mentions alpha",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        }
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-logical",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
    )
    assert len(build.request.events) == 1
    assert build.request.events[0].logical_event_id == make_event_id("sess-logical", "D1:1")


def test_dialog_graph_upsert_v1_logical_event_id_avoids_collision() -> None:
    turns = [
        {"dia_id": "D1:1", "speaker": "User", "text": "alpha"},
    ]
    events = [
        {
            "summary": "Alpha event A",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        },
        {
            "summary": "Alpha event B",
            "event_type": "Atomic",
            "source_turn_ids": ["D1:1"],
            "evidence_status": "mapped",
        },
    ]
    build = build_dialog_graph_upsert_v1(
        tenant_id="t",
        session_id="sess-collision",
        user_tokens=["u:alice"],
        turns=turns,
        memory_domain="dialog",
        events_raw=events,
    )
    assert len(build.request.events) == 2
    assert all(ev.logical_event_id is None for ev in build.request.events)
