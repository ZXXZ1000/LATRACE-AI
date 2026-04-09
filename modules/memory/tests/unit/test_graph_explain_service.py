from __future__ import annotations

import asyncio
from typing import Any, Dict

from modules.memory.application.graph_service import GraphService
from modules.memory.infra.neo4j_store import Neo4jStore


class _StoreStub:
    def __init__(self) -> None:
        self.first_meeting_args: Dict[str, Any] | None = None
        self.event_evidence_args: Dict[str, Any] | None = None
        self.first_meeting_calls: int = 0
        self.event_evidence_calls: int = 0

    async def query_first_meeting(self, **kwargs):
        self.first_meeting_args = kwargs
        self.first_meeting_calls += 1
        return {
            "event_id": "event-1",
            "summary": "Met in lobby",
            "t_abs_start": "2025-01-01T00:00:00+00:00",
            "place_id": "place-1",
            "evidence_ids": ["ev-1"],
        }

    async def query_event_evidence(self, **kwargs):
        self.event_evidence_args = kwargs
        self.event_evidence_calls += 1
        return {
            "event": {"id": "ev-1"},
            "entities": [{"id": "ent-1"}],
            "places": [{"id": "place-1"}],
            "timeslices": [{"id": "ts-1"}],
            "evidences": [{"id": "e1"}],
            "utterances": [{"id": "utt-1"}],
            "utterance_speakers": [{"utterance_id": "utt-1", "entity_id": "ent-1"}],
            "knowledge": [{"id": "k1"}],
        }


def test_explain_first_meeting_shapes_result():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)
    stub = _StoreStub()
    svc.store = stub  # type: ignore[assignment]

    res = asyncio.run(
        svc.explain_first_meeting(
            tenant_id="tenant-a",
            me_id="me",
            other_id="alice",
        )
    )
    assert res["found"] is True
    assert res["event_id"] == "event-1"
    assert res["place_id"] == "place-1"
    assert stub.first_meeting_args == {
        "tenant_id": "tenant-a",
        "me_id": "me",
        "other_id": "alice",
    }


def test_explain_first_meeting_no_result():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)

    class _EmptyStore:
        async def query_first_meeting(self, **kwargs):
            return {}

    svc.store = _EmptyStore()  # type: ignore[assignment]

    res = asyncio.run(
        svc.explain_first_meeting(
            tenant_id="tenant-b",
            me_id="me",
            other_id="bob",
        )
    )
    assert res["found"] is False
    assert res["event_id"] is None
    assert res["evidence_ids"] == []


def test_explain_event_evidence_shapes_result():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)
    stub = _StoreStub()
    svc.store = stub  # type: ignore[assignment]

    res = asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-c",
            event_id="ev-1",
        )
    )
    assert res["event"]["id"] == "ev-1"
    assert res["entities"][0]["id"] == "ent-1"
    assert res["places"][0]["id"] == "place-1"
    assert res["timeslices"][0]["id"] == "ts-1"
    assert res["evidences"][0]["id"] == "e1"
    assert res["utterances"][0]["id"] == "utt-1"
    assert res["utterance_speakers"][0]["entity_id"] == "ent-1"
    assert res["knowledge"][0]["id"] == "k1"
    assert stub.event_evidence_args == {
        "tenant_id": "tenant-c",
        "event_id": "ev-1",
    }


def test_explain_event_evidence_no_result():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)

    class _EmptyStore:
        async def query_event_evidence(self, **kwargs):
            return {}

    svc.store = _EmptyStore()  # type: ignore[assignment]

    res = asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-d",
            event_id="ev-missing",
        )
    )
    assert res["event"] is None
    assert res["entities"] == []
    assert res["evidences"] == []
    assert res["knowledge"] == []


def test_explain_first_meeting_uses_cache():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)
    stub = _StoreStub()
    svc.store = stub  # type: ignore[assignment]
    svc._explain_cache_enabled = True  # type: ignore[attr-defined]
    svc._explain_cache_ttl_s = 60.0  # type: ignore[attr-defined]
    svc._explain_cache_max = 8  # type: ignore[attr-defined]

    # First call populates cache
    res1 = asyncio.run(
        svc.explain_first_meeting(
            tenant_id="tenant-cache",
            me_id="me",
            other_id="alice",
        )
    )
    # Second call with same key should hit cache (no extra store call)
    res2 = asyncio.run(
        svc.explain_first_meeting(
            tenant_id="tenant-cache",
            me_id="me",
            other_id="alice",
        )
    )
    assert stub.first_meeting_calls == 1
    assert res1 == res2


def test_explain_event_evidence_uses_cache():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)
    stub = _StoreStub()
    svc.store = stub  # type: ignore[assignment]
    svc._explain_cache_enabled = True  # type: ignore[attr-defined]
    svc._explain_cache_ttl_s = 60.0  # type: ignore[attr-defined]
    svc._explain_cache_max = 8  # type: ignore[attr-defined]

    res1 = asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-cache",
            event_id="ev-1",
        )
    )
    res2 = asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-cache",
            event_id="ev-1",
        )
    )
    assert stub.event_evidence_calls == 1
    assert res1 == res2


def test_explain_event_evidence_cache_key_includes_scope_filters():
    base_store = Neo4jStore({})
    svc = GraphService(base_store)
    stub = _StoreStub()
    svc.store = stub  # type: ignore[assignment]
    svc._explain_cache_enabled = True  # type: ignore[attr-defined]
    svc._explain_cache_ttl_s = 60.0  # type: ignore[attr-defined]
    svc._explain_cache_max = 8  # type: ignore[attr-defined]

    asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-cache",
            event_id="ev-1",
            user_ids=["u:a"],
            memory_domain="work",
        )
    )
    asyncio.run(
        svc.explain_event_evidence(
            tenant_id="tenant-cache",
            event_id="ev-1",
            user_ids=["u:b"],
            memory_domain="work",
        )
    )
    assert stub.event_evidence_calls == 2
