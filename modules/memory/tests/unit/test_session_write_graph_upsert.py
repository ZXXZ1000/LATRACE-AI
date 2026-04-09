from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.application.service import MemoryService
from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.session_write import session_write


class _CapturingStore:
    def __init__(self, svc: MemoryService, *, fail_graph: bool = False) -> None:
        self._svc = svc
        self.fail_graph = bool(fail_graph)
        self.graph_reqs: List[GraphUpsertRequest] = []
        self.publish_calls: List[Dict[str, Any]] = []

    async def search(self, *args, **kwargs):
        return await self._svc.search(*args, **kwargs)

    async def write(self, *args, **kwargs):
        return await self._svc.write(*args, **kwargs)

    async def delete(self, *args, **kwargs):
        return await self._svc.delete(*args, **kwargs)

    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:
        if self.fail_graph:
            raise RuntimeError("neo4j_timeout")
        self.graph_reqs.append(body)

    async def publish_entries(self, *args, **kwargs):
        self.publish_calls.append({"args": args, "kwargs": kwargs})
        return await self._svc.publish_entries(*args, **kwargs)


class _GraphServiceCapturingStore(_CapturingStore):
    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:
        if self.fail_graph:
            raise RuntimeError("neo4j_timeout")
        # Simulate GraphService side-effect: assign vector IDs to TKG nodes.
        for ev in body.events or []:
            if getattr(ev, "text_vector_id", None) is None:
                ev.text_vector_id = f"tkg_event_vec::{ev.id}"
        for ent in body.entities or []:
            if getattr(ent, "text_vector_id", None) is None:
                ent.text_vector_id = f"tkg_entity_vec::{ent.id}"
        self.graph_reqs.append(body)


def test_session_write_graph_upsert_best_effort_calls_graph_upsert() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        store = _GraphServiceCapturingStore(svc)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
        ]

        r = await session_write(
            store,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-g1",
            turns=turns,
            extract=False,
            graph_policy="best_effort",
        )
        assert r["status"] == "ok"
        assert r["trace"]["graph_upsert_status"] == "ok"
        assert len(store.graph_reqs) == 1
        req = store.graph_reqs[0]
        assert len(req.events) == 0
        assert len(req.utterances) >= 1
        assert any(e.rel_type == "CONTAINS_EVIDENCE" for e in req.edges)

    asyncio.run(_run())


def test_session_write_graph_upsert_require_fails_when_graph_fails() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        store = _CapturingStore(svc, fail_graph=True)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
        ]

        r = await session_write(
            store,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-g2",
            turns=turns,
            extract=False,
            graph_policy="require",
        )
        assert r["status"] == "failed"

        # marker should be written as failed (retryable).
        marker = next(e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker")
        assert str(marker.metadata.get("status")) == "failed"

    asyncio.run(_run())


def test_session_write_can_disable_default_graph_upsert() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        store = _GraphServiceCapturingStore(svc)

        turns = [{"dia_id": "D1:1", "speaker": "User", "text": "Hello"}]
        r = await session_write(
            store,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-g3",
            turns=turns,
            extract=False,
            graph_upsert=False,
        )
        assert r["status"] == "ok"
        assert r["trace"]["graph_upsert_status"] == "skipped"
        assert store.graph_reqs == []
        # When graph_upsert is disabled, no TKG vector index entries should be written either.
        assert not any(
            (e.metadata.get("source") == "tkg_dialog_utterance_index_v1")
            for e in vec.dump().values()
        )

    asyncio.run(_run())


def test_session_write_publishes_graph_vector_ids() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        store = _GraphServiceCapturingStore(svc)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
        ]

        def _extractor(_turns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
            return {
                "events": [
                    {
                        "summary": "User expresses preference for sci-fi movies.",
                        "source_turn_ids": ["D1:1"],
                        "event_confidence": 0.9,
                        "evidence_confidence": 0.9,
                    }
                ],
                "knowledge": [],
            }

        r = await session_write(
            store,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-g4",
            turns=turns,
            extract=True,
            tkg_extractor=_extractor,
            graph_policy="require",
            write_facts=False,
        )
        assert r["status"] == "ok"
        assert store.graph_reqs
        req = store.graph_reqs[-1]
        assert req.events
        ev_vec_id = getattr(req.events[0], "text_vector_id", None)
        assert ev_vec_id, "Event text_vector_id should be set by GraphService"
        assert store.publish_calls
        entry_ids = list((store.publish_calls[-1]["kwargs"] or {}).get("entry_ids") or [])
        assert str(ev_vec_id) in [str(x) for x in entry_ids]

    asyncio.run(_run())


def test_session_write_consumes_llm_participants_and_mentions_in_mainline() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        store = _GraphServiceCapturingStore(svc)

        turns = [
            {"dia_id": "D1:1", "speaker": "Alice", "text": "I met Bob today at lunch.", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Okay.", "timestamp_iso": "2025-01-01T00:01:00+00:00"},
        ]

        def _extractor(_turns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
            return {
                "events": [
                    {
                        "summary": "Alice met Bob at lunch.",
                        "source_turn_ids": ["D1:1"],
                        "event_confidence": 0.9,
                        "evidence_confidence": 0.9,
                        "participants": ["Alice", "Bob"],
                    }
                ],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "fact",
                        "statement": "Bob asked Alice to share updates with Carol.",
                        "source_sample_id": "sess-mention-mainline",
                        "source_turn_ids": ["D1:1"],
                        "mentions": ["Bob", "Carol"],
                    }
                ],
            }

        r = await session_write(
            store,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-mention-mainline",
            turns=turns,
            extract=True,
            tkg_extractor=_extractor,
            graph_policy="require",
        )

        assert r["status"] == "ok"
        assert store.graph_reqs
        req = store.graph_reqs[-1]
        bob_entities = [ent for ent in req.entities if getattr(ent, "name", None) == "Bob"]
        carol_entities = [ent for ent in req.entities if getattr(ent, "name", None) == "Carol"]
        assert bob_entities, "expected participant/mention entity for Bob"
        assert carol_entities, "expected fact mention entity for Carol"
        bob_id = str(bob_entities[0].id)
        carol_id = str(carol_entities[0].id)

        assert any(e.rel_type == "INVOLVES" and str(e.dst_id) == bob_id and str(e.src_type) == "Event" for e in req.edges)
        assert any(e.rel_type == "MENTIONS" and str(e.src_type) == "Knowledge" and str(e.dst_id) == bob_id for e in req.edges)
        assert any(e.rel_type == "MENTIONS" and str(e.src_type) == "Knowledge" and str(e.dst_id) == carol_id for e in req.edges)
        assert "mention_entity" not in dict(r.get("trace") or {})

    asyncio.run(_run())
