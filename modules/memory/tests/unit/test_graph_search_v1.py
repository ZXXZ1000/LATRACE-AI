from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional


class _StubStore:
    def __init__(self) -> None:
        self.search_calls: List[Dict[str, Any]] = []
        self.evidence_calls: List[Dict[str, str]] = []
        self.candidates: List[Dict[str, Any]] = [
            {"event_id": "e1", "summary": "one", "score": 2.0},
            {"event_id": "e2", "summary": "two", "score": 1.0},
        ]

    async def search_event_candidates(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.search_calls.append({"tenant_id": tenant_id, "query": query, "limit": limit, "source_id": source_id})
        return list(self.candidates)

    async def query_event_evidence(self, *, tenant_id: str, event_id: str) -> Dict[str, Any]:
        self.evidence_calls.append({"tenant_id": tenant_id, "event_id": event_id})
        return {
            "event": {"id": event_id, "summary": f"event-{event_id}"},
            "entities": [],
            "places": [],
            "timeslices": [],
            "evidences": [],
            "utterances": [],
        }


def test_graph_search_v1_empty_query_returns_empty():
    from modules.memory.application.graph_service import GraphService

    store = _StubStore()
    svc = GraphService(store)  # type: ignore[arg-type]
    res = asyncio.run(svc.search_events_v1(tenant_id="t1", query="   ", topk=5))
    assert res == {"query": "", "items": []}
    assert store.search_calls == []
    assert store.evidence_calls == []


def test_graph_search_v1_includes_evidence_bundle():
    from modules.memory.application.graph_service import GraphService

    store = _StubStore()
    svc = GraphService(store)  # type: ignore[arg-type]
    res = asyncio.run(svc.search_events_v1(tenant_id="t1", query="hello", topk=1, include_evidence=True))
    assert res["query"] == "hello"
    assert len(res["items"]) == 1
    assert res["items"][0]["event"]["id"] == "e1"
    assert store.search_calls == [{"tenant_id": "t1", "query": "hello", "limit": 1, "source_id": None}]
    assert store.evidence_calls == [{"tenant_id": "t1", "event_id": "e1"}]


def test_graph_search_v1_skips_evidence_when_disabled():
    from modules.memory.application.graph_service import GraphService

    store = _StubStore()
    svc = GraphService(store)  # type: ignore[arg-type]
    res = asyncio.run(svc.search_events_v1(tenant_id="t1", query="hello", topk=2, include_evidence=False))
    assert len(res["items"]) == 2
    assert all("event" not in it for it in res["items"])
    assert store.search_calls == [{"tenant_id": "t1", "query": "hello", "limit": 2, "source_id": None}]
    assert store.evidence_calls == []

