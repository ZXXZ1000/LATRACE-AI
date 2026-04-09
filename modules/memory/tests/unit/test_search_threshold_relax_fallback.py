from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore


class _FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:  # pragma: no cover
        return None

    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None = None) -> List[Dict[str, Any]]:
        self.calls.append({"query": query, "filters": dict(filters), "topk": topk, "threshold": threshold})
        # simulate: config threshold filters everything, but no-threshold returns something
        if threshold is not None:
            return []
        e = MemoryEntry(kind="episodic", modality="text", contents=["event clip=0"], metadata={"tenant_id": "t", "user_id": ["u"]})
        e.id = "id-1"
        return [{"id": "id-1", "score": 0.01, "payload": e}]

    async def fetch_text_corpus(self, filters: Dict[str, Any], *, limit: int = 500) -> List[Dict[str, Any]]:  # pragma: no cover
        return []


def test_search_relaxes_config_threshold_when_empty(monkeypatch):
    vec = _FakeVectorStore()
    gra = InMemGraphStore({})
    aud = AuditStore()
    svc = MemoryService(vec, gra, aud)

    # Force config-default ann.threshold=0.1
    monkeypatch.setattr(
        "modules.memory.application.service.MemoryService._get_cached_config",
        lambda self: {"memory": {"search": {"ann": {"threshold": 0.1, "relax_threshold_on_empty": True}}}},
    )

    async def _run():
        filters = SearchFilters.model_validate({"tenant_id": "t", "user_id": ["u"], "user_match": "all"})
        res = await svc.search("event", topk=5, filters=filters, expand_graph=False)
        assert res.hits, "expected relaxed-threshold fallback to return hits"
        assert vec.calls and vec.calls[0]["threshold"] == 0.1
        assert any(c["threshold"] is None for c in vec.calls), "expected second call without threshold"

    asyncio.run(_run())
