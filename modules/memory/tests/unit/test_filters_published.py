from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


def test_search_default_excludes_unpublished() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e_pub = MemoryEntry(
            id="e_pub",
            kind="semantic",
            modality="text",
            contents=["alpha memory"],
            metadata={"user_id": ["u1"], "memory_domain": "dialog"},
            published=True,
        )
        e_unpub = MemoryEntry(
            id="e_unpub",
            kind="semantic",
            modality="text",
            contents=["alpha memory"],
            metadata={"user_id": ["u1"], "memory_domain": "dialog"},
            published=False,
        )
        await vec.upsert_vectors([e_pub, e_unpub])

        res = await svc.search(
            "alpha",
            topk=5,
            filters=SearchFilters(user_id=["u1"], memory_domain="dialog"),
            expand_graph=False,
        )
        ids = [h.id for h in res.hits]
        assert "e_pub" in ids
        assert "e_unpub" not in ids

    asyncio.run(_run())
