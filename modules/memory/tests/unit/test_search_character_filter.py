from __future__ import annotations

import asyncio
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


def test_search_filters_by_character_id():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # two image entries, different characters
        e1 = MemoryEntry(kind="semantic", modality="image", contents=["img_a"], metadata={"character_id": "Alice", "user_id": ["u"], "memory_domain": "home"})
        e2 = MemoryEntry(kind="semantic", modality="image", contents=["img_b"], metadata={"character_id": "Bob", "user_id": ["u"], "memory_domain": "home"})
        await svc.write([e1, e2])
        # filter for Alice only
        filters = SearchFilters(modality=["image"], character_id=["Alice"], memory_domain="home", user_id=["u"])
        res = await svc.search("", filters=filters, topk=10, expand_graph=False)
        assert res.hits, "expected hits filtered by character_id"
        for h in res.hits:
            assert h.entry.metadata.get("character_id") == "Alice"
    asyncio.run(_run())
