from __future__ import annotations

import asyncio
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


def test_query_character_prefix_expands_to_filter_and_widens_modalities():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        e1 = MemoryEntry(kind="semantic", modality="image", contents=["img_a"], metadata={"character_id": "Alice"})
        e2 = MemoryEntry(kind="semantic", modality="audio", contents=["aud_b"], metadata={"character_id": "Bob"})
        await svc.write([e1, e2])
        res = await svc.search("character:Alice", topk=10, filters=SearchFilters(modality=["image","audio"]), expand_graph=False)
        assert res.hits, "character query should produce hits"
        for h in res.hits:
            assert h.entry.metadata.get("character_id") == "Alice"
    asyncio.run(_run())
