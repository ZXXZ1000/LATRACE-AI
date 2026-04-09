from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_search_cache_lru_eviction():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        svc.set_search_cache(enabled=True, ttl_seconds=60, max_entries=2)

        # seed some entries to have hits
        await svc.write([
            MemoryEntry(kind="semantic", modality="text", contents=["A"], metadata={"source": "mem0"}),
            MemoryEntry(kind="semantic", modality="text", contents=["B"], metadata={"source": "mem0"}),
            MemoryEntry(kind="semantic", modality="text", contents=["C"], metadata={"source": "mem0"}),
        ])

        # fill cache with Q1 and Q2
        await svc.search("A", topk=1, filters=SearchFilters(modality=["text"]))  # Q1
        await svc.search("B", topk=1, filters=SearchFilters(modality=["text"]))  # Q2
        # access Q1 again to make it most recently used
        await svc.search("A", topk=1, filters=SearchFilters(modality=["text"]))
        # add Q3 to trigger eviction (capacity=2)
        await svc.search("C", topk=1, filters=SearchFilters(modality=["text"]))

        # Now cache should contain Q1 and Q3, and Q2 should be evicted
        # We cannot inspect the private cache directly; re-run Q2 and expect a miss → it will not be cached if LRU wrong
        # Validate by timing access paths is brittle; instead, we do a functional check:
        # run Q2 twice now; the first run should be a miss (vector search happens), the second run should be a hit (no vector search).
        # We wrap vector search to count calls.
        calls = {"n": 0}
        orig = vec.search_vectors

        async def wrap(q, f, k, t):
            calls["n"] += 1
            return await orig(q, f, k, t)

        vec.search_vectors = wrap  # type: ignore
        await svc.search("B", topk=1, filters=SearchFilters(modality=["text"]))
        await svc.search("B", topk=1, filters=SearchFilters(modality=["text"]))
        assert calls["n"] == 1

    asyncio.run(_run())

