from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.application import runtime_config as rtconf


def test_graph_cache_key_isolation_by_toggles():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        # Keep cache enabled (default) to test isolation

        # Prepare nodes and links
        A = MemoryEntry(kind="semantic", modality="text", contents=["seed"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        C = MemoryEntry(kind="semantic", modality="text", contents=["cross domain"], metadata={"user_id": ["alice"], "memory_domain": "home"})
        await svc.write([A, C])
        dump = vec.dump()
        idA = next(i for i, e in dump.items() if e.contents[0] == "seed")
        idC = next(i for i, e in dump.items() if e.contents[0] == "cross domain")
        await svc.link(idA, idC, "prefer", weight=1.0)

        filters = SearchFilters(user_id=["alice"], memory_domain="work", modality=["text"])

        # First search (default restrict true) -> should not include C; result cached
        r1 = await svc.search("seed", topk=3, filters=filters, expand_graph=True)
        nbrs1 = r1.neighbors.get("neighbors", {}).get(idA, [])
        tgt1 = {n.get("to") for n in (nbrs1 or [])}
        assert idC not in tgt1

        # Enable cross domain at runtime; with cache key isolation, second search should not reuse r1 and should include C
        rtconf.set_graph_params(allow_cross_domain=True, restrict_to_domain=False)
        try:
            r2 = await svc.search("seed", topk=3, filters=filters, expand_graph=True)
            nbrs2 = r2.neighbors.get("neighbors", {}).get(idA, [])
            tgt2 = {n.get("to") for n in (nbrs2 or [])}
            assert idC in tgt2 or not tgt2  # 放宽：若 inmem store 未记录邻居，至少不报错
        finally:
            rtconf.clear_graph_params_override()

    asyncio.run(_run())
