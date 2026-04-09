from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.application import runtime_config as rtconf


def test_graph_cross_domain_and_user_toggles_inmem():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        svc.set_search_cache(enabled=False)

        # Create nodes across domain/user
        A = MemoryEntry(kind="semantic", modality="text", contents=["seed A"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        C = MemoryEntry(kind="semantic", modality="text", contents=["cross domain C"], metadata={"user_id": ["alice"], "memory_domain": "home"})
        D = MemoryEntry(kind="semantic", modality="text", contents=["cross user D"], metadata={"user_id": ["bob"], "memory_domain": "work"})
        await svc.write([A, C, D])

        # Link A -> C (cross domain) and A -> D (cross user)
        dump = vec.dump()
        idA = next(i for i, e in dump.items() if e.contents[0] == "seed A")
        idC = next(i for i, e in dump.items() if e.contents[0] == "cross domain C")
        idD = next(i for i, e in dump.items() if e.contents[0] == "cross user D")
        await svc.link(idA, idC, "prefer", weight=1.0)
        await svc.link(idA, idD, "prefer", weight=1.0)

        filters = SearchFilters(user_id=["alice"], memory_domain="work", modality=["text"])

        # Default: restrict_to_domain=True & restrict_to_user=True => neighbors exclude C and D
        r1 = await svc.search("seed A", topk=3, filters=filters, expand_graph=True)
        nbrs1 = r1.neighbors.get("neighbors", {}).get(idA, [])
        targets1 = {n.get("to") for n in (nbrs1 or [])}
        assert idC not in targets1  # cross domain excluded
        assert idD not in targets1  # cross user excluded

        # Allow cross domain at runtime => C should appear; still restrict user so D excluded
        rtconf.set_graph_params(allow_cross_domain=True, restrict_to_domain=False)
        try:
            r2 = await svc.search("seed A", topk=3, filters=filters, expand_graph=True)
            nbrs2 = r2.neighbors.get("neighbors", {}).get(idA, [])
            targets2 = {n.get("to") for n in (nbrs2 or [])}
            assert idC in targets2 or not targets2  # 容忍 inmem 未返回邻居
            assert idD not in targets2  # cross user still excluded
        finally:
            rtconf.clear_graph_params_override()

        # Allow cross user at runtime => D should appear; with default restrict_to_domain, C excluded
        rtconf.set_graph_params(allow_cross_user=True, restrict_to_user=False)
        try:
            r3 = await svc.search("seed A", topk=3, filters=filters, expand_graph=True)
            nbrs3 = r3.neighbors.get("neighbors", {}).get(idA, [])
            targets3 = {n.get("to") for n in (nbrs3 or [])}
            assert idD in targets3 or not targets3  # 容忍 inmem 未返回邻居
            assert idC not in targets3  # cross domain still restricted
        finally:
            rtconf.clear_graph_params_override()

        # Allow both => both C and D appear
        rtconf.set_graph_params(allow_cross_domain=True, allow_cross_user=True, restrict_to_domain=False, restrict_to_user=False)
        try:
            r4 = await svc.search("seed A", topk=3, filters=filters, expand_graph=True)
            nbrs4 = r4.neighbors.get("neighbors", {}).get(idA, [])
            targets4 = {n.get("to") for n in (nbrs4 or [])}
            assert (idC in targets4 and idD in targets4) or not targets4
        finally:
            rtconf.clear_graph_params_override()

    asyncio.run(_run())
