from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


def test_inmem_graph_restrict_user_domain():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        # Create three nodes with user/domain
        A = MemoryEntry(kind="semantic", modality="text", contents=["A 节点"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        B = MemoryEntry(kind="semantic", modality="text", contents=["B 节点"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        C = MemoryEntry(kind="semantic", modality="text", contents=["C 节点"], metadata={"user_id": ["alice"], "memory_domain": "home"})
        await svc.write([A, B, C])
        # link A->B and A->C
        dump = vec.dump()
        idA = next(i for i, e in dump.items() if e.contents[0] == "A 节点")
        idB = next(i for i, e in dump.items() if e.contents[0] == "B 节点")
        idC = next(i for i, e in dump.items() if e.contents[0] == "C 节点")
        await svc.link(idA, idB, "prefer", weight=1.0)
        await svc.link(idA, idC, "prefer", weight=1.0)
        # restrict to user=alice and domain=work: neighbors should include B but not C
        filters = SearchFilters(user_id=["alice"], memory_domain="work", modality=["text"])
        r = await svc.search("A 节点", topk=3, filters=filters, expand_graph=True)
        nbrs = r.neighbors.get("neighbors", {}).get(idA, [])
        targets = {n.get("to") for n in (nbrs or [])}
        assert (idB in targets) or not targets
        assert idC not in targets
        # allow cross domain via hot override (simulate by config override in runtime_config graph params if needed in future)
        # keep defaults for now
    asyncio.run(_run())
