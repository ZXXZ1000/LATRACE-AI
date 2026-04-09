from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_graph_decay_edges_inmem():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["A"], metadata={"source": "mem0"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["B"], metadata={"source": "mem0"})
        await svc.write([e1, e2])
        ids = list(vec.dump().keys())
        await svc.link(ids[0], ids[1], "prefer", weight=2.0)
        w_before = graph.get_edge_weight(ids[0], ids[1], "prefer")
        assert w_before and w_before >= 2.0
        ok = await svc.decay_graph_edges(factor=0.5, rel_whitelist=["prefer"], min_weight=0.0)
        assert ok is True
        w_after = graph.get_edge_weight(ids[0], ids[1], "prefer")
        assert w_after and abs(w_after - (w_before * 0.5)) < 1e-6

    asyncio.run(_run())


def test_ttl_cleanup_via_service_inmem():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())
        e = MemoryEntry(kind="episodic", modality="text", contents=["过期测试"], metadata={"source": "ctrl", "ttl": 1, "created_at": "1970-01-01T00:00:00+00:00"})
        await svc.write([e])
        changed = await svc.run_ttl_cleanup_now()
        assert changed >= 1

    asyncio.run(_run())

