from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_write_entries_and_link_returns_version():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        service = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(kind="episodic", modality="text", contents=["18:02 进入客厅并开灯"], metadata={"source": "ctrl"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["更偏好晚上关灯"], metadata={"source": "mem0"})

        version = await service.write([e1, e2], links=[])
        assert version.value.startswith("v-ADD-batch")

        stored = vec.dump()
        assert len(stored) == 2
        ids = list(stored.keys())

        await service.link(ids[0], ids[1], "prefer", weight=0.5)
        edges = graph.dump_edges()
        assert any(src == ids[0] and dst == ids[1] and rel == "prefer" for (src, dst, rel, _w) in edges)

    asyncio.run(_run())
