from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_search_mvp_returns_hits_and_hints():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        service = MemoryService(vec, graph, audit)

        # Seed entries
        e1 = MemoryEntry(kind="episodic", modality="text", contents=["打开客厅主灯"], metadata={"source": "ctrl"})
        e2 = MemoryEntry(kind="episodic", modality="text", contents=["关闭客厅主灯"], metadata={"source": "ctrl"})
        e3 = MemoryEntry(kind="semantic", modality="text", contents=["更偏好在晚上开灯"], metadata={"source": "mem0"})
        await service.write([e1, e2, e3])

        res = await service.search(
            query="开灯 客厅",
            topk=2,
            filters=SearchFilters(modality=["text"]),
            expand_graph=False,
        )
        assert len(res.hits) >= 1
        # top result should be related to "开灯"
        top = res.hits[0].entry.contents[0]
        assert "开灯" in top or "打开" in top
        assert isinstance(res.hints, str)

    asyncio.run(_run())
