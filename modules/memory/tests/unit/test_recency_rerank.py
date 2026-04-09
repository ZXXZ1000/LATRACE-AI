from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_recency_affects_ranking_when_delta_weight_positive():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        # same content, different timestamps
        old_ts = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        new_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

        e_old = MemoryEntry(kind="episodic", modality="text", contents=["客厅 开灯"], metadata={"source": "ctrl", "timestamp": old_ts})
        e_new = MemoryEntry(kind="episodic", modality="text", contents=["客厅 开灯"], metadata={"source": "ctrl", "timestamp": new_ts})
        await svc.write([e_old, e_new])

        res = await svc.search("客厅 开灯", topk=2, filters=SearchFilters(modality=["text"]))
        assert len(res.hits) == 2
        top = res.hits[0].entry.metadata.get("timestamp")
        # newer item should rank higher with positive delta weight
        assert top == new_ts

    asyncio.run(_run())

