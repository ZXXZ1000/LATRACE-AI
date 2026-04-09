from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_time_range_and_entities_filters_inmem():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        ts_old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        ts_new = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        e1 = MemoryEntry(kind="episodic", modality="text", contents=["厨房 做饭"], metadata={"source": "ctrl", "timestamp": ts_old, "entities": ["room:kitchen"]})
        e2 = MemoryEntry(kind="episodic", modality="text", contents=["客厅 看电影"], metadata={"source": "ctrl", "timestamp": ts_new, "entities": ["room:living"]})
        await svc.write([e1, e2])

        # time range gte now-2days should only include e2
        gte = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        res_time = await svc.search("客厅", topk=5, filters=SearchFilters(modality=["text"], time_range={"gte": gte}))
        assert all(h.entry.metadata.get("timestamp") >= gte for h in res_time.hits)

        # entities filter should only match living room
        res_ent = await svc.search("客厅", topk=5, filters=SearchFilters(modality=["text"], entities=["room:living"]))
        assert all("room:living" in (h.entry.metadata.get("entities") or []) for h in res_ent.hits)

    asyncio.run(_run())

