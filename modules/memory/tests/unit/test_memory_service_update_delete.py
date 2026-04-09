from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from modules.memory.application.service import MemoryService
from modules.memory.application.ttl_jobs import run_ttl_cleanup
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_update_and_soft_delete_and_hard_delete():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e = MemoryEntry(kind="semantic", modality="text", contents=["偏好早上开窗"], metadata={"source": "mem0"})
        await svc.write([e])
        # get id back from store
        stored = vec.dump()
        assert len(stored) == 1
        eid = next(iter(stored.keys()))

        # update contents
        new_text = ["偏好晚上关灯"]
        v2 = await svc.update(eid, {"contents": new_text})
        assert v2.value.startswith("v-UPDATE-")
        post = vec.dump()[eid]
        assert post.contents == new_text
        assert post.metadata.get("updated_at") is not None
        assert post.metadata.get("hash") is not None

        # soft delete
        v3 = await svc.delete(eid, soft=True)
        assert v3.value.startswith("v-DELETE-")
        post2 = vec.dump()[eid]
        assert post2.metadata.get("is_deleted") is True

        # hard delete
        v4 = await svc.delete(eid, soft=False)
        assert v4.value.startswith("v-DELETE-")
        assert eid not in vec.dump()
        # graph node should be removed too
        assert eid not in graph.dump_nodes()

    asyncio.run(_run())


def test_ttl_cleanup_soft_deletes_expired():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        # write two entries, one with small ttl and older created_at
        e1 = MemoryEntry(kind="episodic", modality="text", contents=["打开客厅灯"], metadata={"source": "ctrl", "ttl": 1, "created_at": "1970-01-01T00:00:00+00:00"})
        e2 = MemoryEntry(kind="episodic", modality="text", contents=["关闭客厅灯"], metadata={"source": "ctrl", "ttl": 0})
        await svc.write([e1, e2])

        now = datetime.now(timezone.utc)
        changed = await run_ttl_cleanup(vec, now=now)
        assert changed >= 1
        # expired entry should be marked deleted
        dumped = vec.dump()
        deleted = [m for m in dumped.values() if m.metadata.get("is_deleted") is True]
        assert len(deleted) >= 1

    asyncio.run(_run())
