from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_audit_persistence_and_rollback_update_and_delete():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        # use in-memory sqlite
        audit = AuditStore({"sqlite_path": ":memory:"})
        svc = MemoryService(vec, graph, audit)

        e = MemoryEntry(kind="semantic", modality="text", contents=["原始内容"], metadata={"source": "mem0"})
        await svc.write([e])
        eid = next(iter(vec.dump().keys()))

        # update and capture version ID pattern
        v_upd = await svc.update(eid, {"contents": ["已更新内容"]})
        assert v_upd.value.startswith("v-UPDATE-")
        # rollback the update
        ok = await svc.rollback_version(v_upd.value)
        assert ok is True
        after_rb = vec.dump()[eid]
        assert after_rb.contents == ["原始内容"]

        # soft delete and rollback
        v_del = await svc.delete(eid, soft=True)
        assert v_del.value.startswith("v-DELETE-")
        ok2 = await svc.rollback_version(v_del.value)
        assert ok2 is True
        after_rb2 = vec.dump()[eid]
        assert after_rb2.metadata.get("is_deleted") is not True

    asyncio.run(_run())

