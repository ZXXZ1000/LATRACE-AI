from __future__ import annotations

import asyncio
import json

from modules.memory.application.service import MemoryService
from modules.memory.application.decider_mem0 import Mem0UpdateDecider
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def make_llm(action: str, id_: str | None = None):
    def llm(messages, response_format=None):
        # Return a mem0-style JSON decision
        if action in ("UPDATE", "DELETE") and id_ is None:
            # by prompt convention, id "0" represents the first neighbor
            tid = "0"
        else:
            tid = id_ or "0"
        obj = {"memory": [{"id": tid, "text": "", "event": action}]}
        return json.dumps(obj, ensure_ascii=False)

    return llm


def test_mem0_decider_update_delete_none_add():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        # seed existing
        base = MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨"], metadata={"source": "mem0"})
        await svc.write([base])
        base_id = next(iter(vec.dump().keys()))

        # UPDATE
        svc.set_update_decider(Mem0UpdateDecider(make_llm("UPDATE")).decide)
        await svc.write([MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨 加料"], metadata={"source": "mem0"})])
        assert any("加料" in c for c in vec.dump()[base_id].contents)

        # DELETE
        svc.set_update_decider(Mem0UpdateDecider(make_llm("DELETE")).decide)
        await svc.write([MemoryEntry(kind="semantic", modality="text", contents=["不喜欢奶酪披萨"], metadata={"source": "mem0"})])
        assert vec.dump()[base_id].metadata.get("is_deleted") is True

        # NONE
        svc.set_update_decider(Mem0UpdateDecider(make_llm("NONE")).decide)
        await svc.write([MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨"], metadata={"source": "mem0"})])
        # no new entries created
        assert len(vec.dump()) == 1

        # ADD
        svc.set_update_decider(Mem0UpdateDecider(make_llm("ADD")).decide)
        await svc.write([MemoryEntry(kind="semantic", modality="text", contents=["新增偏好 巧克力"], metadata={"source": "mem0"})])
        assert len(vec.dump()) == 2

    asyncio.run(_run())

