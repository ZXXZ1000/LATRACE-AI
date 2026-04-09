from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def _setup_service():
    vec = InMemVectorStore()
    graph = InMemGraphStore()
    audit = AuditStore()
    svc = MemoryService(vec, graph, audit)
    return svc, vec, graph


def test_decider_update_delete_none_add_paths():
    async def run_case(action: str):
        svc, vec, _ = _setup_service()
        # seed old memory
        old = MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨"], metadata={"source": "mem0"})
        await svc.write([old])
        old_id = next(iter(vec.dump().keys()))

        def decider(existing, new):
            # always operate on first neighbor if present
            target = existing[0].id if existing else None
            return (action, target)

        svc.set_update_decider(decider)

        if action == "UPDATE":
            new = MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨 加料"], metadata={"source": "mem0"})
            await svc.write([new])
            dumped = vec.dump()
            assert len(dumped) == 1
            entry = dumped[old_id]
            assert any("加料" in c for c in entry.contents)
        elif action == "DELETE":
            new = MemoryEntry(kind="semantic", modality="text", contents=["不喜欢奶酪披萨"], metadata={"source": "mem0"})
            await svc.write([new])
            dumped = vec.dump()
            assert len(dumped) == 1
            entry = dumped[old_id]
            assert entry.metadata.get("is_deleted") is True
        elif action == "NONE":
            new = MemoryEntry(kind="semantic", modality="text", contents=["喜欢奶酪披萨"], metadata={"source": "mem0"})
            await svc.write([new])
            dumped = vec.dump()
            assert len(dumped) == 1
        elif action == "ADD":
            new = MemoryEntry(kind="semantic", modality="text", contents=["新增偏好 巧克力"], metadata={"source": "mem0"})
            await svc.write([new])
            dumped = vec.dump()
            assert len(dumped) == 2

    for a in ("UPDATE", "DELETE", "NONE", "ADD"):
        asyncio.run(run_case(a))

