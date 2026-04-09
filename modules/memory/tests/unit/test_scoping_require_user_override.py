from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.application import runtime_config as rtconf
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


def test_require_user_override_blocks_search_without_user():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        e = MemoryEntry(kind="semantic", modality="text", contents=["灯 光"], metadata={"memory_domain": "work"})
        await svc.write([e])
        # default require_user=False => should return hit
        res0 = await svc.search("灯", topk=1, filters=SearchFilters(modality=["text"]))
        assert len(res0.hits) >= 1
        # set require_user=True => no user provided => empty
        rtconf.set_scoping_params(require_user=True)
        res1 = await svc.search("灯", topk=1, filters=SearchFilters(modality=["text"]))
        assert len(res1.hits) == 0
        rtconf.clear_scoping_override()
    asyncio.run(_run())

