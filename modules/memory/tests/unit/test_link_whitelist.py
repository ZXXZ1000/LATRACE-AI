from __future__ import annotations

import pytest

from modules.memory.application.service import MemoryService, SafetyError
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry


@pytest.mark.anyio
async def test_link_disallowed_relation_raises():
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    # write two nodes
    a = MemoryEntry(kind="semantic", modality="text", contents=["A"], metadata={"source": "mem0"})
    b = MemoryEntry(kind="semantic", modality="text", contents=["B"], metadata={"source": "mem0"})
    await svc.write([a, b])
    ids = list(svc.vectors.dump().keys())  # type: ignore
    with pytest.raises(SafetyError):
        await svc.link(ids[0], ids[1], "unknown_rel")

