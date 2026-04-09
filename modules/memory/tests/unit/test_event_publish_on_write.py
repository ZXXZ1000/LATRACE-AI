from __future__ import annotations

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


@pytest.mark.anyio
async def test_memory_ready_event_published_on_write():
    captured = []

    def _publish(event: str, payload: dict) -> None:
        if str(event) == "memory_ready":
            captured.append(payload)

    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    svc.set_event_publisher(_publish)
    e1 = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={"source": "mem0", "clip_id": 1})
    e2 = MemoryEntry(kind="episodic", modality="text", contents=["world"], metadata={"source": "mem0", "clip_id": 1})
    ver = await svc.write([e1, e2])
    assert ver.value
    assert len(captured) >= 1
    evt = captured[-1]
    assert evt.get("count") == 2
    assert 1 in evt.get("clip_ids")
