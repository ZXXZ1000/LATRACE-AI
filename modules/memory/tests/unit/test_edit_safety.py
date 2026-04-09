from __future__ import annotations

import pytest

from modules.memory.application.service import MemoryService, SafetyError
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


@pytest.mark.anyio
async def test_delete_soft_allowed_but_hard_requires_confirmation():
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    e = MemoryEntry(kind="semantic", modality="text", contents=["安全 删除"], metadata={"source": "mem0"})
    await svc.write([e])
    eid = next(iter(svc.vectors.dump().keys()))  # type: ignore[attr-defined]

    # soft delete allowed
    v = await svc.delete(eid, soft=True, reason="test")
    assert v.value.startswith("v-DELETE-")

    # hard delete without confirm should raise
    svc.set_safety_policy(require_confirm_hard_delete=True, require_reason_delete=True)
    with pytest.raises(SafetyError):
        await svc.delete(eid, soft=False, reason=None)

    # provide confirmer that approves
    svc.set_safety_confirmer(lambda ctx: True)
    v2 = await svc.delete(eid, soft=False, reason="cleanup")
    assert v2.value.startswith("v-DELETE-")


@pytest.mark.anyio
async def test_link_equivalence_requires_confirmation():
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    a = MemoryEntry(kind="semantic", modality="text", contents=["A"], metadata={"source": "mem0"})
    b = MemoryEntry(kind="semantic", modality="text", contents=["B"], metadata={"source": "mem0"})
    await svc.write([a, b])
    ids = list(svc.vectors.dump().keys())  # type: ignore[attr-defined]
    # default sensitive rels include 'equivalence'
    with pytest.raises(SafetyError):
        await svc.link(ids[0], ids[1], "equivalence")

    # approve via confirmer
    svc.set_safety_confirmer(lambda ctx: True)
    ok = await svc.link(ids[0], ids[1], "equivalence")
    assert ok is True


@pytest.mark.anyio
async def test_update_non_sensitive_allowed():
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    e = MemoryEntry(kind="semantic", modality="text", contents=["旧"], metadata={"source": "mem0"})
    await svc.write([e])
    eid = next(iter(svc.vectors.dump().keys()))  # type: ignore[attr-defined]
    v = await svc.update(eid, {"contents": ["新"]}, reason="change")
    assert v.value.startswith("v-UPDATE-")
