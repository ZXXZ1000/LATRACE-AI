from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService, SafetyError
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_batch_delete_policy_via_service_loop():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # write three
        items = [
            MemoryEntry(kind="semantic", modality="text", contents=[f"X{i}"], metadata={"source": "mem0"})
            for i in range(3)
        ]
        await svc.write(items)
        ids = list(svc.vectors.dump().keys())  # type: ignore[attr-defined]

        # turn on strict hard delete confirmation+reason
        svc.set_safety_policy(require_confirm_hard_delete=True, require_reason_delete=True)

        # without confirm/reason, each hard delete raises
        errs = 0
        for mid in ids:
            try:
                await svc.delete(mid, soft=False)
            except SafetyError:
                errs += 1
        assert errs == len(ids)

        # with confirm+reason, succeed
        for mid in ids:
            v = await svc.delete(mid, soft=False, reason="cleanup", confirm=True)
            assert v.value.startswith("v-DELETE-")

    asyncio.run(_run())


def test_batch_equivalence_links_require_confirm_via_loop():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        await svc.write([
            MemoryEntry(kind="semantic", modality="text", contents=["A"], metadata={"source": "mem0"}),
            MemoryEntry(kind="semantic", modality="text", contents=["B"], metadata={"source": "mem0"}),
            MemoryEntry(kind="semantic", modality="text", contents=["C"], metadata={"source": "mem0"}),
        ])
        ids = list(svc.vectors.dump().keys())  # type: ignore[attr-defined]
        pairs = [(ids[0], ids[1]), (ids[1], ids[2])]

        # without confirm, all should raise
        raised = 0
        for s, d in pairs:
            try:
                await svc.link(s, d, "equivalence")
            except SafetyError:
                raised += 1
        assert raised == len(pairs)

        # with confirm, succeed
        okc = 0
        for s, d in pairs:
            if await svc.link(s, d, "equivalence", confirm=True):
                okc += 1
        assert okc == len(pairs)

    asyncio.run(_run())

