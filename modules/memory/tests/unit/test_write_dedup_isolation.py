from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


def test_write_dedup_does_not_merge_across_tenant_id() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["hello world"],
            metadata={"source": "sdk", "tenant_id": "t1", "user_id": ["u:1"], "memory_domain": "dialog"},
        )
        e2 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["hello world"],
            metadata={"source": "sdk", "tenant_id": "t2", "user_id": ["u:1"], "memory_domain": "dialog"},
        )
        await svc.write([e1])
        await svc.write([e2])

        dumped = vec.dump()
        assert len(dumped) == 2
        tenants = sorted(str(v.metadata.get("tenant_id")) for v in dumped.values())
        assert tenants == ["t1", "t2"]

    asyncio.run(_run())


def test_write_dedup_does_not_merge_across_user_id() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["same content"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog"},
        )
        e2 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["same content"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:2"], "memory_domain": "dialog"},
        )
        await svc.write([e1])
        await svc.write([e2])

        dumped = vec.dump()
        assert len(dumped) == 2
        users = sorted((v.metadata.get("user_id") or [""])[0] for v in dumped.values())
        assert users == ["u:1", "u:2"]

    asyncio.run(_run())


def test_write_dedup_does_not_merge_across_memory_domain() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["same content"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog"},
        )
        e2 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["same content"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "work"},
        )
        await svc.write([e1])
        await svc.write([e2])

        dumped = vec.dump()
        assert len(dumped) == 2
        domains = sorted(str(v.metadata.get("memory_domain")) for v in dumped.values())
        assert domains == ["dialog", "work"]

    asyncio.run(_run())


def test_write_dedup_merges_within_same_subject_scope() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["merge me"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "tag": "a"},
        )
        e2 = MemoryEntry(
            kind="semantic",
            modality="text",
            contents=["merge me"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "tag": "b"},
        )
        await svc.write([e1])
        await svc.write([e2])

        dumped = vec.dump()
        assert len(dumped) == 1
        only = next(iter(dumped.values()))
        assert only.contents == ["merge me"]
        assert only.metadata.get("tag") == "b"

    asyncio.run(_run())


def test_write_dedup_episodic_prefers_session_local_by_run_id() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["same event"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "run_id": "s1"},
        )
        e2 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["same event"],
            metadata={"source": "sdk", "tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "run_id": "s2"},
        )
        await svc.write([e1])
        await svc.write([e2])

        dumped = vec.dump()
        assert len(dumped) == 2
        runs = sorted(str(v.metadata.get("run_id")) for v in dumped.values())
        assert runs == ["s1", "s2"]

    asyncio.run(_run())

