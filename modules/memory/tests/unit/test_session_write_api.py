from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid
from modules.memory.session_write import session_write


def test_session_write_idempotent_skip_existing_marker() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        def extractor(_turns):
            return {
                "events": [],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "fact",
                        "statement": "User likes sci-fi movies.",
                        "scope": "permanent",
                        "status": "n/a",
                        "importance": "medium",
                        "source_sample_id": "sess-1",
                        "source_turn_ids": ["D1:1"],
                    }
                ],
            }

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        r1 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-1",
            turns=turns,
            tkg_extractor=extractor,
            overwrite_existing=False,
        )
        assert r1["status"] == "ok"
        dumped1 = vec.dump()
        n1 = len(dumped1)

        r2 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-1",
            turns=turns,
            tkg_extractor=extractor,
            overwrite_existing=False,
        )
        assert r2["status"] == "skipped_existing"
        assert len(vec.dump()) == n1

    asyncio.run(_run())


def test_session_write_overwrite_existing_purges_graph() -> None:
    async def _run() -> None:
        class _GraphWithPurge(InMemGraphStore):
            def __init__(self) -> None:
                super().__init__()
                self.calls = []

            async def purge_source_except_events(  # type: ignore[override]
                self,
                *,
                tenant_id: str,
                source_id: str,
                keep_event_ids: list[str],
            ) -> dict:
                self.calls.append(("purge", tenant_id, source_id, list(keep_event_ids or [])))
                return {"events": 0}

            async def upsert_graph_v0(self, **_kwargs) -> None:  # type: ignore[override]
                self.calls.append(("upsert",))

        vec = InMemVectorStore()
        graph = _GraphWithPurge()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        def extractor(_turns):
            return {
                "events": [
                    {
                        "summary": "User mentions a fact",
                        "event_type": "Atomic",
                        "source_turn_ids": ["D1:1"],
                        "evidence_status": "mapped",
                    }
                ],
                "knowledge": [],
            }

        turns = [{"dia_id": "D1:1", "speaker": "User", "text": "Hello"}]
        r = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-ovr",
            turns=turns,
            tkg_extractor=extractor,
            overwrite_existing=True,
        )
        assert r["status"] == "ok"
        assert graph.calls
        assert graph.calls[0][0] == "upsert"
        assert graph.calls[1][:2] == ("purge", "t")
        assert graph.calls[1][2] == "dialog::sess-ovr"

    asyncio.run(_run())


def test_session_write_overwrite_existing_replaces_facts() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        def extractor_v1(_turns):
            return {
                "events": [],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "fact",
                        "statement": "User likes sci-fi movies.",
                        "scope": "permanent",
                        "status": "n/a",
                        "importance": "medium",
                        "source_sample_id": "sess-2",
                        "source_turn_ids": ["D1:1"],
                    },
                    {
                        "op": "ADD",
                        "type": "task",
                        "statement": "User will watch Dune tomorrow.",
                        "scope": "temporary",
                        "status": "open",
                        "importance": "low",
                        "source_sample_id": "sess-2",
                        "source_turn_ids": ["D1:2"],
                    },
                ],
            }

        def extractor_v2(_turns):
            return {
                "events": [],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "preference",
                        "statement": "User prefers fantasy movies.",
                        "scope": "until_changed",
                        "status": "n/a",
                        "importance": "high",
                        "source_sample_id": "sess-2",
                        "source_turn_ids": ["D1:1"],
                    }
                ],
            }

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-2",
            turns=turns,
            tkg_extractor=extractor_v1,
            overwrite_existing=False,
        )

        # read marker fact ids
        marker = next(
            e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker"
        )
        old_fact_ids = list(marker.metadata.get("fact_ids") or [])
        assert len(old_fact_ids) == 2
        assert old_fact_ids[0] in vec.dump()
        assert old_fact_ids[1] in vec.dump()

        r2 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-2",
            turns=turns,
            tkg_extractor=extractor_v2,
            overwrite_existing=True,
        )
        assert r2["status"] == "ok"
        assert r2["deleted_fact_ids"] == 1
        # stale fact hard-deleted (fact_idx=1)
        assert old_fact_ids[1] not in vec.dump()

        # marker updated with new fact ids
        marker2 = next(
            e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker"
        )
        new_fact_ids = list(marker2.metadata.get("fact_ids") or [])
        assert len(new_fact_ids) == 1
        assert new_fact_ids[0] in vec.dump()
        fact_entry = vec.dump()[new_fact_ids[0]]
        assert "fantasy" in (fact_entry.contents[0] if fact_entry.contents else "").lower()

    asyncio.run(_run())


def test_session_write_overwrite_existing_requires_graph_success() -> None:
    async def _run() -> None:
        class _FailGraph(InMemGraphStore):
            async def purge_source(self, *, tenant_id: str, source_id: str, delete_orphan_entities: bool = False) -> dict:  # type: ignore[override]
                return {"events": 0}

            async def upsert_graph_v0(self, **_kwargs) -> None:  # type: ignore[override]
                raise RuntimeError("graph down")

        vec = InMemVectorStore()
        graph = _FailGraph()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        def extractor(_turns):
            return {
                "events": [
                    {
                        "summary": "User mentions a fact",
                        "event_type": "Atomic",
                        "source_turn_ids": ["D1:1"],
                        "evidence_status": "mapped",
                    }
                ],
                "knowledge": [],
            }

        turns = [{"dia_id": "D1:1", "speaker": "User", "text": "Hello"}]
        res = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-ovr-fail",
            turns=turns,
            tkg_extractor=extractor,
            overwrite_existing=True,
            graph_policy="best_effort",
        )
        assert res["status"] == "failed"

    asyncio.run(_run())


def test_session_write_fact_ids_are_scoped_by_session_id_even_without_sample_id() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        def extractor_missing_sample_id(_turns):
            return {
                "events": [],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "fact",
                        "statement": "User likes sci-fi movies.",
                        "scope": "permanent",
                        "status": "n/a",
                        "importance": "medium",
                        # Intentionally no sample_id/source_sample_id.
                        "source_turn_ids": ["D1:1"],
                    }
                ],
            }

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        r1 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-no-sample-1",
            turns=turns,
            tkg_extractor=extractor_missing_sample_id,
        )
        assert r1["status"] == "ok"
        fact1 = next(e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "fact")
        assert fact1.id == generate_uuid("locomo.facts", "fact:sess-no-sample-1:0")

        await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-no-sample-2",
            turns=turns,
            tkg_extractor=extractor_missing_sample_id,
        )
        facts = [e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "fact"]
        ids = {f.id for f in facts}
        assert generate_uuid("locomo.facts", "fact:sess-no-sample-1:0") in ids
        assert generate_uuid("locomo.facts", "fact:sess-no-sample-2:0") in ids

    asyncio.run(_run())


def test_dedup_skip_prevents_cross_write_merge() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e1 = MemoryEntry(
            id="e1",
            kind="episodic",
            modality="text",
            contents=["same turn"],
            metadata={"tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "run_id": "s1"},
        )
        e2 = MemoryEntry(
            id="e2",
            kind="episodic",
            modality="text",
            contents=["same turn"],
            metadata={"tenant_id": "t", "user_id": ["u:1"], "memory_domain": "dialog", "run_id": "s1", "dedup_skip": True},
        )

        await svc.write([e1])
        assert len(vec.dump()) == 1

        await svc.write([e2])
        assert len(vec.dump()) == 2

    asyncio.run(_run())
