from __future__ import annotations

import asyncio
import importlib

from modules.memory.domain.dialog_text_pipeline_v1 import build_fact_uuid
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.application.service import MemoryService
from modules.memory.session_write import session_write

session_write_module = importlib.import_module("modules.memory.session_write")


def test_session_write_uses_tenant_scoped_fact_ids_when_shard_namespace_enabled(monkeypatch) -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        monkeypatch.setattr(
            session_write_module,
            "load_memory_config",
            lambda: {
                "memory": {
                    "vector_store": {
                        "sharding": {
                            "enabled": True,
                            "namespace_ids_by_tenant": True,
                        }
                    }
                }
            },
        )

        def extractor_missing_sample_id(_turns):
            return {
                "events": [],
                "knowledge": [
                    {
                        "op": "ADD",
                        "type": "fact",
                        "statement": "User likes sci-fi movies.",
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
            tenant_id="tenant-a",
            user_tokens=["u:alice"],
            session_id="same-session",
            turns=turns,
            tkg_extractor=extractor_missing_sample_id,
        )
        await session_write(
            svc,
            tenant_id="tenant-b",
            user_tokens=["u:bob"],
            session_id="same-session",
            turns=turns,
            tkg_extractor=extractor_missing_sample_id,
        )

        fact_ids = {
            entry.id
            for entry in vec.dump().values()
            if str(entry.metadata.get("node_type") or "") == "fact"
        }
        assert build_fact_uuid(sample_id="same-session", fact_idx=0, tenant_id="tenant-a", namespace_by_tenant=True) in fact_ids
        assert build_fact_uuid(sample_id="same-session", fact_idx=0, tenant_id="tenant-b", namespace_by_tenant=True) in fact_ids

    asyncio.run(_run())
