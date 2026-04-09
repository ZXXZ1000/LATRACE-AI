from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.session_write import session_write


def test_session_write_requires_llm_fact_extractor_by_default(monkeypatch) -> None:
    async def _run() -> None:
        # Ensure environment behaves as "no LLM configured" for this test,
        # regardless of real API keys on the developer machine.
        monkeypatch.setattr(
            "modules.memory.application.dialog_tkg_unified_extractor_v1.build_dialog_tkg_unified_extractor_v1_from_env",
            lambda session_id, reference_time_iso=None: None,
        )

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        r = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-llm-required",
            turns=turns,
        )
        assert r["status"] == "failed"
        assert "LLM TKG unified extractor is not configured" in str(r.get("trace", {}).get("error") or "")

        marker = next(e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker")
        assert str(marker.metadata.get("status")) == "failed"

    asyncio.run(_run())
