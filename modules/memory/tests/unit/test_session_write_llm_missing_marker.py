"""Tests for session_write marker status when LLM is missing (best_effort mode).

When llm_policy="best_effort" and LLM config is missing:
- First write should set marker status to "completed_no_llm" (not "completed")
- Re-write should re-process (not skip) because status != "completed"
- This allows proper entity extraction when LLM becomes available
"""

from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.session_write import session_write


def test_session_write_llm_missing_best_effort_sets_completed_no_llm_marker(monkeypatch) -> None:
    """When LLM is missing with llm_policy=best_effort, marker should be 'completed_no_llm'."""
    async def _run() -> None:
        # Mock LLM extractor to return None (simulating missing LLM config)
        monkeypatch.setattr(
            "modules.memory.application.dialog_tkg_unified_extractor_v1.build_dialog_tkg_unified_extractor_v1_from_env",
            lambda session_id, reference_time_iso=None, adapter=None: None,
        )

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        # First write with llm_policy=best_effort (should succeed with completed_no_llm marker)
        r = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-llm-missing",
            turns=turns,
            llm_policy="best_effort",  # Don't fail on missing LLM
        )

        assert r["status"] == "ok", f"Expected status 'ok', got {r['status']}"
        assert r["trace"].get("facts_skipped_reason") == "llm_missing"

        # Check marker status is "completed_no_llm" (not "completed")
        marker = next(
            (e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker"),
            None
        )
        assert marker is not None, "Session marker not found"
        assert str(marker.metadata.get("status")) == "completed_no_llm", \
            f"Expected marker status 'completed_no_llm', got '{marker.metadata.get('status')}'"

    asyncio.run(_run())


def test_session_write_reprocesses_completed_no_llm_marker(monkeypatch) -> None:
    """Re-write should re-process (not skip) when marker is 'completed_no_llm'."""
    async def _run() -> None:
        # First call: mock LLM extractor to return None
        call_count = {"value": 0}

        def mock_extractor_first_none(session_id, reference_time_iso=None, adapter=None):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return None  # First call: no LLM
            # Second call: return a working extractor
            def extractor(turns):
                return {"events": [], "knowledge": [{"statement": "User likes sci-fi movies", "source_turn_ids": ["D1:1"]}]}
            return extractor

        monkeypatch.setattr(
            "modules.memory.application.dialog_tkg_unified_extractor_v1.build_dialog_tkg_unified_extractor_v1_from_env",
            mock_extractor_first_none,
        )

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        # First write (no LLM) - should set marker to "completed_no_llm"
        r1 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-reprocess",
            turns=turns,
            llm_policy="best_effort",
        )
        assert r1["status"] == "ok"
        assert r1["trace"].get("facts_skipped_reason") == "llm_missing"

        # Second write (with LLM) - should re-process (not skip)
        r2 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-reprocess",
            turns=turns,
            llm_policy="best_effort",
        )

        # Should NOT be skipped - status should be "ok", not "skipped_existing"
        assert r2["status"] == "ok", \
            f"Expected re-processing with status 'ok', got '{r2['status']}' (skipped_reason: {r2.get('trace', {}).get('skipped_reason')})"

        # facts_skipped_reason should be None (LLM was available this time)
        assert r2["trace"].get("facts_skipped_reason") is None, \
            f"Expected no facts_skipped_reason on re-process, got {r2['trace'].get('facts_skipped_reason')}"

    asyncio.run(_run())


def test_session_write_skips_completed_marker_as_before(monkeypatch) -> None:
    """Sessions with proper 'completed' marker should still be skipped on re-write."""
    async def _run() -> None:
        # Mock LLM extractor to always return a working extractor
        def mock_extractor(session_id, reference_time_iso=None, adapter=None):
            def extractor(turns):
                return {"events": [], "knowledge": [{"statement": "User likes sci-fi movies", "source_turn_ids": ["D1:1"]}]}
            return extractor

        monkeypatch.setattr(
            "modules.memory.application.dialog_tkg_unified_extractor_v1.build_dialog_tkg_unified_extractor_v1_from_env",
            mock_extractor,
        )

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        # First write (with LLM) - should set marker to "completed"
        r1 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-completed",
            turns=turns,
            llm_policy="best_effort",
        )
        assert r1["status"] == "ok"
        assert r1["trace"].get("facts_skipped_reason") is None

        # Verify marker is "completed" (not "completed_no_llm")
        marker = next(
            (e for e in vec.dump().values() if str(e.metadata.get("node_type") or "") == "session_marker"),
            None
        )
        assert marker is not None
        assert str(marker.metadata.get("status")) == "completed"

        # Second write - should be skipped
        r2 = await session_write(
            svc,
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-completed",
            turns=turns,
            llm_policy="best_effort",
        )

        assert r2["status"] == "skipped_existing", \
            f"Expected 'skipped_existing' for completed marker, got '{r2['status']}'"

    asyncio.run(_run())
