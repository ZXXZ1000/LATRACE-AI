from __future__ import annotations

import pytest
from typing import Any, Dict

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


class _VecStub:
    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None):
        # Build two episodic and one semantic entries
        e1 = MemoryEntry(id="e1", kind="episodic", modality="text", contents=["A happens"], metadata={"timestamp": 10.0, "clip_id": 0})
        e2 = MemoryEntry(id="e2", kind="episodic", modality="text", contents=["B happens"], metadata={"timestamp": 20.0, "clip_id": 1})
        s1 = MemoryEntry(id="s1", kind="semantic", modality="text", contents=["object TV"], metadata={})
        return [
            {"id": "e1", "score": 0.9, "payload": e1},
            {"id": "e2", "score": 0.8, "payload": e2},
            {"id": "s1", "score": 0.5, "payload": s1},
        ]

    async def upsert_vectors(self, entries):
        return None


class _GraphStub:
    async def expand_neighbors(self, *args, **kwargs):
        # Simple neighbors for e1
        return {"neighbors": {"e1": [{"to": "e2", "rel": "TEMPORAL_NEXT", "weight": 1.0, "hop": 1}]}}

    async def merge_nodes_edges(self, entries, edges=None):
        return None


class _AuditNoop:
    async def add_batch(self, *args, **kwargs):
        return "v1"


@pytest.mark.anyio
async def test_timeline_summary_basic():
    svc = MemoryService(_VecStub(), _GraphStub(), _AuditNoop())
    res = await svc.timeline_summary(query="what happened?", filters=SearchFilters(memory_scope="vh::abcd1234"))
    assert isinstance(res, dict)
    assert res.get("segments") >= 1
    events = res.get("events") or []
    # Expect first event corresponds to clip 0 with A happens
    assert any("A happens" in (ev.get("description") or "") for ev in events)
    # Neighbors present for first seed
    assert any(isinstance(ev.get("neighbors"), list) for ev in events)

