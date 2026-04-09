from __future__ import annotations

import pytest
from typing import Any, Dict, List

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


class _VecStub:
    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None):
        # very naive matcher: if query contains keywords, return corresponding entries
        hits: List[Dict[str, Any]] = []
        if "电视" in query or "TV" in query:
            e = MemoryEntry(id="tv1", kind="semantic", modality="text", contents=["电视 TV 设备"], metadata={})
            hits.append({"id": "tv1", "score": 0.8, "payload": e})
        if "沙发" in query or "sofa" in query:
            e = MemoryEntry(id="sofa1", kind="semantic", modality="text", contents=["沙发 sofa 家具"], metadata={})
            hits.append({"id": "sofa1", "score": 0.75, "payload": e})
        return hits

    async def upsert_vectors(self, entries):
        return None


class _GraphStub:
    async def expand_neighbors(self, *args, **kwargs):
        # build simple relations for tv1 and sofa1
        return {
            "neighbors": {
                "tv1": [{"to": "sofa1", "rel": "CO_OCCURS", "weight": 0.9, "hop": 1}],
                "sofa1": [{"to": "tv1", "rel": "APPEARS_IN", "weight": 0.8, "hop": 1}],
            }
        }

    async def merge_nodes_edges(self, entries, edges=None):
        return None


class _AuditNoop:
    async def add_batch(self, *args, **kwargs):
        return "v1"


@pytest.mark.anyio
async def test_object_search_basic():
    svc = MemoryService(_VecStub(), _GraphStub(), _AuditNoop())
    res = await svc.object_search(objects=["电视","沙发"], scene="客厅", filters=SearchFilters(memory_scope="vh::abcd"))
    assert isinstance(res, dict)
    items = res.get("items") or []
    # two objects should appear
    names = [it.get("object") for it in items]
    assert "电视" in names and "沙发" in names
    # each item should have hits
    assert all(len((it.get("hits") or [])) >= 1 for it in items)
    # and relations list present
    assert any(isinstance((it.get("relations") or []), list) for it in items)

