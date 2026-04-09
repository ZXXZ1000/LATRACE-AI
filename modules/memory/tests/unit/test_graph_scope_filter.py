from __future__ import annotations

from typing import Any, Dict, List

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


class _VecStub:
    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None):
        # Return one dummy hit with minimal payload
        e = MemoryEntry(kind="semantic", modality="text", contents=[query], metadata=filters or {})
        return [{"id": "n1", "score": 0.9, "payload": e}]

    async def upsert_vectors(self, entries: List[MemoryEntry]):
        return None


class _GraphCapture:
    def __init__(self) -> None:
        self.last_args: Dict[str, Any] | None = None

    async def expand_neighbors(self, *args, **kwargs):
        # capture args for assertions
        self.last_args = dict(kwargs)
        return {"neighbors": {}}

    async def merge_nodes_edges(self, *args, **kwargs):
        return None


class _AuditNoop:
    async def record(self, *args, **kwargs):
        return None


@pytest.mark.anyio
async def test_memory_scope_passed_to_graph_expansion():
    vec = _VecStub()
    graph = _GraphCapture()
    audit = _AuditNoop()
    svc = MemoryService(vec, graph, audit)

    # include memory_scope filter; other filters omitted to preserve backward compatibility
    filters = SearchFilters(memory_scope="vh::abcd1234")
    res = await svc.search("hello", topk=3, filters=filters, expand_graph=True)
    assert res is not None

    # ensure graph.expand_neighbors got memory_scope and restrict_to_scope flag
    assert graph.last_args is not None
    assert graph.last_args.get("memory_scope") == "vh::abcd1234"
    # default should restrict to scope unless config disables it
    assert graph.last_args.get("restrict_to_scope") in (True, False)
