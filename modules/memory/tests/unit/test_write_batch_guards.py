from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
try:
    os.makedirs(".cache/mpl", exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
except Exception:
    pass

from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.application.service import MemoryService


class _DummyVector:
    def __init__(self) -> None:
        self.calls: List[int] = []

    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:
        self.calls.append(len(entries))

    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int = 5, threshold: Optional[float] = None) -> list:
        # no neighbors to force ADD
        return []

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


class _DummyGraph:
    def __init__(self) -> None:
        self.node_calls: List[int] = []
        self.edge_calls: int = 0

    async def merge_nodes_edges_batch(self, entries: List[MemoryEntry], edges: Optional[List[Edge]] = None, *, chunk_size: int = 500) -> None:
        if entries:
            self.node_calls.append(len(entries))
        if edges:
            self.edge_calls += len(edges)

    async def merge_nodes_edges(self, entries: List[MemoryEntry], edges: Optional[List[Edge]] = None) -> None:
        if entries:
            self.node_calls.append(len(entries))
        if edges:
            self.edge_calls += len(edges)

    async def merge_rel(self, src_id: str, dst_id: str, rel_type: str, *, weight: Optional[float] = None) -> None:
        self.edge_calls += 1

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


class _DummyAudit:
    async def add_batch(self, event: str, entries: List[MemoryEntry]) -> str:
        return "v-1"

    async def add_one(self, event: str, obj_id: str, payload: Dict[str, Any], reason: Optional[str] = None) -> str:
        return "v-1"


def _new_entry(i: int, *, text_len: int = 8) -> MemoryEntry:
    return MemoryEntry(kind="semantic", modality="text", contents=["x" * text_len], metadata={"clip_id": i})


async def _aw(coro):
    return await coro


def test_write_is_chunked_by_items(event_loop=None):
    vec = _DummyVector()
    gra = _DummyGraph()
    aud = _DummyAudit()
    svc = MemoryService(vec, gra, aud)
    # force small chunk size
    svc._batch_flush_chunk_items = 10
    # build 25 entries
    entries = [_new_entry(i) for i in range(25)]
    # run write (no links)
    asyncio.run(_aw(svc.write(entries, [])))
    # vector writes in chunks [10,10,5]
    assert vec.calls == [10, 10, 5]
    # graph nodes also chunked
    assert gra.node_calls == [10, 10, 5]
    # no edges
    assert gra.edge_calls == 0


def test_enqueue_flush_on_bytes_threshold(event_loop=None):
    vec = _DummyVector()
    gra = _DummyGraph()
    aud = _DummyAudit()
    svc = MemoryService(vec, gra, aud)
    svc.enable_write_batching(enabled=True, max_items=1000)
    # very small bytes budget to trigger flush quickly
    svc._batch_max_bytes = 200
    # two entries each ~150 bytes -> should flush on second enqueue
    e1 = [_new_entry(1, text_len=150)]
    e2 = [_new_entry(2, text_len=150)]
    asyncio.run(_aw(svc.enqueue_write(e1)))
    # not flushed yet (under bytes threshold)
    assert len(svc._batch_entries) == 1
    asyncio.run(_aw(svc.enqueue_write(e2)))
    # flushed and cleared
    assert len(svc._batch_entries) == 0
    # vector writes called once with combined entries (chunking handled inside write)
    assert sum(vec.calls) >= 2
