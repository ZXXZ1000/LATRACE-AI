from __future__ import annotations

import asyncio

from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.contracts.memory_models import MemoryEntry


async def _prep_store():
    s = InMemVectorStore()
    e1 = MemoryEntry(
        id="e1",
        kind="semantic",
        modality="text",
        contents=["我 喜欢 披萨 在 工作"],
        metadata={"user_id": ["alice"], "memory_domain": "work", "run_id": "r1"},
    )
    e2 = MemoryEntry(
        id="e2",
        kind="semantic",
        modality="text",
        contents=["我 喜欢 披萨 在 家庭"],
        metadata={"user_id": ["bob"], "memory_domain": "home", "run_id": "r2"},
    )
    e3 = MemoryEntry(
        id="e3",
        kind="semantic",
        modality="text",
        contents=["我们 都 喜欢 披萨 在 家庭"],
        metadata={"user_id": ["alice", "bob"], "memory_domain": "home", "run_id": "r3"},
    )
    await s.upsert_vectors([e1, e2, e3])
    return s


def test_inmem_filter_user_domain_any():
    async def _run():
        s = await _prep_store()
        filters = {
            "user_id": ["alice"],
            "user_match": "any",
            "memory_domain": "work",
        }
        res = await s.search_vectors("披萨", filters, topk=10, threshold=None)
        ids = [r["id"] for r in res]
        assert ids == ["e1"], f"expected only e1, got {ids}"
    asyncio.run(_run())


def test_inmem_filter_user_domain_all():
    async def _run():
        s = await _prep_store()
        filters = {
            "user_id": ["alice", "bob"],
            "user_match": "all",
            "memory_domain": "home",
        }
        res = await s.search_vectors("披萨", filters, topk=10, threshold=None)
        ids = [r["id"] for r in res]
        assert ids == ["e3"], f"expected only e3, got {ids}"
    asyncio.run(_run())


def test_inmem_filter_run_id_exact():
    async def _run():
        s = await _prep_store()
        filters = {
            "run_id": "r1",
        }
        res = await s.search_vectors("披萨", filters, topk=10, threshold=None)
        ids = [r["id"] for r in res]
        assert ids == ["e1"], f"expected only e1 by run_id, got {ids}"
    asyncio.run(_run())

