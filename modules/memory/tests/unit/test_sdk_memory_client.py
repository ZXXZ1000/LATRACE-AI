from __future__ import annotations

import asyncio

from modules.memory.client import Memory
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_sdk_basic_add_search_update_history():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        m = Memory(svc)

        # add single text (no infer), with scoping context
        r = await m.add("我 喜欢 科幻 电影", user_id="alice", memory_domain="home", run_id="s1", infer=False)
        assert r.get("results") and r["results"][0]["id"]
        mid = r["results"][0]["id"]

        # search within domain
        s = await m.search("科幻", user_id="alice", memory_domain="home", scope="domain", topk=1)
        assert s.get("results") and s["results"][0]["id"] == mid

        # update content
        await m.update(mid, "我 很 喜欢 科幻 电影", reason="refine")

        # history
        hist = await m.history(mid)
        assert any(h.get("event") == "UPDATE" for h in hist)
    asyncio.run(_run())

