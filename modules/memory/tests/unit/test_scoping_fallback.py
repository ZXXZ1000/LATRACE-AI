from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


async def _build_service() -> MemoryService:
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    # write three entries across user/domain/run
    e1 = MemoryEntry(kind="semantic", modality="text", contents=["work pizza"], metadata={"user_id": ["alice"], "memory_domain": "work", "run_id": "r1"})
    e2 = MemoryEntry(kind="semantic", modality="text", contents=["home pasta"], metadata={"user_id": ["alice"], "memory_domain": "home", "run_id": "r2"})
    e3 = MemoryEntry(kind="semantic", modality="text", contents=["work salad"], metadata={"user_id": ["bob"], "memory_domain": "work", "run_id": "r3"})
    await svc.write([e1, e2, e3])
    return svc


def test_scope_session_fallback_to_domain():
    async def _run():
        svc = await _build_service()
        # ask for pizza: session run_id is wrong, should fallback to domain(work) and user(alice)
        filters = SearchFilters(user_id=["alice"], memory_domain="work", run_id="rX")
        res = await svc.search("pizza", topk=5, filters=filters, expand_graph=False, scope="session")
        assert res.hits, "expected fallback hits for domain scope"
        assert res.trace.get("scope_used") in {"domain", "user"}
        assert any("pizza" in h.entry.contents[0] for h in res.hits)
    asyncio.run(_run())


def test_scope_default_domain_fallback_to_user_when_no_domain():
    async def _run():
        svc = await _build_service()
        # only user provided, domain missing → fallback to user
        filters = SearchFilters(user_id=["alice"])  # default scope is domain, not viable; fallback user
        res = await svc.search("home pasta", topk=5, filters=filters, expand_graph=False)
        assert res.hits, "expected hits for user scope"
        assert res.trace.get("scope_used") in {"user", "domain"}
        assert any("pasta" in h.entry.contents[0] for h in res.hits)
    asyncio.run(_run())


def test_scope_user_match_all_requires_both_users():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        e = MemoryEntry(kind="semantic", modality="text", contents=["shared note"], metadata={"user_id": ["alice", "bob"], "memory_domain": "home", "run_id": "r4"})
        await svc.write([e])
        # any: either user matches
        filters_any = SearchFilters(user_id=["alice"], memory_domain="home", user_match="any")
        r_any = await svc.search("shared", topk=5, filters=filters_any, expand_graph=False)
        assert r_any.hits, "any should match"
        # all: require both user present in entry
        filters_all = SearchFilters(user_id=["alice", "bob"], memory_domain="home", user_match="all")
        r_all = await svc.search("shared", topk=5, filters=filters_all, expand_graph=False)
        assert r_all.hits, "all should match for shared entry"
        # all: user set not fully included → no hits
        filters_all_fail = SearchFilters(user_id=["alice", "carol"], memory_domain="home", user_match="all")
        r_fail = await svc.search("shared", topk=5, filters=filters_all_fail, expand_graph=False)
        assert not r_fail.hits, "all should fail when entry doesn't include all users"
    asyncio.run(_run())


def test_cache_key_isolation_by_scope():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        svc.set_search_cache(enabled=True, ttl_seconds=60, max_entries=16)
        # write entries for alice in work domain
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["pizza scope"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        await svc.write([e1])

        calls = {"n": 0}
        orig = svc.vectors.search_vectors

        async def wrap(q, f, k, t):
            calls["n"] += 1
            return await orig(q, f, k, t)

        svc.vectors.search_vectors = wrap  # type: ignore
        # first: domain scope → miss, then cached
        await svc.search("pizza", topk=1, filters=SearchFilters(user_id=["alice"], memory_domain="work"), expand_graph=False, scope="domain")
        # second: user scope（不同 scope，应当是缓存未命中）
        await svc.search("pizza", topk=1, filters=SearchFilters(user_id=["alice"]), expand_graph=False, scope="user")
        assert calls["n"] == 2, f"expected vector search twice due to scope-isolated cache, got {calls['n']}"
    asyncio.run(_run())
