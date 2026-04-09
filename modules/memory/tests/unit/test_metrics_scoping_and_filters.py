from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.application.metrics import get_metrics


def _val(m: dict, key: str) -> int:
    try:
        return int(m.get(key, 0))
    except Exception:
        return 0


def test_metrics_scope_filter_and_cache_hits():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        svc.set_search_cache(enabled=True, ttl_seconds=60, max_entries=128)
        # seed entries across domains/users
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["alpha"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["beta"], metadata={"user_id": ["bob"], "memory_domain": "home"})
        await svc.write([e1, e2])

        before = get_metrics()

        # Search in domain scope for alice/work, expect scope_used=domain, filter user+domain applied
        filters = SearchFilters(user_id=["alice"], memory_domain="work")
        await svc.search("alpha", topk=1, filters=filters, expand_graph=False, scope="domain")

        after1 = get_metrics()
        assert _val(after1, "search_scope_used_domain_total") >= _val(before, "search_scope_used_domain_total") + 1
        assert _val(after1, "search_filter_applied_user_total") >= _val(before, "search_filter_applied_user_total") + 1
        assert _val(after1, "search_filter_applied_domain_total") >= _val(before, "search_filter_applied_domain_total") + 1
        # domain distribution for work should increase (at least by 1 hit)
        assert _val(after1, "domain_distribution_work_total") >= _val(before, "domain_distribution_work_total") + 1

        # second run should hit cache for the same scope
        await svc.search("alpha", topk=1, filters=filters, expand_graph=False, scope="domain")
        after2 = get_metrics()
        assert _val(after2, "search_cache_hits_scope_domain_total") >= _val(after1, "search_cache_hits_scope_domain_total") + 1

    asyncio.run(_run())


def test_sampling_log_has_scope_and_context():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        captured = []
        svc.set_search_sampler(lambda s: captured.append(s), enabled=True, rate=1.0)
        e = MemoryEntry(kind="semantic", modality="text", contents=["gamma"], metadata={"user_id": ["u"], "memory_domain": "home"})
        await svc.write([e])
        filters = SearchFilters(user_id=["u"], memory_domain="home")
        await svc.search("gamma", topk=1, filters=filters, expand_graph=False, scope="domain")
        assert captured, "expected at least one sampling record"
        s = captured[-1]
        assert "scope" in s and s["scope"] == "domain"
        assert s.get("user_ids") == ["u"]
        assert s.get("memory_domain") == "home"
    asyncio.run(_run())

