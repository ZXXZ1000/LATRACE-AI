from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.application import runtime_config as rtconf
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


async def _svc_with(weights: dict) -> MemoryService:
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    rtconf.set_rerank_weights({
        "alpha_vector": 0.25,
        "beta_bm25": 0.25,
        "gamma_graph": 0.25,
        "delta_recency": 0.25,  # 保持权重和为1，便于测试 boosts
        **weights,
    })
    return svc


def test_user_boost_prefers_matching_user():
    async def _run():
        svc = await _svc_with({"user_boost": 1.0, "domain_boost": 0.0, "session_boost": 0.0})
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["same note"], metadata={"user_id": ["alice"], "memory_domain": "work"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["same note"], metadata={"user_id": ["bob"], "memory_domain": "work"})
        await svc.write([e1, e2])
        res = await svc.search("same", topk=2, filters=SearchFilters(user_id=["alice"], memory_domain="work"), expand_graph=False)
        assert res.hits, "expected hits"
        ids = [h.entry.metadata.get("user_id") for h in res.hits]
        assert ["alice"] in ids  # 至少包含匹配用户
        rtconf.clear_rerank_weights_override()
    asyncio.run(_run())


def test_domain_boost_prefers_matching_domain():
    async def _run():
        svc = await _svc_with({"user_boost": 0.0, "domain_boost": 1.0, "session_boost": 0.0})
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["same token"], metadata={"user_id": ["u"], "memory_domain": "work"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["same token"], metadata={"user_id": ["u"], "memory_domain": "home"})
        await svc.write([e1, e2])
        res = await svc.search("same", topk=2, filters=SearchFilters(user_id=["u"], memory_domain="work"), expand_graph=False)
        assert res.hits, "expected hits"
        domains = [h.entry.metadata.get("memory_domain") for h in res.hits]
        assert "work" in domains
        rtconf.clear_rerank_weights_override()
    asyncio.run(_run())


def test_session_boost_prefers_matching_run():
    async def _run():
        svc = await _svc_with({"user_boost": 0.0, "domain_boost": 0.0, "session_boost": 1.0})
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["tok"], metadata={"user_id": ["u"], "memory_domain": "work", "run_id": "r1"})
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["tok"], metadata={"user_id": ["u"], "memory_domain": "work", "run_id": "r2"})
        await svc.write([e1, e2])
        res = await svc.search("tok", topk=2, filters=SearchFilters(user_id=["u"], memory_domain="work", run_id="r1"), expand_graph=False, scope="session")
        assert res.hits, "expected hits"
        runs = [h.entry.metadata.get("run_id") for h in res.hits]
        assert "r1" in runs
        rtconf.clear_rerank_weights_override()
    asyncio.run(_run())
