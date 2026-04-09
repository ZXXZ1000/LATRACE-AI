from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.application import runtime_config as rtconf
from modules.memory.application.config import (
    SEARCH_TOPK_HARD_LIMIT,
    GRAPH_MAX_HOPS_HARD_LIMIT,
    GRAPH_NEIGHBOR_CAP_HARD_LIMIT,
)
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_search_topk_is_clamped_to_hard_limit():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        # 写入少量条目
        entries = [
            MemoryEntry(kind="semantic", modality="text", contents=[f"item {i}"], metadata={"source": "mem0"})
            for i in range(5)
        ]
        await svc.write(entries)

        # 请求一个远大于硬上限的 topk，不应抛错，且实际使用的 topk 不超过硬上限
        res = await svc.search("item", topk=SEARCH_TOPK_HARD_LIMIT * 10, filters=SearchFilters(modality=["text"]))
        assert len(res.hits) <= SEARCH_TOPK_HARD_LIMIT

    asyncio.run(_run())


def test_graph_params_override_respects_hard_limits():
    # 设置超大 hops 和邻居 cap，期望被 clamp 到安全上限
    rtconf.clear_graph_params_override()
    rtconf.set_graph_params(
        rel_whitelist=["APPEARS_IN"],
        max_hops=GRAPH_MAX_HOPS_HARD_LIMIT * 10,
        neighbor_cap_per_seed=GRAPH_NEIGHBOR_CAP_HARD_LIMIT * 10,
        restrict_to_user=True,
        restrict_to_domain=True,
    )
    params = rtconf.get_graph_params_override()
    assert params.get("max_hops") <= GRAPH_MAX_HOPS_HARD_LIMIT
    assert params.get("neighbor_cap_per_seed") <= GRAPH_NEIGHBOR_CAP_HARD_LIMIT
