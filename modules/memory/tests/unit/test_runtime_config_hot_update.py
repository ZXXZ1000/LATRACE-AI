from __future__ import annotations

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.application import runtime_config as rtconf
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


@pytest.mark.anyio
async def test_rerank_and_graph_whitelist_hot_update():
    vec = InMemVectorStore()
    graph = InMemGraphStore()
    audit = AuditStore()
    svc = MemoryService(vec, graph, audit)

    e1 = MemoryEntry(kind="semantic", modality="text", contents=["我 喜欢 奶酪 披萨"], metadata={"source": "mem0"})
    e2 = MemoryEntry(kind="semantic", modality="text", contents=["我 不 喜欢 奶酪 披萨"], metadata={"source": "mem0"})
    await svc.write([e1, e2])

    # 建两条不同关系
    await svc.link(e1.id, e2.id, "prefer", weight=2.0)
    await svc.link(e2.id, e1.id, "executed", weight=3.0)

    # 默认搜索（无覆盖）：
    res0 = await svc.search("奶酪 披萨", topk=2, expand_graph=True)
    assert len(res0.hits) >= 2
    # 默认邻域包含 prefer/executed 至少一种
    n0 = res0.neighbors.get("neighbors", {})
    assert isinstance(n0, dict)

    # 热更新：仅允许 prefer 关系 + 调整权重（强调 BM25，弱化向量）
    rtconf.set_graph_params(rel_whitelist=["prefer"], max_hops=1, neighbor_cap_per_seed=5)
    rtconf.set_rerank_weights({"alpha_vector": 0.0, "beta_bm25": 1.0, "gamma_graph": 0.0, "delta_recency": 0.0})

    res1 = await svc.search("奶酪 披萨", topk=2, expand_graph=True)
    n1 = res1.neighbors.get("neighbors", {})
    # 所有邻居的关系应仅包含 prefer
    for nid, lst in (n1 or {}).items():
        for item in lst or []:
            assert item.get("rel") == "prefer"

    # 恢复（清理覆盖），避免影响其他测试
    rtconf.clear_rerank_weights_override()
    rtconf.clear_graph_params_override()

