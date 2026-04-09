from __future__ import annotations

"""
L4/L5 检索对标（InMem 版逻辑回归）：

目的：
- 在不依赖 Neo4j 的前提下，用 InMemVectorStore/InMemGraphStore 验证
  “聚会对话检测” 与 “无出门日” 两类否定/语义场景的最小可实现性。
"""

from typing import List

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import Edge, MemoryEntry, SearchFilters
from modules.memory.infra.audit_store import InMemAuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


def _mk_service() -> MemoryService:
    return MemoryService(InMemVectorStore({}), InMemGraphStore({}), InMemAuditStore())


@pytest.mark.anyio
async def test_l4_l5_party_talk_to_inmem() -> None:
    """InMem 版：聚会中是否与某人有对话（存在/否定）。"""
    svc = _mk_service()

    # 我、李四、王五作为结构化节点
    entries: List[MemoryEntry] = [
        MemoryEntry(id="me", kind="semantic", modality="structured", contents=["Me"], metadata={"entity_type": "person"}),
        MemoryEntry(id="li-si", kind="semantic", modality="structured", contents=["Li Si"], metadata={"entity_type": "person"}),
        MemoryEntry(id="wang-wu", kind="semantic", modality="structured", contents=["Wang Wu"], metadata={"entity_type": "person"}),
        MemoryEntry(id="party-1", kind="semantic", modality="structured", contents=["Party"], metadata={"entity_type": "event"}),
    ]
    edges = [
        Edge(src_id="party-1", dst_id="me", rel_type="appears_in", weight=1.0),
        Edge(src_id="party-1", dst_id="li-si", rel_type="appears_in", weight=1.0),
        Edge(src_id="party-1", dst_id="wang-wu", rel_type="appears_in", weight=1.0),
        Edge(src_id="party-1", dst_id="li-si", rel_type="said_by", weight=1.0),
        Edge(src_id="party-1", dst_id="wang-wu", rel_type="said_by", weight=1.0),
    ]
    await svc.write(entries, links=edges)

    # 图优先检索：以 party-1 为种子，查看 TALK/SAID 边覆盖哪些人
    out = svc.search_graph(
        {
            "seeds": ["party-1"],
            "filters": SearchFilters(),  # InMemGraphStore 不看 filters 时仍会展开
            "rel_whitelist": ["SAID_BY", "said_by"],
            "max_hops": 1,
            "neighbor_cap_per_seed": 10,
        }
    )
    neighbors = (out.get("neighbors") or {}).get("party-1") or []
    ids = {n["to"] for n in neighbors}
    # 李四、王五都在“说话”邻居里（存在性）；否定逻辑由 Neo4j 小图脚本覆盖
    assert "li-si" in ids and "wang-wu" in ids
