from __future__ import annotations

"""
L1–L3 检索对标场景集成测试

这些测试不是为了覆盖所有细节算法，而是验证当前 MemoryService + InMem* 存储
在小图数据上是否能支撑 docs/时空知识记忆系统构建理论/记忆检索与推理对标清单
中 L1–L3 的典型问题（在工程上“可表达且可跑通”）。
"""

from typing import Any, Dict, List

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import Edge, MemoryEntry, SearchFilters
from modules.memory.infra.audit_store import InMemAuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


def _mk_service() -> MemoryService:
    """构造一个完全内存型的 MemoryService，用于集成测试。"""
    vectors = InMemVectorStore({})
    graph = InMemGraphStore({})
    audit = InMemAuditStore()
    return MemoryService(vectors, graph, audit)


@pytest.mark.anyio
async def test_l1_places_by_time_range_basic() -> None:
    """L1 基础事实：某个时间范围内我去了哪些地方？"""
    svc = _mk_service()

    # 构造三条 episodic 文本记忆，其中两条在查询时间窗内
    entries: List[MemoryEntry] = [
        MemoryEntry(
            id="e-park",
            kind="episodic",
            modality="text",
            contents=["上午我去了公园散步"],
            metadata={
                "timestamp": 1_000.0,
                "entities": ["place:park"],
                "user_id": ["u"],
                "memory_domain": "home",
            },
        ),
        MemoryEntry(
            id="e-market",
            kind="episodic",
            modality="text",
            contents=["下午又去了一趟超市"],
            metadata={
                "timestamp": 2_000.0,
                "entities": ["place:market"],
                "user_id": ["u"],
                "memory_domain": "home",
            },
        ),
        # 不在时间窗内，应被 time_range 过滤掉
        MemoryEntry(
            id="e-old",
            kind="episodic",
            modality="text",
            contents=["上个月去了朋友家"],
            metadata={
                "timestamp": 100.0,
                "entities": ["place:friend_home"],
                "user_id": ["u"],
                "memory_domain": "home",
            },
        ),
    ]

    await svc.write(entries, links=None)

    out: Dict[str, Any] = await svc.list_places_by_time_range(
        query="",
        filters=SearchFilters(user_id=["u"], memory_domain="home"),
        start_time=900.0,
        end_time=2_500.0,
        topk_search=50,
    )

    places = out.get("places") or []
    ids = {p["id"] for p in places}
    assert "place:park" in ids and "place:market" in ids, "时间窗内的地点应被召回"
    assert "place:friend_home" not in ids, "时间窗外的地点不应出现"


@pytest.mark.anyio
async def test_l1_meeting_participants_via_search() -> None:
    """L1 基础事实：昨天下午跟我开会的人是谁？"""
    svc = _mk_service()

    base_ts = 10_000.0
    meeting = MemoryEntry(
        id="ev-meeting",
        kind="episodic",
        modality="text",
        contents=["昨天下午在会议室和 Alice 以及 Bob 开会"],
        metadata={
            "timestamp": base_ts + 3_600.0,
            "event_type": "meeting",
            "entities": ["person:me", "person:alice", "person:bob"],
            "room": "meeting_room",
            "user_id": ["u"],
            "memory_domain": "work",
        },
    )
    noise = MemoryEntry(
        id="ev-other",
        kind="episodic",
        modality="text",
        contents=["独自一人在家看书"],
        metadata={
            "timestamp": base_ts + 1_800.0,
            "entities": ["person:me"],
            "room": "living_room",
            "user_id": ["u"],
            "memory_domain": "work",
        },
    )
    await svc.write([meeting, noise], links=None)

    res = await svc.search(
        query="开会",
        topk=5,
        filters=SearchFilters(
            user_id=["u"],
            memory_domain="work",
            modality=["text"],
            time_range={"gte": base_ts, "lte": base_ts + 86_400.0},
        ),
        expand_graph=False,
    )
    assert res.hits, "应至少召回一次会议事件"
    hit = res.hits[0]
    md = hit.entry.metadata
    assert md.get("event_type") == "meeting"
    participants = [e for e in (md.get("entities") or []) if str(e).startswith("person:")]
    assert set(participants) == {"person:me", "person:alice", "person:bob"}


@pytest.mark.anyio
async def test_l2_entity_event_anchor_inmem() -> None:
    """L2 时序/状态：通过 entity_event_anchor 锚定“回家”相关事件的时间范围。"""
    svc = _mk_service()

    entries = [
        MemoryEntry(
            id="ev-arrive-home",
            kind="episodic",
            modality="text",
            contents=["男子下班回到家"],
            metadata={"timestamp": 5_000.0, "clip_id": 1, "user_id": ["u"], "memory_domain": "home"},
        ),
        MemoryEntry(
            id="ev-open-tv",
            kind="episodic",
            modality="text",
            contents=["男子回家后打开电视看新闻"],
            metadata={"timestamp": 5_100.0, "clip_id": 1, "user_id": ["u"], "memory_domain": "home"},
        ),
    ]
    await svc.write(entries, links=None)

    out = await svc.entity_event_anchor(entity="男子", action="回到家", filters=SearchFilters(user_id=["u"], memory_domain="home"))
    triples = out.get("triples") or []
    assert triples, "应能为实体+动作返回至少一条时间锚定"
    tr = triples[0].get("time_range") or []
    assert len(tr) == 2
    assert tr[0] is not None or tr[1] is not None, "时间范围应包含起点或终点"


@pytest.mark.anyio
async def test_l3_cooccurs_partner_for_bob() -> None:
    """L3 多跳/共现：找到经常和 Bob 一起出现、带有 distinguishing 属性的实体。"""
    svc = _mk_service()

    # 三个角色节点：Bob + 两个共现对象，其中 one_with_glasses 权重大
    bob = MemoryEntry(
        id="ent-bob",
        kind="semantic",
        modality="structured",
        contents=["Bob"],
        metadata={"entity_type": "person", "user_id": ["u"], "memory_domain": "home"},
    )
    with_glasses = MemoryEntry(
        id="ent-glasses",
        kind="semantic",
        modality="structured",
        contents=["戴眼镜的男人"],
        metadata={"entity_type": "person", "has_glasses": True, "user_id": ["u"], "memory_domain": "home"},
    )
    other = MemoryEntry(
        id="ent-other",
        kind="semantic",
        modality="structured",
        contents=["其他朋友"],
        metadata={"entity_type": "person", "has_glasses": False, "user_id": ["u"], "memory_domain": "home"},
    )

    edges = [
        Edge(src_id="ent-bob", dst_id="ent-glasses", rel_type="co_occurs", weight=3.0),
        Edge(src_id="ent-bob", dst_id="ent-other", rel_type="co_occurs", weight=1.0),
    ]
    await svc.write([bob, with_glasses, other], links=edges)

    # 使用 graph-first 检索接口，直接以 Bob 为种子展开 CO_OCCURS 邻居
    out = svc.search_graph(
        {
            "seeds": ["ent-bob"],
            "filters": SearchFilters(user_id=["u"], memory_domain="home"),
            "rel_whitelist": ["CO_OCCURS"],
            "max_hops": 1,
            "neighbor_cap_per_seed": 10,
        }
    )
    neighbors = (out.get("neighbors") or {}).get("ent-bob") or []
    assert neighbors, "应能展开 Bob 的共现邻居"

    # 权重最高的共现对象应是戴眼镜的男人
    neighbors_sorted = sorted(neighbors, key=lambda n: float(n.get("weight", 0.0)), reverse=True)
    top = neighbors_sorted[0]
    assert top["to"] == "ent-glasses"
