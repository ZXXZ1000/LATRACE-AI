from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_v06_l1_list_places_by_time_range_basic():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        now = datetime.now(timezone.utc)
        ts_old = (now - timedelta(days=10)).timestamp()
        ts_fri_morning = (now - timedelta(days=2, hours=6)).timestamp()
        ts_fri_evening = (now - timedelta(days=2, hours=1)).timestamp()

        # 早期一次出行（不在目标时间范围）
        e0 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["十天前在酒店"],
            metadata={"timestamp": ts_old, "entities": ["place:hotel"]},
        )
        # “上周五”早上去超市、晚上在家
        e1 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["早上 去 超市"],
            metadata={"timestamp": ts_fri_morning, "entities": ["place:supermarket"]},
        )
        e2 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["晚上 在 家 看书"],
            metadata={"timestamp": ts_fri_evening, "entities": ["place:home"]},
        )
        await svc.write([e0, e1, e2])

        # 选取覆盖 e1/e2 而排除 e0 的时间窗口
        start = ts_fri_morning - 60.0
        end = ts_fri_evening + 60.0
        res = await svc.list_places_by_time_range(
            query="",
            filters=SearchFilters(modality=["text"]),
            start_time=start,
            end_time=end,
            topk_search=50,
        )
        places = {p["id"]: p["count"] for p in (res.get("places") or [])}
        # 期望 only home/supermarket in window; hotel should not appear
        assert "place:supermarket" in places
        assert "place:home" in places
        assert "place:hotel" not in places

    asyncio.run(_run())


def test_v06_l2_timeline_summary_respects_time_range():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        # 三个 episodic 事件分布在不同时间
        now = datetime.now(timezone.utc)
        t1 = (now - timedelta(hours=3)).timestamp()
        t2 = (now - timedelta(hours=2)).timestamp()
        t3 = (now - timedelta(hours=1)).timestamp()

        e1 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["早上 起床"],
            metadata={"timestamp": t1, "clip_id": 0},
        )
        e2 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["中午 做饭"],
            metadata={"timestamp": t2, "clip_id": 1},
        )
        e3 = MemoryEntry(
            kind="episodic",
            modality="text",
            contents=["晚上 看电影"],
            metadata={"timestamp": t3, "clip_id": 2},
        )
        await svc.write([e1, e2, e3])

        # 仅查看最近两小时（应排除“早上 起床”）
        start = (now - timedelta(hours=2, minutes=30)).timestamp()
        end = now.timestamp()
        timeline = await svc.timeline_summary(
            query="",
            filters=SearchFilters(modality=["text"]),
            start_time=start,
            end_time=end,
            max_segments=10,
        )
        events = timeline.get("events") or []
        # 摘要中不应出现“早上 起床”，但应包含“做饭”或“看电影”
        descriptions = " ".join(ev.get("description") or "" for ev in events)
        assert "早上 起床" not in descriptions
        assert ("做饭" in descriptions) or ("看电影" in descriptions)

    asyncio.run(_run())
