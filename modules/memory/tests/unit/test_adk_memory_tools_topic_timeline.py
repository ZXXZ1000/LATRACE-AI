from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import topic_timeline


class _TopicTimelineApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_topic_timeline_default_include_and_limit_mapping() -> None:
    api = _TopicTimelineApiStub(
        resp={
            "topic": "项目推进",
            "topic_id": "tpk_1",
            "topic_path": "work/project_alpha",
            "status": "active",
            "timeline": [{"event_id": "evt-1", "summary": "推进会"}],
            "total": 1,
        }
    )
    res = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api, topic="项目推进", limit=999))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["total"] == 1
    call = api.calls[0]
    assert call["with_quotes"] is False
    assert call["with_entities"] is False
    assert call["limit"] == 20
    assert call["user_tokens"] == ["u:t1"]
    assert dbg and dbg["heavy_expand"] is False


def test_topic_timeline_include_quotes_marks_heavy_expand_and_session_passthrough() -> None:
    api = _TopicTimelineApiStub(
        resp={
            "topic": "项目推进",
            "topic_id": "tpk_1",
            "topic_path": "work/project_alpha",
            "status": "active",
            "timeline": [{"event_id": "evt-1", "summary": "推进会", "quotes": [{"text": "尽快推进"}]}],
            "total": 1,
            "trace": {"source": "graph_topic_filter"},
        }
    )
    res = asyncio.run(
        topic_timeline(
            tenant_id="t1",
            topic_timeline_api=api,
            topic="项目推进",
            include=["quotes"],
            session_id="sess-1",
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert api.calls and api.calls[0]["with_quotes"] is True
    assert api.calls[0]["with_entities"] is False
    assert api.calls[0]["session_id"] == "sess-1"
    assert dbg and dbg["heavy_expand"] is True
    assert dbg["source_mode"] == "graph_filter"


def test_topic_timeline_invalid_time_range_and_missing_input_prechecked() -> None:
    api = _TopicTimelineApiStub()

    res_missing = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api))
    llm_m = res_missing.to_llm_dict()
    dbg_m = res_missing.to_debug_dict()
    assert llm_m["matched"] is False
    assert "缺少话题或关键词" in (llm_m["message"] or "")
    assert dbg_m and dbg_m["error_type"] == "invalid_input"

    res_bad = asyncio.run(
        topic_timeline(
            tenant_id="t1",
            topic_timeline_api=api,
            topic="项目推进",
            time_range={"start": "bad-iso"},
        )
    )
    llm_b = res_bad.to_llm_dict()
    dbg_b = res_bad.to_debug_dict()
    assert llm_b["matched"] is False
    assert "时间范围格式无效" in (llm_b["message"] or "")
    assert dbg_b and dbg_b["error_type"] == "invalid_input"
    assert api.calls == []


def test_topic_timeline_empty_timeline_maps_no_match() -> None:
    api = _TopicTimelineApiStub(resp={"topic": "项目推进", "topic_id": "tpk_1", "topic_path": "work/project_alpha", "status": "empty", "timeline": [], "total": 0})
    res = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api, topic="项目推进"))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "未找到相关时间线记录" in (llm["message"] or "")
    assert dbg and dbg["api_route"] == "/memory/v1/topic-timeline"


def test_topic_timeline_503_and_504_error_mapping() -> None:
    api_503 = _TopicTimelineApiStub(resp={"status_code": 503, "body": "temporarily unavailable"})
    res_503 = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api_503, topic="项目推进"))
    llm_503 = res_503.to_llm_dict()
    dbg_503 = res_503.to_debug_dict()
    assert llm_503["matched"] is False
    assert "服务暂时不可用" in (llm_503["message"] or "")
    assert dbg_503 and dbg_503["error_type"] == "rate_limit"
    assert dbg_503["retryable"] is True

    api_504 = _TopicTimelineApiStub(resp={"status_code": 504, "body": {"detail": "timeline_timeout"}})
    res_504 = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api_504, topic="项目推进"))
    llm_504 = res_504.to_llm_dict()
    dbg_504 = res_504.to_debug_dict()
    assert llm_504["matched"] is False
    assert "时间线查询超时" in (llm_504["message"] or "")
    assert dbg_504 and dbg_504["error_type"] == "timeout"
    assert dbg_504["retryable"] is True


def test_topic_timeline_retrieval_trace_maps_source_mode_and_fallback() -> None:
    api = _TopicTimelineApiStub(
        resp={
            "topic": "模糊话题",
            "topic_id": None,
            "topic_path": None,
            "status": "active",
            "timeline": [{"event_id": "evt-9", "summary": "通过检索回退命中"}],
            "total": 1,
            "trace": {"source": "retrieval_dialog_v2", "retrieval_evidence": 2},
        }
    )
    res = asyncio.run(topic_timeline(tenant_id="t1", topic_timeline_api=api, keywords=["推进", "项目"]))
    dbg = res.to_debug_dict()
    llm = res.to_llm_dict()
    assert llm["matched"] is True
    assert llm["data"]["topic"]["keywords"] == ["推进", "项目"]
    assert dbg and dbg["source_mode"] == "retrieval_rag"
    assert dbg["fallback_used"] is True

