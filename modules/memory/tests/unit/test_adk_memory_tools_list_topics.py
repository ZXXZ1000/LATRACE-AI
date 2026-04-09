from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import list_topics


class _ListTopicsApiStub:
    def __init__(self, responses: List[Dict[str, Any]] | None = None, *, exc: Exception | None = None) -> None:
        self.responses = [dict(x) for x in (responses or [])]
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        if self.responses:
            return dict(self.responses.pop(0))
        return {"topics": [], "total": 0, "has_more": False}


def test_list_topics_success_and_status_thresholds_go_to_debug_only() -> None:
    api = _ListTopicsApiStub(
        responses=[
            {
                "topics": [{"topic_path": "work/project-alpha", "display_name": None, "event_count": 3}],
                "total": 1,
                "has_more": False,
                "status_thresholds": {"ongoing_days": 3, "paused_days": 14},
            }
        ]
    )
    out = asyncio.run(
        list_topics(
            tenant_id="t1",
            list_topics_api=api,
            query="项目",
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["topics"][0]["topic_path"] == "work/project-alpha"
    assert "status_thresholds" not in (llm["data"] or {})
    assert dbg and "status_thresholds" in (dbg.get("resolution_meta") or {})


def test_list_topics_negative_min_events_is_clamped_to_zero() -> None:
    api = _ListTopicsApiStub(
        responses=[{"topics": [{"topic_path": "x"}], "total": 1, "has_more": False}]
    )
    out = asyncio.run(
        list_topics(
            tenant_id="t1",
            list_topics_api=api,
            min_events=-3,
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is True
    assert api.calls and api.calls[0]["min_events"] == 0
    assert dbg and bool((dbg.get("resolution_meta") or {}).get("query", {}).get("min_events_clamped")) is True


def test_list_topics_invalid_cursor_falls_back_to_first_page_with_notice() -> None:
    api = _ListTopicsApiStub(
        responses=[{"topics": [{"topic_path": "work/project-alpha"}], "total": 1, "has_more": False}]
    )
    out = asyncio.run(
        list_topics(
            tenant_id="t1",
            list_topics_api=api,
            cursor="bad_cursor",
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is True
    assert "分页游标无效" in (llm["message"] or "")
    assert api.calls and api.calls[0].get("cursor") is None
    assert dbg and bool((dbg.get("resolution_meta") or {}).get("query", {}).get("cursor_invalid")) is True


def test_list_topics_empty_maps_to_no_match() -> None:
    api = _ListTopicsApiStub(responses=[{"topics": [], "total": 0, "has_more": False}])
    out = asyncio.run(
        list_topics(
            tenant_id="t1",
            list_topics_api=api,
        )
    )
    llm = out.to_llm_dict()
    assert llm["matched"] is False
    assert "未找到话题候选" in (llm["message"] or "")
    assert llm["data"]["total"] == 0


def test_list_topics_503_maps_retryable() -> None:
    api = _ListTopicsApiStub(responses=[{"status_code": 503, "body": "temporarily unavailable"}])
    out = asyncio.run(
        list_topics(
            tenant_id="t1",
            list_topics_api=api,
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is False
    assert "暂时不可用" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "rate_limit"
    assert dbg["retryable"] is True
