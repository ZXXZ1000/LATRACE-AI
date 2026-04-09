from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import list_entities


class _ListEntitiesApiStub:
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
        return {"entities": [], "total": 0, "has_more": False}


def test_list_entities_auto_page_aggregates_up_to_next_page() -> None:
    api = _ListEntitiesApiStub(
        responses=[
            {
                "entities": [{"id": "e1", "name": "张三"}],
                "total": 2,
                "has_more": True,
                "next_cursor": "c:1",
            },
            {
                "entities": [{"id": "e2", "name": "李四"}],
                "total": 2,
                "has_more": False,
            },
        ]
    )

    out = asyncio.run(
        list_entities(
            tenant_id="t1",
            list_entities_api=api,
            query="张",
            auto_page=True,
            limit=20,
        )
    )

    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is True
    assert len((llm["data"] or {}).get("entities") or []) == 2
    assert len(api.calls) == 2
    assert dbg and dbg["api_route"] == "/memory/v1/entities"
    assert dbg.get("pages_fetched") == 2


def test_list_entities_invalid_mentioned_since_is_prechecked() -> None:
    api = _ListEntitiesApiStub()
    out = asyncio.run(
        list_entities(
            tenant_id="t1",
            list_entities_api=api,
            mentioned_since="昨天",
        )
    )

    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is False
    assert "时间格式无效" in (llm["message"] or "")
    assert api.calls == []
    assert dbg and dbg["error_type"] == "invalid_input"


def test_list_entities_invalid_cursor_falls_back_to_first_page_with_notice() -> None:
    api = _ListEntitiesApiStub(
        responses=[
            {"entities": [{"id": "e1", "name": "张三"}], "total": 1, "has_more": False},
        ]
    )
    out = asyncio.run(
        list_entities(
            tenant_id="t1",
            list_entities_api=api,
            cursor="bad_cursor",
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is True
    assert "分页游标无效" in (llm["message"] or "")
    assert api.calls and api.calls[0].get("cursor") is None
    assert dbg and bool((dbg.get("resolution_meta") or {}).get("query", {}).get("cursor_invalid")) is True


def test_list_entities_empty_maps_to_no_match() -> None:
    api = _ListEntitiesApiStub(responses=[{"entities": [], "total": 0, "has_more": False}])
    out = asyncio.run(
        list_entities(
            tenant_id="t1",
            list_entities_api=api,
        )
    )
    llm = out.to_llm_dict()
    assert llm["matched"] is False
    assert "未找到实体候选" in (llm["message"] or "")
    assert llm["data"]["total"] == 0


def test_list_entities_503_maps_retryable() -> None:
    api = _ListEntitiesApiStub(responses=[{"status_code": 503, "body": "temporarily unavailable"}])
    out = asyncio.run(
        list_entities(
            tenant_id="t1",
            list_entities_api=api,
        )
    )
    llm = out.to_llm_dict()
    dbg = out.to_debug_dict()
    assert llm["matched"] is False
    assert "暂时不可用" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "rate_limit"
    assert dbg["retryable"] is True
