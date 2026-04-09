from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import time_since


class _ResolverStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


class _TimeSinceApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_time_since_entity_only_success_maps_result() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    api = _TimeSinceApiStub(
        resp={
            "resolved_entity": {"id": "ent-1", "name": "张三"},
            "entity_id": "ent-1",
            "last_mentioned": "2026-02-01T00:00:00+00:00",
            "days_ago": 2.5,
            "summary": "讨论项目推进",
        }
    )

    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["needs_disambiguation"] is False
    assert llm["data"]["entity"]["id"] == "ent-1"
    assert llm["data"]["last_mentioned"] == "2026-02-01T00:00:00+00:00"
    assert llm["message"] is None
    assert api.calls and api.calls[0]["entity_id"] == "ent-1"
    assert api.calls[0]["user_tokens"] == ["u:t1"]
    assert dbg and dbg["filter_semantics"] == "entity_only"


def test_time_since_and_semantics_adds_message_and_debug_marker() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    api = _TimeSinceApiStub(
        resp={
            "resolved_entity": {"id": "ent-2", "name": "张三"},
            "entity_id": "ent-2",
            "topic_id": "tpk_1",
            "topic_path": "work/project_alpha",
            "last_mentioned": "2026-02-02T00:00:00+00:00",
            "days_ago": 1.0,
            "summary": "项目推进会",
        }
    )

    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
            topic="项目推进",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert "最近一次" in (llm["message"] or "")
    assert dbg and dbg["filter_semantics"] == "AND"


def test_time_since_missing_query_condition_is_prechecked() -> None:
    resolver = _ResolverStub()
    api = _TimeSinceApiStub()
    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "缺少查询条件" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"
    assert api.calls == []


def test_time_since_entity_ambiguity_stops_before_api_call() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    api = _TimeSinceApiStub()

    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
        )
    )
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert api.calls == []


def test_time_since_last_mentioned_null_maps_no_match() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
    api = _TimeSinceApiStub(
        resp={
            "resolved_entity": {"id": "ent-3", "name": "张三"},
            "entity_id": "ent-3",
            "last_mentioned": None,
            "days_ago": None,
            "summary": None,
        }
    )
    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
        )
    )
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert "没有找到相关记录" in (llm["message"] or "")


def test_time_since_504_timeout_maps_retryable() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "张三"}})
    api = _TimeSinceApiStub(resp={"status_code": 504, "body": {"detail": "time_since_timeout"}})
    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "查询超时" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "timeout"
    assert dbg["retryable"] is True


def test_time_since_invalid_time_range_is_prechecked() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-5", "name": "张三"}})
    api = _TimeSinceApiStub()
    res = asyncio.run(
        time_since(
            tenant_id="t1",
            resolver=resolver,
            time_since_api=api,
            entity="张三",
            time_range={"start": "bad-iso"},
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "时间范围格式无效" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"
    assert api.calls == []

