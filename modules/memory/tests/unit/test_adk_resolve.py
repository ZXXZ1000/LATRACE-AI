from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.resolve import _resolve_if_needed


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


def test_resolve_if_needed_direct_id_bypasses_resolver() -> None:
    resolver = _ResolverStub(resp={"found": True})
    out = asyncio.run(
        _resolve_if_needed(
            resolver=resolver,
            tool_name="entity_profile",
            entity_id="ent-1",
        )
    )
    assert out.should_stop is False
    assert out.entity_id == "ent-1"
    assert out.resolution_meta and out.resolution_meta["entity"]["match_source"] == "direct_id"
    assert resolver.calls == []


def test_resolve_if_needed_missing_entity_returns_invalid_input() -> None:
    resolver = _ResolverStub()
    out = asyncio.run(_resolve_if_needed(resolver=resolver, tool_name="entity_profile", entity="  "))
    assert out.should_stop is True
    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "缺少实体参数" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"


def test_resolve_if_needed_success_returns_resolved_entity_id() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-2", "name": "张三", "match_source": "exact"},
        }
    )
    out = asyncio.run(
        _resolve_if_needed(
            resolver=resolver,
            tool_name="relations",
            entity="张三",
            user_tokens=["u:a"],
            resolve_limit=3,
        )
    )
    assert out.should_stop is False
    assert out.entity_id == "ent-2"
    assert out.resolution_meta and out.resolution_meta["entity"]["resolved_id"] == "ent-2"
    assert resolver.calls and resolver.calls[0]["name"] == "张三"
    assert resolver.calls[0]["limit"] == 3
    assert resolver.calls[0]["user_tokens"] == ["u:a"]


def test_resolve_if_needed_ambiguity_stops_with_candidates() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-3", "name": "张三（同事）"}],
        }
    )
    out = asyncio.run(_resolve_if_needed(resolver=resolver, tool_name="quotes", entity="张三"))
    assert out.should_stop is True
    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert len((llm["data"] or {}).get("candidates") or []) == 2
    assert dbg and dbg["resolution_meta"]["entity"]["candidates"]


def test_resolve_if_needed_not_found_stops_with_no_match() -> None:
    resolver = _ResolverStub(resp={"found": False, "resolved_entity": None})
    out = asyncio.run(_resolve_if_needed(resolver=resolver, tool_name="time_since", entity="不存在的人"))
    assert out.should_stop is True
    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "没有找到" in (llm["message"] or "")


def test_resolve_if_needed_exception_maps_to_retryable_error() -> None:
    resolver = _ResolverStub(exc=TimeoutError("resolver timeout"))
    out = asyncio.run(_resolve_if_needed(resolver=resolver, tool_name="entity_status", entity="张三"))
    assert out.should_stop is True
    res = out.terminal_result
    assert res is not None
    dbg = res.to_debug_dict()
    assert dbg and dbg["error_type"] == "timeout"
    assert dbg["retryable"] is True


def test_resolve_if_needed_clamps_resolve_limit_to_five() -> None:
    resolver = _ResolverStub(resp={"found": False, "resolved_entity": None})
    asyncio.run(_resolve_if_needed(resolver=resolver, tool_name="entity_profile", entity="张三", resolve_limit=999))
    assert resolver.calls
    assert resolver.calls[0]["limit"] == 5

