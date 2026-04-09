from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import entity_profile


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


class _EntityProfileApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_entity_profile_default_include_and_limit_mapping() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    api = _EntityProfileApiStub(
        resp={
            "found": True,
            "entity": {"id": "ent-1", "name": "张三", "type": "PERSON"},
            "facts": [{"k": "v"}],
            "relations": [{"entity_id": "ent-2"}],
            "recent_events": [{"event_id": "evt-1"}],
        }
    )

    res = asyncio.run(
        entity_profile(
            tenant_id="t1",
            resolver=resolver,
            entity_profile_api=api,
            entity="张三",
            limit=999,
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["entity"]["id"] == "ent-1"
    assert "quotes" not in llm["data"]
    assert "states" not in llm["data"]
    assert api.calls
    call = api.calls[0]
    assert call["entity_id"] == "ent-1"
    assert call["include_relations"] is True
    assert call["include_events"] is True
    assert call["include_quotes"] is False
    assert call["include_states"] is False
    assert call["facts_limit"] == 50 and call["relations_limit"] == 50 and call["events_limit"] == 50 and call["quotes_limit"] == 50
    assert call["user_tokens"] == ["u:t1"]
    assert dbg and dbg["resolution_meta"]["query"]["limit"] == 50


def test_entity_profile_direct_entity_id_bypasses_resolver() -> None:
    resolver = _ResolverStub()
    api = _EntityProfileApiStub(
        resp={"found": True, "entity": {"id": "ent-2", "name": "李四"}, "facts": [], "relations": [], "recent_events": []}
    )

    res = asyncio.run(
        entity_profile(
            tenant_id="t1",
            resolver=resolver,
            entity_profile_api=api,
            entity_id="ent-2",
            include=["states"],
            limit=5,
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is True
    assert resolver.calls == []
    assert api.calls and api.calls[0]["include_states"] is True
    assert api.calls[0]["include_relations"] is False
    assert "states" in llm["data"]


def test_entity_profile_entity_ambiguity_stops_before_api_call() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    api = _EntityProfileApiStub()

    res = asyncio.run(
        entity_profile(
            tenant_id="t1",
            resolver=resolver,
            entity_profile_api=api,
            entity="张三",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert api.calls == []


def test_entity_profile_not_found_maps_to_no_match() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "王五"}})
    api = _EntityProfileApiStub(resp={"found": False, "entity": None, "facts": [], "relations": [], "recent_events": []})

    res = asyncio.run(
        entity_profile(
            tenant_id="t1",
            resolver=resolver,
            entity_profile_api=api,
            entity="王五",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "没有找到相关实体" in (llm["message"] or "")


def test_entity_profile_503_temporarily_unavailable_maps_retryable() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "赵六"}})
    api = _EntityProfileApiStub(resp={"status_code": 503, "body": "temporarily unavailable"})

    res = asyncio.run(
        entity_profile(
            tenant_id="t1",
            resolver=resolver,
            entity_profile_api=api,
            entity="赵六",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "服务暂时不可用" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "rate_limit"
    assert dbg["retryable"] is True

