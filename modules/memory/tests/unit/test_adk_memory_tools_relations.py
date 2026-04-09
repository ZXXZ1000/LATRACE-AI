from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import relations


class _ResolverStub:
    def __init__(self, resp: Dict[str, Any] | None = None) -> None:
        self.resp = dict(resp or {})
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.resp)


class _RelationsApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_relations_success_maps_data() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    api = _RelationsApiStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-1", "name": "张三"},
            "entity_id": "ent-1",
            "relations": [{"entity_id": "ent-2", "name": "李四", "relation_type": "co_occurs_with"}],
            "total": 1,
        }
    )

    res = asyncio.run(
        relations(
            tenant_id="t1",
            resolver=resolver,
            relations_api=api,
            entity="张三",
            limit=999,
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["total"] == 1
    assert api.calls and api.calls[0]["limit"] == 50
    assert dbg and dbg["entity_resolved"] is True


def test_relations_found_true_but_empty_relations_maps_no_match_not_entity_missing() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    api = _RelationsApiStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-2", "name": "张三"},
            "entity_id": "ent-2",
            "relations": [],
            "total": 0,
        }
    )
    res = asyncio.run(relations(tenant_id="t1", resolver=resolver, relations_api=api, entity="张三"))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "未找到关系" in (llm["message"] or "")
    assert dbg and dbg["entity_resolved"] is True


def test_relations_unsupported_relation_type_is_prechecked() -> None:
    resolver = _ResolverStub()
    api = _RelationsApiStub()
    res = asyncio.run(
        relations(
            tenant_id="t1",
            resolver=resolver,
            relations_api=api,
            entity="张三",
            relation_type="friend_of",
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "暂不支持该关系类型" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"
    assert resolver.calls == []
    assert api.calls == []


def test_relations_entity_ambiguity_stops_before_api_call() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    api = _RelationsApiStub()
    res = asyncio.run(relations(tenant_id="t1", resolver=resolver, relations_api=api, entity="张三"))
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert api.calls == []


def test_relations_timeout_and_invalid_time_range_mapping() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
    api = _RelationsApiStub(resp={"status_code": 504, "body": {"detail": "relations_timeout"}})

    res_timeout = asyncio.run(relations(tenant_id="t1", resolver=resolver, relations_api=api, entity="张三"))
    llm_t = res_timeout.to_llm_dict()
    dbg_t = res_timeout.to_debug_dict()
    assert llm_t["matched"] is False
    assert "查询超时" in (llm_t["message"] or "")
    assert dbg_t and dbg_t["error_type"] == "timeout"

    api2 = _RelationsApiStub()
    res_bad_range = asyncio.run(
        relations(
            tenant_id="t1",
            resolver=resolver,
            relations_api=api2,
            entity="张三",
            time_range={"start": "bad-iso"},
        )
    )
    llm_b = res_bad_range.to_llm_dict()
    dbg_b = res_bad_range.to_debug_dict()
    assert llm_b["matched"] is False
    assert "时间范围格式无效" in (llm_b["message"] or "")
    assert dbg_b and dbg_b["error_type"] == "invalid_input"
    assert api2.calls == []

