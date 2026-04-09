from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.state_property_vocab import StatePropertyVocabManager
from modules.memory.adk.state_tools import status_changes


class _ResolverStub:
    def __init__(self, resp: Dict[str, Any] | None = None) -> None:
        self.resp = dict(resp or {})
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return dict(self.resp)


class _FetcherStub:
    def __init__(self, responses: List[Dict[str, Any]] | None = None) -> None:
        self.responses = [dict(x) for x in (responses or [])]
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.responses:
            return self.responses.pop(0)
        return {"vocab_version": "v1", "properties": []}


class _StateChangesStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def _vocab_resp() -> Dict[str, Any]:
    return {
        "vocab_version": "v1",
        "properties": [
            {"name": "work_status", "allow_raw_value": True},
            {"name": "occupation", "allow_raw_value": True},
        ],
    }


def test_status_changes_default_route_and_order_limit_normalization() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateChangesStub(resp={"items": [{"value": "在职", "valid_from": "2026-01-01T00:00:00+00:00"}]})

    res = asyncio.run(
        status_changes(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_what_changed=endpoint,
            entity="张三",
            property="工作状态",
            order="weird",
            limit=999,
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert len(llm["data"]["items"]) == 1
    assert endpoint.calls and endpoint.calls[0]["order"] == "desc"
    assert endpoint.calls[0]["limit"] == 50
    assert dbg and dbg["api_route"] == "/memory/state/what-changed"
    assert dbg["resolution_meta"]["query"]["order_normalized"] is True


def test_status_changes_empty_items_maps_to_no_match_but_keeps_data_shape() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateChangesStub(resp={"items": []})

    res = asyncio.run(
        status_changes(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_what_changed=endpoint,
            entity="张三",
            property="工作状态",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "未找到状态变化记录" in (llm["message"] or "")
    assert llm["data"]["items"] == []


def test_status_changes_invalid_time_range_is_prechecked_and_stops() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateChangesStub(resp={"items": []})

    res = asyncio.run(
        status_changes(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_what_changed=endpoint,
            entity="张三",
            property="工作状态",
            time_range={"start": "not-iso"},
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "时间范围格式无效" in (llm["message"] or "")
    assert endpoint.calls == []
    assert dbg and dbg["error_type"] == "invalid_input"


def test_status_changes_entity_ambiguity_stops_before_vocab_and_endpoint() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateChangesStub(resp={"items": []})

    res = asyncio.run(
        status_changes(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_what_changed=endpoint,
            entity="张三",
            property="工作状态",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert fetcher.calls == []
    assert endpoint.calls == []


def test_status_changes_property_ambiguity_stops_before_endpoint() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-5", "name": "张三"}})
    fetcher = _FetcherStub(
        responses=[
            {
                "vocab_version": "v1",
                "properties": [
                    {"name": "work status", "allow_raw_value": True},
                    {"name": "work_status", "allow_raw_value": True},
                ],
            }
        ]
    )
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateChangesStub(resp={"items": []})

    res = asyncio.run(
        status_changes(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_what_changed=endpoint,
            entity="张三",
            property="work-status",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert endpoint.calls == []

