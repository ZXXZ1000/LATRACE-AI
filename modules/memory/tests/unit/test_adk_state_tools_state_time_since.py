from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.state_property_vocab import StatePropertyVocabManager
from modules.memory.adk.state_tools import state_time_since


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


class _StateTimeSinceStub:
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


def test_state_time_since_success_preserves_seconds_ago() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateTimeSinceStub(
        resp={
            "subject_id": "ent-1",
            "property": "work_status",
            "last_changed_at": "2026-02-01T00:00:00+00:00",
            "value": "在职",
            "seconds_ago": 3600,
        }
    )

    res = asyncio.run(
        state_time_since(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_time_since_api=endpoint,
            entity="张三",
            property="工作状态",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["seconds_ago"] == 3600
    assert llm["data"]["property"]["canonical"] == "work_status"
    assert endpoint.calls and endpoint.calls[0]["start_iso"] is None and endpoint.calls[0]["end_iso"] is None
    assert dbg and dbg["api_route"] == "/memory/state/time-since"


def test_state_time_since_404_maps_to_no_match() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateTimeSinceStub(resp={"status_code": 404, "body": {"detail": "state_not_found"}})

    res = asyncio.run(
        state_time_since(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_time_since_api=endpoint,
            entity="张三",
            property="工作状态",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "未找到状态变化记录" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "not_found"


def test_state_time_since_invalid_time_range_is_prechecked() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateTimeSinceStub(resp={})

    res = asyncio.run(
        state_time_since(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_time_since_api=endpoint,
            entity="张三",
            property="工作状态",
            time_range={"end": "bad-iso"},
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "时间范围格式无效" in (llm["message"] or "")
    assert endpoint.calls == []
    assert dbg and dbg["error_type"] == "invalid_input"


def test_state_time_since_entity_or_property_ambiguity_stops_before_endpoint() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateTimeSinceStub(resp={})

    res = asyncio.run(
        state_time_since(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_time_since_api=endpoint,
            entity="张三",
            property="工作状态",
        )
    )
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert endpoint.calls == []


def test_state_time_since_missing_last_changed_at_keeps_match_and_warns() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    endpoint = _StateTimeSinceStub(
        resp={
            "subject_id": "ent-4",
            "property": "work_status",
            "last_changed_at": None,
            "value": "在职",
            "seconds_ago": None,
        }
    )

    res = asyncio.run(
        state_time_since(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_time_since_api=endpoint,
            entity="张三",
            property="工作状态",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert "时间信息不完整" in (llm["message"] or "")
    assert dbg and dbg["resolution_meta"]["state_time_missing"] is True

