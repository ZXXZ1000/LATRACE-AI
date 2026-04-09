from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.state_property_vocab import StatePropertyVocabManager
from modules.memory.adk.state_tools import entity_status


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


class _FetcherStub:
    def __init__(self, responses: List[Dict[str, Any]] | None = None) -> None:
        self.responses = [dict(x) for x in (responses or [])]
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.responses:
            return self.responses.pop(0)
        return {"vocab_version": "v1", "properties": []}


class _StateEndpointStub:
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
            {"name": "occupation", "allow_raw_value": True},
            {"name": "work_status", "allow_raw_value": True},
        ],
    }


def test_entity_status_current_path_success() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三", "match_source": "exact"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    current = _StateEndpointStub(resp={"item": {"subject_id": "ent-1", "property": "occupation", "value": "工程师"}})
    at_time = _StateEndpointStub()

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="职位",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["needs_disambiguation"] is False
    assert llm["data"]["when"] is None
    assert llm["data"]["entity"]["id"] == "ent-1"
    assert llm["data"]["property"]["canonical"] == "occupation"
    assert len(current.calls) == 1
    assert at_time.calls == []
    assert dbg and dbg["api_route"] == "/memory/state/current"


def test_entity_status_at_time_path_success_with_iso_when() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    current = _StateEndpointStub()
    at_time = _StateEndpointStub(resp={"item": {"subject_id": "ent-2", "property": "occupation", "value": "工程师"}})

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="职位",
            when="2026-02-01T12:00:00Z",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["when"] == "2026-02-01T12:00:00+00:00"
    assert current.calls == []
    assert len(at_time.calls) == 1
    assert at_time.calls[0]["t_iso"] == "2026-02-01T12:00:00+00:00"
    assert dbg and dbg["api_route"] == "/memory/state/at_time"


def test_entity_status_invalid_when_is_prechecked_and_stops_before_endpoint() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    current = _StateEndpointStub()
    at_time = _StateEndpointStub()

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="职位",
            when="上周五",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "时间格式无效" in (llm["message"] or "")
    assert current.calls == []
    assert at_time.calls == []
    assert dbg and dbg["error_type"] == "invalid_input"


def test_entity_status_entity_ambiguity_stops_before_vocab_and_state_calls() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    current = _StateEndpointStub()
    at_time = _StateEndpointStub()

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="职位",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert fetcher.calls == []
    assert current.calls == []
    assert at_time.calls == []


def test_entity_status_property_ambiguity_stops_before_state_calls() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "张三"}})
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
    current = _StateEndpointStub()
    at_time = _StateEndpointStub()

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="work-status",
        )
    )

    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert set((llm["data"] or {}).get("property_candidates") or []) == {"work status", "work_status"}
    assert current.calls == []
    assert at_time.calls == []


def test_entity_status_404_state_not_found_maps_to_no_match() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-5", "name": "张三"}})
    fetcher = _FetcherStub(responses=[_vocab_resp()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    current = _StateEndpointStub(resp={"status_code": 404, "body": {"detail": "state_not_found"}})
    at_time = _StateEndpointStub()

    res = asyncio.run(
        entity_status(
            tenant_id="t1",
            resolver=resolver,
            vocab_manager=mgr,
            state_current=current,
            state_at_time=at_time,
            entity="张三",
            property="职位",
        )
    )

    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "未找到该状态" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "not_found"

