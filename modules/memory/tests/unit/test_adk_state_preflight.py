from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.state_preflight import prepare_state_query_preflight
from modules.memory.adk.state_property_vocab import StatePropertyVocabManager


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
    def __init__(self, responses: List[Dict[str, Any]] | None = None, *, exc: Exception | None = None) -> None:
        self.responses = [dict(x) for x in (responses or [])]
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        if self.responses:
            return self.responses.pop(0)
        return {"vocab_version": "v1", "properties": []}


def _sample_vocab_response() -> Dict[str, Any]:
    return {
        "vocab_version": "v1",
        "properties": [
            {"name": "occupation", "description": "工作/职位", "value_type": "string", "allow_raw_value": True},
            {"name": "work_status", "description": "工作状态", "value_type": "string", "allow_raw_value": True},
        ],
    }


def test_prepare_state_query_preflight_direct_entity_id_and_property_canonical_bypass_helpers() -> None:
    resolver = _ResolverStub()
    fetcher = _FetcherStub()
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    out = asyncio.run(
        prepare_state_query_preflight(
            tool_name="entity_status",
            tenant_id="t1",
            vocab_manager=mgr,
            resolver=resolver,
            entity_id="ent-1",
            property_text="职位",
            property_canonical="occupation",
        )
    )

    assert out.should_stop is False
    assert out.entity_id == "ent-1"
    assert out.property_canonical == "occupation"
    assert resolver.calls == []
    assert fetcher.calls == []
    assert out.resolution_meta["entity"]["resolved_id"] == "ent-1"
    assert out.resolution_meta["property"]["match_source"] == "direct_canonical"


def test_prepare_state_query_preflight_missing_property_returns_invalid_input() -> None:
    resolver = _ResolverStub()
    fetcher = _FetcherStub()
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    out = asyncio.run(
        prepare_state_query_preflight(
            tool_name="entity_status",
            tenant_id="t1",
            vocab_manager=mgr,
            resolver=resolver,
            entity_id="ent-1",
            property_text="  ",
        )
    )

    assert out.should_stop is True
    assert out.entity_id == "ent-1"
    assert out.property_canonical is None
    assert fetcher.calls == []

    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is False
    assert "缺少状态属性" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"


def test_prepare_state_query_preflight_vocab_load_failure_returns_retryable_terminal() -> None:
    resolver = _ResolverStub(
        resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三", "match_source": "exact"}}
    )
    fetcher = _FetcherStub(exc=TimeoutError("timeout"))
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    out = asyncio.run(
        prepare_state_query_preflight(
            tool_name="entity_status",
            tenant_id="t1",
            vocab_manager=mgr,
            resolver=resolver,
            entity="张三",
            property_text="职位",
        )
    )

    assert out.should_stop is True
    assert out.entity_id == "ent-2"
    assert fetcher.calls
    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "词表" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "timeout"
    assert dbg["retryable"] is True


def test_prepare_state_query_preflight_property_ambiguity_returns_disambiguation() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})
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

    out = asyncio.run(
        prepare_state_query_preflight(
            tool_name="entity_status",
            tenant_id="t1",
            vocab_manager=mgr,
            resolver=resolver,
            entity="张三",
            property_text="work-status",
        )
    )

    assert out.should_stop is True
    assert out.entity_id == "ent-3"
    res = out.terminal_result
    assert res is not None
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert set((llm["data"] or {}).get("property_candidates") or []) == {"work status", "work_status"}
    assert dbg and dbg["resolution_meta"]["entity"]["resolved_id"] == "ent-3"
    assert dbg["resolution_meta"]["property"]["needs_disambiguation"] is True


def test_prepare_state_query_preflight_success_returns_entity_and_canonical_property() -> None:
    resolver = _ResolverStub(
        resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "张三", "match_source": "exact"}}
    )
    fetcher = _FetcherStub(responses=[_sample_vocab_response()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    out = asyncio.run(
        prepare_state_query_preflight(
            tool_name="entity_status",
            tenant_id="t1",
            vocab_manager=mgr,
            resolver=resolver,
            entity="张三",
            property_text="职位",
            user_tokens=["u:a"],
        )
    )

    assert out.should_stop is False
    assert out.entity_id == "ent-4"
    assert out.property_canonical == "occupation"
    assert fetcher.calls and fetcher.calls[0]["user_tokens"] == ["u:a"]
    assert out.resolution_meta["entity"]["resolved_id"] == "ent-4"
    assert out.resolution_meta["property"]["canonical"] == "occupation"
    assert out.resolution_meta["property"]["match_source"] == "alias_exact"

