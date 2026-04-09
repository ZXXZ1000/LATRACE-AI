from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import quotes


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


class _QuotesApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_quotes_entity_only_success_defaults_and_infers_graph_filter() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}})
    api = _QuotesApiStub(
        resp={
            "resolved_entity": {"id": "ent-1", "name": "张三"},
            "entity_id": "ent-1",
            "quotes": [{"text": "我们这周推进项目", "when": "2026-02-01T00:00:00+00:00"}],
            "total": 1,
        }
    )
    res = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api, entity="张三", limit=99))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["entity"]["id"] == "ent-1"
    assert llm["data"]["total"] == 1
    assert api.calls and api.calls[0]["limit"] == 10
    assert api.calls[0]["user_tokens"] == ["u:t1"]
    assert dbg and dbg["source_mode"] == "graph_filter"
    assert dbg["api_route"] == "/memory/v1/quotes"


def test_quotes_topic_only_success_with_retrieval_trace_marks_fallback() -> None:
    resolver = _ResolverStub()
    api = _QuotesApiStub(
        resp={
            "topic_id": "tpk_x",
            "topic_path": "work/project_alpha",
            "quotes": [{"text": "项目推进卡在资源", "when": "2026-02-02T00:00:00+00:00"}],
            "total": 1,
            "trace": {"source": "retrieval_dialog_v2", "retrieval_evidence": 3},
        }
    )
    res = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api, topic="项目推进"))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["topic"]["input"] == "项目推进"
    assert llm["data"]["quotes"][0]["text"] == "项目推进卡在资源"
    assert resolver.calls == []
    assert dbg and dbg["source_mode"] == "retrieval_rag"
    assert dbg["fallback_used"] is True


def test_quotes_entity_plus_topic_path_success() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-2", "name": "张三"}})
    api = _QuotesApiStub(
        resp={
            "resolved_entity": {"id": "ent-2", "name": "张三"},
            "entity_id": "ent-2",
            "topic_id": "tpk_2",
            "topic_path": "work/project_alpha",
            "quotes": [{"text": "张三说项目推进要加快", "when": "2026-02-03T00:00:00+00:00"}],
            "total": 1,
            "trace": {"source": "graph_topic_filter"},
        }
    )
    res = asyncio.run(
        quotes(
            tenant_id="t1",
            resolver=resolver,
            quotes_api=api,
            entity="张三",
            topic="项目推进",
            time_range={"start": "2026-02-01T00:00:00+00:00"},
        )
    )
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is True
    assert llm["data"]["entity"]["id"] == "ent-2"
    assert llm["data"]["topic"]["topic_path"] == "work/project_alpha"
    assert api.calls and api.calls[0]["time_range"]["start_iso"] == "2026-02-01T00:00:00+00:00"
    assert dbg and dbg["source_mode"] == "graph_filter"


def test_quotes_entity_ambiguity_stops_before_api_call() -> None:
    resolver = _ResolverStub(
        resp={
            "found": True,
            "resolved_entity": {"id": "ent-top1", "name": "张三"},
            "candidates": [{"id": "ent-top1", "name": "张三"}, {"id": "ent-9", "name": "张三（同事）"}],
        }
    )
    api = _QuotesApiStub()
    res = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api, entity="张三"))
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert api.calls == []


def test_quotes_invalid_time_range_and_missing_query_prechecked() -> None:
    resolver = _ResolverStub()
    api = _QuotesApiStub()

    res_missing = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api))
    llm_m = res_missing.to_llm_dict()
    dbg_m = res_missing.to_debug_dict()
    assert llm_m["matched"] is False
    assert "缺少实体或话题参数" in (llm_m["message"] or "")
    assert dbg_m and dbg_m["error_type"] == "invalid_input"

    res_bad = asyncio.run(
        quotes(
            tenant_id="t1",
            resolver=resolver,
            quotes_api=api,
            topic="项目推进",
            time_range={"start": "bad-iso"},
        )
    )
    llm_b = res_bad.to_llm_dict()
    dbg_b = res_bad.to_debug_dict()
    assert llm_b["matched"] is False
    assert "时间范围格式无效" in (llm_b["message"] or "")
    assert dbg_b and dbg_b["error_type"] == "invalid_input"
    assert api.calls == []


def test_quotes_503_and_504_map_to_different_errors() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-3", "name": "张三"}})

    api_503 = _QuotesApiStub(resp={"status_code": 503, "body": "temporarily unavailable"})
    res_503 = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api_503, entity="张三"))
    llm_503 = res_503.to_llm_dict()
    dbg_503 = res_503.to_debug_dict()
    assert llm_503["matched"] is False
    assert "服务暂时不可用" in (llm_503["message"] or "")
    assert dbg_503 and dbg_503["error_type"] == "rate_limit"
    assert dbg_503["retryable"] is True

    api_504 = _QuotesApiStub(resp={"status_code": 504, "body": {"detail": "quotes_timeout"}})
    res_504 = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api_504, entity="张三"))
    llm_504 = res_504.to_llm_dict()
    dbg_504 = res_504.to_debug_dict()
    assert llm_504["matched"] is False
    assert "原话查询超时" in (llm_504["message"] or "")
    assert dbg_504 and dbg_504["error_type"] == "timeout"
    assert dbg_504["retryable"] is True


def test_quotes_empty_quotes_maps_no_match() -> None:
    resolver = _ResolverStub(resp={"found": True, "resolved_entity": {"id": "ent-4", "name": "张三"}})
    api = _QuotesApiStub(resp={"resolved_entity": {"id": "ent-4", "name": "张三"}, "entity_id": "ent-4", "quotes": [], "total": 0})
    res = asyncio.run(quotes(tenant_id="t1", resolver=resolver, quotes_api=api, entity="张三"))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "未找到相关原话" in (llm["message"] or "")
    assert dbg and dbg["source_mode"] == "graph_filter"

