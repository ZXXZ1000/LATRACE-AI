from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.memory_tools import explain


class _ExplainApiStub:
    def __init__(self, resp: Dict[str, Any] | None = None, *, exc: Exception | None = None) -> None:
        self.resp = dict(resp or {})
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        return dict(self.resp)


def test_explain_success_maps_bundle_and_default_scope() -> None:
    api = _ExplainApiStub(
        resp={
            "found": True,
            "event_id": "evt-1",
            "event": {"id": "evt-1", "summary": "讨论项目推进"},
            "entities": [{"id": "ent-1", "name": "张三"}],
            "places": [],
            "timeslices": [],
            "evidences": [{"id": "evd-1"}],
            "utterances": [{"id": "utt-1", "raw_text": "尽快推进"}],
            "utterance_speakers": [{"utterance_id": "utt-1", "entity_id": "ent-1"}],
            "knowledge": [{"id": "k-1", "summary": "项目要加快"}],
        }
    )

    res = asyncio.run(explain(tenant_id="t1", explain_api=api, event_id="evt-1"))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()

    assert llm["matched"] is True
    assert llm["data"]["event_id"] == "evt-1"
    assert llm["data"]["event"]["id"] == "evt-1"
    assert llm["data"]["evidences"][0]["id"] == "evd-1"
    assert api.calls and api.calls[0]["user_tokens"] == ["u:t1"]
    assert dbg and dbg["source_mode"] == "graph_filter"
    assert dbg["api_route"] == "/memory/v1/explain"


def test_explain_not_found_maps_matched_false_but_keeps_structure() -> None:
    api = _ExplainApiStub(
        resp={
            "found": False,
            "event_id": "evt-missing",
            "event": None,
            "entities": [],
            "places": [],
            "timeslices": [],
            "evidences": [],
            "utterances": [],
            "utterance_speakers": [],
            "knowledge": [],
        }
    )
    res = asyncio.run(explain(tenant_id="t1", explain_api=api, event_id="evt-missing"))
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert "未找到相关事件证据" in (llm["message"] or "")
    assert llm["data"]["event_id"] == "evt-missing"


def test_explain_blank_event_id_is_prechecked() -> None:
    api = _ExplainApiStub()
    res = asyncio.run(explain(tenant_id="t1", explain_api=api, event_id="   "))
    llm = res.to_llm_dict()
    dbg = res.to_debug_dict()
    assert llm["matched"] is False
    assert "缺少事件ID" in (llm["message"] or "")
    assert dbg and dbg["error_type"] == "invalid_input"
    assert api.calls == []


def test_explain_forwards_scope_and_memory_domain() -> None:
    api = _ExplainApiStub(
        resp={
            "found": True,
            "event_id": "evt-2",
            "event": {"id": "evt-2"},
            "entities": [],
            "places": [],
            "timeslices": [],
            "evidences": [],
            "utterances": [],
            "utterance_speakers": [],
            "knowledge": [],
        }
    )
    res = asyncio.run(
        explain(
            tenant_id="t1",
            explain_api=api,
            event_id="evt-2",
            user_tokens=["u:alice"],
            memory_domain="work",
        )
    )
    assert res.to_llm_dict()["matched"] is True
    assert api.calls and api.calls[0]["user_tokens"] == ["u:alice"]
    assert api.calls[0]["memory_domain"] == "work"


def test_explain_503_and_500_error_mapping() -> None:
    api_503 = _ExplainApiStub(resp={"status_code": 503, "body": "temporarily unavailable"})
    res_503 = asyncio.run(explain(tenant_id="t1", explain_api=api_503, event_id="evt-3"))
    llm_503 = res_503.to_llm_dict()
    dbg_503 = res_503.to_debug_dict()
    assert llm_503["matched"] is False
    assert "服务暂时不可用" in (llm_503["message"] or "")
    assert dbg_503 and dbg_503["error_type"] == "rate_limit"
    assert dbg_503["retryable"] is True

    api_500 = _ExplainApiStub(resp={"status_code": 500, "body": {"detail": "boom"}})
    res_500 = asyncio.run(explain(tenant_id="t1", explain_api=api_500, event_id="evt-3"))
    llm_500 = res_500.to_llm_dict()
    dbg_500 = res_500.to_debug_dict()
    assert llm_500["matched"] is False
    assert "服务暂时不可用" in (llm_500["message"] or "")
    assert dbg_500 and dbg_500["error_type"] == "server_error"
