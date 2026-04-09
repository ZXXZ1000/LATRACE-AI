from __future__ import annotations

from typing import Any, Dict

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _RuntimeStub:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    async def entity_profile(self, **kwargs):
        from modules.memory.adk import ToolResult

        self.calls.append(("entity_profile", dict(kwargs)))
        return ToolResult.success(data={"entity": {"name": kwargs.get("entity")}})

    async def aclose(self) -> None:
        self.closed = True


def _auth_disabled_settings() -> Dict[str, Any]:
    return {
        "enabled": False,
        "header": "X-API-Token",
        "token": "",
        "tenant_id": "",
        "token_map": {},
        "signing": {"required": False},
    }


def _setup(monkeypatch):
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    client = TestClient(srv.app)
    return srv, client


def test_agentic_tools_default_openai(monkeypatch):
    _, client = _setup(monkeypatch)
    res = client.get("/memory/agentic/tools", headers={"X-Tenant-ID": "t1"})
    assert res.status_code == 200
    payload = res.json()
    assert payload["format"] == "openai"
    assert payload["tool_names"] == [
        "entity_profile",
        "topic_timeline",
        "time_since",
        "quotes",
        "relations",
    ]
    assert payload["count"] == 5
    assert payload["tools"][0]["type"] == "function"


def test_agentic_tools_mcp_with_whitelist(monkeypatch):
    _, client = _setup(monkeypatch)
    res = client.get(
        "/memory/agentic/tools",
        params=[("format", "mcp"), ("tool_whitelist", "explain"), ("tool_whitelist", "entity_status")],
        headers={"X-Tenant-ID": "t1"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["format"] == "mcp"
    assert payload["tool_names"] == ["explain", "entity_status"]
    assert payload["count"] == 2
    assert payload["tools"][0]["name"] == "explain"
    assert "inputSchema" in payload["tools"][0]


def test_agentic_query_single_tool_success(monkeypatch):
    srv, client = _setup(monkeypatch)

    async def _fake_route(**kwargs):
        _ = kwargs
        return {
            "has_tool_call": True,
            "finish_reason": "tool_calls",
            "model": "gpt-4o-mini",
            "tool_name": "entity_profile",
            "tool_args": {"entity": "张三"},
            "tool_args_raw": '{"entity":"张三"}',
            "tool_args_invalid": False,
            "tool_call_id": "call_1",
        }

    runtime = _RuntimeStub()

    def _fake_runtime(**kwargs):
        _ = kwargs
        return runtime

    monkeypatch.setattr(srv, "_agentic_route_tool_call", _fake_route)
    monkeypatch.setattr(srv, "_create_agentic_runtime", _fake_runtime)

    res = client.post(
        "/memory/agentic/query",
        headers={"X-Tenant-ID": "t1"},
        json={"query": "张三最近在忙什么？"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["tool_used"] == "entity_profile"
    assert payload["tool_args"] == {"entity": "张三"}
    assert payload["result"]["matched"] is True
    assert payload["result"]["data"]["entity"]["name"] == "张三"
    assert runtime.calls == [("entity_profile", {"entity": "张三"})]
    assert runtime.closed is True


def test_agentic_query_no_tool_call_returns_no_match(monkeypatch):
    srv, client = _setup(monkeypatch)

    async def _fake_route(**kwargs):
        _ = kwargs
        return {
            "has_tool_call": False,
            "finish_reason": "stop",
            "model": "gpt-4o-mini",
        }

    def _never_runtime(**kwargs):  # pragma: no cover
        _ = kwargs
        raise AssertionError("runtime should not be created when router returns no tool call")

    monkeypatch.setattr(srv, "_agentic_route_tool_call", _fake_route)
    monkeypatch.setattr(srv, "_create_agentic_runtime", _never_runtime)

    res = client.post(
        "/memory/agentic/query",
        headers={"X-Tenant-ID": "t1"},
        json={"query": "你还记得吗"},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["tool_used"] is None
    assert payload["result"]["matched"] is False
    assert payload["result"]["message"] == "未识别到可执行的记忆工具"


def test_agentic_query_invalid_router_args_becomes_no_match(monkeypatch):
    srv, client = _setup(monkeypatch)

    async def _fake_route(**kwargs):
        _ = kwargs
        return {
            "has_tool_call": True,
            "finish_reason": "tool_calls",
            "model": "gpt-4o-mini",
            "tool_name": "explain",
            "tool_args": {},
            "tool_args_raw": "{}",
            "tool_args_invalid": False,
            "tool_call_id": "call_2",
        }

    def _never_runtime(**kwargs):  # pragma: no cover
        _ = kwargs
        raise AssertionError("runtime should not be created for invalid router args")

    monkeypatch.setattr(srv, "_agentic_route_tool_call", _fake_route)
    monkeypatch.setattr(srv, "_create_agentic_runtime", _never_runtime)

    res = client.post(
        "/memory/agentic/query",
        headers={"X-Tenant-ID": "t1"},
        json={"query": "依据是什么", "tool_whitelist": ["explain"]},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["tool_used"] == "explain"
    assert payload["result"]["matched"] is False
    assert payload["result"]["message"] == "router_returned_invalid_tool_args"


def test_agentic_execute_unknown_tool_returns_no_match(monkeypatch):
    _, client = _setup(monkeypatch)
    res = client.post(
        "/memory/agentic/execute",
        headers={"X-Tenant-ID": "t1"},
        json={"tool_name": "unknown_tool", "args": {}},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["tool_used"] == "unknown_tool"
    assert payload["result"]["matched"] is False
    assert payload["meta"]["tool_found"] is False


def test_agentic_execute_validates_required_args(monkeypatch):
    _, client = _setup(monkeypatch)
    res = client.post(
        "/memory/agentic/execute",
        headers={"X-Tenant-ID": "t1"},
        json={"tool_name": "explain", "args": {}},
    )
    assert res.status_code == 400
    assert "missing_required_args:event_id" in str(res.json().get("detail"))
