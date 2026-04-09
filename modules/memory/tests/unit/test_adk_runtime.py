from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.infra_adapter import HttpMemoryInfraAdapter
from modules.memory.adk.runtime import MemoryAdkRuntime, create_memory_runtime


class _FakeInfra:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

    async def aclose(self) -> None:
        self.exited = True

    async def resolve_entity(self, **kwargs):
        self.calls.append({"fn": "resolve_entity", "kwargs": dict(kwargs)})
        return {"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}}

    async def entity_profile_api(self, **kwargs):
        self.calls.append({"fn": "entity_profile_api", "kwargs": dict(kwargs)})
        return {
            "found": True,
            "entity": {"id": "ent-1", "name": "张三"},
            "facts": [{"k": "v"}],
            "relations": [],
            "recent_events": [],
        }

    async def time_since_api(self, **kwargs):
        self.calls.append({"fn": "time_since_api", "kwargs": dict(kwargs)})
        return {"last_mentioned": "2026-02-01T00:00:00Z", "days_ago": 2, "summary": "ok", "trace": {"source": "graph_entity_events"}}

    async def relations_api(self, **kwargs):
        self.calls.append({"fn": "relations_api", "kwargs": dict(kwargs)})
        return {"found": True, "resolved_entity": {"id": "ent-1", "name": "张三"}, "relations": []}

    async def quotes_api(self, **kwargs):
        self.calls.append({"fn": "quotes_api", "kwargs": dict(kwargs)})
        return {"quotes": [], "total": 0}

    async def topic_timeline_api(self, **kwargs):
        self.calls.append({"fn": "topic_timeline_api", "kwargs": dict(kwargs)})
        return {"timeline": [], "total": 0}

    async def explain_api(self, **kwargs):
        self.calls.append({"fn": "explain_api", "kwargs": dict(kwargs)})
        return {"found": True, "event_id": kwargs.get("event_id"), "event": {"id": kwargs.get("event_id")}, "evidences": []}

    async def list_entities_api(self, **kwargs):
        self.calls.append({"fn": "list_entities_api", "kwargs": dict(kwargs)})
        return {"entities": [{"id": "ent-1", "name": "张三"}], "total": 1, "has_more": False}

    async def list_topics_api(self, **kwargs):
        self.calls.append({"fn": "list_topics_api", "kwargs": dict(kwargs)})
        return {"topics": [{"topic_path": "work/project-alpha"}], "total": 1, "has_more": False}

    async def state_current_api(self, **kwargs):
        self.calls.append({"fn": "state_current_api", "kwargs": dict(kwargs)})
        return {"item": {"subject_id": kwargs.get("subject_id"), "property": kwargs.get("property"), "value": "工程师"}}

    async def state_at_time_api(self, **kwargs):
        self.calls.append({"fn": "state_at_time_api", "kwargs": dict(kwargs)})
        return {"item": {"subject_id": kwargs.get("subject_id"), "property": kwargs.get("property"), "value": "工程师"}}

    async def state_what_changed_api(self, **kwargs):
        self.calls.append({"fn": "state_what_changed_api", "kwargs": dict(kwargs)})
        return {"items": []}

    async def state_time_since_api(self, **kwargs):
        self.calls.append({"fn": "state_time_since_api", "kwargs": dict(kwargs)})
        return {"subject_id": kwargs.get("subject_id"), "property": kwargs.get("property"), "last_changed_at": None, "seconds_ago": None}

    async def state_properties_api(self, **kwargs):
        self.calls.append({"fn": "state_properties_api", "kwargs": dict(kwargs)})
        return {
            "vocab_version": "v1",
            "properties": [
                {"name": "occupation", "allow_raw_value": True},
                {"name": "work_status", "allow_raw_value": True},
            ],
        }


def test_create_memory_runtime_is_sync_and_sets_default_user_tokens() -> None:
    runtime = create_memory_runtime(base_url="http://127.0.0.1:8000", tenant_id="tenant_a")
    assert isinstance(runtime, MemoryAdkRuntime)
    assert isinstance(runtime.infra, HttpMemoryInfraAdapter)
    assert runtime.default_user_tokens == ["u:tenant_a"]
    asyncio.run(runtime.aclose())


def test_runtime_async_context_manager_delegates_to_infra() -> None:
    async def _run() -> None:
        infra = _FakeInfra()
        runtime = MemoryAdkRuntime(tenant_id="t1", infra=infra)  # type: ignore[arg-type]
        async with runtime:
            pass
        assert infra.entered is True
        assert infra.exited is True

    asyncio.run(_run())


def test_runtime_entity_profile_uses_default_and_override_user_tokens() -> None:
    async def _run() -> None:
        infra = _FakeInfra()
        runtime = MemoryAdkRuntime(tenant_id="t1", infra=infra, user_tokens=["u:default"])  # type: ignore[arg-type]

        res1 = await runtime.entity_profile(entity="张三")
        res2 = await runtime.entity_profile(entity="张三", user_tokens=["u:override"])

        assert res1.to_llm_dict()["matched"] is True
        assert res2.to_llm_dict()["matched"] is True

        resolve_calls = [c for c in infra.calls if c["fn"] == "resolve_entity"]
        assert resolve_calls[0]["kwargs"]["user_tokens"] == ["u:default"]
        assert resolve_calls[1]["kwargs"]["user_tokens"] == ["u:override"]

        profile_calls = [c for c in infra.calls if c["fn"] == "entity_profile_api"]
        assert profile_calls[0]["kwargs"]["user_tokens"] == ["u:default"]
        assert profile_calls[1]["kwargs"]["user_tokens"] == ["u:override"]

    asyncio.run(_run())


def test_runtime_entity_status_uses_vocab_and_user_tokens_override() -> None:
    async def _run() -> None:
        infra = _FakeInfra()
        runtime = MemoryAdkRuntime(tenant_id="t1", infra=infra, user_tokens=["u:default"])  # type: ignore[arg-type]

        out = await runtime.entity_status(entity="张三", property="occupation", user_tokens=["u:alice"])
        llm = out.to_llm_dict()

        assert llm["matched"] is True
        vocab_calls = [c for c in infra.calls if c["fn"] == "state_properties_api"]
        assert vocab_calls and vocab_calls[0]["kwargs"]["user_tokens"] == ["u:alice"]
        state_calls = [c for c in infra.calls if c["fn"] == "state_current_api"]
        assert state_calls and state_calls[0]["kwargs"]["subject_id"] == "ent-1"

    asyncio.run(_run())


def test_runtime_exposes_openai_and_mcp_tool_exports() -> None:
    infra = _FakeInfra()
    runtime = MemoryAdkRuntime(tenant_id="t1", infra=infra)  # type: ignore[arg-type]
    defs = runtime.get_tool_definitions(enabled_only=True)
    openai_tools = runtime.get_openai_tools()
    mcp_tools = runtime.get_mcp_tools()

    assert defs
    assert all(d.default_enabled for d in defs)
    assert openai_tools and openai_tools[0].get("type") == "function"
    assert mcp_tools and "inputSchema" in mcp_tools[0]


def test_runtime_list_discovery_tools_work() -> None:
    async def _run() -> None:
        infra = _FakeInfra()
        runtime = MemoryAdkRuntime(tenant_id="t1", infra=infra, user_tokens=["u:default"])  # type: ignore[arg-type]
        e = await runtime.list_entities(query="张")
        t = await runtime.list_topics(query="项目")
        assert e.to_llm_dict()["matched"] is True
        assert t.to_llm_dict()["matched"] is True
        assert any(c["fn"] == "list_entities_api" for c in infra.calls)
        assert any(c["fn"] == "list_topics_api" for c in infra.calls)

    asyncio.run(_run())
