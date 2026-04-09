from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx

from modules.memory.adk.infra_adapter import HttpMemoryInfraAdapter


def test_infra_adapter_builds_headers_and_payload_for_all_endpoints() -> None:
    async def _run() -> None:
        calls: List[Dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = None
            if request.content:
                body = json.loads(request.content.decode("utf-8"))
            calls.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "params": dict(request.url.params),
                    "params_user_tokens": list(request.url.params.get_list("user_tokens")),
                    "headers": dict(request.headers),
                    "body": body,
                }
            )
            if request.url.path == "/memory/v1/state/properties":
                return httpx.Response(200, json={"vocab_version": "v1", "properties": []})
            return httpx.Response(200, json={"ok": request.url.path})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://memory.local") as client:
            adapter = HttpMemoryInfraAdapter(
                base_url="http://memory.local",
                tenant_id="tenant_1",
                auth_token="token_abc",
                client=client,
            )

            await adapter.resolve_entity(name="张三", user_tokens=["u:alice"], limit=5, debug=False)
            await adapter.entity_profile_api(entity_id="ent-1", user_tokens=["u:alice"], include_states=True, facts_limit=10)
            await adapter.time_since_api(entity_id="ent-1", topic="项目推进", user_tokens=["u:alice"], limit=9)
            await adapter.relations_api(entity_id="ent-1", relation_type="co_occurs_with", user_tokens=["u:alice"], limit=8)
            await adapter.quotes_api(entity_id="ent-1", topic="项目推进", user_tokens=["u:alice"], limit=7)
            await adapter.topic_timeline_api(topic="项目推进", keywords=["进展"], user_tokens=["u:alice"], with_entities=True, limit=6)
            await adapter.explain_api(event_id="evt-1", user_tokens=["u:alice"])
            await adapter.list_entities_api(query="张", type="PERSON", user_tokens=["u:alice"], limit=5, cursor="c:5")
            await adapter.list_topics_api(query="项目", parent_path="work", min_events=1, user_tokens=["u:alice"], limit=4, cursor="c:4")
            await adapter.state_current_api(subject_id="ent-1", property="occupation", user_tokens=["u:alice"])
            await adapter.state_at_time_api(subject_id="ent-1", property="occupation", t_iso="2026-02-01T00:00:00Z", user_tokens=["u:alice"])
            await adapter.state_what_changed_api(subject_id="ent-1", property="occupation", start_iso="2026-01-01T00:00:00Z", limit=20)
            await adapter.state_time_since_api(subject_id="ent-1", property="occupation", end_iso="2026-02-01T00:00:00Z")
            await adapter.state_properties_api(user_tokens=["u:alice", "u:bob"], limit=120)

        assert len(calls) == 14
        headers0 = {k.lower(): v for k, v in calls[0]["headers"].items()}
        assert headers0["x-tenant-id"] == "tenant_1"
        assert headers0["authorization"] == "Bearer token_abc"
        assert calls[0]["body"] == {"name": "张三", "user_tokens": ["u:alice"], "limit": 5, "debug": False}
        assert "tenant_id" not in (calls[0]["body"] or {})
        assert calls[7]["method"] == "GET"
        assert calls[7]["path"] == "/memory/v1/entities"
        assert calls[7]["params"]["type"] == "PERSON"
        assert calls[8]["method"] == "GET"
        assert calls[8]["path"] == "/memory/v1/topics"
        assert calls[8]["params"]["min_events"] == "1"
        assert calls[13]["method"] == "GET"
        assert calls[13]["path"] == "/memory/v1/state/properties"
        assert calls[13]["params"]["limit"] == "120"
        # Query(list[str]) should be encoded as repeated query params on GET.
        assert calls[13]["params_user_tokens"] == ["u:alice", "u:bob"]

    asyncio.run(_run())


def test_infra_adapter_http_error_payload_prefers_json_then_text() -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/memory/v1/time-since":
                return httpx.Response(400, json={"detail": "missing_core_requirements"})
            if request.url.path == "/memory/v1/explain":
                return httpx.Response(503, text="temporarily unavailable")
            return httpx.Response(404, text="not found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://memory.local") as client:
            adapter = HttpMemoryInfraAdapter(base_url="http://memory.local", tenant_id="t1", client=client)
            r1 = await adapter.time_since_api(entity="张三")
            r2 = await adapter.explain_api(event_id="evt-1")

        assert r1["status_code"] == 400
        assert r1["body"] == {"detail": "missing_core_requirements"}
        assert r2["status_code"] == 503
        assert isinstance(r2["body"], str)
        assert "temporarily unavailable" in r2["body"]

    asyncio.run(_run())


def test_infra_adapter_timeout_maps_to_504() -> None:
    async def _run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout", request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="http://memory.local") as client:
            adapter = HttpMemoryInfraAdapter(base_url="http://memory.local", tenant_id="t1", client=client)
            out = await adapter.resolve_entity(name="张三")

        assert out["status_code"] == 504
        assert "timeout" in str(out["body"]).lower()

    asyncio.run(_run())
