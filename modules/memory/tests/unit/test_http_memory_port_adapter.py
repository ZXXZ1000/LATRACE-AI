from __future__ import annotations

import asyncio
import json

import httpx

from modules.memory.adapters.http_memory_port import HttpMemoryPort
from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


def test_http_memory_port_search_write_delete_payload_and_parsing() -> None:
    async def _run() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "data": request.content,
                    "headers": dict(request.headers),
                    "params": dict(request.url.params),
                }
            )
            if request.url.path.endswith("/search"):
                return httpx.Response(
                    200,
                    json={
                        "hits": [
                            {
                                "id": "h1",
                                "score": 0.5,
                                "entry": {
                                    "id": "e1",
                                    "kind": "semantic",
                                    "modality": "text",
                                    "contents": ["hello"],
                                    "metadata": {"source": "x"},
                                },
                            }
                        ],
                        "neighbors": {},
                        "hints": "",
                        "trace": {"ok": True},
                    },
                )
            if request.url.path.endswith("/write"):
                return httpx.Response(200, json={"value": "v-ADD-batch-1"})
            if request.url.path.endswith("/delete"):
                return httpx.Response(200, json={"value": "v-DELETE-1"})
            return httpx.Response(404, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as sess:
            port = HttpMemoryPort(
                base_url="http://example",
                tenant_id="t1",
                headers={"Authorization": "Bearer x"},
                session=sess,
                timeout_s=3.0,
            )

            sr = await port.search(
                "hello",
                topk=5,
                filters=SearchFilters(user_id=["u:1"], user_match="all"),
                expand_graph=False,
            )
            assert len(sr.hits) == 1
            assert sr.hits[0].id == "h1"
            assert sr.hits[0].entry.id == "e1"
            assert sr.trace.get("ok") is True

            ver = await port.write([MemoryEntry(kind="semantic", modality="text", contents=["x"])], links=None, upsert=True)
            assert ver.value == "v-ADD-batch-1"

            vdel = await port.delete("e1", soft=True, reason="test")
            assert vdel.value == "v-DELETE-1"

        # basic request assertions
        assert len(calls) == 3
        headers0 = {k.lower(): v for k, v in calls[0]["headers"].items()}
        assert headers0["x-tenant-id"] == "t1"
        assert headers0["authorization"] == "Bearer x"
        payload0 = json.loads(calls[0]["data"].decode("utf-8"))
        assert payload0["query"] == "hello"
        assert payload0["topk"] == 5
        assert payload0["expand_graph"] is False
        assert payload0["filters"]["user_id"] == ["u:1"]

    asyncio.run(_run())


def test_http_memory_port_graph_upsert_v0_payload_and_ok_check() -> None:
    async def _run() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "data": request.content,
                    "headers": dict(request.headers),
                    "params": dict(request.url.params),
                }
            )
            if request.url.path.endswith("/graph/v0/upsert"):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(404, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as sess:
            port = HttpMemoryPort(
                base_url="http://example",
                tenant_id="t1",
                headers={"Authorization": "Bearer x"},
                session=sess,
                timeout_s=3.0,
            )

            req = GraphUpsertRequest(segments=[], edges=[])
            await port.graph_upsert_v0(req)

        assert len(calls) == 1
        assert calls[0]["method"].upper() == "POST"
        headers0 = {k.lower(): v for k, v in calls[0]["headers"].items()}
        assert headers0["x-tenant-id"] == "t1"
        payload0 = json.loads(calls[0]["data"].decode("utf-8"))
        assert payload0["segments"] == []
        assert payload0["edges"] == []

    asyncio.run(_run())


def test_http_memory_port_graph_query_endpoints_use_get_params_and_parse_items() -> None:
    async def _run() -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "data": request.content,
                    "headers": dict(request.headers),
                    "params": dict(request.url.params),
                }
            )
            if request.url.path.endswith("/graph/v0/events"):
                return httpx.Response(200, json={"items": [{"id": "ev1"}]})
            if request.url.path.endswith("/graph/v0/places"):
                return httpx.Response(200, json={"items": [{"id": "p1"}]})
            if "/graph/v0/events/" in str(request.url):
                return httpx.Response(200, json={"item": {"id": "ev1", "detail": True}})
            if "/graph/v0/places/" in str(request.url):
                return httpx.Response(200, json={"item": {"id": "p1", "detail": True}})
            if "/graph/v0/explain/event/" in str(request.url):
                return httpx.Response(200, json={"item": {"event": {"id": "ev1"}, "utterances": [{"id": "utt1"}]}})
            return httpx.Response(404, json={})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as sess:
            port = HttpMemoryPort(base_url="http://example", tenant_id="t1", session=sess)
            items = await port.graph_list_events(tenant_id="t1", segment_id="s1", limit=12)
            assert items and items[0]["id"] == "ev1"
            places = await port.graph_list_places(tenant_id="t1", name="room", limit=9)
            assert places and places[0]["id"] == "p1"
            ev = await port.graph_event_detail(tenant_id="t1", event_id="ev1")
            assert ev.get("detail") is True
            pl = await port.graph_place_detail(tenant_id="t1", place_id="p1")
            assert pl.get("detail") is True
            ex = await port.graph_explain_event_evidence(tenant_id="t1", event_id="ev1")
            assert ex.get("event", {}).get("id") == "ev1"

        # validate GET usage and query params
        assert calls[0]["method"].upper() == "GET"
        assert calls[0]["params"]["segment_id"] == "s1"
        assert calls[0]["params"]["limit"] == "12"
        assert calls[1]["method"].upper() == "GET"
        assert calls[1]["params"]["name"] == "room"
        assert calls[1]["params"]["limit"] == "9"
        assert calls[2]["method"].upper() == "GET"
        assert calls[3]["method"].upper() == "GET"
        assert calls[4]["method"].upper() == "GET"

    asyncio.run(_run())
