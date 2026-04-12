from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.contracts.graph_models import Event, GraphUpsertRequest, MediaSegment
from modules.memory.media_session_write import media_session_write


class _CapturingMediaStore:
    def __init__(self) -> None:
        self.graph_reqs: List[GraphUpsertRequest] = []
        self.publish_calls: List[Dict[str, Any]] = []
        self.purge_calls: List[Dict[str, Any]] = []

    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:
        for ev in body.events or []:
            if getattr(ev, "text_vector_id", None) is None:
                ev.text_vector_id = f"tkg_event_vec::{ev.id}"
        self.graph_reqs.append(body)

    async def publish_entries(self, *args, **kwargs):
        self.publish_calls.append({"args": args, "kwargs": kwargs})
        return {"vectors": len(list(kwargs.get("entry_ids") or [])), "graph": len(list(kwargs.get("graph_node_ids") or []))}

    async def graph_purge_source_except_graph_v0(self, *, tenant_id: str, source_id: str, keep_node_ids: List[str]) -> Dict[str, int]:
        self.purge_calls.append(
            {"tenant_id": tenant_id, "source_id": source_id, "keep_node_ids": list(keep_node_ids or [])}
        )
        return {"segments": 1, "events": 2}


def _media_graph_request() -> GraphUpsertRequest:
    return GraphUpsertRequest(
        segments=[
            MediaSegment(
                id="seg-1",
                tenant_id="tenant-a",
                source_id="demo.mp4",
                t_media_start=0.0,
                t_media_end=3.0,
                user_id=["u:1"],
                memory_domain="media",
                published=True,
            )
        ],
        events=[
            Event(
                id="evt-1",
                tenant_id="tenant-a",
                summary="Alice says hello",
                source="demo.mp4",
                user_id=["u:1"],
                memory_domain="media",
                published=True,
            )
        ],
    )


def test_media_session_write_publishes_graph_vectors() -> None:
    async def _run() -> None:
        store = _CapturingMediaStore()
        req = _media_graph_request()

        res = await media_session_write(
            store,
            tenant_id="tenant-a",
            user_tokens=["u:1"],
            memory_domain="media",
            graph_request=req,
            source_id="demo.mp4",
            overwrite_existing=False,
        )

        assert res["status"] == "ok"
        assert store.graph_reqs
        assert req.events[0].published is False
        assert req.segments[0].published is False
        assert store.publish_calls
        publish_kwargs = store.publish_calls[-1]["kwargs"]
        assert publish_kwargs["tenant_id"] == "tenant-a"
        assert "tkg_event_vec::evt-1" in publish_kwargs["entry_ids"]
        assert "evt-1" in publish_kwargs["graph_node_ids"]

    asyncio.run(_run())


def test_media_session_write_overwrite_existing_purges_stale_source_nodes() -> None:
    async def _run() -> None:
        store = _CapturingMediaStore()
        req = _media_graph_request()

        res = await media_session_write(
            store,
            tenant_id="tenant-a",
            user_tokens=["u:1"],
            memory_domain="media",
            graph_request=req,
            source_id="demo.mp4",
            overwrite_existing=True,
        )

        assert res["status"] == "ok"
        assert store.purge_calls
        purge_call = store.purge_calls[-1]
        assert purge_call["tenant_id"] == "tenant-a"
        assert purge_call["source_id"] == "demo.mp4"
        assert "seg-1" in purge_call["keep_node_ids"]
        assert "evt-1" in purge_call["keep_node_ids"]

    asyncio.run(_run())
