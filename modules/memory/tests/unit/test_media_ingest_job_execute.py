from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

from modules.memory.contracts.graph_models import Event, GraphUpsertRequest, MediaSegment
from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore


def _graph_request() -> GraphUpsertRequest:
    return GraphUpsertRequest(
        segments=[
            MediaSegment(
                id="seg-1",
                tenant_id="tenant-a",
                source_id="demo.mp4",
                t_media_start=0.0,
                t_media_end=2.0,
                user_id=["u:1"],
                memory_domain="media",
            )
        ],
        events=[
            Event(
                id="evt-1",
                tenant_id="tenant-a",
                summary="hello",
                source="demo.mp4",
                user_id=["u:1"],
                memory_domain="media",
            )
        ],
    )


def test_run_media_ingest_job_executes_compile_and_write(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv
    import modules.media_graph_compiler as media_compiler

    db_path = tmp_path / "ingest_jobs.db"
    store = AsyncIngestJobStore({"sqlite_path": str(db_path)})
    monkeypatch.setattr(srv, "ingest_store", store)

    compile_calls: List[Dict[str, Any]] = []
    write_calls: List[Dict[str, Any]] = []

    def _compile_video_stub(request):  # type: ignore[no-untyped-def]
        compile_calls.append({"source_id": request.source.source_id, "tenant_id": request.routing.tenant_id})
        return SimpleNamespace(graph_request=_graph_request())

    async def _media_write_stub(*args, **kwargs):  # type: ignore[no-untyped-def]
        write_calls.append({"args": args, "kwargs": kwargs})
        return {
            "status": "ok",
            "written_entries": 1,
            "graph_nodes_written": 2,
            "trace": {"timing_ms": {"graph_upsert_ms": 10, "publish_ms": 5}},
        }

    monkeypatch.setattr(media_compiler, "compile_video", _compile_video_stub)
    monkeypatch.setattr(srv, "media_session_write", _media_write_stub)

    payload = {
        "routing": {"user_id": ["u:1"], "memory_domain": "media"},
        "source_ref": {"source_id": "demo.mp4", "file_path": "/tmp/demo.mp4"},
        "overwrite_existing": True,
    }

    record, created = asyncio.run(
        store.create_job(
            session_id="media::demo.mp4",
            commit_id="c1",
            tenant_id="tenant-a",
            api_key_id=None,
            request_id="req-1",
            user_tokens=["u:1"],
            memory_domain="media",
            llm_policy="best_effort",
            turns=[],
            base_turn_id=None,
            client_meta={},
            payload_raw=json.dumps(payload),
            job_type="media_video",
        )
    )
    assert created is True

    asyncio.run(srv._run_ingest_job_from_record(record))

    updated = asyncio.run(store.get_job(record.job_id))
    assert updated is not None
    assert updated.status == "COMPLETED"
    assert updated.metrics["graph_nodes_written"] == 2
    assert compile_calls == [{"source_id": "demo.mp4", "tenant_id": "tenant-a"}]
    assert write_calls
    assert write_calls[-1]["kwargs"]["source_id"] == "demo.mp4"
    assert write_calls[-1]["kwargs"]["overwrite_existing"] is True
