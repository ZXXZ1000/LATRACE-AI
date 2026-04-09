from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401

from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore


def _auth_disabled_settings() -> Dict[str, Any]:
    return {
        "enabled": False,
        "header": "X-API-Token",
        "token": "",
        "tenant_id": "",
        "token_map": {},
        "signing": {"required": False},
    }


def test_run_ingest_job_normalizes_turns_and_persists_stage2_marks(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))

    async def _fake_session_write(*_args, **_kwargs):
        return {
            "status": "ok",
            "written_entries": 1,
            "trace": {"graph_ids": {"event_ids": ["e1"], "timeslice_id": "ts1"}},
        }

    monkeypatch.setattr(srv, "session_write", _fake_session_write)

    record, _created = asyncio.run(
        srv.ingest_store.create_job(
            session_id="sess-1",
            commit_id=None,
            tenant_id="t1",
            api_key_id=None,
            request_id="r1",
            turns=[{"role": "user", "content": "hi"}],  # no turn_id/text provided
            user_tokens=["u:1"],
            base_turn_id=None,
            client_meta={"memory_policy": "user", "user_id": "u1"},
            memory_domain="dialog",
            llm_policy="best_effort",
            payload_raw=None,
        )
    )

    asyncio.run(
        srv._run_ingest_job(
            job_id=record.job_id,
            tenant_id="t1",
            user_tokens=["u:1"],
            memory_domain="dialog",
            llm_policy="best_effort",
            payload=None,
        )
    )

    updated = asyncio.run(srv.ingest_store.get_job(record.job_id))
    assert updated is not None
    assert updated.status == "COMPLETED"
    assert updated.stage2_marks
    assert updated.stage2_marks[0].get("turn_id") == "t0001"

