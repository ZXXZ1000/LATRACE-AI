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


def test_run_ingest_job_passes_overwrite_existing(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))

    captured: Dict[str, Any] = {}

    async def _fake_session_write(*_args, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "trace": {}, "written_entries": 0}

    monkeypatch.setattr(srv, "session_write", _fake_session_write)

    record, _created = asyncio.run(
        srv.ingest_store.create_job(
            session_id="sess-1",
            commit_id=None,
            tenant_id="t1",
            api_key_id=None,
            request_id="r1",
            turns=[{"turn_id": "t1", "role": "user", "text": "hi"}],
            user_tokens=["u:1"],
            base_turn_id=None,
            client_meta={"overwrite_existing": True, "stage2_enabled": False},
            memory_domain="dialog",
            llm_policy="require",
            payload_raw=None,
        )
    )

    asyncio.run(
        srv._run_ingest_job(
            job_id=record.job_id,
            tenant_id="t1",
            user_tokens=["u:1"],
            memory_domain="dialog",
            llm_policy="require",
            payload=None,
        )
    )

    assert captured.get("overwrite_existing") is True
