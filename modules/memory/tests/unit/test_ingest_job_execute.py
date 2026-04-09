from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient

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


def _create_job(store: AsyncIngestJobStore, turns: List[Dict[str, Any]]):
    return asyncio.run(
        store.create_job(
            session_id="sess-1",
            commit_id=None,
            tenant_id="t1",
            api_key_id=None,
            request_id="r1",
            turns=turns,
            user_tokens=["u:1"],
            base_turn_id=None,
            client_meta={},
            memory_domain="dialog",
            llm_policy="require",
        )
    )


def test_ingest_job_execute_runs_job(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))

    called: List[Dict[str, Any]] = []

    async def _enqueue_stub(_record):  # type: ignore[no-untyped-def]
        called.append({"job_id": _record.job_id})
        return True

    monkeypatch.setattr(srv, "_enqueue_ingest_job", _enqueue_stub)

    record, _created = _create_job(srv.ingest_store, [{"turn_id": "t1", "role": "user", "text": "hi"}])

    client = TestClient(srv.app)
    res = client.post(
        "/ingest/jobs/execute",
        headers={"X-Tenant-ID": "t1"},
        json={"job_id": record.job_id},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["job_id"] == record.job_id
    assert payload.get("enqueued") is True
    assert called
