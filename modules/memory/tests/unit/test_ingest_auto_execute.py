from __future__ import annotations

from typing import Any, Dict

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


def test_ingest_wait_not_supported(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))
    async def _enqueue_stub(_record):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(srv, "_enqueue_ingest_job", _enqueue_stub)

    client = TestClient(srv.app)
    body = {
        "session_id": "sess-1",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "turns": [{"role": "user", "content": "hi"}],
        "commit_id": "c1",
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/ingest/dialog/v1?wait=true&wait_timeout_ms=2000", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 400
    assert res.json().get("detail") == "wait_not_supported"


def test_ingest_default_returns_202(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))
    async def _enqueue_stub(_record):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(srv, "_enqueue_ingest_job", _enqueue_stub)

    client = TestClient(srv.app)
    body = {
        "session_id": "sess-1",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "turns": [{"role": "user", "content": "hi"}],
        "commit_id": "c1",
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/ingest/dialog/v1", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 202
    data = res.json()
    assert data["job_id"]
    assert data.get("status_url")
    assert data.get("enqueue") is True


def test_ingest_enqueue_failure_returns_503(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))
    async def _enqueue_stub(_record):  # type: ignore[no-untyped-def]
        return False

    monkeypatch.setattr(srv, "_enqueue_ingest_job", _enqueue_stub)

    client = TestClient(srv.app)
    body = {
        "session_id": "sess-1",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "turns": [{"role": "user", "content": "hi"}],
        "commit_id": "c1",
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/ingest/dialog/v1", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 503
    data = res.json()
    assert data["job_id"]
    assert data.get("status") == "ENQUEUE_FAILED"
