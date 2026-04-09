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


def test_ingest_dialog_v1_creates_job_and_status(monkeypatch, tmp_path) -> None:
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
        "turns": [
            {"turn_id": "t0001", "role": "user", "text": "hi"},
            {"turn_id": "t0002", "role": "assistant", "text": "hello"},
        ],
        "commit_id": "c1",
        "cursor": {"base_turn_id": None},
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/ingest/dialog/v1", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 202
    job_id = res.json().get("job_id")
    assert job_id

    job = client.get(f"/ingest/jobs/{job_id}", headers={"X-Tenant-ID": "t1"})
    assert job.status_code == 200
    payload = job.json()
    assert payload["status"] in {"RECEIVED", "STAGE2_RUNNING", "STAGE3_RUNNING", "COMPLETED"}
    assert payload["session_id"] == "sess-1"

    sess = client.get("/ingest/sessions/sess-1", headers={"X-Tenant-ID": "t1"})
    assert sess.status_code == 200
    sdata = sess.json()
    assert sdata["cursor_committed"] == "t0002"


def test_retrieval_dialog_v2_calls_retrieval(monkeypatch) -> None:
    from modules.memory.api import server as srv

    called: Dict[str, Any] = {}

    async def _retrieval_stub(_store, **kwargs):
        called.update(kwargs)
        return {"evidence": [{"id": "e1"}], "debug": {"executed_calls": []}}

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    monkeypatch.setattr(srv, "retrieval", _retrieval_stub)

    client = TestClient(srv.app)
    body = {
        "query": "who am i",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "run_id": "sess-1",
        "strategy": "dialog_v2_test",
        "topk": 12,
        "debug": True,
        "with_answer": False,
        "task": "L1",
        "backend": "tkg",
        "tkg_explain": True,
        "enable_event_route": False,
        "enable_evidence_route": False,
        "enable_knowledge_route": False,
        "enable_entity_route": False,
        "enable_time_route": False,
        "dialog_v2_test_ablation": {
            "disabled_routes": ["event", "time"],
            "disabled_backlinks": ["event"],
            "disabled_signals": ["event", "timestamp", "recency"],
            "source_native_only": True,
        },
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/retrieval/dialog/v2", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 200
    assert called["strategy"] == "dialog_v2_test"
    assert called["topk"] == 12
    assert called["run_id"] == "sess-1"
    assert called["task"] == "L1"
    assert called["enable_event_route"] is False
    assert called["enable_evidence_route"] is False
    assert called["enable_knowledge_route"] is False
    assert called["enable_entity_route"] is False
    assert called["enable_time_route"] is False
    assert called["dialog_v2_test_ablation"] == {
        "disabled_routes": ["event", "time"],
        "disabled_backlinks": ["event"],
        "disabled_signals": ["event", "timestamp", "recency"],
        "source_native_only": True,
    }


def test_retrieval_dialog_v2_default_topk_comes_from_config(monkeypatch) -> None:
    from modules.memory.api import server as srv

    called: Dict[str, Any] = {}

    async def _retrieval_stub(_store, **kwargs):
        called.update(kwargs)
        return {"evidence": [{"id": "e1"}], "debug": {"executed_calls": []}}

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    monkeypatch.setattr(srv, "retrieval", _retrieval_stub)

    client = TestClient(srv.app)
    body = {
        "query": "who am i",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "run_id": "sess-1",
        "strategy": "dialog_v2",
        "debug": True,
        "with_answer": False,
        "task": "L1",
        "backend": "tkg",
        "tkg_explain": True,
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }
    res = client.post("/retrieval/dialog/v2", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 200
    # default topk is configured in modules/memory/config/memory.config.yaml (memory.api.topk_defaults.retrieval)
    assert called["topk"] == 15


def test_retrieval_dialog_v2_with_answer_uses_client_meta_adapter(monkeypatch) -> None:
    from modules.memory.api import server as srv

    called: Dict[str, Any] = {}

    class _FakeAdapter:
        def generate(self, messages, response_format=None):  # type: ignore[no-untyped-def]
            return "ok"

    async def _retrieval_stub(_store, **kwargs):
        called.update(kwargs)
        return {"evidence": [], "debug": {"executed_calls": []}}

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    monkeypatch.setattr(srv, "retrieval", _retrieval_stub)
    monkeypatch.setattr(srv, "build_llm_from_byok", lambda **_: _FakeAdapter())

    client = TestClient(srv.app)
    body = {
        "query": "who am i",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "run_id": "sess-1",
        "with_answer": True,
        "client_meta": {
            "memory_policy": "user",
            "user_id": "u:1",
            "llm_mode": "byok",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": "sk-test",
        },
    }
    res = client.post("/retrieval/dialog/v2", headers={"X-Tenant-ID": "t1"}, json=body)
    assert res.status_code == 200
    assert called.get("qa_generate") is not None
    assert called["qa_generate"]("sys", "user") == "ok"
