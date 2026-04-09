from __future__ import annotations

from typing import Any, Dict

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


def _auth_disabled_settings() -> Dict[str, Any]:
    return {
        "enabled": False,
        "header": "X-API-Token",
        "token": "",
        "tenant_id": "",
        "token_map": {},
        "signing": {"required": False},
    }


def _default_body(with_answer: bool) -> Dict[str, Any]:
    return {
        "query": "who am i",
        "user_tokens": ["u:1"],
        "memory_domain": "dialog",
        "run_id": "sess-1",
        "strategy": "dialog_v2",
        "topk": 5,
        "debug": False,
        "with_answer": with_answer,
        "task": "GENERAL",
        "backend": "tkg",
        "client_meta": {"memory_policy": "user", "user_id": "u:1", "llm_mode": "platform"},
    }


def test_with_answer_disabled(monkeypatch) -> None:
    from modules.memory.api import server as srv

    async def _retrieval_stub(_store, **_kwargs):
        return {"evidence": [], "debug": {"executed_calls": []}}

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    monkeypatch.setattr(srv, "retrieval", _retrieval_stub)
    monkeypatch.setattr(srv, "WITH_ANSWER_SETTINGS", {"enabled": False, "required_scope": ""})

    client = TestClient(srv.app)
    res = client.post("/retrieval/dialog/v2", headers={"X-Tenant-ID": "t1"}, json=_default_body(with_answer=True))
    assert res.status_code == 403
    assert res.json().get("detail") == "with_answer_disabled"


def test_with_answer_requires_scope(monkeypatch) -> None:
    from modules.memory.api import server as srv

    async def _retrieval_stub(_store, **_kwargs):
        return {"evidence": [], "debug": {"executed_calls": []}}

    def _auth_ctx(_request):
        return {"tenant_id": "t1", "subject": "k1", "scopes": ["memory.read"]}

    monkeypatch.setattr(srv, "retrieval", _retrieval_stub)
    monkeypatch.setattr(srv, "_authenticate_request", _auth_ctx)
    monkeypatch.setattr(srv, "WITH_ANSWER_SETTINGS", {"enabled": True, "required_scope": "memory.qa"})

    client = TestClient(srv.app)
    res = client.post("/retrieval/dialog/v2", headers={"X-Tenant-ID": "t1"}, json=_default_body(with_answer=True))
    assert res.status_code == 403
    assert res.json().get("detail") == "with_answer_forbidden"
