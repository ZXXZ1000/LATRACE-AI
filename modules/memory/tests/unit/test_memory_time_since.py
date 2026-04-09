from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_events_args = None
        self.events_response = [
            {"id": "evt-1", "summary": "Met Bob", "t_abs_start": "2026-01-20T00:00:00Z"},
        ]

    async def list_events(self, **kwargs):
        self.last_events_args = kwargs
        return list(self.events_response)


def _setup(monkeypatch):
    from modules.memory.api import server as srv

    stub = _GraphStub()
    monkeypatch.setattr(srv, "graph_svc", stub)
    monkeypatch.setattr(
        srv,
        "_auth_settings",
        lambda: {
            "enabled": False,
            "header": "X-API-Token",
            "token": "",
            "tenant_id": "",
            "token_map": {},
        },
    )
    client = TestClient(srv.app)
    return stub, client


def test_memory_time_since_entity_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/time-since",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"entity_id": "ent-1", "user_tokens": ["u:alice"], "limit": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["entity_id"] == "ent-1"
    assert payload["summary"] == "Met Bob"
    assert stub.last_events_args["entity_id"] == "ent-1"
