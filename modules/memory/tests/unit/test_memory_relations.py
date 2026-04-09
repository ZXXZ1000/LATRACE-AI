from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_resolve_args = None
        self.last_rel_args = None
        self.resolve_response = [{"entity_id": "ent-1", "name": "Alice", "type": "PERSON"}]
        self.relations_response = [
            {
                "entity_id": "ent-2",
                "name": "Bob",
                "type": "PERSON",
                "strength": 3,
                "first_mentioned": "2026-01-01T00:00:00Z",
                "last_mentioned": "2026-01-20T00:00:00Z",
                "evidence_event_ids": ["evt-1", "evt-2"],
            }
        ]

    async def resolve_entities(self, **kwargs):
        self.last_resolve_args = kwargs
        return list(self.resolve_response)

    async def entity_relations_by_events(self, **kwargs):
        self.last_rel_args = kwargs
        return list(self.relations_response)


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


def test_memory_relations_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/relations",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"entity": "Alice", "user_tokens": ["u:alice"], "limit": 10},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is True
    assert payload["relations"][0]["entity_id"] == "ent-2"
    assert payload["relations"][0]["strength"] == 3
    assert stub.last_resolve_args == {"tenant_id": "tenant-x", "name": "Alice", "limit": 5}
