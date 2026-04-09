from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_resolve_args = None
        self.last_detail_args = None
        self.last_facts_args = None
        self.last_rel_args = None
        self.last_events_args = None
        self.resolve_response = [
            {"entity_id": "ent-1", "name": "Alice", "type": "PERSON"},
        ]
        self.detail_response = {"id": "ent-1", "name": "Alice", "type": "PERSON", "cluster_label": "Alice"}
        self.facts_response = [{"id": "k-1", "summary": "Alice likes apples"}]
        self.relations_response = [{"entity_id": "ent-2", "name": "Bob", "type": "PERSON", "weight": 2.0}]
        self.events_response = [{"id": "evt-1", "summary": "Met Bob", "t_abs_start": "2026-01-20T00:00:00Z"}]

    async def resolve_entities(self, **kwargs):
        self.last_resolve_args = kwargs
        return list(self.resolve_response)

    async def entity_detail(self, **kwargs):
        self.last_detail_args = kwargs
        return dict(self.detail_response)

    async def entity_facts(self, **kwargs):
        self.last_facts_args = kwargs
        return list(self.facts_response)

    async def entity_relations(self, **kwargs):
        self.last_rel_args = kwargs
        return list(self.relations_response)

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


def test_memory_entity_profile_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/entity-profile",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"entity": "Alice", "user_tokens": ["u:alice"]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is True
    assert payload["entity"]["id"] == "ent-1"
    assert payload["facts"][0]["summary"] == "Alice likes apples"
    assert payload["relations"][0]["entity_id"] == "ent-2"
    assert payload["recent_events"][0]["event_id"] == "evt-1"
    assert stub.last_resolve_args == {
        "tenant_id": "tenant-x",
        "name": "Alice",
        "limit": 5,
    }
