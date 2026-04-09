from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_resolve_args = None
        self.last_events_args = None
        self.last_explain_args = None
        self.resolve_response = [{"entity_id": "ent-1", "name": "Alice", "type": "PERSON"}]
        self.events_response = [{"id": "evt-1"}]
        self.explain_response = {
            "event": {"id": "evt-1", "t_abs_start": "2026-01-20T00:00:00Z"},
            "utterances": [
                {"id": "utt-1", "raw_text": "你好", "t_media_start": 1.0},
            ],
            "utterance_speakers": [{"utterance_id": "utt-1", "entity_id": "ent-1"}],
        }

    async def resolve_entities(self, **kwargs):
        self.last_resolve_args = kwargs
        return list(self.resolve_response)

    async def list_events(self, **kwargs):
        self.last_events_args = kwargs
        return list(self.events_response)

    async def explain_event_evidence(self, **kwargs):
        self.last_explain_args = kwargs
        return dict(self.explain_response)


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


def test_memory_quotes_entity_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/quotes",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"entity": "Alice", "user_tokens": ["u:alice"], "limit": 5},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["entity_id"] == "ent-1"
    assert payload["quotes"][0]["text"] == "你好"
    assert payload["quotes"][0]["speaker_id"] == "ent-1"
    assert stub.last_resolve_args == {"tenant_id": "tenant-x", "name": "Alice", "limit": 5}
    assert stub.last_events_args["entity_id"] == "ent-1"
