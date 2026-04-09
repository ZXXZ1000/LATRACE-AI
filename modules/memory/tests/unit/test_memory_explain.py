from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_explain_args = None
        self.explain_response = {
            "event": {"id": "evt-1", "summary": "Met Bob"},
            "entities": [{"id": "ent-1", "name": "Alice"}],
            "places": [],
            "timeslices": [],
            "evidences": [{"id": "evd-1"}],
            "utterances": [{"id": "utt-1", "raw_text": "你好"}],
            "utterance_speakers": [{"utterance_id": "utt-1", "entity_id": "ent-1"}],
            "knowledge": [{"id": "k-1", "summary": "Alice met Bob"}],
        }

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


def test_memory_explain_endpoint_returns_bundle(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/explain",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"event_id": "evt-1"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is True
    assert payload["event_id"] == "evt-1"
    assert payload["event"]["id"] == "evt-1"
    assert payload["evidences"][0]["id"] == "evd-1"
    assert payload["utterances"][0]["id"] == "utt-1"
    assert payload["knowledge"][0]["id"] == "k-1"
    assert stub.last_explain_args == {
        "tenant_id": "tenant-x",
        "event_id": "evt-1",
        "user_ids": ["u:tenant-x"],
        "memory_domain": None,
    }


def test_memory_explain_endpoint_forwards_optional_scope_filters(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/explain",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"event_id": "evt-1", "user_tokens": ["u:alice"], "memory_domain": "work"},
    )
    assert resp.status_code == 200
    assert stub.last_explain_args == {
        "tenant_id": "tenant-x",
        "event_id": "evt-1",
        "user_ids": ["u:alice"],
        "memory_domain": "work",
    }


def test_memory_explain_endpoint_not_found(monkeypatch):
    stub, client = _setup(monkeypatch)
    stub.explain_response = {
        "event": None,
        "entities": [],
        "places": [],
        "timeslices": [],
        "evidences": [],
        "utterances": [],
        "utterance_speakers": [],
        "knowledge": [],
    }
    resp = client.post(
        "/memory/v1/explain",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"event_id": "evt-missing"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is False
    assert payload["event_id"] == "evt-missing"
    assert payload["event"] is None


def test_memory_explain_endpoint_blank_event_id_short_circuit(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/explain",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"event_id": "   "},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["found"] is False
    assert payload["event_id"] is None
    assert stub.last_explain_args is None
