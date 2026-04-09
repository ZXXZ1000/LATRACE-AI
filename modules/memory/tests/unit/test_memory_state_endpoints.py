from __future__ import annotations

from datetime import datetime

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_current_args = None
        self.last_at_args = None
        self.last_changes_args = None
        self.last_what_changed_args = None
        self.last_time_since_args = None
        self.last_pending_list_args = None
        self.last_pending_approve_args = None
        self.last_pending_reject_args = None
        self.current_response = {"id": "state-1", "value": "employed"}
        self.at_response = {"id": "state-2", "value": "employed"}
        self.changes_response = [
            {"id": "state-1", "value": "employed", "valid_from": "2026-01-01T00:00:00Z"},
            {"id": "state-2", "value": "unemployed", "valid_from": "2026-01-02T00:00:00Z"},
        ]
        self.pending_list_response = [{"id": "pending-1"}]
        self.pending_approve_response = {"pending": {"id": "pending-1"}, "applied": False}
        self.pending_reject_response = {"pending": {"id": "pending-1"}}

    async def get_current_state(self, **kwargs):
        self.last_current_args = kwargs
        return dict(self.current_response)

    async def get_state_at_time(self, **kwargs):
        self.last_at_args = kwargs
        return dict(self.at_response)

    async def get_state_changes(self, **kwargs):
        self.last_changes_args = kwargs
        self.last_what_changed_args = kwargs
        self.last_time_since_args = kwargs
        return list(self.changes_response)

    async def list_pending_states(self, **kwargs):
        self.last_pending_list_args = kwargs
        return list(self.pending_list_response)

    async def approve_pending_state(self, **kwargs):
        self.last_pending_approve_args = kwargs
        return dict(self.pending_approve_response)

    async def reject_pending_state(self, **kwargs):
        self.last_pending_reject_args = kwargs
        return dict(self.pending_reject_response)


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


def test_memory_state_current_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/current",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"subject_id": "entity-1", "property": "job_status"},
    )
    assert resp.status_code == 200
    assert resp.json()["item"]["id"] == "state-1"
    assert stub.last_current_args == {
        "tenant_id": "tenant-x",
        "subject_id": "entity-1",
        "property": "job_status",
    }


def test_memory_state_at_time_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/at_time",
        headers={"X-Tenant-ID": "tenant-y"},
        json={"subject_id": "entity-2", "property": "mood", "t_iso": "2026-01-01T10:00:00Z"},
    )
    assert resp.status_code == 200
    assert resp.json()["item"]["id"] == "state-2"
    assert stub.last_at_args["tenant_id"] == "tenant-y"
    assert stub.last_at_args["subject_id"] == "entity-2"
    assert stub.last_at_args["property"] == "mood"
    assert isinstance(stub.last_at_args["t"], datetime)


def test_memory_state_changes_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/changes",
        headers={"X-Tenant-ID": "tenant-z"},
        json={
            "subject_id": "entity-3",
            "property": "relationship_status",
            "start_iso": "2026-01-01T00:00:00Z",
            "end_iso": "2026-01-31T00:00:00Z",
            "limit": 10,
            "order": "asc",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == [
        {"id": "state-1", "value": "employed", "valid_from": "2026-01-01T00:00:00Z"},
        {"id": "state-2", "value": "unemployed", "valid_from": "2026-01-02T00:00:00Z"},
    ]
    assert stub.last_changes_args["tenant_id"] == "tenant-z"
    assert stub.last_changes_args["subject_id"] == "entity-3"
    assert stub.last_changes_args["property"] == "relationship_status"
    assert isinstance(stub.last_changes_args["start"], datetime)
    assert isinstance(stub.last_changes_args["end"], datetime)
    assert stub.last_changes_args["limit"] == 10
    assert stub.last_changes_args["order"] == "asc"


def test_memory_state_what_changed_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/what-changed",
        headers={"X-Tenant-ID": "tenant-w"},
        json={"subject_id": "entity-7", "property": "job_status", "limit": 2},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == [
        {"id": "state-1", "value": "employed", "valid_from": "2026-01-01T00:00:00Z"},
        {"id": "state-2", "value": "unemployed", "valid_from": "2026-01-02T00:00:00Z"},
    ]
    assert stub.last_what_changed_args["tenant_id"] == "tenant-w"
    assert stub.last_what_changed_args["subject_id"] == "entity-7"
    assert stub.last_what_changed_args["property"] == "job_status"
    assert stub.last_what_changed_args["limit"] == 2


def test_memory_state_time_since_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/time-since",
        headers={"X-Tenant-ID": "tenant-w"},
        json={"subject_id": "entity-7", "property": "job_status"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["property"] == "job_status"
    assert body["last_changed_at"] is not None
    assert isinstance(body["seconds_ago"], int)


def test_memory_state_pending_list_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/pending/list",
        headers={"X-Tenant-ID": "tenant-p"},
        json={"subject_id": "entity-9", "property": "job_status", "status": "pending", "limit": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["items"] == [{"id": "pending-1"}]
    assert stub.last_pending_list_args["tenant_id"] == "tenant-p"
    assert stub.last_pending_list_args["subject_id"] == "entity-9"
    assert stub.last_pending_list_args["property"] == "job_status"
    assert stub.last_pending_list_args["status"] == "pending"
    assert stub.last_pending_list_args["limit"] == 5


def test_memory_state_pending_approve_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/pending/approve",
        headers={"X-Tenant-ID": "tenant-p"},
        json={"pending_id": "pending-1", "apply": False, "note": "ok"},
    )
    assert resp.status_code == 200
    assert resp.json()["pending"]["id"] == "pending-1"
    assert stub.last_pending_approve_args == {
        "tenant_id": "tenant-p",
        "pending_id": "pending-1",
        "apply": False,
        "note": "ok",
    }


def test_memory_state_pending_reject_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/state/pending/reject",
        headers={"X-Tenant-ID": "tenant-p"},
        json={"pending_id": "pending-1", "note": "bad"},
    )
    assert resp.status_code == 200
    assert resp.json()["pending"]["id"] == "pending-1"
    assert stub.last_pending_reject_args == {
        "tenant_id": "tenant-p",
        "pending_id": "pending-1",
        "note": "bad",
    }
