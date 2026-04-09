from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


class _GraphStub:
    def __init__(self) -> None:
        self.last_topic_args = None
        self.topic_response = [
            {
                "id": "evt-1",
                "summary": "计划日本旅行",
                "t_abs_start": "2026-01-20T00:00:00Z",
                "event_confidence": 0.86,
                "evidence_count": 2,
            }
        ]

    async def topic_timeline(self, **kwargs):
        self.last_topic_args = kwargs
        return list(self.topic_response)


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


def test_memory_topic_timeline_endpoint(monkeypatch):
    stub, client = _setup(monkeypatch)
    resp = client.post(
        "/memory/v1/topic-timeline",
        headers={"X-Tenant-ID": "tenant-x"},
        json={"topic_path": "travel/japan", "user_tokens": ["u:alice"], "limit": 10},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["topic_path"] == "travel/japan"
    assert payload["timeline"][0]["event_id"] == "evt-1"
    assert payload["timeline"][0]["evidence_count"] == 2
    assert stub.last_topic_args == {
        "tenant_id": "tenant-x",
        "topic_id": None,
        "topic_path": "travel/japan",
        "tags": None,
        "keywords": None,
        "start": None,
        "end": None,
        "user_ids": ["u:alice"],
        "memory_domain": None,
        "limit": 10,
        "event_ids": None,
    }
