from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient
import pytest


@pytest.fixture(autouse=True, scope="function")
def _restore_server_module():
    """Restore server module after each test to prevent pollution."""
    original_module = sys.modules.get("modules.memory.api.server")
    yield
    if original_module is not None:
        sys.modules["modules.memory.api.server"] = original_module
    else:
        sys.modules.pop("modules.memory.api.server", None)


@pytest.fixture(autouse=True)
def _patch_store(monkeypatch):
    class _StubStore:
        async def cleanup_expired(self, *, tenant_id: str, buffer_hours: float, limit: int, dry_run: bool = False):
            return {"nodes": 2, "edges": 1, "dry_run": dry_run}

        async def export_srot(self, *, tenant_id: str, rel_types=None, min_confidence=None, limit=1000, cursor=None):
            items = [{"subject": "a", "relation": "REL", "object": "b", "time_origin": "media", "t_ref": "now"}]
            return {"items": items, "next_cursor": None}

    # Force reimport the server module to get fresh state
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")

    monkeypatch.setattr(srv, "graph_svc", type("S", (), {"store": _StubStore()})())
    # Disable auth to avoid interference from other tests
    monkeypatch.setattr(
        srv,
        "_auth_settings",
        lambda: {
            "enabled": False,
            "header": "X-API-Token",
            "token": "",
            "tenant_id": "",
            "token_map": {},
            "signing": {"required": False},
        },
    )
    return None


def _client():
    from modules.memory.api.server import app
    return TestClient(app, headers={"X-Tenant-ID": "t"})


def test_ttl_cleanup_api():
    c = _client()
    resp = c.post("/graph/v0/admin/ttl/cleanup", json={"buffer_hours": 1, "limit": 10, "dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["deleted"]["nodes"] == 2
    assert resp.json()["deleted"]["dry_run"] is True


def test_export_srot_api():
    c = _client()
    resp = c.get("/graph/v0/admin/export_srot?rel=REL&min_confidence=0.5&limit=10")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["relation"] == "REL"
