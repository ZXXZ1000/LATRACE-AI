from __future__ import annotations

import importlib
import sys
from typing import Dict

import pytest


@pytest.fixture(autouse=True)
def _restore_server_module():
    """Restore server module after each test to prevent pollution."""
    original_module = sys.modules.get("modules.memory.api.server")
    yield
    if original_module is not None:
        sys.modules["modules.memory.api.server"] = original_module
    else:
        sys.modules.pop("modules.memory.api.server", None)


def _reload_server(monkeypatch, env: Dict[str, str]):
    """Reload server module with custom env to ensure clean state."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    monkeypatch.setitem(sys.modules, "modules.memory.api.server", srv)
    return srv


def test_min_auth_write_requires_token(monkeypatch):
    # Reload server with auth enabled and rate limit disabled to avoid interference
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_TOKEN": "t-secret",
            "MEMORY_API_TENANT_ID": "test-tenant",
            "MEMORY_API_RATE_LIMIT_ENABLED": "false",  # disable rate limit
        },
    )
    from starlette.testclient import TestClient

    c = TestClient(srv.app)
    payload = {"entries": [], "links": [], "upsert": True}

    # without token -> 401
    r = c.post("/write", json=payload)
    assert r.status_code == 401

    # with wrong token -> 401
    r = c.post("/write", json=payload, headers={"X-API-Token": "wrong"})
    assert r.status_code == 401

    # with correct token -> 200 (even with empty payload)
    r = c.post("/write", json=payload, headers={"X-API-Token": "t-secret"})
    assert r.status_code == 200

