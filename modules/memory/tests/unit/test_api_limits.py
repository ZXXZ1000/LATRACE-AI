from __future__ import annotations

import importlib
import sys
from typing import Dict

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _restore_server_module():
    """Restore server module after each test to prevent pollution."""
    # Capture original module if it exists
    original_module = sys.modules.get("modules.memory.api.server")
    yield
    # Restore original module after test
    if original_module is not None:
        sys.modules["modules.memory.api.server"] = original_module
    else:
        sys.modules.pop("modules.memory.api.server", None)


def _reload_server(monkeypatch, env: Dict[str, str]):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    monkeypatch.setitem(sys.modules, "modules.memory.api.server", srv)
    return srv


def test_request_rejected_when_content_length_exceeds_limit(monkeypatch):
    srv = _reload_server(monkeypatch, {"MEMORY_API_MAX_REQUEST_BYTES": "100"})
    client = TestClient(srv.app)

    oversized_payload = {"entries": ["x" * 256]}
    res = client.post("/write", json=oversized_payload)

    assert res.status_code == 413
    assert "too large" in res.text


def test_rate_limit_blocks_mutations_immediately(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_RATE_LIMIT_ENABLED": "true",
            "MEMORY_API_RATE_LIMIT_PER_MINUTE": "0",
            "MEMORY_API_RATE_LIMIT_BURST": "0",
        },
    )
    client = TestClient(srv.app)

    res = client.post("/write", json={"entries": []})

    assert res.status_code == 429
    assert "rate limit" in res.text
