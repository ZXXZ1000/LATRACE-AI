"""Integration coverage for API security guards.

These tests exercise the FastAPI layer with production-like auth/signing
settings to ensure HTTP wiring, metrics, and request validation work
together (beyond the per-function unit specs).
"""
from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import sys
import time
from typing import Dict

from starlette.testclient import TestClient


def _reload_server(monkeypatch, env: Dict[str, str]):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    monkeypatch.setitem(sys.modules, "modules.memory.api.server", srv)
    return srv


def _stub_model(payload: Dict[str, object]):
    class _Model:
        def model_dump(self):
            return payload

    return _Model()


def test_search_requires_auth_when_enabled(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_CONFIG_PROFILE": "production",
            "MEMORY_API_TOKEN": "tenant-token",
            "MEMORY_API_TENANT_ID": "tenant-auth",
        },
    )

    async def _search_stub(*args, **kwargs):
        return _stub_model({"ok": True})

    monkeypatch.setattr(srv.svc, "search", _search_stub)
    client = TestClient(srv.app)

    no_auth = client.post("/search", json={"query": "hi"})
    authed = client.post("/search", headers={"X-API-Token": "tenant-token"}, json={"query": "hi"})

    assert no_auth.status_code == 401
    assert authed.status_code == 200
    assert authed.json()["ok"] is True


def test_write_requires_signature_and_counts_failures(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_CONFIG_PROFILE": "production",
            "MEMORY_API_TOKEN": "writer-token",
            "MEMORY_API_TENANT_ID": "tenant-sign",
            "MEMORY_API_SIGNING_SECRET": "sigsecret",
        },
    )

    async def _write_stub(*args, **kwargs):
        return _stub_model({"version": "v-test"})

    monkeypatch.setattr(srv.svc, "write", _write_stub)
    client = TestClient(srv.app)
    body = {"entries": []}
    ts = int(time.time())
    raw = json.dumps(body, separators=(",", ":")).encode()
    sig = hmac.new(b"sigsecret", f"{ts}./write".encode() + b"." + raw, hashlib.sha256).hexdigest()

    missing = client.post("/write", headers={"X-API-Token": "writer-token"}, json=body)
    signed = client.post(
        "/write",
        headers={
            "X-API-Token": "writer-token",
            "X-Signature": sig,
            "X-Signature-Ts": str(ts),
            "Content-Type": "application/json",
        },
        content=raw,
    )

    metrics = srv.get_metrics()
    assert metrics.get("signature_failures_total", 0) >= 1
    assert missing.status_code == 401
    assert signed.status_code == 200
    assert signed.json()["version"] == "v-test"
