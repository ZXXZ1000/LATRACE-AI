import asyncio
import base64
import hashlib
import hmac
import http.server
import importlib
import json
import sys
import threading
import time
from typing import Dict

import jwt
import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def _restore_server_module():
    """Restore server module after each test to prevent pollution."""
    original_module = sys.modules.get("modules.memory.api.server")
    yield
    if original_module is not None:
        sys.modules["modules.memory.api.server"] = original_module
    else:
        sys.modules.pop("modules.memory.api.server", None)


class _JWKSHandler(http.server.BaseHTTPRequestHandler):
    jwks: Dict[str, object] = {}

    def do_GET(self) -> None:  # pragma: no cover - exercised via client
        body = json.dumps(self.jwks).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # pragma: no cover
        return



def _start_jwks_server(jwks: Dict[str, object]) -> http.server.HTTPServer:
    _JWKSHandler.jwks = jwks
    server = http.server.HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server



def _reload_server(monkeypatch, env: Dict[str, str]):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    monkeypatch.setitem(sys.modules, "modules.memory.api.server", srv)
    return srv


async def _stub_model(payload: Dict[str, object]):
    class _Model:
        def model_dump(self):
            return payload

    return _Model()



def test_jwt_auth_allows_search(monkeypatch):
    secret = "supersecret"
    {
        "keys": [
            {
                "kty": "oct",
                "k": base64.urlsafe_b64encode(secret.encode()).rstrip(b"=").decode(),
                "alg": "HS256",
                "kid": "kid1",
            }
        ]
    }
    # Avoid binding local HTTP server in sandbox; JWKS fetch is bypassed via monkeypatch.
    token = jwt.encode(
        {"sub": "user1", "aud": "aud1", "iss": "issuer1", "tenant_id": "tenantA"},
        secret,
        algorithm="HS256",
        headers={"kid": "kid1"},
    )
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_JWKS_URL": "http://jwks.local/jwks.json",
            "MEMORY_API_JWT_AUD": "aud1",
            "MEMORY_API_JWT_ISS": "issuer1",
            "MEMORY_API_SIGNING_REQUIRED": "false",
            "MEMORY_CONFIG_PROFILE": "production",
        },
    )

    monkeypatch.setattr(srv, "_decode_jwt_token", lambda tok, settings: {"sub": "user1", "tenant_id": "tenantA"})

    async def _search_stub(*args, **kwargs):
        return await _stub_model({"ok": True, "args": args})

    monkeypatch.setattr(srv.svc, "search", _search_stub)
    client = TestClient(srv.app)
    res = client.post("/search", headers={"X-API-Token": token}, json={"query": "hi", "topk": 1})

    assert res.status_code == 200
    assert res.json()["ok"] is True



def test_signature_required_for_write(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_TOKEN": "static-token",
            "MEMORY_API_TENANT_ID": "tenant-sign",
            "MEMORY_API_SIGNING_REQUIRED": "true",
            "MEMORY_API_SIGNING_SECRET": "sigsecret",
        },
    )

    async def _write_stub(*args, **kwargs):
        return await _stub_model({"version": "v-test"})

    monkeypatch.setattr(srv.svc, "write", _write_stub)
    client = TestClient(srv.app)
    body = {"entries": []}
    ts = int(time.time())
    raw = json.dumps(body, separators=(",", ":"))
    raw_bytes = raw.encode()
    payload = f"{ts}./write".encode() + b"." + raw_bytes
    sig = hmac.new(b"sigsecret", payload, hashlib.sha256).hexdigest()

    ok = client.post(
        "/write",
        headers={
            "X-API-Token": "static-token",
            "X-Signature": sig,
            "X-Signature-Ts": str(ts),
            "Content-Type": "application/json",
        },
        content=raw_bytes,
    )
    missing = client.post("/write", headers={"X-API-Token": "static-token"}, json=body)

    assert ok.status_code == 200, ok.text
    assert missing.status_code == 401



def test_circuit_breaker_short_circuits_after_timeout(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_HIGH_COST_TIMEOUT_SECONDS": "0.01",
        },
    )

    async def _slow_search(*args, **kwargs):
        await asyncio.sleep(0.05)
        return await _stub_model({"ok": False})

    monkeypatch.setattr(srv.svc, "search", _slow_search)
    srv.HIGH_COST_TIMEOUT = 0.01
    srv.HIGH_COST_BREAKERS["search"] = srv.SimpleCircuitBreaker(1, 30)
    client = TestClient(srv.app)

    first = client.post("/search", headers={"X-Tenant-ID": "tenantX"}, json={"query": "slow"})
    second = client.post("/search", headers={"X-Tenant-ID": "tenantX"}, json={"query": "slow"})

    assert first.status_code == 504
    assert second.status_code == 503


def test_scope_enforced_for_jwt(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_JWKS_URL": "http://jwks.local",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )

    monkeypatch.setattr(
        srv,
        "_decode_jwt_token",
        lambda tok, settings: {"sub": "user1", "tenant_id": "tenantA", "scopes": ["memory.read"]},
    )

    async def _search_stub(*args, **kwargs):
        return await _stub_model({"ok": True})

    async def _write_stub(*args, **kwargs):
        return await _stub_model({"version": "v-scope"})

    monkeypatch.setattr(srv.svc, "search", _search_stub)
    monkeypatch.setattr(srv.svc, "write", _write_stub)

    client = TestClient(srv.app)
    token = "jwt-token"

    res_read = client.post("/search", headers={"X-API-Token": token}, json={"query": "hi", "topk": 1})
    res_write = client.post("/write", headers={"X-API-Token": token}, json={"entries": [], "links": [], "upsert": True})

    assert res_read.status_code == 200
    assert res_write.status_code == 403


def test_jwt_without_scopes_allows_legacy_access(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_JWKS_URL": "http://jwks.local",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )

    monkeypatch.setattr(
        srv,
        "_decode_jwt_token",
        lambda tok, settings: {"sub": "user1", "tenant_id": "tenantA"},
    )

    async def _write_stub(*args, **kwargs):
        return await _stub_model({"version": "v-legacy"})

    monkeypatch.setattr(srv.svc, "write", _write_stub)

    client = TestClient(srv.app)
    res = client.post("/write", headers={"X-API-Token": "jwt-token"}, json={"entries": [], "links": [], "upsert": True})

    assert res.status_code == 200


def test_authorization_header_fallback(monkeypatch):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_TOKEN": "static-token",
            "MEMORY_API_TENANT_ID": "tenant-sign",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )

    async def _search_stub(*args, **kwargs):
        return await _stub_model({"ok": True})

    monkeypatch.setattr(srv.svc, "search", _search_stub)
    client = TestClient(srv.app)

    res = client.post("/search", headers={"Authorization": "Bearer static-token"}, json={"query": "hi", "topk": 1})

    assert res.status_code == 200
    assert res.headers.get("x-request-id")
