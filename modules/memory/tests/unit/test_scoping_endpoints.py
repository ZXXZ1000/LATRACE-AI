from __future__ import annotations

from fastapi.testclient import TestClient


def test_scoping_endpoints_unique_and_working(monkeypatch):
    # Patch server.svc to avoid external deps and startup network calls
    from modules.memory.api import server as srv
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.inmem_vector_store import InMemVectorStore
    from modules.memory.infra.inmem_graph_store import InMemGraphStore
    from modules.memory.infra.audit_store import AuditStore

    srv.svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())

    # X-Tenant-ID header is required when auth is disabled (fallback to header-based tenant)
    headers = {"X-Tenant-ID": "test-tenant"}
    client = TestClient(srv.app, headers=headers)

    # GET returns dict (may be empty)
    r = client.get("/config/search/scoping")
    assert r.status_code == 200 and isinstance(r.json(), dict)

    # POST sets override and persists
    body = {"default_scope": "domain", "user_match_mode": "any", "fallback_order": ["session", "domain", "user"], "require_user": False}
    r2 = client.post("/config/search/scoping", json=body)
    assert r2.status_code == 200 and r2.json().get("ok") is True

    r3 = client.get("/config/search/scoping")
    data = r3.json()
    assert data.get("default_scope") == "domain"
    assert data.get("user_match_mode") == "any"

