from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Dict

from fastapi.testclient import TestClient

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


def _reload_server(monkeypatch, env: Dict[str, str] | None = None):
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    monkeypatch.setitem(sys.modules, "modules.memory.api.server", srv)
    return srv


def _entry(entry_id: str, tenant_id: str, text: str) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        kind="episodic",
        modality="text",
        contents=[text],
        metadata={"tenant_id": tenant_id},
    )


def _seed_job(store: AsyncIngestJobStore, *, tenant_id: str, session_id: str, commit_id: str) -> None:
    asyncio.run(
        store.create_job(
            session_id=session_id,
            commit_id=commit_id,
            tenant_id=tenant_id,
            api_key_id=None,
            request_id=None,
            user_tokens=[f"u:{tenant_id}"],
            memory_domain="dialog",
            llm_policy="require",
            turns=[{"turn_id": "1", "text": "hello"}],
            base_turn_id=None,
            client_meta={"sdk": "test"},
            payload_raw=None,
        )
    )


def _install_test_service(monkeypatch, srv, tmp_path):
    service = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    ingest_store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    monkeypatch.setattr(srv, "_svc", service, raising=False)
    monkeypatch.setattr(srv, "_graph_svc", None, raising=False)
    monkeypatch.setattr(srv, "ingest_store", ingest_store, raising=False)
    srv._tenant_clear_locks.clear()
    return service, ingest_store


class _LegacyVectorStub:
    async def count_by_filter(self, *, tenant_id: str, api_key_id=None) -> int:
        return 2

    async def list_ids_by_filter(self, *, tenant_id: str, api_key_id=None):
        return ["point-uuid-1", "point-uuid-2"]

    async def list_entry_ids_by_filter(self, *, tenant_id: str, api_key_id=None):
        return ["legacy-1", "legacy-2"]

    async def delete_by_filter(self, *, tenant_id: str, api_key_id=None) -> int:
        return 2


class _LegacyGraphStub:
    async def count_tenant_nodes(self, tenant_id: str) -> int:
        return 0

    async def count_legacy_memory_nodes_by_ids(self, ids):
        return len(list(ids))

    async def purge_tenant(self, tenant_id: str) -> int:
        return 0

    async def purge_legacy_memory_nodes_by_ids(self, ids):
        return len(list(ids))


def test_memory_clear_scope_mapping_prefers_exact_rule(monkeypatch):
    srv = _reload_server(monkeypatch)

    assert srv._required_scope_for_path("/memory/v1/clear") == "memory.clear"
    assert srv._lookup_scope_mapping("/memory/v1/clear") == ("memory.clear", True)


def test_memory_clear_dry_run_confirm_and_idempotent(monkeypatch, tmp_path):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "false",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )
    service, ingest_store = _install_test_service(monkeypatch, srv, tmp_path)

    tenant_a_entries = [_entry("vec-a1", "tenant-a", "alpha"), _entry("vec-a2", "tenant-a", "beta")]
    tenant_b_entries = [_entry("vec-b1", "tenant-b", "gamma")]
    asyncio.run(service.vectors.upsert_vectors(tenant_a_entries + tenant_b_entries))
    asyncio.run(service.graph.merge_nodes_edges(tenant_a_entries + tenant_b_entries, []))
    _seed_job(ingest_store, tenant_id="tenant-a", session_id="sess-a", commit_id="commit-a")
    _seed_job(ingest_store, tenant_id="tenant-b", session_id="sess-b", commit_id="commit-b")

    client = TestClient(srv.app)

    dry_run = client.post(
        "/memory/v1/clear",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"reason": "preview"},
    )
    assert dry_run.status_code == 200
    body = dry_run.json()
    assert body["dry_run"] is True
    assert body["status"] == "completed"
    assert body["estimated_vectors"] == 2
    assert body["estimated_graph_nodes"] == 2
    assert body["estimated_ingest_jobs"] == 1
    assert body["cleared_vectors"] == 0
    assert body["cleared_graph_nodes"] == 0
    assert body["cleared_ingest_jobs"] == 0
    assert asyncio.run(service.vectors.count_by_filter(tenant_id="tenant-a")) == 2
    assert asyncio.run(service.graph.count_tenant_nodes("tenant-a")) == 2
    assert asyncio.run(ingest_store.count_by_tenant("tenant-a")) == 1

    cleared = client.post(
        "/memory/v1/clear",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"reason": "wipe", "confirm": True},
    )
    assert cleared.status_code == 200
    cleared_body = cleared.json()
    assert cleared_body["dry_run"] is False
    assert cleared_body["status"] == "completed"
    assert cleared_body["cleared_vectors"] == 2
    assert cleared_body["cleared_graph_nodes"] == 2
    assert cleared_body["cleared_ingest_jobs"] == 1
    assert asyncio.run(service.vectors.count_by_filter(tenant_id="tenant-a")) == 0
    assert asyncio.run(service.graph.count_tenant_nodes("tenant-a")) == 0
    assert asyncio.run(ingest_store.count_by_tenant("tenant-a")) == 0
    assert asyncio.run(service.vectors.count_by_filter(tenant_id="tenant-b")) == 1
    assert asyncio.run(service.graph.count_tenant_nodes("tenant-b")) == 1
    assert asyncio.run(ingest_store.count_by_tenant("tenant-b")) == 1

    repeat = client.post(
        "/memory/v1/clear",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"reason": "repeat", "confirm": True},
    )
    assert repeat.status_code == 200
    repeat_body = repeat.json()
    assert repeat_body["cleared_vectors"] == 0
    assert repeat_body["cleared_graph_nodes"] == 0
    assert repeat_body["cleared_ingest_jobs"] == 0


def test_memory_clear_requires_explicit_scope_when_auth_enabled(monkeypatch, tmp_path):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "true",
            "MEMORY_API_JWKS_URL": "http://jwks.local",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )
    _install_test_service(monkeypatch, srv, tmp_path)

    client = TestClient(srv.app)

    monkeypatch.setattr(
        srv,
        "_decode_jwt_token",
        lambda tok, settings: {"sub": "key-1", "tenant_id": "tenant-a", "scopes": ["memory.read"]},
    )
    forbidden = client.post(
        "/memory/v1/clear",
        headers={"X-API-Token": "jwt-token"},
        json={"reason": "wipe", "confirm": True},
    )
    assert forbidden.status_code == 403

    monkeypatch.setattr(
        srv,
        "_decode_jwt_token",
        lambda tok, settings: {"sub": "key-1", "tenant_id": "tenant-a", "scopes": ["memory.clear"]},
    )
    allowed = client.post(
        "/memory/v1/clear",
        headers={"X-API-Token": "jwt-token"},
        json={"reason": "wipe", "confirm": True},
    )
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "completed"


def test_memory_clear_includes_legacy_graph_nodes_via_vector_ids(monkeypatch, tmp_path):
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_API_AUTH_ENABLED": "false",
            "MEMORY_API_SIGNING_REQUIRED": "false",
        },
    )
    service = MemoryService(_LegacyVectorStub(), _LegacyGraphStub(), AuditStore())
    ingest_store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "legacy_ingest_jobs.db")})
    monkeypatch.setattr(srv, "_svc", service, raising=False)
    monkeypatch.setattr(srv, "ingest_store", ingest_store, raising=False)
    srv._tenant_clear_locks.clear()
    client = TestClient(srv.app)

    dry_run = client.post(
        "/memory/v1/clear",
        headers={"X-Tenant-ID": "tenant-legacy"},
        json={"reason": "legacy preview"},
    )
    assert dry_run.status_code == 200
    dry_body = dry_run.json()
    assert dry_body["estimated_vectors"] == 2
    assert dry_body["estimated_graph_nodes"] == 2

    cleared = client.post(
        "/memory/v1/clear",
        headers={"X-Tenant-ID": "tenant-legacy"},
        json={"reason": "legacy clear", "confirm": True},
    )
    assert cleared.status_code == 200
    clear_body = cleared.json()
    assert clear_body["cleared_vectors"] == 2
    assert clear_body["cleared_graph_nodes"] == 2
