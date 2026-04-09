import importlib
import sys
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from modules.memory.contracts.graph_models import PendingEquiv
from modules.memory.infra.equiv_store import EquivStore


@pytest.fixture(autouse=True, scope="function")
def _restore_server_module():
    """Restore server module after each test to prevent pollution."""
    original_module = sys.modules.get("modules.memory.api.server")
    yield
    if original_module is not None:
        sys.modules["modules.memory.api.server"] = original_module
    else:
        sys.modules.pop("modules.memory.api.server", None)


class _StubEquivStore(EquivStore):
    def __init__(self):
        self.pending_records: List[Dict[str, Any]] = []
        self.approved: List[str] = []
        self.rejected: List[str] = []

    def list_pending(self, *, tenant_id: str, status: str = "pending", limit: int = 200):
        return [p for p in self.pending_records if p.get("tenant_id") == tenant_id and p.get("status") == status][:limit]

    def upsert_pending(self, *, tenant_id: str, records: List[PendingEquiv]):
        for r in records:
            self.pending_records.append(r.model_dump())

    def approve(self, *, tenant_id: str, pending_id: str, reviewer: str | None = None):
        self.approved.append(pending_id)
        return {"merged": 1}

    def reject(self, *, tenant_id: str, pending_id: str, reviewer: str | None = None):
        self.rejected.append(pending_id)
        return {"updated": 1}


@pytest.fixture(autouse=True)
def _patch_equiv_store(monkeypatch):
    stub = _StubEquivStore()
    # Force reimport the server module to get fresh state
    sys.modules.pop("modules.memory.api.server", None)
    srv = importlib.import_module("modules.memory.api.server")
    
    monkeypatch.setattr(srv, "equiv_store", stub)
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
    return stub


def _client(headers: Dict[str, str] | None = None) -> TestClient:
    from modules.memory.api.server import app
    return TestClient(app, headers=headers or {"X-Tenant-ID": "t"})


def test_equiv_api_pending_and_approve_flow():
    c = _client()
    payload = {"pending_equivs": [{"id": "peq1", "entity_id": "a", "candidate_id": "b", "confidence": 0.8}]}
    resp = c.post("/graph/v0/admin/equiv/pending", json=payload)
    assert resp.status_code == 200
    assert resp.json()["count"] == 1

    # list
    resp = c.get("/graph/v0/admin/equiv/pending")
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1

    # approve
    resp = c.post("/graph/v0/admin/equiv/approve", json={"pending_id": "peq1", "reviewer": "r1"})
    assert resp.status_code == 200
    assert resp.json()["merged"] == 1

    # reject another (even if not present, stub returns updated=1 to cover path)
    resp = c.post("/graph/v0/admin/equiv/reject", json={"pending_id": "peq2", "reviewer": "r2"})
    assert resp.status_code == 200
    assert resp.json()["updated"] == 1
