from __future__ import annotations

from typing import Any, Dict

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401
from fastapi.testclient import TestClient


def _auth_disabled_settings() -> Dict[str, Any]:
    return {
        "enabled": False,
        "header": "X-API-Token",
        "token": "",
        "tenant_id": "",
        "token_map": {},
        "signing": {"required": False},
    }


def test_api_list_returns_routes_and_scopes(monkeypatch) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    client = TestClient(srv.app)

    res = client.get("/api/list", headers={"X-Tenant-ID": "t1"})
    assert res.status_code == 200
    data = res.json()
    assert data.get("version") == "1.0"
    categories = data.get("categories") or []
    assert isinstance(categories, list) and categories

    endpoints = []
    for c in categories:
        endpoints.extend(list(c.get("endpoints") or []))
    paths = {e.get("path") for e in endpoints}
    assert "/search" in paths

    search_ep = [e for e in endpoints if e.get("path") == "/search"][0]
    assert "POST" in (search_ep.get("methods") or [])
    assert search_ep.get("auth_scope") in ("memory.read", None)


def test_api_list_hides_admin_by_default(monkeypatch) -> None:
    from modules.memory.api import server as srv

    monkeypatch.setattr(srv, "_auth_settings", _auth_disabled_settings)
    client = TestClient(srv.app)

    res = client.get("/api/list", headers={"X-Tenant-ID": "t1"})
    assert res.status_code == 200
    data = res.json()
    endpoints = []
    for c in data.get("categories") or []:
        endpoints.extend(list(c.get("endpoints") or []))
    paths = {e.get("path") for e in endpoints}
    assert "/write" not in paths

    res2 = client.get("/api/list?include_internal=1", headers={"X-Tenant-ID": "t1"})
    assert res2.status_code == 200
    data2 = res2.json()
    endpoints2 = []
    for c in data2.get("categories") or []:
        endpoints2.extend(list(c.get("endpoints") or []))
    paths2 = {e.get("path") for e in endpoints2}
    assert "/write" in paths2

