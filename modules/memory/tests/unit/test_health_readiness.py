from __future__ import annotations

from typing import Any, Dict, Optional

import pytest


class _OkStore:
    async def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


class _Resp:
    def __init__(self, status_code: int, json_data: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = int(status_code)
        self._json_data = dict(json_data or {})

    def json(self) -> Dict[str, Any]:
        return dict(self._json_data)


@pytest.mark.anyio
async def test_health_check_includes_openrouter_and_disk_ok(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore

    # OpenRouter OK + credits OK (>= min threshold)
    import httpx

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc: BaseException, tb) -> None:  # type: ignore[override]
            return None

        async def get(self, url: str, headers: Optional[Dict[str, str]] = None):
            if url.endswith("/auth/key"):
                return _Resp(200, {"data": {"label": "test"}})
            if url.endswith("/credits"):
                # Match real OpenRouter shape (observed): total_credits + total_usage
                return _Resp(200, {"data": {"total_credits": 16.0, "total_usage": 12.5}})
            return _Resp(404, {})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    # Disk OK
    import shutil
    import os
    from collections import namedtuple

    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900 * 1024 * 1024))
    monkeypatch.setattr(os, "access", lambda p, mode: True)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-1234")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MEMORY_HEALTH_OPENROUTER_MIN_USD", "1.0")
    monkeypatch.setenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "0")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_MIN_FREE_MB", "500")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    assert h["status"] == "ok"
    assert h["dependencies"]["llm_provider"]["status"] == "ok"
    assert h["dependencies"]["llm_provider"]["auth"]["status"] == "ok"
    assert h["dependencies"]["llm_provider"]["balance"]["status"] == "ok"
    assert h["dependencies"]["llm_provider"]["balance"]["remaining_usd"] == pytest.approx(3.5)
    assert h["dependencies"]["disk"]["status"] == "ok"


@pytest.mark.anyio
async def test_health_check_fails_when_openrouter_key_missing(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "0")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    # Disk OK to isolate LLM failure
    import shutil
    import os
    from collections import namedtuple

    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900 * 1024 * 1024))
    monkeypatch.setattr(os, "access", lambda p, mode: True)

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    assert h["status"] == "fail"
    assert h["dependencies"]["llm_provider"]["status"] == "fail"
    assert h["dependencies"]["llm_provider"]["auth"]["error"] == "API_KEY_MISSING"


@pytest.mark.anyio
async def test_health_check_fails_when_balance_low(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore
    import httpx
    import shutil
    import os
    from collections import namedtuple

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url: str, **kwargs):
            if url.endswith("/auth/key"):
                return _Resp(200, {"data": {"label": "ok"}})
            if url.endswith("/credits"):
                # Total 10, Usage 9.5 => Remaining 0.5 < 1.0
                return _Resp(200, {"data": {"total_credits": 10.0, "total_usage": 9.5}})
            return _Resp(404)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900*1024*1024))
    monkeypatch.setattr(os, "access", lambda p, m: True)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_HEALTH_OPENROUTER_MIN_USD", "1.0")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    assert h["status"] == "fail"
    assert h["dependencies"]["llm_provider"]["balance"]["status"] == "fail"
    assert h["dependencies"]["llm_provider"]["balance"]["error"] == "BALANCE_BELOW_THRESHOLD"
    assert h["dependencies"]["llm_provider"]["balance"]["remaining_usd"] == pytest.approx(0.5)

@pytest.mark.anyio
async def test_health_check_handles_credits_api_failure(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore
    import httpx
    import shutil
    import os
    from collections import namedtuple

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url: str, **kwargs):
            if url.endswith("/auth/key"):
                return _Resp(200, {"data": {"label": "ok"}})
            if url.endswith("/credits"):
                return _Resp(500, {})  # API Error
            return _Resp(404)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900*1024*1024))
    monkeypatch.setattr(os, "access", lambda p, m: True)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    assert h["status"] == "fail"
    assert h["dependencies"]["llm_provider"]["balance"]["status"] == "fail"
    # Ensure undocumented error code matches what we added to docs
    assert h["dependencies"]["llm_provider"]["balance"]["error"] == "CREDITS_API_FAILED"

@pytest.mark.anyio
async def test_health_check_fails_when_disk_inaccessible(monkeypatch) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore
    import os

    # Emulate disk path not accessible
    monkeypatch.setattr(os, "access", lambda p, m: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True) # exists but no permission
    
    # LLM Mock to be OK
    import httpx
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url: str, **kwargs):
            if url.endswith("/auth/key"):
                return _Resp(200, {"data": {"label": "ok"}})
            if url.endswith("/credits"):
                return _Resp(200, {"data": {"total_credits": 10, "total_usage": 0}})
            return _Resp(404)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", "/root/protected")

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    assert h["status"] == "fail"
    assert h["dependencies"]["disk"]["status"] == "fail"
    assert h["dependencies"]["disk"]["error"] == "PATH_NOT_ACCESSIBLE"

@pytest.mark.anyio
async def test_health_check_timestamp_format(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore
    import httpx
    import shutil
    import os
    import collections
    
    # Mock everything to OK
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url: str, **kwargs):
             return _Resp(200, {"data": {"label": "ok", "total_credits": 100, "total_usage": 0}})
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    Usage = collections.namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900*1024*1024))
    monkeypatch.setattr(os, "access", lambda p, m: True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    ts = h["timestamp"]
    assert ts.endswith("Z")
    assert "T" in ts
    # e.g., 2026-01-07T14:48:00Z and len is 20
    assert len(ts) == 20

@pytest.mark.anyio
async def test_health_http_returns_503_when_unhealthy(monkeypatch, tmp_path) -> None:
    import importlib
    from fastapi.testclient import TestClient

    srv = importlib.import_module("modules.memory.api.server")

    # Override global service with deterministic inmem instance
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore

    import shutil
    import os
    from collections import namedtuple

    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900 * 1024 * 1024))
    monkeypatch.setattr(os, "access", lambda p, mode: True)
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # force unhealthy
    monkeypatch.setenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "0")

    srv._svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    c = TestClient(srv.app)
    r = c.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "fail"


@pytest.mark.anyio
async def test_health_check_credits_api_failed(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore

    import httpx

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc: BaseException, tb) -> None:  # type: ignore[override]
            return None

        async def get(self, url: str, headers: Optional[Dict[str, str]] = None):
            if url.endswith("/auth/key"):
                return _Resp(200, {"data": {"label": "test"}})
            if url.endswith("/credits"):
                return _Resp(500, {})
            return _Resp(404, {})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-1234")
    monkeypatch.setenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "0")
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(tmp_path))

    # Disk OK
    import shutil
    import os
    from collections import namedtuple

    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(1000, 100, 900 * 1024 * 1024))
    monkeypatch.setattr(os, "access", lambda p, mode: True)

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    llm = h["dependencies"]["llm_provider"]
    assert h["status"] == "fail"
    assert llm["status"] == "fail"
    assert llm["balance"]["error"] == "CREDITS_API_FAILED"


@pytest.mark.anyio
async def test_health_check_disk_path_not_accessible(monkeypatch, tmp_path) -> None:
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.audit_store import AuditStore

    missing_path = tmp_path / "missing"
    monkeypatch.setenv("MEMORY_HEALTH_DISK_PATH", str(missing_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "0")

    svc = MemoryService(_OkStore(), _OkStore(), AuditStore({"sqlite_path": ":memory:"}))
    h = await svc.health_check()
    disk = h["dependencies"]["disk"]
    assert h["status"] == "fail"
    assert disk["status"] == "fail"
    assert disk["error"] == "PATH_NOT_ACCESSIBLE"
