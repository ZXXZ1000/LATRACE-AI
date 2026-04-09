from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


@pytest.mark.asyncio
async def test_usage_wal_flush_sends_internal_key_header(monkeypatch) -> None:
    from modules.memory.infra import usage_wal as wal_mod

    sent_requests: List[Dict[str, Any]] = []

    class _Resp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc: BaseException, tb) -> None:  # type: ignore[override]
            return None

        async def post(self, url: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
            sent_requests.append({"url": url, "json": dict(json or {}), "headers": dict(headers or {})})
            return _Resp(200)

    monkeypatch.setattr(wal_mod.httpx, "AsyncClient", _FakeAsyncClient)

    settings = wal_mod.UsageWALSettings(
        enabled=True,
        sqlite_path=":memory:",
        sink_url="http://control-plane/internal/usage/events",
        sink_auth_internal_key="secret",
        sink_auth_internal_header="X-Internal-Key",
        sink_auth_authorization=None,
        flush_interval_s=0.1,
        batch_size=10,
        timeout_s=1.0,
    )
    wal = wal_mod.UsageWAL(settings)
    wal.append({"event_id": "evt_1", "event_type": "llm", "tenant_id": "t1", "metrics": {"prompt_tokens": 1}})

    sent = await wal.flush_once()
    assert sent == 1
    assert len(sent_requests) == 1
    assert sent_requests[0]["headers"]["X-Internal-Key"] == "secret"


@pytest.mark.asyncio
async def test_usage_wal_flush_sends_authorization_header(monkeypatch) -> None:
    from modules.memory.infra import usage_wal as wal_mod

    sent_requests: List[Dict[str, Any]] = []

    class _Resp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc: BaseException, tb) -> None:  # type: ignore[override]
            return None

        async def post(self, url: str, json: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
            sent_requests.append({"url": url, "json": dict(json or {}), "headers": dict(headers or {})})
            return _Resp(200)

    monkeypatch.setattr(wal_mod.httpx, "AsyncClient", _FakeAsyncClient)

    settings = wal_mod.UsageWALSettings(
        enabled=True,
        sqlite_path=":memory:",
        sink_url="http://control-plane/internal/usage/events",
        sink_auth_internal_key=None,
        sink_auth_internal_header="X-Internal-Key",
        sink_auth_authorization="Bearer internal-token",
        flush_interval_s=0.1,
        batch_size=10,
        timeout_s=1.0,
    )
    wal = wal_mod.UsageWAL(settings)
    wal.append({"event_id": "evt_2", "event_type": "llm", "tenant_id": "t1", "metrics": {"prompt_tokens": 1}})

    sent = await wal.flush_once()
    assert sent == 1
    assert len(sent_requests) == 1
    assert sent_requests[0]["headers"]["Authorization"] == "Bearer internal-token"

