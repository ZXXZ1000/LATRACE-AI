from __future__ import annotations

import importlib
import sys
from typing import Dict

import pytest


@pytest.fixture(autouse=True)
def _restore_server_module():
    original_module = sys.modules.get("modules.memory.api.server")
    yield
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


@pytest.mark.anyio
async def test_internal_retry_disabled_does_not_schedule_retry(monkeypatch, tmp_path) -> None:
    srv = _reload_server(
        monkeypatch,
        {
            "MEMORY_INGEST_JOB_DB_PATH": str(tmp_path / "ingest_jobs.db"),
            "MEMORY_INGEST_INTERNAL_RETRY_ENABLED": "false",
            "MEMORY_INGEST_MAX_RETRY_ATTEMPTS": "3",
            "MEMORY_STAGE2_ENABLED": "true",
        },
    )

    # Force stage2 extractor missing
    monkeypatch.setattr(srv, "build_turn_mark_extractor_v1_from_env", lambda *args, **kwargs: None)

    record, created = await srv.ingest_store.create_job(
        session_id="s1",
        commit_id=None,
        tenant_id="t1",
        api_key_id=None,
        request_id="r1",
        turns=[{"turn_id": "t1", "role": "user", "content": "hi"}],
        user_tokens=["u:t1"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created is True
    assert record.status == "RECEIVED"

    await srv._run_ingest_job(
        job_id=record.job_id,
        tenant_id="t1",
        user_tokens=["u:t1"],
        memory_domain="dialog",
        llm_policy="require",
    )

    updated = await srv.ingest_store.get_job(record.job_id)
    assert updated is not None
    assert updated.status == "STAGE2_FAILED"
    assert updated.next_retry_at in (None, "")
    # Service no longer schedules internal retry tasks; worker drives retries.
