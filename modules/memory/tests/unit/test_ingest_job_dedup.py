from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

fastapi = pytest.importorskip("fastapi")  # noqa: F401

from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore


def _create_job(store: AsyncIngestJobStore, turns: List[Dict[str, Any]]):
    return asyncio.run(
        store.create_job(
            session_id="sess-1",
            commit_id=None,
            tenant_id="t1",
            api_key_id=None,
            request_id="r1",
            turns=turns,
            user_tokens=["u:1"],
            base_turn_id=None,
            client_meta={},
            memory_domain="dialog",
            llm_policy="require",
        )
    )


def test_run_ingest_job_noops_when_already_running(monkeypatch, tmp_path) -> None:
    from modules.memory.api import server as srv

    db_path = tmp_path / "ingest_jobs.db"
    monkeypatch.setattr(srv, "ingest_store", AsyncIngestJobStore({"sqlite_path": str(db_path)}))

    record, _created = _create_job(srv.ingest_store, [{"turn_id": "t1", "role": "user", "text": "hi"}])
    asyncio.run(srv.ingest_store.update_status(record.job_id, status="STAGE2_RUNNING", stage="stage2", attempt_inc=True))
    before = asyncio.run(srv.ingest_store.get_job(record.job_id))
    assert before is not None

    asyncio.run(
        srv._run_ingest_job(
            job_id=record.job_id,
            tenant_id="t1",
            user_tokens=["u:1"],
            memory_domain="dialog",
            llm_policy="require",
            payload=None,
            claim=False,
        )
    )

    after = asyncio.run(srv.ingest_store.get_job(record.job_id))
    assert after is not None
    assert after.status == "STAGE2_RUNNING"
    assert dict(after.attempts or {}) == dict(before.attempts or {})

