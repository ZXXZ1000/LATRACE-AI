from __future__ import annotations

import asyncio

import pytest

from modules.memory.application.ingest_executor import IngestExecutor, IngestExecutorConfig
from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore


@pytest.mark.asyncio
async def test_ingest_executor_runs_job(tmp_path) -> None:
    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    record, _ = await store.create_job(
        session_id="s1",
        commit_id=None,
        tenant_id="t1",
        api_key_id=None,
        request_id=None,
        user_tokens=["u:1"],
        memory_domain="dialog",
        llm_policy="require",
        turns=[{"role": "user", "content": "hi"}],
        base_turn_id=None,
        client_meta={"memory_policy": "user", "user_id": "u:1"},
        payload_raw=None,
    )
    ran = asyncio.Event()

    async def _run_job(rec):  # type: ignore[no-untyped-def]
        await store.update_status(rec.job_id, status="COMPLETED", stage="stage3")
        ran.set()

    cfg = IngestExecutorConfig(
        worker_count=1,
        global_concurrency=1,
        per_tenant_concurrency=1,
        job_timeout_s=5,
        recover_stale_s=0,
    )
    executor = IngestExecutor(store=store, run_job=_run_job, config=cfg)
    await executor.start()
    assert await executor.enqueue(record.job_id, tenant_id=record.tenant_id)
    await asyncio.wait_for(ran.wait(), timeout=2)
    await executor.stop()

    updated = await store.get_job(record.job_id)
    assert updated is not None
    assert updated.status == "COMPLETED"


@pytest.mark.asyncio
async def test_ingest_executor_timeout_requeues(tmp_path) -> None:
    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    record, _ = await store.create_job(
        session_id="s1",
        commit_id=None,
        tenant_id="t1",
        api_key_id=None,
        request_id=None,
        user_tokens=["u:1"],
        memory_domain="dialog",
        llm_policy="require",
        turns=[{"role": "user", "content": "hi"}],
        base_turn_id=None,
        client_meta={"memory_policy": "user", "user_id": "u:1"},
        payload_raw=None,
    )

    async def _run_job(_rec):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1.2)

    cfg = IngestExecutorConfig(
        worker_count=1,
        global_concurrency=1,
        per_tenant_concurrency=1,
        job_timeout_s=1,
        retry_delay_s=5,
        recover_stale_s=0,
        max_retries=3,
    )
    executor = IngestExecutor(store=store, run_job=_run_job, config=cfg)
    await executor.start()
    assert await executor.enqueue(record.job_id, tenant_id=record.tenant_id)

    deadline = asyncio.get_event_loop().time() + 3.0
    updated = None
    while asyncio.get_event_loop().time() < deadline:
        updated = await store.get_job(record.job_id)
        if updated and updated.status == "RECEIVED" and (updated.last_error or {}).get("code") == "timeout":
            break
        await asyncio.sleep(0.05)

    await executor.stop()

    assert updated is not None
    assert updated.status == "RECEIVED"
    assert (updated.last_error or {}).get("code") == "timeout"
    assert updated.next_retry_at is not None


@pytest.mark.asyncio
async def test_ingest_executor_timeout_marks_failed_when_retries_exhausted(tmp_path) -> None:
    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    record, _ = await store.create_job(
        session_id="s1",
        commit_id=None,
        tenant_id="t1",
        api_key_id=None,
        request_id=None,
        user_tokens=["u:1"],
        memory_domain="dialog",
        llm_policy="require",
        turns=[{"role": "user", "content": "hi"}],
        base_turn_id=None,
        client_meta={"memory_policy": "user", "user_id": "u:1"},
        payload_raw=None,
    )

    async def _run_job(_rec):  # type: ignore[no-untyped-def]
        await asyncio.sleep(1.2)

    cfg = IngestExecutorConfig(
        worker_count=1,
        global_concurrency=1,
        per_tenant_concurrency=1,
        job_timeout_s=1,
        retry_delay_s=1,
        recover_stale_s=0,
        max_retries=0,
    )
    executor = IngestExecutor(store=store, run_job=_run_job, config=cfg)
    await executor.start()
    assert await executor.enqueue(record.job_id, tenant_id=record.tenant_id)

    deadline = asyncio.get_event_loop().time() + 3.0
    updated = None
    while asyncio.get_event_loop().time() < deadline:
        updated = await store.get_job(record.job_id)
        if updated and updated.status == "STAGE3_FAILED":
            break
        await asyncio.sleep(0.05)

    await executor.stop()

    assert updated is not None
    assert updated.status == "STAGE3_FAILED"
    assert (updated.last_error or {}).get("code") == "timeout"
