from __future__ import annotations

from typing import Any, Dict, List

import pytest


@pytest.mark.anyio
async def test_ingest_job_prune_payload_on_completed(tmp_path) -> None:
    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    turns: List[Dict[str, Any]] = [{"turn_id": "t1", "role": "user", "content": "hi"}]
    record, created = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant",
        api_key_id=None,
        request_id=None,
        turns=turns,
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={"llm_api_key": "secret", "x": "y"},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created is True
    assert record.status == "RECEIVED"
    assert record.turns
    assert record.client_meta

    await store.update_status(record.job_id, status="COMPLETED")
    ok = await store.prune_payload(record.job_id)
    assert ok is True

    pruned = await store.get_job(record.job_id)
    assert pruned is not None
    assert pruned.status == "COMPLETED"
    assert pruned.turns == []
    assert pruned.stage2_marks == []
    assert pruned.stage2_pin_intents == []
    assert pruned.client_meta == {}


@pytest.mark.anyio
async def test_ingest_job_purge_removes_commit_index(tmp_path) -> None:
    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    record, created = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t1"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created is True
    await store.update_status(record.job_id, status="COMPLETED")
    await store.prune_payload(record.job_id)

    purged = await store.purge_jobs(statuses=["COMPLETED"], older_than_iso="9999-12-31T00:00:00+00:00")
    assert purged == 1
    assert await store.get_job(record.job_id) is None

    # Commit idempotency index must also be removed; otherwise re-commit would break.
    record2, created2 = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t2"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created2 is True
    assert record2.job_id != record.job_id


@pytest.mark.anyio
async def test_ingest_job_idempotency_does_not_reuse_corrupt_payload(tmp_path) -> None:
    """Regression: a corrupt/empty job must not permanently block commit_id reuse."""

    import sqlite3
    import json
    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    db_path = tmp_path / "ingest_jobs.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE ingest_jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                commit_id TEXT,
                tenant_id TEXT NOT NULL,
                api_key_id TEXT,
                request_id TEXT,
                user_tokens TEXT NOT NULL,
                memory_domain TEXT NOT NULL,
                llm_policy TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts TEXT NOT NULL,
                next_retry_at TEXT,
                last_error TEXT,
                metrics TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cursor_committed TEXT,
                turns TEXT,
                client_meta TEXT,
                stage2_marks TEXT,
                stage2_pin_intents TEXT,
                payload_raw TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ingest_commit_index (
                session_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                PRIMARY KEY (session_id, commit_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ingest_session_state (
                session_id TEXT PRIMARY KEY,
                latest_job_id TEXT,
                latest_status TEXT,
                cursor_committed TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO ingest_jobs(
                job_id, session_id, commit_id, tenant_id, api_key_id, request_id, user_tokens, memory_domain, llm_policy, status,
                attempts, next_retry_at, last_error, metrics, created_at, updated_at, cursor_committed, turns, client_meta,
                stage2_marks, stage2_pin_intents, payload_raw
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "job_bad",
                "s1",
                "c1",
                "tenant",
                None,
                None,
                json.dumps([], ensure_ascii=False),
                "dialog",
                "require",
                "RECEIVED",
                json.dumps({}, ensure_ascii=False),
                None,
                json.dumps({}, ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                None,
                json.dumps([], ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                json.dumps({"turns": [{"turn_id": "t1", "role": "user", "text": "hi"}], "user_tokens": ["u"]}, ensure_ascii=False),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO ingest_commit_index(session_id, commit_id, job_id) VALUES (?,?,?)",
            ("s1", "c1", "job_bad"),
        )
        conn.commit()
    finally:
        conn.close()

    store = AsyncIngestJobStore({"sqlite_path": str(db_path)})
    record, created = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t2", "role": "user", "content": "hello"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created is True
    assert record.job_id != "job_bad"


@pytest.mark.anyio
async def test_ingest_job_failure_purge_skips_scheduled_retry(tmp_path) -> None:
    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})
    record, created = await store.create_job(
        session_id="s1",
        commit_id=None,
        tenant_id="tenant",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t1"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created is True
    await store.update_status(record.job_id, status="STAGE3_FAILED", next_retry_at="2099-01-01T00:00:00+00:00")

    purged = await store.purge_jobs(
        statuses=["STAGE3_FAILED"],
        older_than_iso="9999-12-31T00:00:00+00:00",
        require_no_retry=True,
    )
    assert purged == 0
    assert await store.get_job(record.job_id) is not None

    # Once no retry is scheduled, GC is allowed to purge.
    await store.update_status(record.job_id, next_retry_at="")
    purged2 = await store.purge_jobs(
        statuses=["STAGE3_FAILED"],
        older_than_iso="9999-12-31T00:00:00+00:00",
        require_no_retry=True,
    )
    assert purged2 == 1
    assert await store.get_job(record.job_id) is None
