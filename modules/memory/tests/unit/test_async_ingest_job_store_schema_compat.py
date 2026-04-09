from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.mark.anyio
async def test_async_ingest_job_store_reads_old_schema_column_order(tmp_path) -> None:
    """Regression: older DBs had missing columns added later via ALTER TABLE.

    SQLite returns SELECT * columns in physical table order, so tuple-index decoding breaks.
    Store must decode rows by column name instead.
    """

    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    db_path = tmp_path / "ingest_jobs_old.db"
    conn = sqlite3.connect(str(db_path))
    try:
        # Old schema: api_key_id/request_id/stage2_* not present.
        conn.execute(
            """
            CREATE TABLE ingest_jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                commit_id TEXT,
                tenant_id TEXT NOT NULL,
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
                client_meta TEXT
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
            CREATE TABLE ingest_commit_index (
                session_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                PRIMARY KEY (session_id, commit_id)
            )
            """
        )
        now = _now_iso()
        conn.execute(
            """
            INSERT INTO ingest_jobs (
                job_id, session_id, commit_id, tenant_id,
                user_tokens, memory_domain, llm_policy, status,
                attempts, next_retry_at, last_error, metrics,
                created_at, updated_at, cursor_committed, turns, client_meta
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "job_legacy",
                "s1",
                "c1",
                "tenant",
                json.dumps(["u:1"], ensure_ascii=False),
                "dialog",
                "best_effort",
                "RECEIVED",
                json.dumps({}, ensure_ascii=False),
                None,
                None,
                json.dumps({"x": 1}, ensure_ascii=False),
                now,
                now,
                None,
                json.dumps([{"turn_id": "t1", "role": "user", "text": "hi"}], ensure_ascii=False),
                json.dumps({"stage2_skip_llm": True}, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # New store should migrate schema (ALTER TABLE) but still read records correctly.
    store = AsyncIngestJobStore({"sqlite_path": str(db_path)})
    rec = await store.get_job("job_legacy")
    assert rec is not None
    assert rec.job_id == "job_legacy"
    assert rec.status == "RECEIVED"
    assert rec.tenant_id == "tenant"
    assert rec.user_tokens == ["u:1"]
    assert rec.turns and rec.turns[0].get("text") == "hi"
    assert rec.client_meta.get("stage2_skip_llm") is True
    assert rec.api_key_id is None
    assert rec.request_id is None

