from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.mark.anyio
async def test_commit_id_isolated_by_tenant(tmp_path) -> None:
    """Different租户相同session+commit应创建独立job，不得复用。"""
    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    store = AsyncIngestJobStore({"sqlite_path": str(tmp_path / "ingest_jobs.db")})

    job_a, created_a = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant_a",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t1", "text": "hi"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created_a is True

    # 相同 session+commit 但不同租户，应当创建新 job。
    job_b, created_b = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant_b",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t2", "text": "hello"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created_b is True
    assert job_b.job_id != job_a.job_id

    # 同租户重复提交应复用。
    job_a2, created_a2 = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant_a",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t3", "text": "reuse"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created_a2 is False
    assert job_a2.job_id == job_a.job_id


@pytest.mark.anyio
async def test_legacy_commit_index_respects_tenant(tmp_path) -> None:
    """旧版 ingest_commit_index 仅(session,commit)的场景下，跨租户不得误复用。"""
    db_path = tmp_path / "legacy.db"
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
                "job_legacy",
                "s1",
                "c1",
                "tenant_a",
                None,
                None,
                json.dumps(["u"], ensure_ascii=False),
                "dialog",
                "require",
                "RECEIVED",
                json.dumps({}, ensure_ascii=False),
                None,
                json.dumps({}, ensure_ascii=False),
                json.dumps({"archived_turns": 1}, ensure_ascii=False),
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                None,
                json.dumps([{"turn_id": "t1", "text": "hi"}], ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                None,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO ingest_commit_index(session_id, commit_id, job_id) VALUES (?,?,?)",
            ("s1", "c1", "job_legacy"),
        )
        conn.commit()
    finally:
        conn.close()

    from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore

    store = AsyncIngestJobStore({"sqlite_path": str(db_path)})
    job_new, created_new = await store.create_job(
        session_id="s1",
        commit_id="c1",
        tenant_id="tenant_b",
        api_key_id=None,
        request_id=None,
        turns=[{"turn_id": "t2", "text": "hello"}],
        user_tokens=["u"],
        base_turn_id=None,
        client_meta={},
        memory_domain="dialog",
        llm_policy="require",
    )
    assert created_new is True
    assert job_new.job_id != "job_legacy"
