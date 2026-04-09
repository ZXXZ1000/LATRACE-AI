"""PostgreSQL-backed async ingest job store.

This module provides a PostgreSQL implementation of the ingest job store,
enabling horizontal scaling of the memory service by removing SQLite dependency.

Uses asyncpg for native async PostgreSQL support with connection pooling.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from .async_ingest_job_store import IngestJobRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_turn_id(turns: List[Dict[str, Any]]) -> Optional[str]:
    ids = [str(t.get("turn_id") or "").strip() for t in turns if str(t.get("turn_id") or "").strip()]
    if not ids:
        return None
    return max(ids)


@dataclass
class PgIngestJobStoreSettings:
    """PostgreSQL connection settings."""
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = "memory"
    pool_min: int = 2
    pool_max: int = 10

    @classmethod
    def from_env(cls) -> "PgIngestJobStoreSettings":
        return cls(
            host=os.getenv("MEMORY_PG_HOST", "localhost"),
            port=int(os.getenv("MEMORY_PG_PORT", "5432")),
            user=os.getenv("MEMORY_PG_USER", "postgres"),
            password=os.getenv("MEMORY_PG_PASSWORD", ""),
            database=os.getenv("MEMORY_PG_DATABASE", "postgres"),
            pool_min=int(os.getenv("MEMORY_PG_POOL_MIN", "2")),
            pool_max=int(os.getenv("MEMORY_PG_POOL_MAX", "10")),
        )


class PgIngestJobStore:
    """PostgreSQL-backed async ingest job store.

    Provides the same interface as AsyncIngestJobStore but uses PostgreSQL
    for persistence, enabling horizontal scaling across multiple pods.

    Uses asyncpg connection pooling for optimal performance.
    """

    def __init__(self, settings: Optional[PgIngestJobStoreSettings] = None) -> None:
        self._settings = settings or PgIngestJobStoreSettings.from_env()
        self._logger = logging.getLogger(__name__)
        self._pool: Optional[asyncpg.Pool] = None
        self._schema_initialized = False

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self._settings.host,
                port=self._settings.port,
                user=self._settings.user,
                password=self._settings.password,
                database=self._settings.database,
                min_size=self._settings.pool_min,
                max_size=self._settings.pool_max,
            )
            await self._init_schema()
        return self._pool

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        if self._schema_initialized:
            return

        pool = self._pool
        if pool is None:
            return

        async with pool.acquire() as conn:
            # Create ingest_jobs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    commit_id TEXT,
                    tenant_id TEXT NOT NULL,
                    api_key_id TEXT,
                    request_id TEXT,
                    user_tokens JSONB NOT NULL DEFAULT '[]',
                    memory_domain TEXT NOT NULL,
                    llm_policy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts JSONB NOT NULL DEFAULT '{"stage2": 0, "stage3": 0}',
                    next_retry_at TIMESTAMPTZ,
                    last_error JSONB,
                    metrics JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    cursor_committed TEXT,
                    turns JSONB,
                    client_meta JSONB,
                    stage2_marks JSONB,
                    stage2_pin_intents JSONB,
                    payload_raw TEXT
                )
            """)

            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status ON ingest_jobs(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_session ON ingest_jobs(session_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ingest_jobs_tenant ON ingest_jobs(tenant_id)")

            # Create ingest_session_state table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_session_state (
                    session_id TEXT PRIMARY KEY,
                    latest_job_id TEXT,
                    latest_status TEXT,
                    cursor_committed TEXT
                )
            """)

            # Create ingest_commit_index_v2 table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingest_commit_index_v2 (
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, session_id, commit_id)
                )
            """)

        self._schema_initialized = True
        self._logger.info("pg_ingest_job_store: schema initialized")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _row_to_record(self, row: asyncpg.Record) -> IngestJobRecord:
        """Convert database row to IngestJobRecord."""
        def _safe_list(val: Any) -> list:
            if val is None:
                return []
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, list) else []
                except Exception:
                    return []
            if isinstance(val, list):
                return val
            return []

        def _safe_dict(val: Any) -> dict:
            if val is None:
                return {}
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
            if isinstance(val, dict):
                return val
            return {}

        def _ts_to_iso(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.isoformat()
            return str(val)

        return IngestJobRecord(
            job_id=str(row["job_id"] or ""),
            session_id=str(row["session_id"] or ""),
            commit_id=str(row["commit_id"]) if row["commit_id"] else None,
            tenant_id=str(row["tenant_id"] or ""),
            api_key_id=str(row["api_key_id"]) if row["api_key_id"] else None,
            request_id=str(row["request_id"]) if row["request_id"] else None,
            user_tokens=_safe_list(row["user_tokens"]),
            memory_domain=str(row["memory_domain"] or ""),
            llm_policy=str(row["llm_policy"] or ""),
            status=str(row["status"] or ""),
            attempts=_safe_dict(row["attempts"]),
            next_retry_at=_ts_to_iso(row["next_retry_at"]),
            last_error=_safe_dict(row["last_error"]),
            metrics=_safe_dict(row["metrics"]),
            created_at=_ts_to_iso(row["created_at"]) or "",
            updated_at=_ts_to_iso(row["updated_at"]) or "",
            cursor_committed=str(row["cursor_committed"]) if row["cursor_committed"] else None,
            turns=_safe_list(row["turns"]),
            client_meta=_safe_dict(row["client_meta"]),
            stage2_marks=_safe_list(row["stage2_marks"]),
            stage2_pin_intents=_safe_list(row["stage2_pin_intents"]),
            payload_raw=str(row["payload_raw"]) if row["payload_raw"] else None,
        )

    async def create_job(
        self,
        *,
        session_id: str,
        commit_id: Optional[str],
        tenant_id: str,
        api_key_id: Optional[str],
        request_id: Optional[str],
        user_tokens: List[str],
        memory_domain: str,
        llm_policy: str,
        turns: List[Dict[str, Any]],
        base_turn_id: Optional[str],
        client_meta: Optional[Dict[str, Any]],
        payload_raw: Optional[str] = None,
    ) -> Tuple[IngestJobRecord, bool]:
        """Create or retrieve an ingest job with commit_id idempotency.

        Uses atomic INSERT ... ON CONFLICT DO NOTHING on commit_index_v2 as
        the concurrency gatekeeper. This ensures only one job is created per
        (tenant_id, session_id, commit_id) tuple even under concurrent requests.
        """
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        cid = str(commit_id or "").strip() or None

        pool = await self._get_pool()
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        cursor_val = _max_turn_id(turns) or (str(base_turn_id).strip() if base_turn_id else None)

        async with pool.acquire() as conn:
            async with conn.transaction():
                # For commit_id idempotency: use atomic INSERT ... ON CONFLICT DO NOTHING
                # to claim the slot. Only the winner proceeds to create the job.
                if cid:
                    # Try to atomically claim the commit_id slot
                    result = await conn.execute(
                        """
                        INSERT INTO ingest_commit_index_v2 (tenant_id, session_id, commit_id, job_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (tenant_id, session_id, commit_id) DO NOTHING
                        """,
                        str(tenant_id), sid, cid, job_id
                    )
                    # Check if we won the race (INSERT 0 1 = success, INSERT 0 0 = conflict)
                    rows_inserted = int(result.split()[-1]) if result else 0

                    if rows_inserted == 0:
                        # Lost the race - fetch the existing job
                        existing_row = await conn.fetchrow(
                            """
                            SELECT job_id FROM ingest_commit_index_v2
                            WHERE tenant_id = $1 AND session_id = $2 AND commit_id = $3
                            """,
                            str(tenant_id), sid, cid
                        )
                        if existing_row:
                            existing = await self._get_job_internal(conn, str(existing_row["job_id"]))
                            if existing is not None and str(existing.tenant_id) == str(tenant_id):
                                # Check if payload is usable (corrupted job recovery)
                                if existing.status != "COMPLETED" and not self._payload_has_core(existing):
                                    # Drop corrupted job and retry
                                    await conn.execute("DELETE FROM ingest_jobs WHERE job_id = $1", existing.job_id)
                                    await conn.execute(
                                        "DELETE FROM ingest_commit_index_v2 WHERE tenant_id = $1 AND session_id = $2 AND commit_id = $3",
                                        str(tenant_id), sid, cid
                                    )
                                    # Re-claim the slot after cleanup - MUST verify success
                                    reclaim_result = await conn.execute(
                                        """
                                        INSERT INTO ingest_commit_index_v2 (tenant_id, session_id, commit_id, job_id)
                                        VALUES ($1, $2, $3, $4)
                                        ON CONFLICT (tenant_id, session_id, commit_id) DO NOTHING
                                        """,
                                        str(tenant_id), sid, cid, job_id
                                    )
                                    reclaim_rows = int(reclaim_result.split()[-1]) if reclaim_result else 0
                                    if reclaim_rows == 0:
                                        # Another request claimed it during cleanup - return their job
                                        winner_row = await conn.fetchrow(
                                            """
                                            SELECT job_id FROM ingest_commit_index_v2
                                            WHERE tenant_id = $1 AND session_id = $2 AND commit_id = $3
                                            """,
                                            str(tenant_id), sid, cid
                                        )
                                        if winner_row:
                                            winner_job = await self._get_job_internal(conn, str(winner_row["job_id"]))
                                            if winner_job is not None:
                                                return winner_job, False
                                        # Edge case: winner disappeared, fall through to create
                                    # Successfully re-claimed, proceed to create job below
                                else:
                                    return existing, False
                        # If we get here with no existing_row, the slot is free - try to claim it
                        if not existing_row:
                            retry_result = await conn.execute(
                                """
                                INSERT INTO ingest_commit_index_v2 (tenant_id, session_id, commit_id, job_id)
                                VALUES ($1, $2, $3, $4)
                                ON CONFLICT (tenant_id, session_id, commit_id) DO NOTHING
                                """,
                                str(tenant_id), sid, cid, job_id
                            )
                            retry_rows = int(retry_result.split()[-1]) if retry_result else 0
                            if retry_rows == 0:
                                # Someone else claimed it - fetch and return their job
                                final_row = await conn.fetchrow(
                                    """
                                    SELECT job_id FROM ingest_commit_index_v2
                                    WHERE tenant_id = $1 AND session_id = $2 AND commit_id = $3
                                    """,
                                    str(tenant_id), sid, cid
                                )
                                if final_row:
                                    final_job = await self._get_job_internal(conn, str(final_row["job_id"]))
                                    if final_job is not None:
                                        return final_job, False

                # Create the job record
                await conn.execute(
                    """
                    INSERT INTO ingest_jobs (
                        job_id, session_id, commit_id, tenant_id, api_key_id, request_id,
                        user_tokens, memory_domain, llm_policy, status, attempts,
                        next_retry_at, last_error, metrics, created_at, updated_at,
                        cursor_committed, turns, client_meta, stage2_marks, stage2_pin_intents, payload_raw
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11::jsonb, $12, $13, $14::jsonb, $15, $16, $17, $18::jsonb, $19::jsonb, $20, $21, $22)
                    """,
                    job_id, sid, cid, str(tenant_id),
                    str(api_key_id).strip() if api_key_id else None,
                    str(request_id).strip() if request_id else None,
                    json.dumps(list(user_tokens)),  # Explicit JSON serialization for JSONB
                    str(memory_domain), str(llm_policy), "RECEIVED",
                    json.dumps({"stage2": 0, "stage3": 0}),
                    None, None,
                    json.dumps({"archived_turns": len(turns), "kept_turns": None, "graph_nodes_written": 0, "vector_points_written": 0}),
                    now, now, cursor_val,
                    json.dumps(list(turns)),
                    json.dumps(dict(client_meta or {})) if client_meta else None,
                    None, None,
                    str(payload_raw) if payload_raw is not None else None,
                )

                # Update session state
                await conn.execute(
                    """
                    INSERT INTO ingest_session_state (session_id, latest_job_id, latest_status, cursor_committed)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id) DO UPDATE SET
                        latest_job_id = $2, latest_status = $3, cursor_committed = $4
                    """,
                    sid, job_id, "RECEIVED", cursor_val
                )

                record = IngestJobRecord(
                    job_id=job_id,
                    session_id=sid,
                    commit_id=cid,
                    tenant_id=str(tenant_id),
                    api_key_id=str(api_key_id).strip() if api_key_id else None,
                    request_id=str(request_id).strip() if request_id else None,
                    user_tokens=list(user_tokens),
                    memory_domain=str(memory_domain),
                    llm_policy=str(llm_policy),
                    status="RECEIVED",
                    attempts={"stage2": 0, "stage3": 0},
                    next_retry_at=None,
                    last_error=None,
                    metrics={"archived_turns": len(turns), "kept_turns": None, "graph_nodes_written": 0, "vector_points_written": 0},
                    created_at=now.isoformat(),
                    updated_at=now.isoformat(),
                    cursor_committed=cursor_val,
                    turns=list(turns),
                    client_meta=dict(client_meta or {}),
                )
                return record, True

    def _payload_has_core(self, job: IngestJobRecord) -> bool:
        """Return True when job has enough payload to execute safely."""
        if not job.user_tokens or not any(str(x).strip() for x in job.user_tokens):
            return False
        if job.turns and any(isinstance(t, dict) and str(t.get("text") or t.get("content") or "").strip() for t in job.turns):
            return True
        raw = str(job.payload_raw or "").strip()
        if not raw:
            return False
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return False
            turns_val = data.get("turns")
            if not isinstance(turns_val, list) or not turns_val:
                return False
            return any(isinstance(t, dict) and str(t.get("text") or t.get("content") or "").strip() for t in turns_val)
        except Exception:
            return False

    async def _get_job_internal(self, conn: asyncpg.Connection, job_id: str) -> Optional[IngestJobRecord]:
        """Internal get_job using existing connection."""
        row = await conn.fetchrow("SELECT * FROM ingest_jobs WHERE job_id = $1", str(job_id))
        if not row:
            return None
        return self._row_to_record(row)

    async def get_job(self, job_id: str) -> Optional[IngestJobRecord]:
        """Get job by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await self._get_job_internal(conn, job_id)

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session state."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT latest_job_id, latest_status, cursor_committed FROM ingest_session_state WHERE session_id = $1",
                str(session_id)
            )
            if not row:
                return {}
            return {
                "latest_job_id": row["latest_job_id"],
                "latest_status": row["latest_status"],
                "cursor_committed": row["cursor_committed"],
            }

    async def count_by_tenant(self, tenant_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS count FROM ingest_jobs WHERE tenant_id = $1",
                str(tenant_id),
            )
            return int(row["count"] or 0) if row is not None else 0

    async def clear_by_tenant(self, tenant_id: str) -> int:
        tid = str(tenant_id or "").strip()
        if not tid:
            return 0
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM ingest_jobs WHERE tenant_id = $1",
                    tid,
                )
                deleted = int(str(result).split()[-1])
                await conn.execute(
                    "DELETE FROM ingest_commit_index_v2 WHERE tenant_id = $1",
                    tid,
                )
                await conn.execute(
                    """
                    DELETE FROM ingest_session_state
                    WHERE session_id NOT IN (SELECT DISTINCT session_id FROM ingest_jobs)
                    """
                )
                return int(deleted)

    async def list_jobs_by_status(self, statuses: List[str]) -> List[IngestJobRecord]:
        """List jobs by status."""
        if not statuses:
            return []
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM ingest_jobs WHERE status = ANY($1)",
                statuses
            )
            return [self._row_to_record(row) for row in rows]

    async def update_status(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
        next_retry_at: Optional[str] = None,
        metrics_patch: Optional[Dict[str, Any]] = None,
        attempt_inc: bool = False,
    ) -> None:
        """Update job status and related fields."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                job = await self._get_job_internal(conn, job_id)
                if job is None:
                    return

                new_status = status if status is not None else job.status
                new_attempts = dict(job.attempts)
                if attempt_inc and stage:
                    new_attempts[stage] = int(new_attempts.get(stage, 0)) + 1
                new_error = dict(error) if error is not None else job.last_error
                new_next_retry = next_retry_at if next_retry_at is not None else job.next_retry_at
                new_metrics = dict(job.metrics)
                if metrics_patch:
                    new_metrics.update(metrics_patch)

                # Parse next_retry_at if string
                retry_ts = None
                if new_next_retry:
                    try:
                        retry_ts = datetime.fromisoformat(str(new_next_retry).replace("Z", "+00:00"))
                    except Exception:
                        pass

                await conn.execute(
                    """
                    UPDATE ingest_jobs SET
                        status = $2,
                        attempts = $3::jsonb,
                        last_error = $4::jsonb,
                        next_retry_at = $5,
                        metrics = $6::jsonb,
                        updated_at = $7
                    WHERE job_id = $1
                    """,
                    str(job_id), new_status,
                    json.dumps(new_attempts),  # Explicit JSON serialization for JSONB
                    json.dumps(new_error) if new_error else None,
                    retry_ts,
                    json.dumps(new_metrics),
                    datetime.now(timezone.utc),
                )

                await conn.execute(
                    """
                    INSERT INTO ingest_session_state (session_id, latest_job_id, latest_status, cursor_committed)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id) DO UPDATE SET
                        latest_job_id = $2, latest_status = $3, cursor_committed = $4
                    """,
                    job.session_id, str(job_id), new_status, job.cursor_committed
                )

    async def try_transition_status(
        self,
        job_id: str,
        *,
        from_statuses: List[str],
        to_status: str,
    ) -> bool:
        """Atomically transition job status when current status matches."""
        if not from_statuses:
            return False
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    """
                    UPDATE ingest_jobs SET status = $2, updated_at = $3
                    WHERE job_id = $1 AND status = ANY($4)
                    """,
                    str(job_id), str(to_status), datetime.now(timezone.utc),
                    [str(s) for s in from_statuses]
                )
                # Check if any row was updated
                if result == "UPDATE 0":
                    return False

                # Update session state
                row = await conn.fetchrow(
                    "SELECT session_id, cursor_committed FROM ingest_jobs WHERE job_id = $1",
                    str(job_id)
                )
                if row:
                    await conn.execute(
                        """
                        INSERT INTO ingest_session_state (session_id, latest_job_id, latest_status, cursor_committed)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (session_id) DO UPDATE SET
                            latest_job_id = $2, latest_status = $3, cursor_committed = $4
                        """,
                        str(row["session_id"]), str(job_id), str(to_status), row["cursor_committed"]
                    )
                return True

    async def update_stage2(
        self,
        job_id: str,
        *,
        marks: List[Dict[str, Any]],
        pin_intents: List[Dict[str, Any]],
        metrics_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update stage2 marks and pin intents."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            job = await self._get_job_internal(conn, job_id)
            if job is None:
                return

            new_metrics = dict(job.metrics)
            if metrics_patch:
                new_metrics.update(metrics_patch)

            await conn.execute(
                """
                UPDATE ingest_jobs SET
                    stage2_marks = $2::jsonb,
                    stage2_pin_intents = $3::jsonb,
                    metrics = $4::jsonb,
                    updated_at = $5
                WHERE job_id = $1
                """,
                str(job_id),
                json.dumps(list(marks or [])),  # Explicit JSON serialization for JSONB
                json.dumps(list(pin_intents or [])),
                json.dumps(new_metrics),
                datetime.now(timezone.utc),
            )

    async def prune_payload(self, job_id: str) -> bool:
        """Clear payload after job completion."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            job = await self._get_job_internal(conn, job_id)
            if job is None or job.status != "COMPLETED":
                return False

            await conn.execute(
                """
                UPDATE ingest_jobs SET
                    turns = '[]',
                    client_meta = '{}',
                    stage2_marks = '[]',
                    stage2_pin_intents = '[]',
                    payload_raw = NULL,
                    updated_at = $2
                WHERE job_id = $1
                """,
                str(job_id), datetime.now(timezone.utc)
            )
            return True

    async def purge_jobs(
        self,
        *,
        statuses: List[str],
        older_than_iso: Optional[str],
        require_no_retry: bool = False,
    ) -> int:
        """Purge old jobs by status."""
        if not statuses:
            return 0

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Build query conditions
                conditions = ["status = ANY($1)"]
                params: List[Any] = [statuses]

                if older_than_iso:
                    try:
                        older_than = datetime.fromisoformat(older_than_iso.replace("Z", "+00:00"))
                        conditions.append(f"updated_at < ${len(params) + 1}")
                        params.append(older_than)
                    except Exception:
                        pass

                if require_no_retry:
                    conditions.append("next_retry_at IS NULL")

                where_clause = " AND ".join(conditions)

                # Get jobs to delete
                rows = await conn.fetch(
                    f"SELECT job_id, session_id, commit_id, tenant_id FROM ingest_jobs WHERE {where_clause}",
                    *params
                )

                purged = 0
                for row in rows:
                    await conn.execute("DELETE FROM ingest_jobs WHERE job_id = $1", row["job_id"])
                    if row["commit_id"]:
                        await conn.execute(
                            "DELETE FROM ingest_commit_index_v2 WHERE tenant_id = $1 AND session_id = $2 AND commit_id = $3",
                            row["tenant_id"], row["session_id"], row["commit_id"]
                        )
                    purged += 1

                return purged
