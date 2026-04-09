"""Async wrapper for IngestJobStore using connection-per-operation pattern.

Each operation creates a new SQLite connection and uses asyncio.to_thread
to avoid blocking. This eliminates lock contention between operations.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_turn_id(turns: List[Dict[str, Any]]) -> Optional[str]:
    ids = [str(t.get("turn_id") or "").strip() for t in turns if str(t.get("turn_id") or "").strip()]
    if not ids:
        return None
    return max(ids)


@dataclass
class IngestJobRecord:
    job_id: str
    session_id: str
    commit_id: Optional[str]
    tenant_id: str
    api_key_id: Optional[str]
    request_id: Optional[str]
    user_tokens: List[str]
    memory_domain: str
    llm_policy: str
    status: str
    attempts: Dict[str, int]
    next_retry_at: Optional[str]
    last_error: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    created_at: str
    updated_at: str
    cursor_committed: Optional[str]
    turns: List[Dict[str, Any]] = field(default_factory=list)
    client_meta: Dict[str, Any] = field(default_factory=dict)
    stage2_marks: List[Dict[str, Any]] = field(default_factory=list)
    stage2_pin_intents: List[Dict[str, Any]] = field(default_factory=list)
    payload_raw: Optional[str] = None


class AsyncIngestJobStore:
    """Async SQLite-backed ingest job store.

    Uses connection-per-operation pattern with asyncio.to_thread.
    This avoids lock contention by not sharing connections between operations.

    Thread safety:
        - _create_lock: asyncio.Lock() protects commit_id idempotency check
        - BEGIN IMMEDIATE: SQLite-level write lock for atomic transactions
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        cfg = settings or {}
        path = cfg.get("sqlite_path") or "modules/memory/outputs/ingest_jobs.db"
        self._path = str(path)
        self._logger = logging.getLogger(__name__)
        if self._path != ":memory:":
            base_dir = os.path.dirname(os.path.abspath(self._path))
            if base_dir and not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
        self._schema_initialized = False
        # Lock for create_job to ensure commit_id idempotency under concurrent requests
        self._create_lock: Optional[asyncio.Lock] = None
        # Initialize schema synchronously at startup
        self._init_schema_sync()

    def _get_create_lock(self) -> asyncio.Lock:
        """Lazily initialize the create lock (must be called from async context)."""
        if self._create_lock is None:
            self._create_lock = asyncio.Lock()
        return self._create_lock

    def _get_conn_sync(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings."""
        conn = sqlite3.connect(
            self._path,
            timeout=60.0,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema_sync(self) -> None:
        """Initialize schema synchronously at startup."""
        if self._schema_initialized:
            return
        conn = self._get_conn_sync()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
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
                CREATE TABLE IF NOT EXISTS ingest_session_state (
                    session_id TEXT PRIMARY KEY,
                    latest_job_id TEXT,
                    latest_status TEXT,
                    cursor_committed TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_commit_index (
                    session_id TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, commit_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_commit_index_v2 (
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, session_id, commit_id)
                )
                """
            )
            conn.commit()
            # Add columns if missing
            cursor = conn.execute("PRAGMA table_info(ingest_jobs)")
            existing = {row[1] for row in cursor.fetchall()}
            for col, col_type in [
                ("stage2_marks", "TEXT"),
                ("stage2_pin_intents", "TEXT"),
                ("api_key_id", "TEXT"),
                ("request_id", "TEXT"),
                ("payload_raw", "TEXT"),
            ]:
                if col not in existing:
                    conn.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {col} {col_type}")
            conn.commit()
            # Best-effort backfill legacy commit index into tenant-scoped v2.
            try:
                cur = conn.execute("SELECT 1 FROM ingest_commit_index_v2 LIMIT 1")
                has_v2 = cur.fetchone() is not None
            except Exception:
                has_v2 = False
            if not has_v2:
                try:
                    cur = conn.execute(
                        """
                        SELECT j.tenant_id, idx.session_id, idx.commit_id, idx.job_id
                        FROM ingest_commit_index AS idx
                        JOIN ingest_jobs AS j ON j.job_id = idx.job_id
                        WHERE j.tenant_id IS NOT NULL
                        """
                    )
                    rows = cur.fetchall()
                    if rows:
                        conn.executemany(
                            "INSERT OR IGNORE INTO ingest_commit_index_v2(tenant_id, session_id, commit_id, job_id) VALUES (?,?,?,?)",
                            rows,
                        )
                        conn.commit()
                        self._logger.info(
                            "async_ingest_job_store: commit_index_v2 backfilled from legacy index",
                            extra={"rows_backfilled": len(rows)},
                        )
                    else:
                        self._logger.info(
                            "async_ingest_job_store: commit_index_v2 backfill found no legacy rows",
                            extra={},
                        )
                except Exception:
                    conn.rollback()
                    self._logger.warning(
                        "async_ingest_job_store: commit_index_v2 backfill failed",
                        exc_info=True,
                    )
            self._schema_initialized = True
        finally:
            conn.close()

    def _create_job_sync(
        self,
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
        payload_raw: Optional[str],
    ) -> Tuple["IngestJobRecord", bool]:
        """Synchronous create_job - runs in thread pool.

        Uses BEGIN IMMEDIATE to acquire write lock at transaction start,
        ensuring atomic check-and-insert for commit_id idempotency.
        """
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        cid = str(commit_id or "").strip() or None

        conn = self._get_conn_sync()
        try:
            # BEGIN IMMEDIATE acquires RESERVED lock immediately, preventing
            # concurrent writes from interleaving. This ensures the read-check
            # and insert are atomic with respect to other writers.
            conn.execute("BEGIN IMMEDIATE")

            def _payload_has_core(job: "IngestJobRecord") -> bool:
                """Return True when job has enough payload to execute safely."""
                if job.user_tokens and any(str(x).strip() for x in job.user_tokens):
                    pass
                else:
                    return False
                if job.turns and any(isinstance(t, dict) and str(t.get("text") or t.get("content") or "").strip() for t in job.turns):
                    return True
                # Allow raw payload to supply turns when stored turns got pruned/corrupted.
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

            # Check for existing job with same commit_id (within transaction)
            if cid:
                cursor = conn.execute(
                    "SELECT job_id FROM ingest_commit_index_v2 WHERE tenant_id=? AND session_id=? AND commit_id=?",
                    (str(tenant_id), sid, cid),
                )
                row = cursor.fetchone()
                if row is None:
                    # Legacy index without tenant isolation (fallback; guarded by tenant check below)
                    cursor = conn.execute(
                        "SELECT job_id FROM ingest_commit_index WHERE session_id=? AND commit_id=?",
                        (sid, cid),
                    )
                    row = cursor.fetchone()
                if row:
                    existing_job = self._get_job_sync_internal(conn, str(row[0]))
                    if existing_job is not None:
                        if str(existing_job.tenant_id or "") != str(tenant_id):
                            existing_job = None
                        if existing_job is not None:
                            # Never let a corrupted/empty job permanently block a commit_id.
                            # If existing payload is unusable, drop it and allow a new job to be created.
                            if existing_job.status != "COMPLETED" and not _payload_has_core(existing_job):
                                conn.execute("DELETE FROM ingest_jobs WHERE job_id=?", (str(existing_job.job_id),))
                                conn.execute(
                                    "DELETE FROM ingest_commit_index_v2 WHERE tenant_id=? AND session_id=? AND commit_id=?",
                                    (str(tenant_id), sid, cid),
                                )
                                conn.execute(
                                    "DELETE FROM ingest_commit_index WHERE session_id=? AND commit_id=?",
                                    (sid, cid),
                                )
                                existing_job = None
                            else:
                                conn.rollback()
                                return existing_job, False

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            now = _now_iso()
            cursor_val = _max_turn_id(turns) or (str(base_turn_id).strip() if base_turn_id else None)
            record = IngestJobRecord(
                job_id=job_id,
                session_id=sid,
                commit_id=cid,
                tenant_id=str(tenant_id),
                api_key_id=(str(api_key_id).strip() if api_key_id else None),
                request_id=(str(request_id).strip() if request_id else None),
                user_tokens=list(user_tokens),
                memory_domain=str(memory_domain),
                llm_policy=str(llm_policy),
                status="RECEIVED",
                attempts={"stage2": 0, "stage3": 0},
                next_retry_at=None,
                last_error=None,
                metrics={
                    "archived_turns": len(turns),
                    "kept_turns": None,
                    "graph_nodes_written": 0,
                    "vector_points_written": 0,
                },
                created_at=now,
                updated_at=now,
                cursor_committed=cursor_val,
                turns=list(turns),
                client_meta=dict(client_meta or {}),
                payload_raw=(str(payload_raw) if payload_raw is not None else None),
            )
            self._upsert_job_sync_internal(conn, record)
            if cid:
                conn.execute(
                    "INSERT OR REPLACE INTO ingest_commit_index_v2(tenant_id, session_id, commit_id, job_id) VALUES (?,?,?,?)",
                    (str(tenant_id), sid, cid, job_id),
                )
            conn.execute(
                """INSERT OR REPLACE INTO ingest_session_state(session_id, latest_job_id, latest_status, cursor_committed)
                   VALUES (?,?,?,?)""",
                (sid, job_id, record.status, cursor_val),
            )
            conn.commit()
            return record, True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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
    ) -> Tuple["IngestJobRecord", bool]:
        """Create or retrieve an ingest job with commit_id idempotency.

        Uses BEGIN IMMEDIATE transaction at SQLite level to ensure the
        read-check and insert are atomic w.r.t. other writers.
        """
        return await asyncio.to_thread(
            self._create_job_sync,
            session_id, commit_id, tenant_id, api_key_id, request_id,
            user_tokens, memory_domain, llm_policy, turns, base_turn_id, client_meta, payload_raw,
        )

    def _get_job_sync(self, job_id: str) -> Optional["IngestJobRecord"]:
        """Synchronous get_job - runs in thread pool."""
        conn = self._get_conn_sync()
        try:
            return self._get_job_sync_internal(conn, job_id)
        finally:
            conn.close()

    def _get_columns_sync(self, conn: sqlite3.Connection) -> List[str]:
        cursor = conn.execute("PRAGMA table_info(ingest_jobs)")
        return [row[1] for row in cursor.fetchall()]

    def _get_job_sync_internal(self, conn: sqlite3.Connection, job_id: str) -> Optional["IngestJobRecord"]:
        columns = self._get_columns_sync(conn)
        cursor = conn.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (str(job_id),))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_record(row, columns)

    async def get_job(self, job_id: str) -> Optional["IngestJobRecord"]:
        return await asyncio.to_thread(self._get_job_sync, job_id)

    def _get_session_sync(self, session_id: str) -> Dict[str, Any]:
        conn = self._get_conn_sync()
        try:
            cursor = conn.execute(
                "SELECT session_id, latest_job_id, latest_status, cursor_committed FROM ingest_session_state WHERE session_id=?",
                (str(session_id),),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            return {"latest_job_id": row[1], "latest_status": row[2], "cursor_committed": row[3]}
        finally:
            conn.close()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._get_session_sync, session_id)

    def _count_by_tenant_sync(self, tenant_id: str) -> int:
        conn = self._get_conn_sync()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM ingest_jobs WHERE tenant_id=?", (str(tenant_id),))
            row = cursor.fetchone()
            return int((row or [0])[0] or 0)
        finally:
            conn.close()

    async def count_by_tenant(self, tenant_id: str) -> int:
        return await asyncio.to_thread(self._count_by_tenant_sync, tenant_id)

    def _clear_by_tenant_sync(self, tenant_id: str) -> int:
        tid = str(tenant_id or "").strip()
        if not tid:
            return 0
        conn = self._get_conn_sync()
        try:
            conn.execute("BEGIN IMMEDIATE")
            session_rows = conn.execute(
                "SELECT DISTINCT session_id FROM ingest_jobs WHERE tenant_id=?",
                (tid,),
            ).fetchall()
            session_ids = [str(row[0]) for row in session_rows if row and str(row[0] or "").strip()]
            cursor = conn.execute("DELETE FROM ingest_jobs WHERE tenant_id=?", (tid,))
            deleted = int(cursor.rowcount or 0)
            conn.execute("DELETE FROM ingest_session_state WHERE session_id NOT IN (SELECT DISTINCT session_id FROM ingest_jobs)")
            try:
                conn.execute(
                    """
                    DELETE FROM ingest_commit_index_v2
                    WHERE tenant_id=?
                    """,
                    (tid,),
                )
            except Exception:
                pass
            if session_ids:
                qs = ",".join("?" for _ in session_ids)
                try:
                    conn.execute(
                        f"DELETE FROM ingest_commit_index WHERE session_id IN ({qs})",
                        tuple(session_ids),
                    )
                except Exception:
                    pass
            conn.commit()
            return deleted
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def clear_by_tenant(self, tenant_id: str) -> int:
        return await asyncio.to_thread(self._clear_by_tenant_sync, tenant_id)

    def _list_jobs_by_status_sync(self, statuses: List[str]) -> List["IngestJobRecord"]:
        if not statuses:
            return []
        conn = self._get_conn_sync()
        try:
            qs = ",".join("?" for _ in statuses)
            columns = self._get_columns_sync(conn)
            cursor = conn.execute(f"SELECT * FROM ingest_jobs WHERE status IN ({qs})", tuple(statuses))
            rows = cursor.fetchall()
            return [self._row_to_record(row, columns) for row in rows]
        finally:
            conn.close()

    async def list_jobs_by_status(self, statuses: List[str]) -> List["IngestJobRecord"]:
        return await asyncio.to_thread(self._list_jobs_by_status_sync, statuses)

    def _update_status_sync(
        self,
        job_id: str,
        status: Optional[str],
        stage: Optional[str],
        error: Optional[Dict[str, Any]],
        next_retry_at: Optional[str],
        metrics_patch: Optional[Dict[str, Any]],
        attempt_inc: bool,
    ) -> None:
        conn = self._get_conn_sync()
        try:
            job = self._get_job_sync_internal(conn, job_id)
            if job is None:
                return
            if status is not None:
                job.status = str(status)
            job.updated_at = _now_iso()
            if attempt_inc and stage:
                job.attempts[stage] = int(job.attempts.get(stage, 0)) + 1
            if error is not None:
                job.last_error = dict(error) if error else None
            if next_retry_at is not None:
                job.next_retry_at = next_retry_at
            if metrics_patch:
                job.metrics.update(metrics_patch)
            self._upsert_job_sync_internal(conn, job)
            conn.execute(
                """INSERT OR REPLACE INTO ingest_session_state(session_id, latest_job_id, latest_status, cursor_committed)
                   VALUES (?,?,?,?)""",
                (job.session_id, job.job_id, job.status, job.cursor_committed),
            )
            conn.commit()
        finally:
            conn.close()

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
        await asyncio.to_thread(
            self._update_status_sync,
            job_id, status, stage, error, next_retry_at, metrics_patch, attempt_inc,
        )

    def _try_transition_status_sync(self, job_id: str, from_statuses: List[str], to_status: str) -> bool:
        if not from_statuses:
            return False
        conn = self._get_conn_sync()
        try:
            qs = ",".join("?" for _ in from_statuses)
            now = _now_iso()
            cur = conn.execute(
                f"UPDATE ingest_jobs SET status=?, updated_at=? WHERE job_id=? AND status IN ({qs})",
                (str(to_status), now, str(job_id), *[str(s) for s in from_statuses]),
            )
            if int(getattr(cur, "rowcount", 0) or 0) <= 0:
                conn.rollback()
                return False
            # Keep ingest_session_state latest_status in sync.
            cursor = conn.execute(
                "SELECT session_id, cursor_committed FROM ingest_jobs WHERE job_id=?",
                (str(job_id),),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    """INSERT OR REPLACE INTO ingest_session_state(session_id, latest_job_id, latest_status, cursor_committed)
                       VALUES (?,?,?,?)""",
                    (str(row[0]), str(job_id), str(to_status), row[1]),
                )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def try_transition_status(self, job_id: str, *, from_statuses: List[str], to_status: str) -> bool:
        """Atomically transition job status when current status matches.

        Used to prevent duplicate concurrent execution (e.g., auto-execute scheduling).
        """
        return await asyncio.to_thread(self._try_transition_status_sync, job_id, from_statuses, to_status)

    def _update_stage2_sync(
        self,
        job_id: str,
        marks: List[Dict[str, Any]],
        pin_intents: List[Dict[str, Any]],
        metrics_patch: Optional[Dict[str, Any]],
    ) -> None:
        conn = self._get_conn_sync()
        try:
            job = self._get_job_sync_internal(conn, job_id)
            if job is None:
                return
            job.stage2_marks = list(marks or [])
            job.stage2_pin_intents = list(pin_intents or [])
            if metrics_patch:
                job.metrics.update(metrics_patch)
            job.updated_at = _now_iso()
            self._upsert_job_sync_internal(conn, job)
            conn.commit()
        finally:
            conn.close()

    async def update_stage2(
        self,
        job_id: str,
        *,
        marks: List[Dict[str, Any]],
        pin_intents: List[Dict[str, Any]],
        metrics_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        await asyncio.to_thread(
            self._update_stage2_sync,
            job_id, marks, pin_intents, metrics_patch,
        )

    def _prune_payload_sync(self, job_id: str) -> bool:
        conn = self._get_conn_sync()
        try:
            job = self._get_job_sync_internal(conn, job_id)
            if job is None:
                return False
            if str(job.status) != "COMPLETED":
                return False
            job.turns = []
            job.client_meta = {}
            job.stage2_marks = []
            job.stage2_pin_intents = []
            job.payload_raw = None
            job.updated_at = _now_iso()
            self._upsert_job_sync_internal(conn, job)
            conn.commit()
            return True
        finally:
            conn.close()

    async def prune_payload(self, job_id: str) -> bool:
        return await asyncio.to_thread(self._prune_payload_sync, job_id)

    def _purge_jobs_sync(
        self,
        statuses: List[str],
        older_than_iso: Optional[str],
        require_no_retry: bool,
    ) -> int:
        if not statuses:
            return 0
        conn = self._get_conn_sync()
        try:
            qs = ",".join("?" for _ in statuses)
            cursor = conn.execute(
                f"SELECT job_id, session_id, commit_id, tenant_id, next_retry_at, updated_at FROM ingest_jobs WHERE status IN ({qs})",
                tuple(statuses),
            )
            rows = cursor.fetchall()
            purged = 0
            for job_id, session_id, commit_id, tenant_id, next_retry_at, updated_at in rows:
                if older_than_iso and str(updated_at or "") >= str(older_than_iso):
                    continue
                if require_no_retry and str(next_retry_at or "").strip():
                    continue
                conn.execute("DELETE FROM ingest_jobs WHERE job_id=?", (str(job_id),))
                if commit_id:
                    conn.execute(
                        "DELETE FROM ingest_commit_index_v2 WHERE tenant_id=? AND session_id=? AND commit_id=?",
                        (str(tenant_id), str(session_id), str(commit_id)),
                    )
                    conn.execute(
                        "DELETE FROM ingest_commit_index WHERE session_id=? AND commit_id=?",
                        (str(session_id), str(commit_id)),
                    )
                purged += 1
            conn.commit()
            return purged
        finally:
            conn.close()

    async def purge_jobs(
        self,
        *,
        statuses: List[str],
        older_than_iso: Optional[str],
        require_no_retry: bool = False,
    ) -> int:
        return await asyncio.to_thread(
            self._purge_jobs_sync, statuses, older_than_iso, require_no_retry,
        )

    def _upsert_job_sync_internal(self, conn: sqlite3.Connection, record: "IngestJobRecord") -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO ingest_jobs(
                job_id, session_id, commit_id, tenant_id, api_key_id, request_id, user_tokens, memory_domain, llm_policy, status,
                attempts, next_retry_at, last_error, metrics, created_at, updated_at, cursor_committed, turns, client_meta,
                stage2_marks, stage2_pin_intents, payload_raw
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.job_id,
                record.session_id,
                record.commit_id,
                record.tenant_id,
                record.api_key_id,
                record.request_id,
                json.dumps(record.user_tokens, ensure_ascii=False),
                record.memory_domain,
                record.llm_policy,
                record.status,
                json.dumps(record.attempts, ensure_ascii=False),
                record.next_retry_at,
                json.dumps(record.last_error, ensure_ascii=False) if record.last_error else None,
                json.dumps(record.metrics, ensure_ascii=False),
                record.created_at,
                record.updated_at,
                record.cursor_committed,
                json.dumps(record.turns, ensure_ascii=False),
                json.dumps(record.client_meta, ensure_ascii=False) if record.client_meta else None,
                json.dumps(record.stage2_marks, ensure_ascii=False) if record.stage2_marks else None,
                json.dumps(record.stage2_pin_intents, ensure_ascii=False) if record.stage2_pin_intents else None,
                record.payload_raw,
            ),
        )

    def _row_to_record(self, row: Tuple[Any, ...], columns: List[str]) -> "IngestJobRecord":
        col_map = {name: idx for idx, name in enumerate(columns)}

        def _get_col(name: str) -> Any:
            idx = col_map.get(name)
            if idx is None or idx >= len(row):
                return None
            return row[idx]

        def _safe_json_list(val: Any) -> list:
            """Safely parse JSON list, handling empty/whitespace strings."""
            if not val or not str(val).strip():
                return []
            try:
                return list(json.loads(val))
            except (json.JSONDecodeError, TypeError):
                return []

        def _safe_json_dict(val: Any) -> dict:
            """Safely parse JSON dict, handling empty/whitespace strings."""
            if not val or not str(val).strip():
                return {}
            try:
                return dict(json.loads(val))
            except (json.JSONDecodeError, TypeError):
                return {}

        return IngestJobRecord(
            job_id=str(_get_col("job_id") or ""),
            session_id=str(_get_col("session_id") or ""),
            commit_id=(str(_get_col("commit_id")) if _get_col("commit_id") is not None else None),
            tenant_id=str(_get_col("tenant_id") or ""),
            api_key_id=(str(_get_col("api_key_id")) if _get_col("api_key_id") else None),
            request_id=(str(_get_col("request_id")) if _get_col("request_id") else None),
            user_tokens=_safe_json_list(_get_col("user_tokens")),
            memory_domain=str(_get_col("memory_domain") or ""),
            llm_policy=str(_get_col("llm_policy") or ""),
            status=str(_get_col("status") or ""),
            attempts=_safe_json_dict(_get_col("attempts")),
            next_retry_at=(str(_get_col("next_retry_at")) if _get_col("next_retry_at") else None),
            last_error=_safe_json_dict(_get_col("last_error")),
            metrics=_safe_json_dict(_get_col("metrics")),
            created_at=str(_get_col("created_at") or ""),
            updated_at=str(_get_col("updated_at") or ""),
            cursor_committed=(str(_get_col("cursor_committed")) if _get_col("cursor_committed") else None),
            turns=_safe_json_list(_get_col("turns")),
            client_meta=_safe_json_dict(_get_col("client_meta")),
            stage2_marks=_safe_json_list(_get_col("stage2_marks")),
            stage2_pin_intents=_safe_json_list(_get_col("stage2_pin_intents")),
            payload_raw=(str(_get_col("payload_raw")) if _get_col("payload_raw") else None),
        )
