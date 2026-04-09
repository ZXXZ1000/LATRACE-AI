from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import sqlite3
import threading
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


class IngestJobStore:
    """SQLite-backed ingest job store (V0).

    - Persisted by default (sqlite file).
    - Stores turns for retry/replay.
    - Supports commit_id idempotency per session.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        cfg = settings or {}
        path = cfg.get("sqlite_path") or "modules/memory/outputs/ingest_jobs.db"
        self._path = str(path)
        if self._path != ":memory:":
            base_dir = os.path.dirname(os.path.abspath(self._path))
            if base_dir and not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_session_state (
                session_id TEXT PRIMARY KEY,
                latest_job_id TEXT,
                latest_status TEXT,
                cursor_committed TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_commit_index (
                session_id TEXT NOT NULL,
                commit_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                PRIMARY KEY (session_id, commit_id)
            )
            """
        )
        self._conn.commit()
        self._ensure_columns(
            "ingest_jobs",
            {
                "stage2_marks": "TEXT",
                "stage2_pin_intents": "TEXT",
                "api_key_id": "TEXT",
                "request_id": "TEXT",
                "payload_raw": "TEXT",
            },
        )

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        for name, col_type in columns.items():
            if name in existing:
                continue
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
        self._conn.commit()

    def create_job(
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
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        cid = str(commit_id or "").strip() or None
        with self._lock:
            def _payload_has_core(job: IngestJobRecord) -> bool:
                if job.user_tokens and any(str(x).strip() for x in job.user_tokens):
                    pass
                else:
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

            if cid:
                existing = self._get_job_id_by_commit(sid, cid)
                if existing:
                    record = self.get_job(existing)
                    if record is not None:
                        if record.status != "COMPLETED" and not _payload_has_core(record):
                            cur = self._conn.cursor()
                            cur.execute("DELETE FROM ingest_jobs WHERE job_id=?", (str(record.job_id),))
                            cur.execute(
                                "DELETE FROM ingest_commit_index WHERE session_id=? AND commit_id=?",
                                (sid, cid),
                            )
                            self._conn.commit()
                        else:
                            return record, False

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            now = _now_iso()
            cursor = _max_turn_id(turns) or (str(base_turn_id).strip() if base_turn_id else None)
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
                cursor_committed=cursor,
                turns=list(turns),
                client_meta=dict(client_meta or {}),
                payload_raw=(str(payload_raw) if payload_raw is not None else None),
            )
            self._upsert_job(record)
            if cid:
                self._insert_commit_index(sid, cid, job_id)
            self._upsert_session_state(sid, record.job_id, record.status, cursor)
            return record, True

    def get_job(self, job_id: str) -> Optional[IngestJobRecord]:
        with self._lock:
            cur = self._conn.cursor()
            columns = self._get_columns(cur)
            cur.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (str(job_id),))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_record(row, columns)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT session_id, latest_job_id, latest_status, cursor_committed FROM ingest_session_state WHERE session_id=?",
                (str(session_id),),
            )
            row = cur.fetchone()
            if not row:
                return {}
            return {"latest_job_id": row[1], "latest_status": row[2], "cursor_committed": row[3]}

    def list_jobs_by_status(self, statuses: List[str]) -> List[IngestJobRecord]:
        if not statuses:
            return []
        with self._lock:
            qs = ",".join("?" for _ in statuses)
            cur = self._conn.cursor()
            columns = self._get_columns(cur)
            cur.execute(f"SELECT * FROM ingest_jobs WHERE status IN ({qs})", tuple(statuses))
            return [self._row_to_record(row, columns) for row in cur.fetchall()]

    def update_status(
        self,
        job_id: str,
        *,
        status: str,
        stage: Optional[str] = None,
        error: Optional[Dict[str, Any]] = None,
        next_retry_at: Optional[str] = None,
        metrics_patch: Optional[Dict[str, Any]] = None,
        attempt_inc: bool = False,
    ) -> None:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return
            job.status = str(status)
            job.updated_at = _now_iso()
            if attempt_inc and stage:
                job.attempts[stage] = int(job.attempts.get(stage, 0)) + 1
            if error:
                job.last_error = dict(error)
            if next_retry_at is not None:
                job.next_retry_at = next_retry_at
            if metrics_patch:
                job.metrics.update(metrics_patch)
            self._upsert_job(job)
            self._upsert_session_state(job.session_id, job.job_id, job.status, job.cursor_committed)

    def update_stage2(
        self,
        job_id: str,
        *,
        marks: List[Dict[str, Any]],
        pin_intents: List[Dict[str, Any]],
        metrics_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return
            job.stage2_marks = list(marks or [])
            job.stage2_pin_intents = list(pin_intents or [])
            if metrics_patch:
                job.metrics.update(metrics_patch)
            job.updated_at = _now_iso()
            self._upsert_job(job)

    def _get_job_id_by_commit(self, session_id: str, commit_id: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT job_id FROM ingest_commit_index WHERE session_id=? AND commit_id=?",
            (str(session_id), str(commit_id)),
        )
        row = cur.fetchone()
        if not row:
            return None
        return str(row[0])

    def _insert_commit_index(self, session_id: str, commit_id: str, job_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO ingest_commit_index(session_id, commit_id, job_id) VALUES (?,?,?)",
            (str(session_id), str(commit_id), str(job_id)),
        )
        self._conn.commit()

    def _upsert_session_state(self, session_id: str, job_id: str, status: str, cursor_committed: Optional[str]) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO ingest_session_state(session_id, latest_job_id, latest_status, cursor_committed)
            VALUES (?,?,?,?)
            """,
            (str(session_id), str(job_id), str(status), cursor_committed),
        )
        self._conn.commit()

    def _upsert_job(self, record: IngestJobRecord) -> None:
        cur = self._conn.cursor()
        cur.execute(
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
        self._conn.commit()

    def _get_columns(self, cur: sqlite3.Cursor) -> List[str]:
        cur.execute("PRAGMA table_info(ingest_jobs)")
        return [row[1] for row in cur.fetchall()]

    def _row_to_record(self, row: Tuple[Any, ...], columns: List[str]) -> IngestJobRecord:
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
