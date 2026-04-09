from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
import asyncio
import json
import os
import sqlite3
import threading
import uuid

import httpx


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UsageWALSettings:
    enabled: bool
    sqlite_path: str
    sink_url: str
    sink_auth_internal_key: Optional[str]
    sink_auth_internal_header: str
    sink_auth_authorization: Optional[str]
    flush_interval_s: float
    batch_size: int
    timeout_s: float

    @classmethod
    def from_env(cls) -> "UsageWALSettings":
        enabled = str(os.getenv("MEMORY_USAGE_WAL_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}
        sqlite_path = os.getenv("MEMORY_USAGE_WAL_PATH", "modules/memory/outputs/usage_wal.db")
        sink_url = os.getenv("MEMORY_USAGE_SINK_URL", "")
        internal_key = (
            os.getenv("MEMORY_USAGE_SINK_INTERNAL_KEY")
            or os.getenv("MEMORY_INTERNAL_KEY")
            or os.getenv("MEMA_INTERNAL_KEY")
            or ""
        ).strip()
        internal_header = str(os.getenv("MEMORY_USAGE_SINK_INTERNAL_HEADER", "X-Internal-Key") or "X-Internal-Key").strip()
        authorization = str(os.getenv("MEMORY_USAGE_SINK_AUTHORIZATION", "") or "").strip() or None
        flush_interval_s = float(os.getenv("MEMORY_USAGE_FLUSH_INTERVAL_S", "5.0") or 5.0)
        batch_size = int(os.getenv("MEMORY_USAGE_BATCH_SIZE", "100") or 100)
        timeout_s = float(os.getenv("MEMORY_USAGE_TIMEOUT_S", "5.0") or 5.0)
        return cls(
            enabled=bool(enabled and sink_url),
            sqlite_path=str(sqlite_path),
            sink_url=str(sink_url),
            sink_auth_internal_key=(internal_key if internal_key else None),
            sink_auth_internal_header=(internal_header if internal_header else "X-Internal-Key"),
            sink_auth_authorization=authorization,
            flush_interval_s=max(0.1, flush_interval_s),
            batch_size=max(1, batch_size),
            timeout_s=max(0.5, timeout_s),
        )


class UsageWAL:
    def __init__(self, settings: UsageWALSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        path = settings.sqlite_path
        if path != ":memory:":
            base_dir = os.path.dirname(os.path.abspath(path))
            if base_dir and not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_wal (
                event_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_wal_status ON usage_wal(status)")
        self._conn.commit()

    def append(self, payload: Dict[str, Any]) -> str:
        event_id = str(payload.get("event_id") or f"evt_{uuid.uuid4().hex}")
        payload["event_id"] = event_id
        now = _now_iso()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO usage_wal(event_id, payload, status, attempts, last_error, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    json.dumps(payload, ensure_ascii=False),
                    "pending",
                    0,
                    None,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return event_id

    def _list_pending(self, limit: int) -> List[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT event_id, payload FROM usage_wal WHERE status='pending' ORDER BY created_at ASC LIMIT ?",
                (int(limit),),
            )
            rows = cur.fetchall()
        out: List[Tuple[str, Dict[str, Any]]] = []
        for event_id, payload in rows:
            try:
                data = json.loads(payload) if payload else {}
            except Exception:
                data = {}
            out.append((str(event_id), data))
        return out

    def _mark_sent(self, event_ids: Iterable[str]) -> None:
        ids = [str(x) for x in event_ids if str(x)]
        if not ids:
            return
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"DELETE FROM usage_wal WHERE event_id IN ({','.join('?' for _ in ids)})",
                tuple(ids),
            )
            self._conn.commit()

    def _mark_failed(self, event_ids: Iterable[str], error: str) -> None:
        ids = [str(x) for x in event_ids if str(x)]
        if not ids:
            return
        now = _now_iso()
        with self._lock:
            cur = self._conn.cursor()
            for event_id in ids:
                cur.execute(
                    "UPDATE usage_wal SET attempts=attempts+1, last_error=?, updated_at=? WHERE event_id=?",
                    (error[:200], now, event_id),
                )
            self._conn.commit()

    async def flush_once(self) -> int:
        if not self.settings.enabled:
            return 0
        pending = self._list_pending(self.settings.batch_size)
        if not pending:
            return 0
        event_ids = [eid for eid, _ in pending]
        payload = {"events": [item for _, item in pending]}
        headers = {"Content-Type": "application/json"}
        if self.settings.sink_auth_internal_key:
            headers[str(self.settings.sink_auth_internal_header or "X-Internal-Key")] = str(
                self.settings.sink_auth_internal_key
            )
        if self.settings.sink_auth_authorization:
            headers["Authorization"] = str(self.settings.sink_auth_authorization)
        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_s) as client:
                resp = await client.post(self.settings.sink_url, json=payload, headers=headers)
            if 200 <= resp.status_code < 300:
                self._mark_sent(event_ids)
                return len(event_ids)
            self._mark_failed(event_ids, f"http_{resp.status_code}")
            return 0
        except Exception as exc:
            self._mark_failed(event_ids, f"{type(exc).__name__}: {exc}")
            return 0

    async def run_flush_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.flush_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.settings.flush_interval_s)
            except asyncio.TimeoutError:
                continue
