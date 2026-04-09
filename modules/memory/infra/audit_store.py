from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import sqlite3
import json


class AuditStore:
    """SQLite-backed audit/history store.

    - Default uses in-memory DB (suitable for tests). Provide settings={"sqlite_path": "/path/to/audit.db"}
      to persist on disk.
    - Schema: events(version TEXT PRIMARY KEY, event TEXT, obj_id TEXT, payload TEXT, reason TEXT, created_at TEXT)
    """

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        path = self.settings.get("sqlite_path", ":memory:")
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                version TEXT PRIMARY KEY,
                event TEXT NOT NULL,
                obj_id TEXT,
                payload TEXT,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    async def add_batch(self, event: str, entries: List[Any]) -> str:
        version = f"v-{event}-batch"
        payload = json.dumps({"entries": [self._safe_dump(e) for e in entries]}, ensure_ascii=False)
        self._insert(version, event, None, payload, None)
        return version

    async def add_one(self, event: str, obj_id: str, payload: Dict[str, Any], *, reason: str | None = None) -> str:
        version = f"v-{event}-{obj_id}"
        self._insert(version, event, obj_id, json.dumps(payload, ensure_ascii=False), reason)
        return version

    def _insert(self, version: str, event: str, obj_id: Optional[str], payload: str, reason: Optional[str]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO events(version, event, obj_id, payload, reason, created_at) VALUES (?,?,?,?,?,?)",
            (version, event, obj_id, payload, reason, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_event(self, version: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT version, event, obj_id, payload, reason, created_at FROM events WHERE version=?", (version,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "version": row[0],
            "event": row[1],
            "obj_id": row[2],
            "payload": json.loads(row[3]) if row[3] else None,
            "reason": row[4],
            "created_at": row[5],
        }

    def list_events_for_obj(self, obj_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT version, event, obj_id, payload, reason, created_at FROM events WHERE obj_id=? ORDER BY created_at ASC",
            (obj_id,),
        )
        out = []
        for row in cur.fetchall():
            out.append(
                {
                    "version": row[0],
                    "event": row[1],
                    "obj_id": row[2],
                    "payload": json.loads(row[3]) if row[3] else None,
                    "reason": row[4],
                    "created_at": row[5],
                }
            )
        return out

    @staticmethod
    def _safe_dump(e: Any) -> Any:
        try:
            # pydantic BaseModel
            return e.model_dump()
        except Exception:
            try:
                return dict(e)
            except Exception:
                return str(e)


# Alias for backwards compatibility - AuditStore defaults to in-memory DB
InMemAuditStore = AuditStore
