from __future__ import annotations

from typing import Dict
from datetime import datetime, timezone, timedelta

from modules.memory.contracts.memory_models import MemoryEntry


def _parse_iso8601(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


async def run_ttl_cleanup(vector_store, *, now: datetime | None = None) -> int:
    """Soft-delete entries whose TTL has expired (ttl>0 and created_at+ttl < now).

    Returns number of entries marked as deleted.
    """
    # vector_store is expected to provide a dump() of {id: MemoryEntry}
    cnt = 0
    now_dt = now or datetime.now(timezone.utc)
    if not hasattr(vector_store, "dump"):
        return 0
    entries: Dict[str, MemoryEntry] = vector_store.dump()  # type: ignore[assignment]
    for eid, entry in entries.items():
        md = entry.metadata or {}
        if md.get("is_deleted") is True:
            continue
        ttl = md.get("ttl")
        if not isinstance(ttl, int) or ttl <= 0:
            continue
        created_iso = md.get("created_at")
        cdt = _parse_iso8601(created_iso) if isinstance(created_iso, str) else None
        if not cdt:
            continue
        if cdt + timedelta(seconds=ttl) < now_dt:
            # mark soft deleted
            md["is_deleted"] = True
            md["deleted_at"] = now_dt.isoformat()
            entry.metadata = md
            await vector_store.upsert_vectors([entry])
            cnt += 1
    return cnt

