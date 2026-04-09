from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.memory.contracts.memory_models import Edge, MemoryEntry


"""
Dialog text pipeline v1 (benchmark-aligned).

Note: for strict compatibility with the LoCoMo benchmark artifacts, we keep:
- deterministic UUID namespaces (`locomo.*`)
- metadata.source = "locomo_text_pipeline"
"""


DEFAULT_DIALOG_V1_SOURCE = "locomo_text_pipeline"


IMPORTANCE_MAP: Dict[str, float] = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.8,
}


def generate_uuid(namespace: str, name: str) -> str:
    """Generate deterministic UUID (uuid5) using the same scheme as the benchmark pipeline."""
    ns = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
    return str(uuid.uuid5(ns, name))


def build_fact_uuid(
    *,
    sample_id: str,
    fact_idx: int,
    tenant_id: str | None = None,
    namespace_by_tenant: bool = False,
) -> str:
    logical_id = f"fact:{str(sample_id)}:{int(fact_idx)}"
    if namespace_by_tenant and str(tenant_id or "").strip():
        logical_id = f"tenant:{str(tenant_id).strip()}|{logical_id}"
    return generate_uuid("locomo.facts", logical_id)


def normalize_importance(value: Any) -> float:
    if value is None:
        return 0.5
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(IMPORTANCE_MAP.get(value.lower(), 0.5))
    return 0.5


def parse_dia_id(dia_id: str) -> Optional[Tuple[int, int]]:
    """Parse 'D8:6' -> (8, 6)."""
    match = re.match(r"^D(\d+):(\d+)$", (dia_id or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def normalize_dia_id(dia_id: str) -> str:
    """Normalize 'D1:3' -> 'D1_3' (benchmark compatibility)."""
    return (dia_id or "").replace(":", "_")


def make_event_id(sample_id: str, dia_id: str) -> str:
    """Build logical event id: 'conv-26_D1_3' (benchmark compatibility)."""
    return f"{sample_id}_{normalize_dia_id(dia_id)}"


def make_timeslice_id(sample_id: str, session_num: int) -> str:
    """Build logical timeslice id: 'conv-26_session_1_ts' (benchmark compatibility)."""
    return f"{sample_id}_session_{int(session_num)}_ts"


def parse_datetime(dt_str: str) -> Tuple[float, str]:
    """Parse LoCoMo date time: '1:56 pm on 8 May, 2023' -> (unix_ts, iso_string)."""
    try:
        match = re.match(
            r"(\d+):(\d+)\s*(am|pm)\s+on\s+(\d+)\s+(\w+),?\s*(\d{4})",
            (dt_str or "").strip(),
            re.I,
        )
        if match:
            hour, minute, ampm, day, month_name, year = match.groups()
            hour_i = int(hour)
            if ampm.lower() == "pm" and hour_i != 12:
                hour_i += 12
            elif ampm.lower() == "am" and hour_i == 12:
                hour_i = 0

            months = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }
            month_i = months.get(month_name.lower(), 1)
            dt = datetime(int(year), month_i, int(day), hour_i, int(minute))
            return dt.timestamp(), dt.isoformat()
    except Exception:
        pass

    default_dt = datetime(2023, 5, 1, 12, 0)
    return default_dt.timestamp(), default_dt.isoformat()


@dataclass(frozen=True)
class DialogEventRecord:
    """Minimal shape for benchmark step1 event record (kept loose for compatibility)."""

    id: str
    text: str
    timestamp_iso: str
    timeslice: str
    user_id: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DialogTimeSliceRecord:
    """Minimal shape for benchmark step1 timeslice record (kept loose for compatibility)."""

    id: str
    label: str
    start_iso: str
    end_iso: str | None
    user_id: str


def event_record_to_entry(
    event: Dict[str, Any], *, tenant_id: str, source: str = DEFAULT_DIALOG_V1_SOURCE
) -> Tuple[MemoryEntry, str]:
    """Convert benchmark step1 event dict -> MemoryEntry(kind=episodic)."""
    logical_id = str(event.get("id") or "")
    event_uuid = generate_uuid("locomo.events", logical_id)
    meta = dict(event.get("metadata") or {})
    entry = MemoryEntry(
        kind="episodic",
        modality="text",
        contents=[str(event.get("text") or "")],
        id=event_uuid,
        metadata={
            "tenant_id": str(tenant_id),
            "memory_domain": "dialog",
            "source": source,
            "event_id": logical_id,
            "timestamp_iso": event.get("timestamp_iso"),
            "timeslice": event.get("timeslice"),
            "speaker": meta.get("speaker"),
            "session": meta.get("session"),
            "turn": meta.get("turn"),
            "sample_id": meta.get("sample_id"),
            "dia_id": meta.get("dia_id"),
            "user_id": event.get("user_id"),
        },
    )
    return entry, event_uuid


def timeslice_record_to_entry(
    ts: Dict[str, Any], *, tenant_id: str, source: str = DEFAULT_DIALOG_V1_SOURCE
) -> Tuple[MemoryEntry, str]:
    """Convert benchmark step1 timeslice dict -> MemoryEntry(kind=semantic, node_type=timeslice)."""
    logical_id = str(ts.get("id") or "")
    ts_uuid = generate_uuid("locomo.timeslices", logical_id)
    label = ts.get("label", f"Session {logical_id}")
    entry = MemoryEntry(
        kind="semantic",
        modality="text",
        contents=[str(label)],
        id=ts_uuid,
        metadata={
            "tenant_id": str(tenant_id),
            "memory_domain": "dialog",
            "source": source,
            "node_type": "timeslice",
            "timeslice_id": logical_id,
            "start_iso": ts.get("start_iso"),
            "end_iso": ts.get("end_iso"),
            "user_id": ts.get("user_id"),
        },
    )
    return entry, ts_uuid


def fact_item_to_entry(
    fact: Dict[str, Any],
    *,
    fact_idx: int,
    tenant_id: str,
    user_prefix: str = "locomo_user_",
    source: str = DEFAULT_DIALOG_V1_SOURCE,
) -> Tuple[Optional[MemoryEntry], Optional[str]]:
    """Convert benchmark FactItem dict -> MemoryEntry(kind=semantic). Returns (entry, uuid) or (None, None) if skipped."""
    op = str(fact.get("op", "ADD")).upper()
    if op in ("KEEP", "DELETE"):
        return None, None
    statement = str(fact.get("statement") or "").strip()
    if not statement:
        return None, None

    sample_id = str(fact.get("source_sample_id") or fact.get("sample_id") or "").strip()
    user_id = f"{user_prefix}{sample_id}" if sample_id else None
    fact_uuid = build_fact_uuid(sample_id=sample_id, fact_idx=int(fact_idx), tenant_id=str(tenant_id))

    entry = MemoryEntry(
        kind="semantic",
        modality="text",
        contents=[statement],
        id=fact_uuid,
        metadata={
            "tenant_id": str(tenant_id),
            "memory_domain": "dialog",
            "source": source,
            "fact_type": fact.get("type") or fact.get("fact_type") or "fact",
            "scope": fact.get("scope") or "permanent",
            "status": fact.get("status") or "n/a",
            "importance": normalize_importance(fact.get("importance")),
            "source_sample_id": sample_id,
            "source_turn_ids": fact.get("source_turn_ids") or [],
            "rationale": fact.get("rationale"),
            "user_id": user_id,
            "speaker": fact.get("speaker"),
            "mentions": fact.get("mentions") or [],
            "temporal_grounding": fact.get("temporal_grounding") or [],
        },
    )
    return entry, fact_uuid


def build_entries_and_links(
    *,
    events_raw: List[Dict[str, Any]],
    facts_raw: List[Dict[str, Any]],
    tenant_id: str,
    user_prefix: str = "locomo_user_",
    source: str = DEFAULT_DIALOG_V1_SOURCE,
) -> Tuple[List[MemoryEntry], List[Edge]]:
    """Build merged entries + links, matching benchmark/scripts/step3_build_graph.py semantics."""
    events = [e for e in events_raw if e.get("kind") == "event"]
    timeslices = [e for e in events_raw if e.get("kind") == "timeslice"]

    # (sample_id, dia_id) -> event_logical_id
    event_index: Dict[Tuple[str, str], str] = {}
    for e in events:
        meta = e.get("metadata") or {}
        sample_id = str(meta.get("sample_id") or "").strip()
        dia_id = str(meta.get("dia_id") or "").strip()
        if sample_id and dia_id:
            event_index[(sample_id, dia_id)] = str(e.get("id"))

    # (sample_id, session_num) -> timeslice_logical_id
    ts_index: Dict[Tuple[str, int], str] = {}
    ts_pattern = re.compile(r"^(.+)_session_(\d+)_ts$")
    for ts in timeslices:
        ts_id = str(ts.get("id") or "")
        match = ts_pattern.match(ts_id)
        if match:
            ts_index[(match.group(1), int(match.group(2)))] = ts_id

    entries: List[MemoryEntry] = []
    links: List[Edge] = []

    ts_uuid_map: Dict[str, str] = {}
    event_uuid_map: Dict[str, str] = {}

    # 1) TimeSlices
    for ts in timeslices:
        ent, ts_uuid = timeslice_record_to_entry(ts, tenant_id=tenant_id, source=source)
        ts_uuid_map[str(ts.get("id") or "")] = ts_uuid
        entries.append(ent)

    # 2) Events + OCCURS_AT edges
    for ev in events:
        ent, ev_uuid = event_record_to_entry(ev, tenant_id=tenant_id, source=source)
        event_uuid_map[str(ev.get("id") or "")] = ev_uuid
        entries.append(ent)

        ts_logical_id = ev.get("timeslice")
        if ts_logical_id and ts_logical_id in ts_uuid_map:
            links.append(Edge(src_id=ev_uuid, dst_id=ts_uuid_map[str(ts_logical_id)], rel_type="OCCURS_AT", weight=1.0))

    # 3) Facts + REFERENCES/PART_OF edges
    for i, fact in enumerate(facts_raw):
        ent, fact_uuid = fact_item_to_entry(
            fact,
            fact_idx=i,
            tenant_id=tenant_id,
            user_prefix=user_prefix,
            source=source,
        )
        if ent is None or fact_uuid is None:
            continue
        entries.append(ent)

        sample_id = str(ent.metadata.get("source_sample_id") or "").strip()
        turn_ids = list(ent.metadata.get("source_turn_ids") or [])

        sessions_for_fact: Set[int] = set()
        for turn_id in turn_ids:
            parsed = parse_dia_id(str(turn_id))
            if not parsed:
                continue
            session_num, _turn_num = parsed
            sessions_for_fact.add(session_num)

            event_logical_id = event_index.get((sample_id, str(turn_id)))
            if event_logical_id and event_logical_id in event_uuid_map:
                links.append(
                    Edge(
                        src_id=fact_uuid,
                        dst_id=event_uuid_map[event_logical_id],
                        rel_type="REFERENCES",
                        weight=1.0,
                    )
                )

        for session_num in sessions_for_fact:
            ts_logical_id = ts_index.get((sample_id, session_num))
            if ts_logical_id and ts_logical_id in ts_uuid_map:
                links.append(
                    Edge(
                        src_id=fact_uuid,
                        dst_id=ts_uuid_map[ts_logical_id],
                        rel_type="PART_OF",
                        weight=1.0,
                    )
                )

    return entries, links
