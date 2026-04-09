from __future__ import annotations

from dataclasses import dataclass
import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.application.config import load_memory_config, get_dialog_event_settings, get_graph_settings
from modules.memory.ports.memory_port import MemoryPort
from modules.memory.domain.dialog_text_pipeline_v1 import (
    build_fact_uuid,
    generate_uuid,
    fact_item_to_entry,
)


UnifiedExtractor = Callable[[List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]


@dataclass(frozen=True)
class SessionWriteResult:
    status: str  # ok | skipped_existing | failed
    session_id: str
    version: str | None
    marker_id: str
    written_entries: int
    written_links: int
    deleted_fact_ids: int
    trace: Dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_entries_published(entries: Sequence[MemoryEntry], published: bool) -> None:
    for e in entries or []:
        try:
            e.published = bool(published)
        except Exception:
            continue


def _set_graph_published(req: GraphUpsertRequest, published: bool) -> None:
    for seq in (
        req.segments,
        req.evidences,
        getattr(req, "utterances", []),
        req.entities,
        req.events,
        req.places,
        req.time_slices,
        getattr(req, "regions", []),
        getattr(req, "states", []),
        getattr(req, "knowledge", []),
    ):
        for node in seq or []:
            try:
                node.published = bool(published)
            except Exception:
                continue
    for edge in req.edges or []:
        try:
            edge.published = bool(published)
        except Exception:
            continue


def _collect_graph_node_ids(req: GraphUpsertRequest) -> List[str]:
    ids: List[str] = []
    for seq in (
        req.segments,
        req.evidences,
        getattr(req, "utterances", []),
        req.entities,
        req.events,
        req.places,
        req.time_slices,
        getattr(req, "regions", []),
        getattr(req, "states", []),
        getattr(req, "knowledge", []),
    ):
        for node in seq or []:
            nid = getattr(node, "id", None)
            if nid:
                ids.append(str(nid))
    return ids


def _collect_graph_vector_ids(req: GraphUpsertRequest) -> List[str]:
    """Collect vector entry IDs assigned during GraphService upsert.

    These IDs are required to publish TKG vectors written via GraphService
    (Event/Entity vectors) so they become visible to retrieval.
    """
    ids: List[str] = []
    # Event vectors
    for ev in req.events or []:
        vid = getattr(ev, "text_vector_id", None)
        if vid:
            ids.append(str(vid))
        vid = getattr(ev, "clip_vector_id", None)
        if vid:
            ids.append(str(vid))
    # Entity vectors
    for ent in req.entities or []:
        vid = getattr(ent, "text_vector_id", None)
        if vid:
            ids.append(str(vid))
        vid = getattr(ent, "face_vector_id", None)
        if vid:
            ids.append(str(vid))
        vid = getattr(ent, "voice_vector_id", None)
        if vid:
            ids.append(str(vid))
    # De-dup while preserving order
    return list(dict.fromkeys([x for x in ids if str(x).strip()]))


def _build_turn_mark_maps(
    turns: Sequence[Dict[str, Any]],
    turn_marks: Optional[Sequence[Dict[str, Any]]],
) -> tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    marks_by_id: Dict[str, Dict[str, Any]] = {}
    if turn_marks:
        for m in turn_marks:
            if not isinstance(m, dict):
                continue
            tid = str(m.get("turn_id") or "").strip()
            if tid:
                marks_by_id[tid] = dict(m)
    marks_by_index: Dict[int, Dict[str, Any]] = {}
    for idx, t in enumerate(turns or [], start=1):
        tid = str(t.get("turn_id") or t.get("dia_id") or "").strip()
        if tid and tid in marks_by_id:
            marks_by_index[idx] = marks_by_id[tid]
    return marks_by_index, marks_by_id


def _parse_turn_index_from_source_id(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value) if value > 0 else None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("t") and s[1:].isdigit():
        try:
            n = int(s[1:])
            return n if n > 0 else None
        except Exception:
            return None
    if ":" in s:
        try:
            tail = s.split(":")[-1]
            n = int(tail)
            return n if n > 0 else None
        except Exception:
            return None
    if "_" in s:
        try:
            tail = s.split("_")[-1]
            n = int(tail)
            return n if n > 0 else None
        except Exception:
            return None
    try:
        n = int(s)
        return n if n > 0 else None
    except Exception:
        return None


def _merge_ttl_seconds(ttls: Sequence[Optional[float]]) -> Optional[int]:
    vals = [float(x) for x in ttls if isinstance(x, (int, float))]
    if not vals:
        return None
    positives = [v for v in vals if v > 0]
    if positives:
        return int(min(positives))
    return 0


def _merge_forget_policy(policies: Sequence[Optional[str]]) -> Optional[str]:
    vals = {str(x).strip() for x in policies if x is not None and str(x).strip()}
    if "until_changed" in vals:
        return "until_changed"
    if "permanent" in vals:
        return "permanent"
    if vals:
        return "temporary"
    return None


def _parse_ttl_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"discard", "drop", "none"}:
        return None
    if s.isdigit():
        return int(s)
    unit = s[-1]
    try:
        num = float(s[:-1])
    except Exception:
        return None
    if unit == "s":
        return int(num)
    if unit == "m":
        return int(num * 60)
    if unit == "h":
        return int(num * 3600)
    if unit == "d":
        return int(num * 86400)
    return None


def _apply_event_policy(
    events_raw: Sequence[Dict[str, Any]],
    *,
    settings: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    out: List[Dict[str, Any]] = []
    stats = {"mapped": 0, "weak": 0, "unmapped": 0, "dropped": 0}
    min_conf = float(settings.get("min_event_confidence", 0.4))
    ttl_mapped = settings.get("ttl_mapped", 0)
    ttl_weak = settings.get("ttl_weak", "7d")
    ttl_unmapped = settings.get("ttl_unmapped", "discard")
    supported_by_topk = int(settings.get("supported_by_topk", 5))
    for ev in events_raw or []:
        if not isinstance(ev, dict):
            continue
        status = str(ev.get("evidence_status") or "").strip().lower() or "mapped"
        if status not in {"mapped", "weak", "unmapped"}:
            status = "mapped"
        ev_conf = None
        try:
            ev_conf = float(ev.get("event_confidence")) if ev.get("event_confidence") is not None else None
        except Exception:
            ev_conf = None
        if ev_conf is not None and ev_conf < min_conf:
            stats["dropped"] += 1
            continue
        if status == "unmapped":
            stats["unmapped"] += 1
            if str(ttl_unmapped).strip().lower() in {"discard", "drop"}:
                stats["dropped"] += 1
                continue
        elif status == "weak":
            stats["weak"] += 1
        else:
            stats["mapped"] += 1
        ttl_val = ttl_mapped if status == "mapped" else ttl_weak if status == "weak" else ttl_unmapped
        ttl_seconds = _parse_ttl_seconds(ttl_val)
        if ttl_seconds is not None:
            ev["ttl_seconds"] = ttl_seconds
        src_ids = list(ev.get("source_turn_ids") or [])
        if supported_by_topk > 0 and len(src_ids) > supported_by_topk:
            ev["supported_turn_ids"] = src_ids[:supported_by_topk]
        else:
            ev["supported_turn_ids"] = list(src_ids)
        out.append(ev)
    return out, stats


def _apply_marks_to_facts(
    facts_raw: Sequence[Dict[str, Any]],
    marks_by_index: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for fact in facts_raw or []:
        if not isinstance(fact, dict):
            continue
        f = dict(fact)
        source_turn_ids = list(f.get("source_turn_ids") or [])
        marks: List[Dict[str, Any]] = []
        for stid in source_turn_ids:
            idx = _parse_turn_index_from_source_id(stid)
            if idx is None:
                continue
            mark = marks_by_index.get(idx)
            if mark:
                marks.append(mark)
        if marks:
            ttl_vals = [m.get("ttl_seconds") for m in marks]
            ttl_seconds = _merge_ttl_seconds(ttl_vals)
            policies = [m.get("forget_policy") for m in marks]
            forget_policy = _merge_forget_policy(policies)
            # importance: keep explicit fact importance if larger
            try:
                cur_imp = float(f.get("importance")) if f.get("importance") is not None else None
            except Exception:
                cur_imp = None
            mark_imp_vals = []
            for m in marks:
                try:
                    mark_imp_vals.append(float(m.get("importance")))
                except Exception:
                    continue
            if mark_imp_vals:
                max_mark_imp = max(mark_imp_vals)
                if cur_imp is None or max_mark_imp > cur_imp:
                    f["importance"] = max_mark_imp
            if ttl_seconds is not None:
                f["ttl_seconds"] = ttl_seconds
            if forget_policy:
                f["forget_policy"] = forget_policy
        out.append(f)
    return out


def _normalize_user_tokens(user_tokens: Sequence[str]) -> List[str]:
    out = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not out:
        raise ValueError("user_tokens must be non-empty")
    # stable order for marker id
    return list(sorted(dict.fromkeys(out)))


def _normalize_turns(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        text = str(t.get("text") or t.get("content") or "").strip()
        if not text:
            continue
        blip_caption = t.get("blip_caption")
        if blip_caption:
            cap = str(blip_caption).strip()
            if cap and "[Image:" not in text:
                # Keep compatibility with benchmark step1_convert_events.py formatting.
                text = f"{text} [Image: {cap}]"
        dia_id = t.get("dia_id")
        turn_id = t.get("turn_id")
        if not dia_id and turn_id:
            dia_id = turn_id
        speaker = t.get("speaker")
        role = t.get("role")
        if not speaker and role:
            speaker = str(role)
        session_idx = t.get("session_idx", t.get("session"))
        try:
            session_idx_norm = int(session_idx) if session_idx is not None else None
        except Exception:
            session_idx_norm = None
        session_date_time = t.get("session_date_time") or t.get("session_datetime") or t.get("session_ref_time")
        out.append(
            {
                "dia_id": (str(dia_id).strip() if dia_id else None),
                "turn_id": (str(turn_id).strip() if turn_id else None),
                "speaker": (str(speaker).strip() if speaker else "Unknown"),
                "text": text,
                "timestamp_iso": t.get("timestamp_iso"),
                "session_idx": session_idx_norm,
                "session_date_time": (str(session_date_time).strip() if session_date_time else None),
                "blip_caption": (str(blip_caption).strip() if blip_caption else None),
            }
        )
    if not out:
        raise ValueError("turns is empty after normalization")
    return out


def _parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _turn_id_for_turn(t: Dict[str, Any], idx: int) -> str:
    dia_id = str(t.get("dia_id") or "").strip()
    if dia_id:
        return dia_id
    turn_id = str(t.get("turn_id") or "").strip()
    if turn_id:
        return turn_id
    return f"t{idx}"


def _build_turn_time_map(turns: Sequence[Dict[str, Any]]) -> Dict[str, datetime]:
    out: Dict[str, datetime] = {}
    for idx, t in enumerate(turns or [], start=1):
        tid = _turn_id_for_turn(t, idx)
        dt = _parse_iso_dt(t.get("timestamp_iso"))
        if dt is not None:
            out[tid] = dt
    return out


def _resolve_state_subject_id(subject_ref: str, speaker_map: Dict[str, str]) -> Optional[str]:
    if not subject_ref:
        return None
    if subject_ref in speaker_map:
        return speaker_map.get(subject_ref)
    low = subject_ref.lower()
    if low in speaker_map:
        return speaker_map.get(low)
    # fallback: case-insensitive match
    for k, v in speaker_map.items():
        if str(k).lower() == low:
            return v
    return None


def _fill_turn_timestamps(
    turns: List[Dict[str, Any]],
    *,
    reference_time_iso: Optional[str],
    turn_interval_seconds: int,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    interval = max(1, int(turn_interval_seconds or 60))
    ref_dt = _parse_iso_dt(reference_time_iso)
    ref_idx: Optional[int] = None
    if ref_dt is None:
        for idx, t in enumerate(turns, start=1):
            ref_dt = _parse_iso_dt(t.get("timestamp_iso"))
            if ref_dt is not None:
                ref_idx = int(idx)
                break
    if ref_dt is None:
        ref_dt = datetime.now(timezone.utc)
        reference_time_iso = ref_dt.isoformat()
    elif reference_time_iso is None and ref_idx is not None:
        # Backfill from the first known turn timestamp to keep turn-1 aligned.
        ref_dt = ref_dt - timedelta(seconds=(ref_idx - 1) * interval)
        reference_time_iso = ref_dt.isoformat()
    else:
        reference_time_iso = ref_dt.isoformat()

    for idx, t in enumerate(turns, start=1):
        if not t.get("timestamp_iso"):
            t["timestamp_iso"] = (ref_dt + timedelta(seconds=(idx - 1) * interval)).isoformat()

    return turns, reference_time_iso


def _marker_search_filters(
    *,
    tenant_id: str,
    user_tokens: List[str],
    user_match: str,
    memory_domain: str,
    session_id: str,
) -> SearchFilters:
    return SearchFilters(
        tenant_id=str(tenant_id),
        user_id=list(user_tokens),
        user_match=("all" if str(user_match or "all").lower() != "any" else "any"),
        memory_domain=str(memory_domain),
        run_id=str(session_id),
        memory_type=["semantic"],
        modality=["text"],
        source=["dialog_session_marker"],
    )


def _make_marker_id(*, tenant_id: str, memory_domain: str, session_id: str, user_tokens: List[str]) -> str:
    # Make collisions across tenants/users impossible.
    key = f"tenant={tenant_id}|domain={memory_domain}|session={session_id}|users={'|'.join(user_tokens)}"
    return generate_uuid("memory.session_marker", key)


def _make_marker_entry(
    *,
    marker_id: str,
    tenant_id: str,
    user_tokens: List[str],
    memory_domain: str,
    session_id: str,
    status: str,
    overwrite_existing: bool,
    event_ids: List[str],
    timeslice_ids: List[str],
    fact_ids: List[str],
    written_links: int,
    version: Optional[str],
    error_reason: Optional[str],
    started_at: str,
    completed_at: Optional[str],
) -> MemoryEntry:
    md: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "user_id": list(user_tokens),
        "memory_domain": memory_domain,
        "run_id": session_id,
        "source": "dialog_session_marker",
        "node_type": "session_marker",
        "status": status,
        "overwrite_existing": bool(overwrite_existing),
        "event_ids": list(event_ids),
        "timeslice_ids": list(timeslice_ids),
        "fact_ids": list(fact_ids),
        "written_links": int(written_links),
        "version": version,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    if error_reason:
        md["error_reason"] = str(error_reason)[:500]
    return MemoryEntry(
        id=marker_id,
        kind="semantic",
        modality="text",
        contents=[f"session_marker {session_id}"],
        metadata=md,
    )


async def session_write(
    store: MemoryPort,
    *,
    tenant_id: str,
    user_tokens: Sequence[str],
    session_id: str,
    turns: Sequence[Dict[str, Any]],
    memory_domain: str = "dialog",
    user_match: str = "all",
    overwrite_existing: bool = False,
    extract: bool = True,
    write_events: bool = True,
    write_facts: bool = True,
    graph_upsert: bool = True,
    graph_policy: str = "best_effort",  # require | best_effort
    llm_policy: str = "require",  # require | best_effort
    tkg_extractor: Optional[UnifiedExtractor] = None,
    reference_time_iso: Optional[str] = None,
    turn_interval_seconds: int = 60,
    extra_facts: Optional[Sequence[Dict[str, Any]]] = None,
    turn_marks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Dialogue session ingestion (TKG-first): turns -> unified LLM extraction (events + knowledge) -> TKG graph + vector indexes.

    Key properties:
    - session_id idempotency via session_marker
    - overwrite_existing deletes stale fact/knowledge index entries (vector-side)
    - TKG is the primary graph: write via graph_upsert_v0
    - vector writes are limited to:
      - fact/knowledge index entries (semantic, benchmark-compatible for fact_search)
      - utterance index entries (semantic, source=tkg_dialog_utterance_index_v1)
    """
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    user_tokens_norm = _normalize_user_tokens(user_tokens)
    turns_norm = _normalize_turns(turns)
    turns_norm, reference_time_iso = _fill_turn_timestamps(
        list(turns_norm),
        reference_time_iso=reference_time_iso,
        turn_interval_seconds=int(turn_interval_seconds),
    )
    marks_by_index, marks_by_id = _build_turn_mark_maps(turns_norm, turn_marks)

    marker_id = _make_marker_id(
        tenant_id=str(tenant_id),
        memory_domain=str(memory_domain),
        session_id=str(session_id),
        user_tokens=user_tokens_norm,
    )

    t_total0 = time.perf_counter()
    timing_ms: Dict[str, int] = {}
    trace: Dict[str, Any] = {
        "tenant_id": str(tenant_id),
        "memory_domain": str(memory_domain),
        "session_id": str(session_id),
        "overwrite_existing": bool(overwrite_existing),
        "extract": bool(extract),
        "write_events": bool(write_events),
        "write_facts": bool(write_facts),
        "graph_upsert": bool(graph_upsert),
        "graph_policy": str(graph_policy),
    }
    if marks_by_id:
        trace["turn_marks"] = len(marks_by_id)
    publish_supported = callable(getattr(store, "publish_entries", None))
    trace["publish_supported"] = publish_supported
    tenant_scoped_fact_ids = False
    try:
        cfg = load_memory_config()
        memory_cfg = cfg.get("memory", {}) or {}
        vector_cfg = memory_cfg.get("vector_store", {}) or {}
        shard_cfg = vector_cfg.get("sharding", {}) or {}
        tenant_scoped_fact_ids = bool(shard_cfg.get("enabled")) and bool(shard_cfg.get("namespace_ids_by_tenant"))
    except Exception:
        tenant_scoped_fact_ids = False
    trace["tenant_scoped_fact_ids"] = bool(tenant_scoped_fact_ids)

    # 1) Idempotency check via marker
    marker_filters = _marker_search_filters(
        tenant_id=str(tenant_id),
        user_tokens=user_tokens_norm,
        user_match=user_match,
        memory_domain=str(memory_domain),
        session_id=str(session_id),
    )
    marker_md: Dict[str, Any] = {}
    marker_status = ""
    # Prefer a strict by-id get (if available) over ANN search: markers must never match across sessions.
    marker_get_fn = getattr(store, "get", None)
    if callable(marker_get_fn):
        try:
            marker_entry = await marker_get_fn(str(marker_id))
            trace["marker_lookup"] = "get"
            if marker_entry is not None:
                marker_md = dict(marker_entry.metadata or {})
                marker_status = str(marker_md.get("status") or "")
            else:
                trace["marker_lookup"] = "get_miss"
        except Exception as e:
            trace["marker_get_error"] = str(e)[:200]
            trace["marker_lookup"] = "get_error"
    else:
        try:
            marker_res = await store.search(
                f"session_marker {session_id}",
                topk=1,
                filters=marker_filters,
                expand_graph=False,
            )
        except Exception as e:
            marker_res = None
            trace["marker_search_error"] = str(e)[:200]

        marker_hit = (marker_res.hits[0] if (marker_res and marker_res.hits) else None)
        marker_md = dict(marker_hit.entry.metadata) if marker_hit else {}
        marker_status = str(marker_md.get("status") or "")
        trace["marker_lookup"] = "search"

    if marker_status == "completed" and not overwrite_existing:
        return SessionWriteResult(
            status="skipped_existing",
            session_id=str(session_id),
            version=str(marker_md.get("version") or "") or None,
            marker_id=marker_id,
            written_entries=0,
            written_links=0,
            deleted_fact_ids=0,
            trace={**trace, "skipped_reason": "marker_completed"},
        ).__dict__

    # 2) Overwrite: remember previous fact_ids; we will delete only the stale ones after a successful write.
    deleted_fact_ids = 0
    old_fact_ids: List[str] = []
    if marker_status == "completed" and overwrite_existing:
        old_fact_ids = [str(x) for x in (marker_md.get("fact_ids") or []) if str(x).strip()]
        trace["overwrite_old_fact_ids"] = len(old_fact_ids)

    # 3) Marker lifecycle: in_progress -> (write) -> completed/failed
    started_at = _now_iso()
    marker_in_progress = _make_marker_entry(
        marker_id=marker_id,
        tenant_id=str(tenant_id),
        user_tokens=user_tokens_norm,
        memory_domain=str(memory_domain),
        session_id=str(session_id),
        status="in_progress",
        overwrite_existing=overwrite_existing,
        event_ids=[],
        timeslice_ids=[],
        fact_ids=[],
        written_links=0,
        version=None,
        error_reason=None,
        started_at=started_at,
        completed_at=None,
    )

    try:
        graph_policy_norm = str(graph_policy or "best_effort").lower()
        if overwrite_existing and write_events:
            graph_store = getattr(store, "graph", None)
            if callable(getattr(graph_store, "upsert_graph_v0", None)):
                graph_policy_norm = "require"
        trace["graph_policy_effective"] = graph_policy_norm
        await store.write([marker_in_progress], links=None, upsert=True)
    except Exception as e:
        # Marker is best-effort; do not block the main write.
        trace["marker_write_in_progress_error"] = str(e)[:200]

    try:
        # 4) Unified TKG extraction (events + knowledge) via single LLM call series.
        facts_raw: List[Dict[str, Any]] = []
        events_raw: List[Dict[str, Any]] = []
        states_raw: List[Dict[str, Any]] = []
        facts_skipped_reason: Optional[str] = None
        events_skipped_reason: Optional[str] = None
        states_skipped_reason: Optional[str] = None
        event_stats: Dict[str, int] = {}

        if not extract:
            facts_skipped_reason = "extract_disabled"
            events_skipped_reason = "extract_disabled"
        elif not write_facts and not write_events:
            facts_skipped_reason = "write_facts_disabled"
            events_skipped_reason = "write_events_disabled"
        else:
            if tkg_extractor is None:
                try:
                    from modules.memory.application.dialog_tkg_unified_extractor_v1 import (
                        build_dialog_tkg_unified_extractor_v1_from_env,
                    )

                    tkg_extractor = build_dialog_tkg_unified_extractor_v1_from_env(
                        session_id=str(session_id),
                        reference_time_iso=reference_time_iso,
                    )
                    if tkg_extractor is not None:
                        trace["tkg_extractor"] = "dialog_tkg_unified_from_env"
                except Exception:
                    tkg_extractor = None

            if tkg_extractor is None:
                if str(llm_policy or "best_effort").lower() == "require":
                    raise RuntimeError("LLM TKG unified extractor is not configured (llm_policy=require).")
                facts_skipped_reason = "llm_missing"
                events_skipped_reason = "llm_missing"
            else:
                t_extract0 = time.perf_counter()
                out = await asyncio.to_thread(tkg_extractor, list(turns_norm))
                timing_ms["extract_ms"] = int((time.perf_counter() - t_extract0) * 1000)
                if isinstance(out, dict):
                    if write_events:
                        events_raw = list(out.get("events") or [])
                    if write_facts:
                        facts_raw = list(out.get("knowledge") or [])
                    states_raw = list(out.get("states") or [])
                else:
                    facts_raw = []
                    events_raw = []
                    states_raw = []

                if not write_facts:
                    facts_skipped_reason = "write_facts_disabled"
                if not write_events:
                    events_skipped_reason = "write_events_disabled"

        if facts_skipped_reason:
            trace["facts_skipped_reason"] = facts_skipped_reason
        if events_skipped_reason:
            trace["events_skipped_reason"] = events_skipped_reason
        if states_skipped_reason:
            trace["states_skipped_reason"] = states_skipped_reason

        if extra_facts:
            facts_raw.extend([dict(x) for x in extra_facts if isinstance(x, dict)])
        if marks_by_index:
            facts_raw = _apply_marks_to_facts(facts_raw, marks_by_index)

        if events_raw:
            settings = get_dialog_event_settings(load_memory_config())
            events_raw, event_stats = _apply_event_policy(events_raw, settings=settings)
            trace["events_count"] = len(events_raw)
            if event_stats:
                trace["event_stats"] = dict(event_stats)
        if states_raw:
            trace["states_count"] = len(states_raw)

        # 4.5) Prepare TKG graph build + utterance vector index entries (pure; no I/O).
        # Note: we upsert the graph BEFORE vector writes to avoid leaving a "vector-only" partial state
        # when graph_policy="require" and the graph write fails.
        tkg_build = None
        tkg_index_entries: List[MemoryEntry] = []
        tkg_index_ids: List[str] = []
        if graph_upsert and write_events:
            try:
                from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1
                from modules.memory.domain.dialog_tkg_vector_index_v1 import build_dialog_tkg_utterance_index_entries_v1

                t_build0 = time.perf_counter()
                tkg_build = build_dialog_graph_upsert_v1(
                    tenant_id=str(tenant_id),
                    session_id=str(session_id),
                    user_tokens=list(user_tokens_norm),
                    turns=list(turns_norm),
                    memory_domain=str(memory_domain),
                    facts_raw=list(facts_raw),
                    events_raw=list(events_raw),
                    turn_marks_by_index=marks_by_index if marks_by_index else None,
                    reference_time_iso=reference_time_iso,
                    turn_interval_seconds=int(turn_interval_seconds),
                    tenant_scoped_fact_ids=bool(tenant_scoped_fact_ids),
                )
                if publish_supported:
                    _set_graph_published(tkg_build.request, False)
                tkg_index = build_dialog_tkg_utterance_index_entries_v1(
                    tenant_id=str(tenant_id),
                    session_id=str(session_id),
                    user_tokens=list(user_tokens_norm),
                    memory_domain=str(memory_domain),
                    turns=list(turns_norm),
                    graph_build=tkg_build,
                )
                timing_ms["build_ms"] = int((time.perf_counter() - t_build0) * 1000)
                tkg_index_entries = list(tkg_index.entries)
                tkg_index_ids = list(tkg_index.index_ids)
                if marks_by_index:
                    for ent in tkg_index_entries:
                        try:
                            idx = ent.metadata.get("turn_index")
                            mark = marks_by_index.get(int(idx)) if idx is not None else None
                        except Exception:
                            mark = None
                        if not mark:
                            continue
                        md = dict(ent.metadata or {})
                        try:
                            md["importance"] = float(mark.get("importance"))
                        except Exception:
                            pass
                        ttl_val = mark.get("ttl_seconds")
                        if isinstance(ttl_val, (int, float)):
                            md["ttl"] = int(ttl_val)
                        forget_policy = mark.get("forget_policy")
                        if forget_policy:
                            md["forgetting_policy"] = str(forget_policy)
                        ent.metadata = md
                trace["tkg_vector_index_entries"] = len(tkg_index_entries)
            except Exception as e:
                trace["tkg_vector_index_error"] = f"{type(e).__name__}: {str(e)[:240]}"
                if str(graph_policy or "best_effort").lower() == "require":
                    raise

        # 4.6) Upsert TKG graph first (graph is the primary truth source).
        graph_status = "skipped"
        graph_error: Optional[str] = None
        purge_success = False
        if graph_upsert:
            try:
                graph_fn = getattr(store, "graph_upsert_v0", None)
                if callable(graph_fn):
                    t_graph0 = time.perf_counter()
                    if tkg_build is None:
                        from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1

                        tkg_build = build_dialog_graph_upsert_v1(
                            tenant_id=str(tenant_id),
                            session_id=str(session_id),
                            user_tokens=list(user_tokens_norm),
                            turns=list(turns_norm),
                            memory_domain=str(memory_domain),
                            facts_raw=list(facts_raw),
                            events_raw=list(events_raw),
                            turn_marks_by_index=marks_by_index if marks_by_index else None,
                            reference_time_iso=reference_time_iso,
                            turn_interval_seconds=int(turn_interval_seconds),
                            tenant_scoped_fact_ids=bool(tenant_scoped_fact_ids),
                        )
                        if publish_supported:
                            _set_graph_published(tkg_build.request, False)
                    await graph_fn(tkg_build.request)
                    timing_ms["graph_upsert_ms"] = int((time.perf_counter() - t_graph0) * 1000)
                    graph_status = "ok"
                    trace["graph_ids"] = tkg_build.graph_ids
                    if tkg_index_ids:
                        trace["tkg_vector_index_ids"] = list(tkg_index_ids)
                    if overwrite_existing and write_events:
                        purge_status = "skipped"
                        purge_error: Optional[str] = None
                        try:
                            graph_store = getattr(store, "graph", None)
                            if graph_store is None:
                                purge_status = "unsupported"
                                if graph_policy_norm == "require":
                                    raise RuntimeError("graph store not available for purge_source")
                            else:
                                from modules.memory.application.graph_service import GraphService

                                cfg = load_memory_config()
                                gating = get_graph_settings(cfg)
                                keep_event_ids = list((tkg_build.graph_ids or {}).get("event_ids") or [])
                                purge_res = await GraphService(graph_store, gating=gating).purge_source_except_events(
                                    tenant_id=str(tenant_id),
                                    source_id=f"dialog::{str(session_id)}",
                                    keep_event_ids=keep_event_ids,
                                )
                                trace["graph_overwrite_purge"] = purge_res
                                purge_status = "ok"
                                purge_success = True
                        except Exception as pe:
                            purge_status = "failed"
                            purge_error = f"{type(pe).__name__}: {str(pe)[:240]}"
                            if graph_policy_norm == "require":
                                trace["graph_overwrite_purge_status"] = purge_status
                                trace["graph_overwrite_purge_error"] = purge_error
                                raise
                        trace["graph_overwrite_purge_status"] = purge_status
                        if purge_error:
                            trace["graph_overwrite_purge_error"] = purge_error
                else:
                    graph_status = "unsupported"
                    if graph_policy_norm == "require":
                        raise RuntimeError("graph_upsert_v0 not supported by this store")
            except Exception as ge:
                if isinstance(ge, NotImplementedError):
                    graph_status = "unsupported"
                else:
                    graph_status = "failed"
                graph_error = f"{type(ge).__name__}: {str(ge)[:240]}"
                if purge_success:
                    trace["graph_upsert_purged_then_failed"] = True
                    trace["graph_upsert_status"] = graph_status
                    trace["graph_upsert_error"] = graph_error
                    raise
                if graph_policy_norm == "require":
                    trace["graph_upsert_status"] = graph_status
                    trace["graph_upsert_error"] = graph_error
                    raise

        trace["graph_upsert_status"] = graph_status
        if graph_error:
            trace["graph_upsert_error"] = graph_error

        # 4.8) Apply state updates (Phase 3 MVP) after graph upsert.
        state_updates = {
            "applied": 0,
            "pending": 0,
            "pending_low_conf": 0,
            "pending_out_of_order": 0,
            "skipped_no_subject": 0,
            "skipped_no_graph": 0,
            "skipped_error": 0,
        }
        if states_raw:
            graph_store = getattr(store, "graph", None)
            if graph_status != "ok" or graph_store is None or not callable(getattr(graph_store, "apply_state_update", None)):
                state_updates["skipped_no_graph"] = len(states_raw)
            else:
                speaker_map = {}
                try:
                    speaker_map = dict(tkg_build.graph_ids.get("speaker_entity_map") or {}) if tkg_build is not None else {}
                except Exception:
                    speaker_map = {}
                # include lower-case keys for matching
                for k, v in list(speaker_map.items()):
                    speaker_map.setdefault(str(k).lower(), v)
                turn_time_map = _build_turn_time_map(turns_norm)
                ref_dt = _parse_iso_dt(reference_time_iso) if reference_time_iso else None
                try:
                    conf_threshold = float(os.getenv("MEMORY_STATE_CONFIDENCE_THRESHOLD", "0.8"))
                except Exception:
                    conf_threshold = 0.8
                for st in states_raw:
                    try:
                        subject_ref = str(st.get("subject_ref") or "").strip()
                        subject_id = _resolve_state_subject_id(subject_ref, speaker_map)
                        if not subject_id:
                            state_updates["skipped_no_subject"] += 1
                            continue
                        conf = st.get("confidence")
                        try:
                            conf_val = float(conf) if conf is not None else None
                        except Exception:
                            conf_val = None
                        low_conf = conf_val is not None and conf_val < conf_threshold
                        # resolve valid_from from source_turn_ids
                        valid_from = None
                        for tid in st.get("source_turn_ids") or []:
                            dt = turn_time_map.get(str(tid))
                            if dt is not None:
                                valid_from = dt
                                break
                        if valid_from is None:
                            valid_from = ref_dt or datetime.now(timezone.utc)
                        res = await graph_store.apply_state_update(
                            tenant_id=str(tenant_id),
                            subject_id=str(subject_id),
                            property=str(st.get("property") or ""),
                            value=str(st.get("value") or ""),
                            raw_value=st.get("raw_value"),
                            confidence=conf_val,
                            valid_from=valid_from,
                            source_event_id=st.get("source_event_id"),
                            user_id=list(user_tokens_norm),
                            memory_domain=str(memory_domain),
                            status="pending" if low_conf else st.get("status"),
                            pending_reason="low_confidence" if low_conf else None,
                            extractor_version=st.get("extractor_version"),
                        )
                        if res.get("pending"):
                            state_updates["pending"] += 1
                            if res.get("reason") == "out_of_order":
                                state_updates["pending_out_of_order"] += 1
                            if low_conf:
                                state_updates["pending_low_conf"] += 1
                        else:
                            state_updates["applied"] += 1
                    except Exception:
                        state_updates["skipped_error"] += 1
                trace["state_updates"] = dict(state_updates)

        # 5) Build vector entries to write:
        # - fact/knowledge index entries (semantic, benchmark-compatible)
        # - utterance index entries (semantic, internal source)
        entries_to_write: List[MemoryEntry] = []

        fact_ids_built: List[str] = []
        if write_facts and facts_raw:
            for i, fact in enumerate(list(facts_raw)):
                fact_for_entry = dict(fact)
                # `fact_item_to_entry` uses sample_id/source_sample_id to build deterministic UUIDs.
                # In our `session_write` contract, the stable sample_id is the session_id (not LLM-provided).
                fact_for_entry["source_sample_id"] = str(session_id)
                fact_for_entry["sample_id"] = str(session_id)
                ent, fid = fact_item_to_entry(
                    fact_for_entry,
                    fact_idx=i,
                    tenant_id=str(tenant_id),
                    user_prefix="locomo_user_",
                    source="locomo_text_pipeline",
                )
                if ent is None or fid is None:
                    continue
                if tenant_scoped_fact_ids:
                    scoped_fact_id = build_fact_uuid(
                        sample_id=str(session_id),
                        fact_idx=int(i),
                        tenant_id=str(tenant_id),
                        namespace_by_tenant=True,
                    )
                    ent.id = scoped_fact_id
                    fid = scoped_fact_id
                md = dict(ent.metadata or {})
                md["tenant_id"] = str(tenant_id)
                md["user_id"] = list(user_tokens_norm)
                md["memory_domain"] = str(memory_domain)
                md["run_id"] = str(session_id)
                md["node_type"] = "fact"
                md["dedup_skip"] = True
                if fact_for_entry.get("importance") is not None:
                    try:
                        md["importance"] = float(fact_for_entry.get("importance"))
                    except Exception:
                        pass
                ttl_val = fact_for_entry.get("ttl_seconds")
                if isinstance(ttl_val, (int, float)):
                    md["ttl"] = int(ttl_val)
                forget_policy = fact_for_entry.get("forget_policy") or fact_for_entry.get("forgetting_policy")
                if forget_policy:
                    md["forgetting_policy"] = str(forget_policy)
                ent.metadata = md
                entries_to_write.append(ent)
                fact_ids_built.append(fid)

        # Add TKG vector index entries (hidden from default /search unless explicitly filtered by source).
        if tkg_index_entries:
            entries_to_write.extend(list(tkg_index_entries))

        if publish_supported:
            _set_entries_published(entries_to_write, False)
        t_vec0 = time.perf_counter()
        ver = await store.write(entries_to_write, links=None, upsert=True)
        timing_ms["vector_write_ms"] = int((time.perf_counter() - t_vec0) * 1000)
        version_value = ver.value if hasattr(ver, "value") else str(ver)
        completed_at = _now_iso()

        event_ids: List[str] = []
        timeslice_ids: List[str] = []
        fact_ids = list(fact_ids_built)

        if tkg_build is not None:
            try:
                event_ids = [str(x) for x in (tkg_build.graph_ids.get("event_ids") or []) if str(x).strip()]
                ts_id = str(tkg_build.graph_ids.get("timeslice_id") or "").strip()
                if ts_id:
                    timeslice_ids = [ts_id]
            except Exception:
                pass

        # 6) Publish entries after successful vector write (graph already upserted if enabled)
        publish_fn = getattr(store, "publish_entries", None)
        if callable(publish_fn):
            entry_ids = [str(e.id) for e in entries_to_write if e.id]
            graph_node_ids: List[str] = []
            if tkg_build is not None and graph_status == "ok":
                try:
                    graph_node_ids = _collect_graph_node_ids(tkg_build.request)
                except Exception:
                    graph_node_ids = []
                try:
                    graph_vec_ids = _collect_graph_vector_ids(tkg_build.request)
                    if graph_vec_ids:
                        entry_ids.extend(list(graph_vec_ids))
                except Exception:
                    pass
            try:
                t_pub0 = time.perf_counter()
                pub_res = await publish_fn(
                    tenant_id=str(tenant_id),
                    entry_ids=entry_ids,
                    graph_node_ids=graph_node_ids,
                    published=True,
                )
                timing_ms["publish_ms"] = int((time.perf_counter() - t_pub0) * 1000)
                trace["publish_updates"] = pub_res
            except Exception as e:
                trace["publish_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                # Fail fast to avoid silently leaving data unpublished and invisible.
                raise

        # Overwrite cleanup: delete old facts that are no longer present in the new extraction result.
        # IMPORTANT: do it only after we finish the new write path, otherwise a failed run could delete data.
        if old_fact_ids:
            new_set = {str(x) for x in fact_ids if str(x).strip()}
            to_delete = [fid for fid in old_fact_ids if fid not in new_set]
            trace["overwrite_delete_candidates"] = len(to_delete)
            t_del0 = time.perf_counter()
            for fid in to_delete:
                await store.delete(str(fid), soft=False, reason="session_write.overwrite_existing")
                deleted_fact_ids += 1
            trace["overwrite_deleted_fact_ids"] = deleted_fact_ids
            timing_ms["overwrite_delete_ms"] = int((time.perf_counter() - t_del0) * 1000)

        # Use "completed_no_llm" status when LLM extraction was skipped due to missing config.
        # This allows re-processing when LLM becomes available (marker check at line 439
        # only skips status="completed", so "completed_no_llm" will pass through).
        marker_status_final = "completed"
        if facts_skipped_reason == "llm_missing":
            marker_status_final = "completed_no_llm"

        timing_ms["total_ms"] = int((time.perf_counter() - t_total0) * 1000)
        trace["timing_ms"] = dict(timing_ms)
        marker_done = _make_marker_entry(
            marker_id=marker_id,
            tenant_id=str(tenant_id),
            user_tokens=user_tokens_norm,
            memory_domain=str(memory_domain),
            session_id=str(session_id),
            status=marker_status_final,
            overwrite_existing=overwrite_existing,
            event_ids=event_ids,
            timeslice_ids=timeslice_ids,
            fact_ids=fact_ids,
            written_links=0,
            version=str(version_value),
            error_reason=None,
            started_at=started_at,
            completed_at=completed_at,
        )
        try:
            await store.write([marker_done], links=None, upsert=True)
        except Exception as e:
            trace["marker_write_completed_error"] = str(e)[:200]

        return SessionWriteResult(
            status="ok",
            session_id=str(session_id),
            version=str(version_value),
            marker_id=marker_id,
            written_entries=len(entries_to_write),
            written_links=0,
            deleted_fact_ids=deleted_fact_ids,
            trace=trace,
        ).__dict__
    except Exception as e:
        # Best-effort: mark failed (no rollback of already written events by default).
        try:
            marker_failed = _make_marker_entry(
                marker_id=marker_id,
                tenant_id=str(tenant_id),
                user_tokens=user_tokens_norm,
                memory_domain=str(memory_domain),
                session_id=str(session_id),
                status="failed",
                overwrite_existing=overwrite_existing,
                event_ids=[],
                timeslice_ids=[],
                fact_ids=[],
                written_links=0,
                version=None,
                error_reason=str(e),
                started_at=started_at,
                completed_at=None,
            )
            await store.write([marker_failed], links=None, upsert=True)
        except Exception as e2:
            trace["marker_write_failed_error"] = str(e2)[:200]

        return SessionWriteResult(
            status="failed",
            session_id=str(session_id),
            version=None,
            marker_id=marker_id,
            written_entries=0,
            written_links=0,
            deleted_fact_ids=deleted_fact_ids,
            trace={**trace, "error": str(e)},
        ).__dict__
