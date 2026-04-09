from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.memory.contracts.graph_models import (
    Entity,
    Event,
    GraphEdge,
    GraphUpsertRequest,
    Knowledge,
    MediaSegment,
    TimeSlice,
    UtteranceEvidence,
)
from modules.memory.domain.dialog_text_pipeline_v1 import build_fact_uuid, generate_uuid, make_event_id
from modules.memory.application.topic_normalizer import enqueue_deferred_event, get_normalization_mode


@dataclass(frozen=True)
class DialogGraphBuildResult:
    request: GraphUpsertRequest
    graph_ids: Dict[str, Any]


def _try_parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _derive_time_bucket(t_start: Optional[datetime], t_end: Optional[datetime]) -> List[str]:
    ts = t_start or t_end
    if not ts:
        return []
    try:
        year = ts.strftime("%Y")
        month = ts.strftime("%Y-%m")
        day = ts.strftime("%Y-%m-%d")
        return [year, month, day]
    except Exception:
        return []


def _normalize_entity_name(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _entity_name_id_key(value: object) -> str:
    norm = _normalize_entity_name(value)
    return norm.casefold() if norm else ""


def _stable_speaker_entity_id(*, tenant_id: str, memory_domain: str, user_tokens: Sequence[str], speaker: str) -> str:
    speaker_key = _entity_name_id_key(speaker) or _normalize_entity_name(speaker)
    key = f"tenant={tenant_id}|domain={memory_domain}|users={'|'.join([str(x) for x in user_tokens])}|speaker={speaker_key}"
    return generate_uuid("tkg.dialog.entity", key)


def build_dialog_graph_upsert_v1(
    *,
    tenant_id: str,
    session_id: str,
    user_tokens: Sequence[str],
    turns: Sequence[Dict[str, Any]],
    memory_domain: str = "dialog",
    facts_raw: Optional[Sequence[Dict[str, Any]]] = None,
    events_raw: Optional[Sequence[Dict[str, Any]]] = None,
    turn_marks_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
    reference_time_iso: Optional[str] = None,
    turn_interval_seconds: int = 60,
    source_id: Optional[str] = None,
    tenant_scoped_fact_ids: bool = False,
) -> DialogGraphBuildResult:
    """Build a TKG GraphUpsertRequest for a dialogue session (pure, no I/O).

    This intentionally does NOT attempt global identity resolution. Speaker entities are
    scoped to (tenant, memory_domain, user_tokens, speaker_label) to avoid cross-user merges.
    """
    if not str(tenant_id or "").strip():
        raise ValueError("tenant_id is required")
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    if not user_tokens:
        raise ValueError("user_tokens is required")

    user_scope = [str(x).strip() for x in user_tokens if str(x).strip()]
    domain_scope = str(memory_domain)
    tenant_scope = str(tenant_id)

    interval = max(1, int(turn_interval_seconds))
    time_origin = "logical"
    source = "dialog_session_write_v1"

    # Base time: use explicit reference if present; else fall back to first turn timestamp; else None.
    ref_dt = _try_parse_iso_dt(reference_time_iso)
    if ref_dt is None:
        for t in turns:
            ref_dt = _try_parse_iso_dt(t.get("timestamp_iso"))
            if ref_dt is not None:
                break

    seg_id = generate_uuid("tkg.dialog.segment", f"{tenant_id}|{session_id}")
    seg_source_id = str(source_id or f"dialog::{session_id}")
    seg_has_physical_time = ref_dt is not None

    # We always create a pseudo "text segment" timeline to anchor utterances by media-time offsets.
    duration_s = max(1, len(list(turns))) * interval
    segment = MediaSegment(
        id=seg_id,
        tenant_id=str(tenant_id),
        source_id=seg_source_id,
        t_media_start=0.0,
        t_media_end=float(duration_s),
        has_physical_time=bool(seg_has_physical_time),
        recorded_at=(ref_dt if seg_has_physical_time else None),
        modality="text",
        time_origin=time_origin,
        provenance={"source": source},
    )

    ts_id = generate_uuid("tkg.dialog.timeslice", f"{tenant_id}|{session_id}")

    # Build utterances/entities + edges.
    entities_by_speaker: Dict[str, str] = {}
    entities_by_name: Dict[str, str] = {}
    entities: List[Entity] = []
    utterances: List[UtteranceEvidence] = []
    events: List[Event] = []
    knowledge: List[Knowledge] = []
    edges: List[GraphEdge] = []

    first_abs: Optional[datetime] = None
    last_abs: Optional[datetime] = None

    def _speaker_entity(speaker: str) -> Tuple[str, Entity]:
        sid = entities_by_speaker.get(speaker)
        if sid:
            # Already created in this request.
            return sid, next(e for e in entities if e.id == sid)
        ent_id = _stable_speaker_entity_id(
            tenant_id=str(tenant_id),
            memory_domain=str(memory_domain),
            user_tokens=list(user_tokens),
            speaker=str(speaker),
        )
        ent = Entity(
            id=ent_id,
            tenant_id=str(tenant_id),
            type="PERSON",
            name=str(speaker),
            cluster_label=str(speaker),
            manual_name=str(speaker),
            time_origin=time_origin,
            provenance={"source": source},
        )
        entities_by_speaker[str(speaker)] = ent_id
        entities.append(ent)
        norm = _normalize_entity_name(speaker)
        if norm:
            entities_by_name.setdefault(norm.casefold(), ent_id)
        return ent_id, ent

    def _entity_for_name(name: object) -> Optional[Tuple[str, Entity]]:
        norm = _normalize_entity_name(name)
        if not norm:
            return None
        key = norm.casefold()
        ent_id = entities_by_name.get(key)
        if ent_id:
            return ent_id, next(e for e in entities if e.id == ent_id)
        ent_id = _stable_speaker_entity_id(
            tenant_id=str(tenant_id),
            memory_domain=str(memory_domain),
            user_tokens=list(user_tokens),
            speaker=norm,
        )
        ent = Entity(
            id=ent_id,
            tenant_id=str(tenant_id),
            type="PERSON",
            name=norm,
            cluster_label=norm,
            manual_name=norm,
            time_origin=time_origin,
            provenance={"source": "dialog_tkg_unified_extractor_v1"},
        )
        entities.append(ent)
        entities_by_name[key] = ent_id
        return ent_id, ent

    def _append_edge_if_absent(edge: GraphEdge) -> None:
        key = (str(edge.src_id), str(edge.dst_id), str(edge.rel_type))
        for existing in edges:
            if (str(existing.src_id), str(existing.dst_id), str(existing.rel_type)) == key:
                return
        edges.append(edge)

    event_ids: List[str] = []
    utterance_ids: List[str] = []
    marks_by_index = dict(turn_marks_by_index or {})

    def _mark_for(idx: int) -> Optional[Dict[str, Any]]:
        return marks_by_index.get(int(idx))

    def _mark_importance(mark: Optional[Dict[str, Any]]) -> Optional[float]:
        if not mark:
            return None
        try:
            return float(mark.get("importance"))
        except Exception:
            return None

    def _mark_ttl(mark: Optional[Dict[str, Any]]) -> Optional[float]:
        if not mark:
            return None
        ttl_val = mark.get("ttl_seconds")
        if isinstance(ttl_val, (int, float)):
            return float(ttl_val)
        return None

    def _mark_forget_policy(mark: Optional[Dict[str, Any]]) -> Optional[str]:
        if not mark:
            return None
        fp = mark.get("forget_policy")
        return str(fp) if fp else None

    for idx, t in enumerate(turns, start=1):
        raw_text = str(t.get("text") or t.get("content") or "").strip()
        if not raw_text:
            continue
        speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"

        start_s = float((idx - 1) * interval)
        end_s = float(idx * interval)
        utt_id = generate_uuid("tkg.dialog.utterance", f"{tenant_id}|{session_id}|{idx}")
        utterance_ids.append(utt_id)

        mark = _mark_for(idx)
        utterances.append(
            UtteranceEvidence(
                id=utt_id,
                tenant_id=str(tenant_id),
                raw_text=raw_text,
                t_media_start=start_s,
                t_media_end=end_s,
                segment_id=seg_id,
                lang=None,
                speaker_track_id=None,
                asr_model_version=None,
                time_origin=time_origin,
                provenance={"source": source},
                importance=_mark_importance(mark),
                ttl=_mark_ttl(mark),
                forgetting_policy=_mark_forget_policy(mark),
            )
        )

        t_abs = _try_parse_iso_dt(t.get("timestamp_iso"))
        if t_abs is None and ref_dt is not None:
            t_abs = ref_dt + timedelta(seconds=(idx - 1) * interval)
        if t_abs is not None:
            first_abs = first_abs or t_abs
            last_abs = t_abs

        # Segment contains utterance
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=seg_id,
                dst_id=utt_id,
                rel_type="CONTAINS_EVIDENCE",
                src_type="MediaSegment",
                dst_type="UtteranceEvidence",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )

        # TimeSlice contains utterance (time anchor)
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=ts_id,
                dst_id=utt_id,
                rel_type="TEMPORALLY_CONTAINS",
                src_type="TimeSlice",
                dst_type="UtteranceEvidence",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )

        ent_id, _ = _speaker_entity(speaker)

        # Utterance -> Speaker entity
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=utt_id,
                dst_id=ent_id,
                rel_type="SPOKEN_BY",
                src_type="UtteranceEvidence",
                dst_type="Entity",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )

    def _build_turn_index_map() -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, t in enumerate(turns, start=1):
            candidates: List[str] = []
            dia_id = str(t.get("dia_id") or "").strip()
            if dia_id:
                candidates.append(dia_id)
                if ":" in dia_id:
                    candidates.append(dia_id.replace(":", "_"))
            turn_id = str(t.get("turn_id") or "").strip()
            if turn_id:
                candidates.append(turn_id)
            candidates.append(f"t{idx}")
            for cid in candidates:
                if cid and cid not in mapping:
                    mapping[cid] = idx
        return mapping

    turn_index_by_id = _build_turn_index_map()

    # Build abstract events (session/topic-level) from events_raw.
    def _parse_turn_index_from_source_id(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return int(value) if value > 0 else None
        s = str(value).strip()
        if not s:
            return None
        mapped = turn_index_by_id.get(s)
        if mapped is not None:
            return mapped
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

    def _logical_event_id_from_turn(turn_index: int) -> Optional[str]:
        if turn_index < 1 or turn_index > len(turns):
            return None
        t = turns[turn_index - 1]
        dia_id = str(t.get("dia_id") or "").strip()
        if dia_id:
            return make_event_id(str(session_id), dia_id)
        return f"{session_id}_t{turn_index}"

    turn_to_event_ids: Dict[int, List[str]] = {}
    event_order: List[Tuple[str, Optional[int], Optional[datetime], Optional[datetime], List[int]]] = []
    event_inputs: List[Tuple[int, Dict[str, Any], List[int]]] = []
    turn_event_counts: Dict[int, int] = {}
    for ev_idx, ev in enumerate(list(events_raw or []), start=1):
        if not isinstance(ev, dict):
            continue
        summary = str(ev.get("summary") or "").strip()
        if not summary:
            continue
        full_turn_ids = list(ev.get("source_turn_ids") or [])
        source_turn_ids = list(ev.get("supported_turn_ids") or full_turn_ids)
        turn_indices = []
        for stid in source_turn_ids:
            tidx = _parse_turn_index_from_source_id(stid)
            if tidx is not None and 1 <= tidx <= len(turns):
                turn_indices.append(tidx)
        if not turn_indices:
            continue
        turn_indices = list(sorted(dict.fromkeys(turn_indices)))
        for tidx in turn_indices:
            turn_event_counts[tidx] = turn_event_counts.get(tidx, 0) + 1
        event_inputs.append((ev_idx, ev, turn_indices))

    for ev_idx, ev, turn_indices in event_inputs:
        summary = str(ev.get("summary") or "").strip()
        full_turn_ids = list(ev.get("source_turn_ids") or [])
        ev_id = generate_uuid("tkg.dialog.event", f"{tenant_id}|{session_id}|event|{ev_idx}")
        event_ids.append(ev_id)

        t_abs_start: Optional[datetime] = None
        t_abs_end: Optional[datetime] = None
        for tidx in turn_indices:
            t = turns[tidx - 1]
            t_abs = _try_parse_iso_dt(t.get("timestamp_iso"))
            if t_abs is None and ref_dt is not None:
                t_abs = ref_dt + timedelta(seconds=(tidx - 1) * interval)
            if t_abs is None:
                continue
            if t_abs_start is None or t_abs < t_abs_start:
                t_abs_start = t_abs
            if t_abs_end is None or t_abs > t_abs_end:
                t_abs_end = t_abs

        logical_event_id = None
        if len(turn_indices) == 1 and turn_event_counts.get(turn_indices[0], 0) == 1:
            logical_event_id = _logical_event_id_from_turn(turn_indices[0])
        try:
            ev_conf = float(ev.get("event_confidence")) if ev.get("event_confidence") is not None else None
        except Exception:
            ev_conf = None
        try:
            evd_conf = float(ev.get("evidence_confidence")) if ev.get("evidence_confidence") is not None else None
        except Exception:
            evd_conf = None
        ttl_val = ev.get("ttl_seconds") if ev.get("ttl_seconds") is not None else ev.get("ttl")
        ttl_val = float(ttl_val) if isinstance(ttl_val, (int, float)) else None
        raw_time_bucket = ev.get("time_bucket")
        time_bucket = (
            [str(x).strip() for x in raw_time_bucket if str(x).strip()]
            if isinstance(raw_time_bucket, list)
            else [str(raw_time_bucket).strip()]
            if isinstance(raw_time_bucket, str) and str(raw_time_bucket).strip()
            else _derive_time_bucket(t_abs_start, t_abs_end)
        )
        tags = (
            [str(x).strip() for x in (ev.get("tags") or []) if str(x).strip()]
            if isinstance(ev.get("tags"), list)
            else []
        )
        keywords = (
            [str(x).strip() for x in (ev.get("keywords") or []) if str(x).strip()]
            if isinstance(ev.get("keywords"), list)
            else []
        )
        tags_vocab_version = str(ev.get("tags_vocab_version") or "").strip() or None

        events.append(
            Event(
                id=ev_id,
                tenant_id=str(tenant_id),
                summary=(summary if len(summary) <= 200 else summary[:197] + "..."),
                desc=str(ev.get("desc") or "").strip() or None,
                t_abs_start=t_abs_start,
                t_abs_end=t_abs_end,
                source=source,
                time_origin=time_origin,
                provenance={"source": source},
                user_id=list(user_tokens),
                memory_domain=str(memory_domain),
                importance=ev_conf,
                ttl=ttl_val,
                forgetting_policy=str(ev.get("forgetting_policy") or ev.get("forget_policy") or "") or None,
                logical_event_id=logical_event_id,
                source_turn_ids=[str(x) for x in full_turn_ids if str(x).strip()],
                evidence_status=str(ev.get("evidence_status") or "").strip() or None,
                evidence_confidence=evd_conf,
                event_confidence=ev_conf,
                evidence_count=int(ev.get("evidence_count") or len(full_turn_ids)),
                topic_id=str(ev.get("topic_id") or "").strip() or None,
                topic_path=str(ev.get("topic_path") or "").strip() or None,
                tags=tags or None,
                keywords=keywords or None,
                time_bucket=time_bucket or None,
                tags_vocab_version=tags_vocab_version,
                event_type=str(ev.get("event_type") or "").strip() or None,
            )
        )
        try:
            if get_normalization_mode() == "async":
                tp = str(ev.get("topic_path") or "").strip()
                if tp.startswith("_uncategorized/"):
                    enqueue_deferred_event(
                        {
                            "id": ev_id,
                            "tenant_id": str(tenant_id),
                            "user_id": list(user_tokens),
                            "memory_domain": str(memory_domain),
                            "summary": summary,
                            "desc": str(ev.get("desc") or "").strip() or None,
                            "topic_id": str(ev.get("topic_id") or "").strip() or None,
                            "topic_path": tp,
                            "tags": tags or None,
                            "keywords": keywords or None,
                            "time_bucket": time_bucket or None,
                            "tags_vocab_version": tags_vocab_version,
                            "source_turn_ids": [str(x) for x in full_turn_ids if str(x).strip()],
                            "time_hint": str(ev.get("time_hint") or "").strip() or None,
                        }
                    )
        except Exception:
            pass

        # TimeSlice covers all events in this session.
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=ts_id,
                dst_id=ev_id,
                rel_type="COVERS_EVENT",
                src_type="TimeSlice",
                dst_type="Event",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=ev_id,
                dst_id=ts_id,
                rel_type="OCCURS_AT",
                src_type="Event",
                dst_type="TimeSlice",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )

        # Segment anchor (optional but helpful for source_id queries).
        edges.append(
            GraphEdge(
                tenant_id=str(tenant_id),
                src_id=ev_id,
                dst_id=seg_id,
                rel_type="SUMMARIZES",
                src_type="Event",
                dst_type="MediaSegment",
                confidence=1.0,
                weight=1.0,
                layer="fact",
                kind="observed",
                source=source,
                time_origin=time_origin,
            )
        )

        # Evidence chain: Event -> UtteranceEvidence
        for tidx in turn_indices:
            utt_id = utterance_ids[tidx - 1]
            edges.append(
                GraphEdge(
                    tenant_id=str(tenant_id),
                    src_id=ev_id,
                    dst_id=utt_id,
                    rel_type="SUPPORTED_BY",
                    src_type="Event",
                    dst_type="UtteranceEvidence",
                    confidence=evd_conf if evd_conf is not None else 1.0,
                    weight=1.0,
                    layer="fact",
                    kind="observed",
                    source=source,
                    time_origin=time_origin,
                )
            )
            turn_to_event_ids.setdefault(tidx, []).append(ev_id)

        # Event -> Speaker entity (from supporting turns)
        involved_speakers: List[str] = []
        for tidx in turn_indices:
            try:
                speaker_raw = turns[tidx - 1].get("speaker") or turns[tidx - 1].get("role") or "Unknown"
            except Exception:
                speaker_raw = "Unknown"
            speaker = str(speaker_raw or "Unknown").strip() or "Unknown"
            if speaker not in involved_speakers:
                involved_speakers.append(speaker)
        for speaker in involved_speakers:
            ent_id = entities_by_speaker.get(speaker) or _stable_speaker_entity_id(
                tenant_id=str(tenant_id),
                memory_domain=str(memory_domain),
                user_tokens=list(user_tokens),
                speaker=speaker,
            )
            if speaker not in entities_by_speaker:
                entities_by_speaker[speaker] = ent_id
                entities.append(
                    Entity(
                        id=ent_id,
                        tenant_id=str(tenant_id),
                        type="PERSON",
                        name=str(speaker),
                        cluster_label=str(speaker),
                        manual_name=str(speaker),
                        time_origin=time_origin,
                        provenance={"source": source},
                    )
                )
                entities_by_name.setdefault(_normalize_entity_name(speaker).casefold(), ent_id)
            edges.append(
                GraphEdge(
                    tenant_id=str(tenant_id),
                    src_id=ev_id,
                    dst_id=ent_id,
                    rel_type="INVOLVES",
                    src_type="Event",
                    dst_type="Entity",
                    confidence=1.0,
                    weight=1.0,
                    layer="fact",
                    kind="observed",
                    source=source,
                    time_origin=time_origin,
                )
            )

        # Event participants from unified extractor (LLM-mainline).
        # This complements speaker-derived INVOLVES edges above.
        participants = [_normalize_entity_name(x) for x in (ev.get("participants") or []) if _normalize_entity_name(x)]
        seen_participants: set[str] = set()
        for participant in participants:
            pkey = participant.casefold()
            if pkey in seen_participants:
                continue
            seen_participants.add(pkey)
            ent = _entity_for_name(participant)
            if ent is None:
                continue
            ent_id, _ = ent
            _append_edge_if_absent(
                GraphEdge(
                    tenant_id=str(tenant_id),
                    src_id=ev_id,
                    dst_id=ent_id,
                    rel_type="INVOLVES",
                    src_type="Event",
                    dst_type="Entity",
                    confidence=evd_conf if evd_conf is not None else (ev_conf if ev_conf is not None else 0.8),
                    weight=1.0,
                    layer="fact",
                    kind="derived",
                    source="dialog_tkg_unified_extractor_v1",
                    time_origin=time_origin,
                )
            )

        event_order.append((ev_id, min(turn_indices) if turn_indices else None, t_abs_start, t_abs_end, turn_indices))

    # NEXT_EVENT uses event order (by earliest turn index or absolute time when available)
    if event_order:
        def _order_key(item: Tuple[str, Optional[int], Optional[datetime], Optional[datetime], List[int]]):
            _, min_turn, t_start, _, _ = item
            if t_start is not None:
                return (0, t_start, min_turn or 0)
            return (1, min_turn or 0, 0)

        ordered = sorted(event_order, key=_order_key)
        for i in range(len(ordered) - 1):
            a = ordered[i][0]
            b = ordered[i + 1][0]
            edges.append(
                GraphEdge(
                    tenant_id=str(tenant_id),
                    src_id=a,
                    dst_id=b,
                    rel_type="NEXT_EVENT",
                    src_type="Event",
                    dst_type="Event",
                    confidence=1.0,
                    weight=1.0,
                    layer="fact",
                    kind="observed",
                    source=source,
                    time_origin=time_origin,
                )
            )

    # Cognition layer: facts -> Knowledge nodes with evidence links.
    # This is the "认知层入口": extracted facts must be traceable back to specific turns/events.
    if facts_raw:
        def _turn_index_from_source_turn_id(v: object) -> Optional[int]:
            if v is None:
                return None
            if isinstance(v, int):
                return int(v) if v > 0 else None
            s = str(v).strip()
            if not s:
                return None
            mapped = turn_index_by_id.get(s)
            if mapped is not None:
                return mapped
            if s.startswith("t") and s[1:].isdigit():
                try:
                    n = int(s[1:])
                    return n if n > 0 else None
                except Exception:
                    return None
            # D1:3 -> 3
            if ":" in s:
                try:
                    tail = s.split(":")[-1]
                    n = int(tail)
                    return n if n > 0 else None
                except Exception:
                    return None
            # D1_3 -> 3
            if "_" in s:
                try:
                    tail = s.split("_")[-1]
                    n = int(tail)
                    return n if n > 0 else None
                except Exception:
                    return None
            # "3"
            try:
                n = int(s)
                return n if n > 0 else None
            except Exception:
                return None

        for fact_idx, fact in enumerate(list(facts_raw)):
            op = str((fact or {}).get("op", "ADD")).upper()
            if op in ("KEEP", "DELETE"):
                continue
            statement = str((fact or {}).get("statement") or "").strip()
            if not statement:
                continue

            sample_id = str((fact or {}).get("source_sample_id") or (fact or {}).get("sample_id") or "").strip() or str(session_id)
            fact_id = build_fact_uuid(
                sample_id=sample_id,
                fact_idx=int(fact_idx),
                tenant_id=str(tenant_id),
                namespace_by_tenant=bool(tenant_scoped_fact_ids),
            )
            try:
                fact_importance = float((fact or {}).get("importance")) if (fact or {}).get("importance") is not None else None
            except Exception:
                fact_importance = None
            ttl_val = (fact or {}).get("ttl_seconds")
            if ttl_val is None:
                ttl_val = (fact or {}).get("ttl")
            ttl_val = float(ttl_val) if isinstance(ttl_val, (int, float)) else None
            forget_policy = (fact or {}).get("forget_policy") or (fact or {}).get("forgetting_policy")

            knowledge.append(
                Knowledge(
                    id=fact_id,
                    tenant_id=str(tenant_id),
                    schema_version="dialog_fact_v1",
                    summary=(statement if len(statement) <= 240 else statement[:237] + "..."),
                    buckets_meta={"fact_type": str((fact or {}).get("type") or (fact or {}).get("fact_type") or "fact")},
                    data={
                        "statement": statement,
                        "fact_type": (fact or {}).get("type") or (fact or {}).get("fact_type") or "fact",
                        "scope": (fact or {}).get("scope") or "permanent",
                        "status": (fact or {}).get("status") or "n/a",
                        "importance": (fact or {}).get("importance"),
                        "source_session_id": str(session_id),
                        "source_sample_id": sample_id,
                        "source_turn_ids": list((fact or {}).get("source_turn_ids") or []),
                        "mentions": list((fact or {}).get("mentions") or []),
                        "temporal_grounding": list((fact or {}).get("temporal_grounding") or []),
                        "rationale": (fact or {}).get("rationale"),
                    },
                    time_origin=time_origin,
                    provenance={"source": "dialog_fact_extractor_v1"},
                    importance=fact_importance,
                    ttl=ttl_val,
                    forgetting_policy=(str(forget_policy) if forget_policy else None),
                )
            )

            mentioned_names = [
                _normalize_entity_name(x)
                for x in list((fact or {}).get("mentions") or [])
                if _normalize_entity_name(x)
            ]
            mentioned_seen: set[str] = set()
            for mname in mentioned_names:
                mkey = mname.casefold()
                if mkey in mentioned_seen:
                    continue
                mentioned_seen.add(mkey)
                ent = _entity_for_name(mname)
                if ent is None:
                    continue
                ent_id, _ = ent
                _append_edge_if_absent(
                    GraphEdge(
                        tenant_id=str(tenant_id),
                        src_id=fact_id,
                        dst_id=ent_id,
                        rel_type="MENTIONS",
                        src_type="Knowledge",
                        dst_type="Entity",
                        confidence=1.0,
                        weight=1.0,
                        layer="semantic",
                        kind="derived",
                        source="dialog_tkg_unified_extractor_v1",
                        time_origin=time_origin,
                    )
                )

            # Session contains this fact/knowledge.
            edges.append(
                GraphEdge(
                    tenant_id=str(tenant_id),
                    src_id=ts_id,
                    dst_id=fact_id,
                    rel_type="CONTAINS",
                    src_type="TimeSlice",
                    dst_type="Knowledge",
                    confidence=1.0,
                    weight=1.0,
                    layer="semantic",
                    kind="derived",
                    source="dialog_fact_extractor_v1",
                    time_origin=time_origin,
                )
            )

            # Link fact to its supporting events/utterances/speakers.
            source_turn_ids = list((fact or {}).get("source_turn_ids") or [])
            for stid in source_turn_ids:
                turn_i = _turn_index_from_source_turn_id(stid)
                if turn_i is None:
                    continue
                if turn_i < 1 or turn_i > len(utterance_ids):
                    continue
                utt_id = utterance_ids[turn_i - 1]
                for ev_id in turn_to_event_ids.get(turn_i, []):
                    # Fact derived from event
                    edges.append(
                        GraphEdge(
                            tenant_id=str(tenant_id),
                            src_id=fact_id,
                            dst_id=ev_id,
                            rel_type="DERIVED_FROM",
                            src_type="Knowledge",
                            dst_type="Event",
                            confidence=1.0,
                            weight=1.0,
                            layer="semantic",
                            kind="derived",
                            source="dialog_fact_extractor_v1",
                            time_origin=time_origin,
                        )
                    )
                # Fact supported by utterance
                edges.append(
                    GraphEdge(
                        tenant_id=str(tenant_id),
                        src_id=fact_id,
                        dst_id=utt_id,
                        rel_type="SUPPORTED_BY",
                        src_type="Knowledge",
                        dst_type="UtteranceEvidence",
                        confidence=1.0,
                        weight=1.0,
                        layer="semantic",
                        kind="derived",
                        source="dialog_fact_extractor_v1",
                        time_origin=time_origin,
                    )
                )
                # Fact stated by speaker (best-effort: use the speaker label in the turn)
                try:
                    speaker_raw = turns[turn_i - 1].get("speaker") or turns[turn_i - 1].get("role") or "Unknown"
                except Exception:
                    speaker_raw = "Unknown"
                speaker = str(speaker_raw or "Unknown").strip() or "Unknown"
                ent_id = entities_by_speaker.get(speaker) or _stable_speaker_entity_id(
                    tenant_id=str(tenant_id),
                    memory_domain=str(memory_domain),
                    user_tokens=list(user_tokens),
                    speaker=speaker,
                )
                if speaker not in entities_by_speaker:
                    entities_by_speaker[speaker] = ent_id
                    entities.append(
                        Entity(
                            id=ent_id,
                            tenant_id=str(tenant_id),
                            type="PERSON",
                            name=str(speaker),
                            cluster_label=speaker,
                            manual_name=speaker,
                            time_origin=time_origin,
                            provenance={"source": source},
                        )
                    )
                    entities_by_name.setdefault(_normalize_entity_name(speaker).casefold(), ent_id)
                edges.append(
                    GraphEdge(
                        tenant_id=str(tenant_id),
                        src_id=fact_id,
                        dst_id=ent_id,
                        rel_type="STATED_BY",
                        src_type="Knowledge",
                        dst_type="Entity",
                        confidence=1.0,
                        weight=1.0,
                        layer="semantic",
                        kind="derived",
                        source="dialog_fact_extractor_v1",
                        time_origin=time_origin,
                    )
                )

    # TimeSlice for the session timeline.
    if first_abs is None and ref_dt is not None:
        first_abs = ref_dt
        last_abs = ref_dt + timedelta(seconds=duration_s)
    if first_abs is not None:
        if last_abs is None or last_abs <= first_abs:
            last_abs = first_abs + timedelta(seconds=duration_s)

    ts = TimeSlice(
        id=ts_id,
        tenant_id=str(tenant_id),
        kind="dialog_session",
        t_abs_start=first_abs,
        t_abs_end=last_abs,
        t_media_start=0.0,
        t_media_end=float(duration_s),
        granularity_level=0,
        parent_id=None,
        time_origin=time_origin,
        provenance={"source": source},
    )

    # Link TimeSlice -> Segment to allow day/session aggregations to reuse the same pattern as other modalities.
    edges.append(
        GraphEdge(
            tenant_id=str(tenant_id),
            src_id=ts_id,
            dst_id=seg_id,
            rel_type="COVERS_SEGMENT",
            src_type="TimeSlice",
            dst_type="MediaSegment",
            confidence=1.0,
            weight=1.0,
            layer="fact",
            kind="observed",
            source=source,
            time_origin=time_origin,
        )
    )

    # Apply scope metadata (tenant/user/domain) to all nodes/edges for isolation.
    def _apply_scope_meta(items: List[Any]) -> None:
        for it in items:
            try:
                it.tenant_id = tenant_scope
            except Exception:
                pass
            try:
                it.user_id = list(user_scope)
            except Exception:
                pass
            try:
                it.memory_domain = domain_scope
            except Exception:
                pass

    _apply_scope_meta([segment])
    _apply_scope_meta(utterances)
    _apply_scope_meta(entities)
    _apply_scope_meta(events)
    _apply_scope_meta([ts])
    _apply_scope_meta(knowledge)
    _apply_scope_meta(edges)

    req = GraphUpsertRequest(
        segments=[segment],
        utterances=utterances,
        entities=entities,
        events=events,
        time_slices=[ts],
        knowledge=knowledge,
        edges=edges,
    )
    return DialogGraphBuildResult(
        request=req,
        graph_ids={
            "segment_id": seg_id,
            "timeslice_id": ts_id,
            "event_ids": list(event_ids),
            "utterance_ids": list(utterance_ids),
            "entity_ids": [str(getattr(ent, "id", "")) for ent in entities if str(getattr(ent, "id", "")).strip()],
            "speaker_entity_map": dict(entities_by_speaker),
        },
    )


__all__ = ["DialogGraphBuildResult", "build_dialog_graph_upsert_v1"]
