from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from modules.memory.application.config import load_memory_config, get_dialog_event_settings
from modules.memory.application.embedding_adapter import build_embedding_from_settings
from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_config, build_llm_from_env
from modules.memory.application.topic_normalizer import normalize_events


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_tkg_event_extractor_system_prompt_v1.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_ALLOWED_EVENT_TYPES = {"Atomic", "Process", "Composite"}
_ALLOWED_EVIDENCE_STATUS = {"mapped", "weak", "unmapped"}


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def parse_event_json(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(_remove_code_blocks(raw))
    except Exception:
        return []
    items = data.get("events") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(dict(it))
    return out


def _turn_id_for_turn(t: Dict[str, Any], idx: int) -> str:
    dia_id = str(t.get("dia_id") or "").strip()
    if dia_id:
        return dia_id
    turn_id = str(t.get("turn_id") or "").strip()
    if turn_id:
        return turn_id
    return f"t{idx}"


def build_event_context(*, session_id: str, turns: Sequence[Dict[str, Any]]) -> Tuple[str, List[str]]:
    lines: List[str] = []
    turn_ids: List[str] = []
    for idx, t in enumerate(turns or [], start=1):
        tid = _turn_id_for_turn(t, idx)
        turn_ids.append(tid)

    lines.append(f"Session ID: {session_id}")
    lines.append("Allowed turn_ids (you must choose from this list only):")
    lines.append(f"{turn_ids}")
    lines.append("")
    lines.append("Dialogue:")
    for idx, t in enumerate(turns or [], start=1):
        tid = _turn_id_for_turn(t, idx)
        speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"
        text = str(t.get("text") or t.get("content") or "").strip()
        if not text:
            continue
        lines.append(f"{tid} {speaker}: {text}")
    return "\n".join(lines), turn_ids


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clamp01_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        s = str(it).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_event(event: Dict[str, Any], *, allowed_turn_ids: set[str]) -> Optional[Dict[str, Any]]:
    summary = str(event.get("summary") or "").strip()
    if not summary:
        return None
    desc = str(event.get("desc") or "").strip() or None
    event_type = str(event.get("event_type") or "").strip() or "Atomic"
    if event_type not in _ALLOWED_EVENT_TYPES:
        event_type = "Atomic"
    raw_turn_ids = event.get("source_turn_ids") or []
    turn_ids = [str(x).strip() for x in raw_turn_ids if str(x).strip()]
    turn_ids = [tid for tid in turn_ids if tid in allowed_turn_ids]
    status = str(event.get("evidence_status") or "").strip().lower() or "mapped"
    if status not in _ALLOWED_EVIDENCE_STATUS:
        status = "mapped" if turn_ids else "unmapped"
    if not turn_ids:
        status = "unmapped"
    ev_conf = _clamp01_or_none(event.get("event_confidence"))
    evd_conf = _clamp01_or_none(event.get("evidence_confidence"))
    participants = [str(x).strip() for x in (event.get("participants") or []) if str(x).strip()]
    topic_id = str(event.get("topic_id") or "").strip() or None
    topic_path = str(event.get("topic_path") or "").strip() or None
    tags = _normalize_str_list(event.get("tags"))
    keywords = _normalize_str_list(event.get("keywords"))
    time_bucket = _normalize_str_list(event.get("time_bucket"))
    tags_vocab_version = str(event.get("tags_vocab_version") or "").strip() or None
    time_hint = str(event.get("time_hint") or "").strip() or None
    return {
        "summary": summary,
        "desc": desc,
        "event_type": event_type,
        "event_confidence": ev_conf,
        "evidence_status": status,
        "evidence_confidence": evd_conf,
        "source_turn_ids": list(turn_ids),
        "evidence_count": len(turn_ids),
        "participants": participants,
        "time_hint": time_hint,
        "topic_id": topic_id,
        "topic_path": topic_path,
        "tags": tags or None,
        "keywords": keywords or None,
        "time_bucket": time_bucket or None,
        "tags_vocab_version": tags_vocab_version,
    }


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0:
        return 0.0
    return dot / denom


def _pick_min_span(hits: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
    hits_sorted = sorted(hits, key=lambda x: x[0])
    if len(hits_sorted) <= k:
        return hits_sorted
    best = None
    for i in range(0, len(hits_sorted) - k + 1):
        window = hits_sorted[i : i + k]
        span = window[-1][0] - window[0][0]
        if best is None or span < best[0]:
            best = (span, window)
    return best[1] if best else hits_sorted[:k]


def _align_events_with_turns(
    *,
    events: List[Dict[str, Any]],
    turns: Sequence[Dict[str, Any]],
    turn_ids: Sequence[str],
    min_score: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    cfg = load_memory_config()
    embed_settings = ((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("embedding", {}) or {}
    embed = build_embedding_from_settings(embed_settings)

    turn_texts = []
    for idx, t in enumerate(turns or [], start=1):
        tid = _turn_id_for_turn(t, idx)
        text = str(t.get("text") or t.get("content") or "").strip()
        turn_texts.append((tid, text))

    turn_vecs: Dict[str, List[float]] = {}
    for tid, text in turn_texts:
        if not text:
            continue
        turn_vecs[tid] = embed(text)

    for ev in events:
        if ev.get("source_turn_ids"):
            continue
        if str(ev.get("evidence_status") or "") == "mapped":
            continue
        query = f"{ev.get('summary') or ''} {ev.get('desc') or ''}".strip()
        if not query:
            ev["evidence_status"] = "unmapped"
            ev["evidence_count"] = 0
            continue
        qv = embed(query)
        scored: List[Tuple[int, float]] = []
        for idx, tid in enumerate(turn_ids, start=1):
            tv = turn_vecs.get(tid)
            if tv is None:
                continue
            score = _cosine(qv, tv)
            if score >= float(min_score):
                scored.append((idx, score))
        if not scored:
            ev["source_turn_ids"] = []
            ev["evidence_status"] = "unmapped"
            ev["evidence_confidence"] = 0.0
            ev["evidence_count"] = 0
            continue
        picked = _pick_min_span(scored, max(1, int(top_k)))
        picked_sorted = sorted(picked, key=lambda x: x[0])
        ev["source_turn_ids"] = [turn_ids[i - 1] for i, _ in picked_sorted if 1 <= i <= len(turn_ids)]
        ev["evidence_status"] = "weak"
        ev["evidence_confidence"] = max(score for _, score in picked_sorted) if picked_sorted else 0.0
        ev["evidence_count"] = len(ev["source_turn_ids"])
    return events


def _segment_turns(turns: Sequence[Dict[str, Any]], *, max_turns: int) -> List[Tuple[int, List[Dict[str, Any]]]]:
    if max_turns <= 0:
        return [list(turns or [])]
    turns_list = list(turns or [])
    if not turns_list:
        return []
    # Preserve session boundaries; only split within each session.
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    order: List[int] = []
    for t in turns_list:
        raw_idx = t.get("session_idx", t.get("session"))
        try:
            idx = int(raw_idx) if raw_idx is not None else 1
        except Exception:
            idx = 1
        if idx not in grouped:
            grouped[idx] = []
            order.append(idx)
        grouped[idx].append(t)

    segments: List[Tuple[int, List[Dict[str, Any]]]] = []
    for idx in order:
        sess_turns = grouped.get(idx, [])
        if len(sess_turns) <= max_turns:
            segments.append((idx, sess_turns))
            continue
        for i in range(0, len(sess_turns), max_turns):
            segments.append((idx, sess_turns[i : i + max_turns]))
    return segments


def build_dialog_tkg_event_extractor_v1_from_env(
    *,
    session_id: str,
    reference_time_iso: Optional[str] = None,
    trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_include_context: bool = False,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]:
    if adapter is None:
        adapter = build_llm_from_config("extract")
    if adapter is None:
        adapter = build_llm_from_env()
    if adapter is None:
        return None

    adapter_local = threading.local()
    adapter_local.adapter = adapter

    def _get_adapter() -> LLMAdapter:
        cached = getattr(adapter_local, "adapter", None)
        if cached is not None:
            return cached
        fresh = build_llm_from_config("extract")
        if fresh is None:
            fresh = build_llm_from_env()
        adapter_local.adapter = fresh or adapter
        return adapter_local.adapter

    settings = get_dialog_event_settings(load_memory_config())

    def _extract_segment(seg_idx: int, sess_idx: int, seg_turns: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        ctx, turn_ids = build_event_context(session_id=session_id, turns=seg_turns)
        if trace_hook is not None:
            try:
                trace_hook(
                    {
                        "stage": "build_event_context",
                        "session_id": str(session_id),
                        "segment_index": int(seg_idx),
                        "turns": len(seg_turns),
                        "context": ctx if bool(trace_include_context) else None,
                    }
                )
            except Exception:
                pass
        local_adapter = _get_adapter()
        raw = local_adapter.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            response_format={"type": "json_object"},
        )
        items = parse_event_json(raw)
        allowed = set(turn_ids)
        normalized: List[Dict[str, Any]] = []
        for it in items:
            ev = _normalize_event(it, allowed_turn_ids=allowed)
            if ev is None:
                continue
            normalized.append(ev)
        aligned = _align_events_with_turns(
            events=normalized,
            turns=seg_turns,
            turn_ids=turn_ids,
            min_score=float(settings.get("alignment_min_score", 0.35)),
            top_k=int(settings.get("alignment_top_k", 3)),
        )
        try:
            aligned = normalize_events(aligned)
        except Exception:
            pass
        return sess_idx, aligned

    def _extract(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        turns_list = list(turns or [])
        segments = _segment_turns(turns_list, max_turns=int(settings.get("segment_max_turns", 80)))
        all_events: List[Dict[str, Any]] = []
        per_session_counts: Dict[int, int] = {}
        max_events = int(settings.get("max_events_per_session", 0))
        concurrency = max(1, int(settings.get("event_extract_concurrency", 1)))
        if concurrency <= 1 or len(segments) <= 1:
            for seg_idx, (sess_idx, seg_turns) in enumerate(segments, start=1):
                _, events = _extract_segment(seg_idx, sess_idx, list(seg_turns))
                if max_events > 0:
                    count = per_session_counts.get(sess_idx, 0)
                    remaining = max(0, max_events - count)
                    if remaining <= 0:
                        continue
                    events = list(events[:remaining])
                    per_session_counts[sess_idx] = count + len(events)
                all_events.extend(events)
        else:
            results: Dict[int, Tuple[int, List[Dict[str, Any]]]] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {
                    ex.submit(_extract_segment, seg_idx, sess_idx, list(seg_turns)): seg_idx
                    for seg_idx, (sess_idx, seg_turns) in enumerate(segments, start=1)
                }
                for future in as_completed(futures):
                    seg_idx = futures[future]
                    results[seg_idx] = future.result()
            for seg_idx in sorted(results.keys()):
                sess_idx, events = results[seg_idx]
                if max_events > 0:
                    count = per_session_counts.get(sess_idx, 0)
                    remaining = max(0, max_events - count)
                    if remaining <= 0:
                        continue
                    events = list(events[:remaining])
                    per_session_counts[sess_idx] = count + len(events)
                all_events.extend(events)
        return all_events

    return _extract


__all__ = [
    "SYSTEM_PROMPT",
    "parse_event_json",
    "build_event_context",
    "build_dialog_tkg_event_extractor_v1_from_env",
]
