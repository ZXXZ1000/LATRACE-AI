from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from modules.memory.application.config import load_memory_config, get_dialog_event_settings
from modules.memory.application.embedding_adapter import build_embedding_from_settings
from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_config, build_llm_from_env
from modules.memory.application.topic_normalizer import normalize_events
import yaml


_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "dialog_tkg_unified_extractor_system_prompt_v1.txt"
)
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_ALLOWED_EVENT_TYPES = {"Atomic", "Process", "Composite"}
_ALLOWED_EVIDENCE_STATUS = {"mapped", "weak", "unmapped"}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def parse_unified_json(raw: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        data = json.loads(_remove_code_blocks(raw))
    except Exception:
        return {"events": [], "knowledge": [], "states": []}
    if not isinstance(data, dict):
        return {"events": [], "knowledge": [], "states": []}
    events = data.get("events")
    knowledge = data.get("knowledge")
    states = data.get("states")
    if not isinstance(events, list):
        events = []
    if not isinstance(knowledge, list):
        knowledge = []
    if not isinstance(states, list):
        states = []
    return {
        "events": [dict(it) for it in events if isinstance(it, dict)],
        "knowledge": [dict(it) for it in knowledge if isinstance(it, dict)],
        "states": [dict(it) for it in states if isinstance(it, dict)],
    }


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _detect_lang(text: str) -> str:
    # Minimal heuristic: CJK -> zh, otherwise en.
    if _contains_cjk(text):
        return "zh"
    return "en"


def _expected_lang_for_turns(turn_ids: Sequence[str], lang_map: Dict[str, str]) -> Optional[str]:
    zh = 0
    en = 0
    for tid in turn_ids:
        lang = lang_map.get(str(tid))
        if lang == "zh":
            zh += 1
        elif lang == "en":
            en += 1
    if zh > 0 and en > 0:
        if zh == en:
            return "mixed"
        return "zh" if zh > en else "en"
    if zh > 0:
        return "zh"
    if en > 0:
        return "en"
    return None


def _text_matches_lang(text: str, expected: str) -> bool:
    if not text:
        return True
    if expected == "zh":
        return _contains_cjk(text)
    if expected == "en":
        return not _contains_cjk(text)
    return True


def _build_turn_lang_map(turns_with_index: Sequence[Tuple[int, Dict[str, Any]]]) -> Dict[str, str]:
    lang_map: Dict[str, str] = {}
    for idx, t in turns_with_index:
        tid = _turn_id_for_turn(t, idx)
        text = str(t.get("text") or t.get("content") or "")
        lang_map[tid] = _detect_lang(text)
    return lang_map


def _validate_language_consistency(
    *,
    events: Sequence[Dict[str, Any]],
    knowledge: Sequence[Dict[str, Any]],
    turn_lang_map: Dict[str, str],
) -> Tuple[bool, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    for ev in events or []:
        turn_ids = [str(x).strip() for x in (ev.get("source_turn_ids") or []) if str(x).strip()]
        expected = _expected_lang_for_turns(turn_ids, turn_lang_map)
        if expected in {None, "mixed"}:
            continue
        summary = str(ev.get("summary") or "").strip()
        desc = str(ev.get("desc") or "").strip()
        if summary and not _text_matches_lang(summary, expected):
            errors.append({"kind": "event", "field": "summary", "expected": expected, "turn_ids": turn_ids, "text": summary[:80]})
            continue
        if desc and not _text_matches_lang(desc, expected):
            errors.append({"kind": "event", "field": "desc", "expected": expected, "turn_ids": turn_ids, "text": desc[:80]})
            continue
    for kn in knowledge or []:
        turn_ids = [str(x).strip() for x in (kn.get("source_turn_ids") or []) if str(x).strip()]
        expected = _expected_lang_for_turns(turn_ids, turn_lang_map)
        if expected in {None, "mixed"}:
            continue
        stmt = str(kn.get("statement") or "").strip()
        if stmt and not _text_matches_lang(stmt, expected):
            errors.append({"kind": "knowledge", "field": "statement", "expected": expected, "turn_ids": turn_ids, "text": stmt[:80]})
            continue
    return (len(errors) == 0), errors


def _schema_stats(events: Sequence[Dict[str, Any]], knowledge: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    stats = {
        "events_total": 0,
        "events_missing_summary": 0,
        "events_missing_source_turns": 0,
        "knowledge_total": 0,
        "knowledge_missing_statement": 0,
        "knowledge_missing_source_turns": 0,
    }
    for ev in events or []:
        stats["events_total"] += 1
        if not str(ev.get("summary") or "").strip():
            stats["events_missing_summary"] += 1
        if not list(ev.get("source_turn_ids") or []):
            stats["events_missing_source_turns"] += 1
    for kn in knowledge or []:
        stats["knowledge_total"] += 1
        if not str(kn.get("statement") or "").strip():
            stats["knowledge_missing_statement"] += 1
        if not list(kn.get("source_turn_ids") or []):
            stats["knowledge_missing_source_turns"] += 1
    return stats


_STATE_VOCAB: Optional[Dict[str, Any]] = None


def _load_state_vocab() -> Dict[str, Any]:
    global _STATE_VOCAB
    if _STATE_VOCAB is not None:
        return _STATE_VOCAB
    root = Path(__file__).resolve().parents[3]
    fp = root / "modules" / "memory" / "vocab" / "state_properties.yaml"
    data: Dict[str, Any] = {}
    try:
        data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    props = {}
    allow_raw = {}
    mvp_props = set()
    for it in data.get("properties") or []:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        allowed = it.get("allowed_values") or []
        props[name] = [str(x) for x in allowed if str(x).strip()]
        allow_raw[name] = bool(it.get("allow_raw_value"))
        if bool(it.get("mvp")):
            mvp_props.add(name)
    _STATE_VOCAB = {"allowed_values": props, "allow_raw": allow_raw, "mvp_props": sorted(mvp_props)}
    return _STATE_VOCAB


def _state_allowed_properties() -> Optional[set[str]]:
    raw = os.getenv("MEMORY_STATE_PROPERTIES")
    if not raw:
        vocab = _load_state_vocab()
        mvp_props = vocab.get("mvp_props") or []
        if mvp_props:
            return set([str(x).strip() for x in mvp_props if str(x).strip()])
        return None
    out = {s.strip() for s in raw.split(",") if s.strip()}
    return out or None


def _normalize_state(item: Dict[str, Any], *, allowed_turn_ids: set[str]) -> Optional[Dict[str, Any]]:
    subject_ref = str(item.get("subject_ref") or item.get("subject") or "").strip()
    prop = str(item.get("property") or "").strip()
    value = str(item.get("value") or "").strip()
    raw_value = str(item.get("raw_value") or "").strip()
    time_hint = str(item.get("time_hint") or "").strip()
    negated = bool(item.get("negated") or False)
    if not subject_ref or not prop or not value or negated:
        return None
    allowlist = _state_allowed_properties()
    if allowlist and prop not in allowlist:
        return None
    vocab = _load_state_vocab()
    allowed_values = set(vocab.get("allowed_values", {}).get(prop) or [])
    allow_raw = bool(vocab.get("allow_raw", {}).get(prop, False))
    if allowed_values and value not in allowed_values:
        if allow_raw:
            raw_value = raw_value or value
            value = "_other"
        else:
            return None
    turn_ids = [str(x).strip() for x in (item.get("source_turn_ids") or []) if str(x).strip()]
    turn_ids = [t for t in turn_ids if t in allowed_turn_ids]
    if not turn_ids:
        return None
    conf_raw = item.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except Exception:
        confidence = None
    out = {
        "subject_ref": subject_ref,
        "property": prop,
        "value": value,
        "raw_value": raw_value or None,
        "confidence": confidence,
        "time_hint": time_hint or None,
        "negated": negated,
        "source_turn_ids": turn_ids,
    }
    return out


def _turn_id_for_turn(t: Dict[str, Any], idx: int) -> str:
    dia_id = str(t.get("dia_id") or "").strip()
    if dia_id:
        return dia_id
    turn_id = str(t.get("turn_id") or "").strip()
    if turn_id:
        return turn_id
    return f"t{idx}"


def _normalize_reference_time(dt_str: str) -> str:
    if not dt_str:
        return ""
    try:
        match = re.match(
            r"(\d+):(\d+)\s*(am|pm)\s+on\s+(\d+)\s+(\w+),?\s*(\d{4})",
            dt_str.strip(),
            re.I,
        )
        if not match:
            return dt_str.strip()
        hour, minute, ampm, day, month_name, year = match.groups()
        hour_i = int(hour)
        minute_i = int(minute)
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
        dt = datetime(int(year), month_i, int(day), hour_i, minute_i)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str.strip()


def _build_context(
    *,
    session_id: str,
    session_idx: int,
    turns_with_index: Sequence[Tuple[int, Dict[str, Any]]],
    reference_time_iso: Optional[str],
) -> Tuple[str, List[str]]:
    speakers: List[str] = []
    for _, t in turns_with_index:
        s = str(t.get("speaker") or t.get("role") or "").strip()
        if s and s not in speakers:
            speakers.append(s)
    speaker_a = speakers[0] if len(speakers) >= 1 else "Speaker_A"
    speaker_b = speakers[1] if len(speakers) >= 2 else "Speaker_B"

    ref = ""
    dt_str = ""
    for _, t in turns_with_index:
        dt_str = str(t.get("session_date_time") or t.get("session_datetime") or t.get("session_ref_time") or "").strip()
        if dt_str:
            break
    if dt_str:
        ref = _normalize_reference_time(dt_str)
    if not ref and reference_time_iso:
        try:
            ref = datetime.fromisoformat(reference_time_iso).strftime("%Y-%m-%d %H:%M")
        except Exception:
            ref = str(reference_time_iso)
    if not ref:
        ref = datetime.now().strftime("%Y-%m-%d %H:%M")

    turn_ids: List[str] = []
    lines: List[str] = []
    lines.append(f"Sample ID: {session_id}")
    lines.append(f"Session Index: {session_idx}")
    lines.append(f"Speaker A Name: {speaker_a}")
    lines.append(f"Speaker B Name: {speaker_b}")
    lines.append(f"Reference Time: {ref}")

    for idx, t in turns_with_index:
        turn_ids.append(_turn_id_for_turn(t, idx))
    lines.append("Allowed turn_ids (you must choose from this list only):")
    lines.append(f"{turn_ids}")
    lines.append("")
    lines.append("Dialogue:")
    for idx, t in turns_with_index:
        tid = _turn_id_for_turn(t, idx)
        speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"
        text = str(t.get("text") or t.get("content") or "").strip()
        cap = str(t.get("blip_caption") or "").strip()
        if cap and "[Image:" not in text:
            text = f"{text} [Image: {cap}]"
        if not text:
            continue
        lines.append(f"{tid} {speaker}: {text}")
    return "\n".join(lines), list(turn_ids)


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


def _normalize_knowledge(item: Dict[str, Any], *, allowed_turn_ids: set[str]) -> Optional[Dict[str, Any]]:
    statement = str(item.get("statement") or "").strip()
    if not statement:
        return None
    raw_turn_ids = item.get("source_turn_ids") or []
    turn_ids = [str(x).strip() for x in raw_turn_ids if str(x).strip()]
    turn_ids = [tid for tid in turn_ids if tid in allowed_turn_ids]
    if not turn_ids:
        return None
    out = dict(item)
    out["statement"] = statement
    out["source_turn_ids"] = list(turn_ids)
    return out


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
    embed: Optional[Callable[[str], List[float]]] = None,
    batch_size: Optional[int] = None,
    embed_sem: Optional[threading.Semaphore] = None,
    embed_concurrency: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not events:
        return events
    pending = [
        ev
        for ev in events
        if (not ev.get("source_turn_ids"))
        and str(ev.get("evidence_status") or "").strip().lower() != "mapped"
    ]
    if not pending:
        return events
    if embed is None:
        cfg = load_memory_config()
        embed_settings = ((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("embedding", {}) or {}
        embed = build_embedding_from_settings(embed_settings)
    try:
        bsz = int(batch_size or 0)
    except Exception:
        bsz = 0
    if bsz <= 0:
        bsz = 128

    def _with_sem(fn: Callable[[], List[List[float]]]) -> List[List[float]]:
        if embed_sem is None:
            return fn()
        embed_sem.acquire()
        try:
            return fn()
        finally:
            embed_sem.release()

    def _encode_batch(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            conc = int(embed_concurrency or 0)
        except Exception:
            conc = 0
        if conc <= 0:
            conc = 1
        encode_batch = getattr(embed, "encode_batch", None)
        if callable(encode_batch):
            out: List[List[float]] = []
            chunks = [texts[i : i + bsz] for i in range(0, len(texts), bsz)] if bsz > 0 else [texts]
            if len(chunks) <= 1 or conc <= 1:
                for chunk in chunks:
                    def _call() -> List[List[float]]:
                        try:
                            return list(encode_batch(chunk, bsz=bsz))
                        except TypeError:
                            return list(encode_batch(chunk))
                    out.extend(_with_sem(_call))
                return out
            # parallelize chunk embedding
            results: Dict[int, List[List[float]]] = {}
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futures = {}
                for idx, chunk in enumerate(chunks):
                    def _call_chunk(c=chunk) -> List[List[float]]:
                        def _call() -> List[List[float]]:
                            try:
                                return list(encode_batch(c, bsz=bsz))
                            except TypeError:
                                return list(encode_batch(c))
                        return _with_sem(_call)
                    futures[ex.submit(_call_chunk)] = idx
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
            for idx in sorted(results.keys()):
                out.extend(results[idx])
            return out
        out: List[List[float]] = []
        if conc <= 1 or len(texts) <= 1:
            for text in texts:
                def _call_one() -> List[List[float]]:
                    return [list(embed(text))]
                out.extend(_with_sem(_call_one))
            return out
        results: Dict[int, List[float]] = {}
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futures = {}
            for idx, text in enumerate(texts):
                def _call_one(t=text) -> List[float]:
                    def _call() -> List[List[float]]:
                        return [list(embed(t))]
                    return _with_sem(_call)[0]
                futures[ex.submit(_call_one)] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        for idx in sorted(results.keys()):
            out.append(results[idx])
        return out

    turn_texts = []
    for idx, t in enumerate(turns or [], start=1):
        tid = turn_ids[idx - 1] if idx - 1 < len(turn_ids) else _turn_id_for_turn(t, idx)
        text = str(t.get("text") or t.get("content") or "").strip()
        turn_texts.append((tid, text))

    turn_vecs: Dict[str, List[float]] = {}
    turn_payload = [(tid, text) for tid, text in turn_texts if text]
    if turn_payload:
        vecs = _encode_batch([text for _, text in turn_payload])
        dim = len(vecs[0]) if vecs else 1
        if len(vecs) < len(turn_payload):
            vecs.extend([[0.0] * dim for _ in range(len(turn_payload) - len(vecs))])
        for (tid, _), vec in zip(turn_payload, vecs):
            turn_vecs[tid] = list(vec)
    if not turn_vecs:
        for ev in pending:
            ev["source_turn_ids"] = []
            ev["evidence_status"] = "unmapped"
            ev["evidence_confidence"] = 0.0
            ev["evidence_count"] = 0
        return events

    queries: List[str] = []
    query_events: List[Dict[str, Any]] = []
    for ev in pending:
        query = f"{ev.get('summary') or ''} {ev.get('desc') or ''}".strip()
        if not query:
            ev["evidence_status"] = "unmapped"
            ev["evidence_count"] = 0
            continue
        queries.append(query)
        query_events.append(ev)
    if not queries:
        return events

    query_vecs = _encode_batch(queries)
    dim = len(query_vecs[0]) if query_vecs else 1
    if len(query_vecs) < len(query_events):
        query_vecs.extend([[0.0] * dim for _ in range(len(query_events) - len(query_vecs))])

    for ev, qv in zip(query_events, query_vecs):
        scored: List[Tuple[int, float]] = []
        for idx, tid in enumerate(turn_ids, start=1):
            tv = turn_vecs.get(tid)
            if tv is None:
                continue
            score = _cosine(list(qv), tv)
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


def _segment_turns(
    turns: Sequence[Dict[str, Any]],
    *,
    max_turns: int,
) -> List[Tuple[int, List[Tuple[int, Dict[str, Any]]]]]:
    turns_list = list(turns or [])
    if not turns_list:
        return []
    max_limit: Optional[int] = int(max_turns) if max_turns and max_turns > 0 else None

    grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    order: List[int] = []
    for idx, t in enumerate(turns_list, start=1):
        raw_idx = t.get("session_idx", t.get("session"))
        try:
            sess_idx = int(raw_idx) if raw_idx is not None else 1
        except Exception:
            sess_idx = 1
        if sess_idx not in grouped:
            grouped[sess_idx] = []
            order.append(sess_idx)
        grouped[sess_idx].append((idx, t))

    segments: List[Tuple[int, List[Tuple[int, Dict[str, Any]]]]] = []
    for sess_idx in order:
        sess_turns = grouped.get(sess_idx, [])
        if max_limit is None or len(sess_turns) <= max_limit:
            segments.append((sess_idx, sess_turns))
            continue
        for i in range(0, len(sess_turns), max_limit or len(sess_turns)):
            segments.append((sess_idx, sess_turns[i : i + (max_limit or len(sess_turns))]))
    return segments


def build_dialog_tkg_unified_extractor_v1_from_env(
    *,
    session_id: str,
    reference_time_iso: Optional[str] = None,
    trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_include_context: bool = False,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[Callable[[List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]]:
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

    fallback_reference_time_iso = str(reference_time_iso or "").strip()
    if not fallback_reference_time_iso:
        fallback_reference_time_iso = datetime.now().isoformat()

    settings = get_dialog_event_settings(load_memory_config())
    align_batch_size = int(settings.get("alignment_embed_batch_size", 128))
    embed_conc = int(settings.get("alignment_embed_concurrency", 8))
    embed_sem = threading.Semaphore(max(1, embed_conc)) if embed_conc > 0 else None
    cfg = load_memory_config()
    embed_settings = ((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("embedding", {}) or {}
    embed = build_embedding_from_settings(embed_settings)

    def _extract_segment(
        seg_idx: int,
        session_idx: int,
        seg_turns: List[Tuple[int, Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ctx, turn_ids = _build_context(
            session_id=str(session_id),
            session_idx=int(session_idx),
            turns_with_index=seg_turns,
            reference_time_iso=fallback_reference_time_iso,
        )
        if trace_hook is not None:
            try:
                trace_hook(
                    {
                        "stage": "build_unified_context",
                        "session_id": str(session_id),
                        "segment_index": int(seg_idx),
                        "session_index": int(session_idx),
                        "turns": len(seg_turns),
                        "context": ctx if bool(trace_include_context) else None,
                        "context_chars": len(ctx),
                    }
                )
            except Exception:
                pass
        local_adapter = _get_adapter()
        lang_map = _build_turn_lang_map(seg_turns)

        def _run_once(prompt_ctx: str, *, attempt: int) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
            raw_local = local_adapter.generate(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_ctx},
                ],
                response_format={"type": "json_object"},
            )
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "llm_raw",
                            "session_id": str(session_id),
                            "segment_index": int(seg_idx),
                            "session_index": int(session_idx),
                            "attempt": int(attempt),
                            "raw": raw_local,
                        }
                    )
                except Exception:
                    pass
            parsed_local = parse_unified_json(raw_local)
            return parsed_local, raw_local

        parsed, raw = _run_once(ctx, attempt=1)
        if trace_hook is not None:
            try:
                trace_hook(
                    {
                        "stage": "llm_schema",
                        "session_id": str(session_id),
                        "segment_index": int(seg_idx),
                        "session_index": int(session_idx),
                        "stats": _schema_stats(parsed.get("events") or [], parsed.get("knowledge") or []),
                    }
                )
            except Exception:
                pass

        lang_ok, lang_errors = _validate_language_consistency(
            events=parsed.get("events") or [],
            knowledge=parsed.get("knowledge") or [],
            turn_lang_map=lang_map,
        )
        if not lang_ok:
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "llm_language_violation",
                            "session_id": str(session_id),
                            "segment_index": int(seg_idx),
                            "session_index": int(session_idx),
                            "errors": list(lang_errors[:5]),
                        }
                    )
                except Exception:
                    pass
            lang_lines = ["Language of each turn_id (STRICT):"]
            for tid, lang in lang_map.items():
                lang_lines.append(f"- {tid}: {lang}")
            retry_ctx = (
                ctx
                + "\n\nVALIDATION ERROR: Output language must match each item's source_turn_ids. "
                + "Regenerate STRICT JSON only.\n"
                + "\n".join(lang_lines)
            )
            parsed, _ = _run_once(retry_ctx, attempt=2)
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "llm_language_retry",
                            "session_id": str(session_id),
                            "segment_index": int(seg_idx),
                            "session_index": int(session_idx),
                            "ok": bool(
                                _validate_language_consistency(
                                    events=parsed.get("events") or [],
                                    knowledge=parsed.get("knowledge") or [],
                                    turn_lang_map=lang_map,
                                )[0]
                            ),
                        }
                    )
                except Exception:
                    pass
        allowed = set(turn_ids)

        events_norm: List[Dict[str, Any]] = []
        for it in parsed.get("events", []):
            ev = _normalize_event(it, allowed_turn_ids=allowed)
            if ev is None:
                continue
            events_norm.append(ev)

        knowledge_norm: List[Dict[str, Any]] = []
        for it in parsed.get("knowledge", []):
            kn = _normalize_knowledge(it, allowed_turn_ids=allowed)
            if kn is None:
                continue
            knowledge_norm.append(kn)

        states_norm: List[Dict[str, Any]] = []
        for it in parsed.get("states", []):
            st = _normalize_state(it, allowed_turn_ids=allowed)
            if st is None:
                continue
            states_norm.append(st)

        events_norm = _align_events_with_turns(
            events=events_norm,
            turns=[t for _, t in seg_turns],
            turn_ids=turn_ids,
            min_score=float(settings.get("alignment_min_score", 0.35)),
            top_k=int(settings.get("alignment_top_k", 3)),
            embed=embed,
            batch_size=align_batch_size,
            embed_sem=embed_sem,
            embed_concurrency=embed_conc,
        )
        try:
            events_norm = normalize_events(events_norm)
        except Exception:
            pass
        return events_norm, knowledge_norm, states_norm

    def _extract(turns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        turns_list = list(turns or [])
        segments = _segment_turns(turns_list, max_turns=int(settings.get("segment_max_turns", 80)))

        all_events: List[Dict[str, Any]] = []
        all_knowledge: List[Dict[str, Any]] = []
        all_states: List[Dict[str, Any]] = []
        per_session_counts: Dict[int, int] = {}
        max_events = int(settings.get("max_events_per_session", 0))
        concurrency = max(1, int(settings.get("event_extract_concurrency", 1)))

        if concurrency <= 1 or len(segments) <= 1:
            for seg_idx, (sess_idx, seg_turns) in enumerate(segments, start=1):
                evs, kns, sts = _extract_segment(seg_idx, sess_idx, list(seg_turns))
                if max_events > 0:
                    count = per_session_counts.get(sess_idx, 0)
                    remaining = max(0, max_events - count)
                    if remaining <= 0:
                        evs = []
                    else:
                        evs = list(evs[:remaining])
                        per_session_counts[sess_idx] = count + len(evs)
                all_events.extend(evs)
                all_knowledge.extend(kns)
                all_states.extend(sts)
        else:
            results: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {
                    ex.submit(_extract_segment, seg_idx, sess_idx, list(seg_turns)): seg_idx
                    for seg_idx, (sess_idx, seg_turns) in enumerate(segments, start=1)
                }
                for future in as_completed(futures):
                    seg_idx = futures[future]
                    results[seg_idx] = future.result()
            for seg_idx in sorted(results.keys()):
                evs, kns, sts = results[seg_idx]
                sess_idx = segments[seg_idx - 1][0]
                if max_events > 0:
                    count = per_session_counts.get(sess_idx, 0)
                    remaining = max(0, max_events - count)
                    if remaining <= 0:
                        evs = []
                    else:
                        evs = list(evs[:remaining])
                        per_session_counts[sess_idx] = count + len(evs)
                all_events.extend(evs)
                all_knowledge.extend(kns)
                all_states.extend(sts)

        if all_knowledge:
            dedup: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
            for it in all_knowledge:
                stmt = str(it.get("statement") or "").strip()
                stids = tuple(str(x).strip() for x in (it.get("source_turn_ids") or []) if str(x).strip())
                if not stmt or not stids:
                    continue
                key = (stmt, stids)
                if key in dedup:
                    continue
                dedup[key] = dict(it)
            all_knowledge = list(dedup.values())

        return {"events": all_events, "knowledge": all_knowledge, "states": all_states}

    return _extract


__all__ = [
    "SYSTEM_PROMPT",
    "parse_unified_json",
    "build_dialog_tkg_unified_extractor_v1_from_env",
]
