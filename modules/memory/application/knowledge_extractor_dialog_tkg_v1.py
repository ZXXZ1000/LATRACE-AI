from __future__ import annotations

"""Dialog TKG v1 knowledge extractor (benchmark-aligned).

This module intentionally mirrors:
- benchmark/archive/v2/prompts/tkg_knowledge_extractor_system_prompt_v1.txt

We keep the prompt here (not importing benchmark code) so that production usage does not depend on `benchmark/`.
Tests assert prompt equality to prevent drift.
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os

from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_env, build_llm_from_config
from modules.memory.application.config import load_memory_config, get_dialog_event_settings
from concurrent.futures import ThreadPoolExecutor, as_completed



_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_tkg_knowledge_extractor_system_prompt_v1.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def parse_knowledge_json(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of knowledge dicts (best-effort, schema-tolerant)."""
    try:
        data = json.loads(_remove_code_blocks(raw))
    except Exception:
        return []
    items = data.get("knowledge")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # Hard contract: statement + non-empty source_turn_ids
        statement = str(it.get("statement") or "").strip()
        stids = it.get("source_turn_ids") or []
        if not statement or not isinstance(stids, list) or not [x for x in stids if str(x).strip()]:
            continue
        out.append(dict(it))
    return out


def _normalize_reference_time(dt_str: str) -> str:
    """Normalize LoCoMo session date_time to 'YYYY-MM-DD HH:MM' (benchmark-aligned)."""
    if not dt_str:
        return ""
    try:
        match = re.match(
            r"(\\d+):(\\d+)\\s*(am|pm)\\s+on\\s+(\\d+)\\s+(\\w+),?\\s*(\\d{4})",
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


def _group_turns_by_session(turns: Sequence[Dict[str, Any]]) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
    """Group turns by session_idx, returning (sorted_session_indices, session_turns, session_date_time)."""
    session_turns: Dict[int, List[Dict[str, Any]]] = {}
    session_dt: Dict[int, str] = {}
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        raw_idx = t.get("session_idx", t.get("session"))
        try:
            idx = int(raw_idx) if raw_idx is not None else 1
        except Exception:
            idx = 1
        session_turns.setdefault(idx, []).append(t)
        dt = t.get("session_date_time") or t.get("session_datetime") or t.get("session_ref_time") or ""
        if dt and idx not in session_dt:
            session_dt[idx] = str(dt)
    if not session_turns:
        session_turns = {1: []}
    indices = sorted(session_turns.keys())
    return indices, session_turns, session_dt


def build_dialogue_context(
    *,
    session_id: str,
    turns: List[Dict[str, Any]],
    reference_time_iso: Optional[str] = None,
) -> str:
    """Build a benchmark-compatible context block for the extractor prompt.

    This mirrors benchmark/archive/v2/step2_extract_knowledge_tkg_v1.py::_build_context formatting:
    - multi-session header
    - per-session reference time normalization
    - optional image caption injection: `... [Image: <blip_caption>]`
    """
    speakers: List[str] = []
    for t in turns or []:
        s = str(t.get("speaker") or t.get("role") or "").strip()
        if s and s not in speakers:
            speakers.append(s)
    speaker_a = speakers[0] if len(speakers) >= 1 else "Speaker_A"
    speaker_b = speakers[1] if len(speakers) >= 2 else "Speaker_B"

    sess_indices, sess_turns, sess_dt = _group_turns_by_session(turns or [])

    parts: List[str] = []
    parts.append(f"Sample ID: {session_id}")
    parts.append(f"Speaker A Name: {speaker_a}")
    parts.append(f"Speaker B Name: {speaker_b}")
    parts.append(f"Sessions included: {list(sess_indices)}")
    parts.append("\n=== DIALOGUE ===")

    fallback_ref = ""
    if reference_time_iso:
        try:
            fallback_ref = datetime.fromisoformat(reference_time_iso).strftime("%Y-%m-%d %H:%M")
        except Exception:
            fallback_ref = str(reference_time_iso)
    if not fallback_ref:
        fallback_ref = datetime.now().strftime("%Y-%m-%d %H:%M")

    for sess_idx in sess_indices:
        dt_str = str(sess_dt.get(sess_idx, "") or "").strip()
        ref = _normalize_reference_time(dt_str) if dt_str else fallback_ref
        if not ref:
            ref = fallback_ref
        parts.append(f"\n[Session {sess_idx}] (Reference Time: {ref})")
        for t in sess_turns.get(sess_idx, []):
            dia_id = str(t.get("dia_id") or "").strip()
            speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"
            text = str(t.get("text") or t.get("content") or "").strip()
            cap = str(t.get("blip_caption") or "").strip()
            if cap and "[Image:" not in text:
                text = f"{text} [Image: {cap}]"
            parts.append(f"{dia_id} {speaker}: {text}")

    return "\n".join(parts)


def build_dialog_tkg_knowledge_extractor_v1_from_env(
    *,
    session_id: str,
    reference_time_iso: Optional[str] = None,
    trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    trace_include_context: bool = False,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]:
    """Build a TKG knowledge extractor using the project LLM adapter (returns None if LLM is not configured)."""
    # Try config-based extract LLM first (preferred)
    if adapter is None:
        adapter = build_llm_from_config("extract")
    # Fallback to env-based if config doesn't provide one
    if adapter is None:
        adapter = build_llm_from_env()
    if adapter is None:
        return None

    def _extract(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        turns_list = list(turns or [])
        sess_indices, sess_turns, _ = _group_turns_by_session(turns_list)
        try:
            max_sessions = int(os.environ.get("MEMORY_DIALOG_TKG_EXTRACT_SESSIONS_PER_CALL", "4") or "4")
        except Exception:
            max_sessions = 4
        max_sessions = max(1, int(max_sessions))

        all_items: List[Dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()

        settings = get_dialog_event_settings(load_memory_config())
        concurrency = max(1, int(settings.get("fact_extract_concurrency", 1)))

        def _run_chunk(chunk_indices: List[int], chunk_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            ctx = build_dialogue_context(
                session_id=str(session_id),
                turns=chunk_turns,
                reference_time_iso=reference_time_iso,
            )
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "build_context",
                            "session_id": str(session_id),
                            "chunk_indices": list(chunk_indices),
                            "turns": len(chunk_turns),
                            "context": ctx if bool(trace_include_context) else None,
                            "context_chars": len(ctx),
                        }
                    )
                except Exception:
                    pass
            raw = adapter.generate(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ctx},
                ],
                response_format={"type": "json_object"},
            )
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "llm_raw",
                            "session_id": str(session_id),
                            "chunk_indices": list(chunk_indices),
                            "raw": raw,
                        }
                    )
                except Exception:
                    pass
            items = parse_knowledge_json(raw)
            if trace_hook is not None:
                try:
                    trace_hook(
                        {
                            "stage": "parsed_items",
                            "session_id": str(session_id),
                            "chunk_indices": list(chunk_indices),
                            "items": list(items),
                        }
                    )
                except Exception:
                    pass
            return list(items)

        chunks: List[Tuple[List[int], List[Dict[str, Any]]]] = []
        for i in range(0, len(sess_indices), max_sessions):
            chunk_indices = sess_indices[i : i + max_sessions]
            chunk_turns: List[Dict[str, Any]] = []
            for idx in chunk_indices:
                chunk_turns.extend(list(sess_turns.get(idx, [])))
            chunks.append((list(chunk_indices), list(chunk_turns)))

        if concurrency <= 1 or len(chunks) <= 1:
            for chunk_indices, chunk_turns in chunks:
                items = _run_chunk(chunk_indices, chunk_turns)
                for it in items:
                    stmt = str(it.get("statement") or "").strip()
                    stids = tuple(str(x).strip() for x in (it.get("source_turn_ids") or []) if str(x).strip())
                    if not stmt or not stids:
                        continue
                    key = (stmt, stids)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_items.append(dict(it))
        else:
            collected: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_run_chunk, idxs, ts): idxs for idxs, ts in chunks}
                for future in as_completed(futures):
                    items = future.result()
                    collected.extend(items)
            for it in collected:
                stmt = str(it.get("statement") or "").strip()
                stids = tuple(str(x).strip() for x in (it.get("source_turn_ids") or []) if str(x).strip())
                if not stmt or not stids:
                    continue
                key = (stmt, stids)
                if key in seen:
                    continue
                seen.add(key)
                all_items.append(dict(it))

        if trace_hook is not None:
            try:
                trace_hook(
                    {
                        "stage": "dedup_done",
                        "session_id": str(session_id),
                        "facts": len(all_items),
                    }
                )
            except Exception:
                pass
        return all_items

    return _extract


__all__ = [
    "SYSTEM_PROMPT",
    "build_dialogue_context",
    "parse_knowledge_json",
    "build_dialog_tkg_knowledge_extractor_v1_from_env",
]
