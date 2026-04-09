from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from modules.memory.application.knowledge_extractor_dialog_tkg_v1 import build_dialogue_context

_BENCHMARK_ROOT = Path(__file__).resolve().parents[4] / "benchmark"
pytestmark = pytest.mark.skipif(
    not _BENCHMARK_ROOT.exists(),
    reason="benchmark directory not present, skipping alignment tests",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_sample(sample_id: str) -> Dict[str, Any]:
    path = _repo_root() / "benchmark" / "data" / "locomo" / "raw" / "locomo10.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    for it in data:
        if isinstance(it, dict) and str(it.get("sample_id") or "") == sample_id:
            return it
    raise AssertionError(f"missing sample: {sample_id}")


def _normalize_reference_time(dt_str: str) -> str:
    # Mirror benchmark/scripts/step2_extract_facts.py::_normalize_reference_time (no import).
    if not dt_str:
        return ""
    try:
        match = re.match(
            r"(\\d+):(\\d+)\\s*(am|pm)\\s+on\\s+(\\d+)\\s+(\\w+),?\\s*(\\d{4})",
            dt_str.strip(),
            re.I,
        )
        if match:
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
        pass
    return dt_str.strip()


def _get_sorted_sessions(sample: Dict[str, Any]) -> List[tuple[int, str, List[Dict[str, Any]]]]:
    # Mirror benchmark/scripts/step2_extract_facts.py::_get_sorted_sessions (no import).
    conv = sample.get("conversation", {})
    assert isinstance(conv, dict)
    sessions: List[tuple[int, str, List[Dict[str, Any]]]] = []
    for k, v in conv.items():
        if not str(k).startswith("session_") or str(k).endswith("_date_time"):
            continue
        try:
            idx = int(str(k).split("_", 1)[1])
        except Exception:
            continue
        if isinstance(v, list):
            dt = str(conv.get(f"{k}_date_time", "") or "")
            sessions.append((idx, dt, list(v)))
    sessions.sort(key=lambda x: x[0])
    return sessions


def _expected_benchmark_v2_build_context(sample: Dict[str, Any], session_indices: List[int]) -> str:
    # Mirror benchmark/archive/v2/step2_extract_knowledge_tkg_v1.py::_build_context (no import).
    sample_id = sample.get("sample_id", "conv-0")
    conv = sample.get("conversation", {})
    assert isinstance(conv, dict)
    speaker_a = conv.get("speaker_a", "Speaker_A")
    speaker_b = conv.get("speaker_b", "Speaker_B")

    parts: List[str] = []
    parts.append(f"Sample ID: {sample_id}")
    parts.append(f"Speaker A Name: {speaker_a}")
    parts.append(f"Speaker B Name: {speaker_b}")
    parts.append(f"Sessions included: {session_indices}")
    parts.append("\n=== DIALOGUE ===")

    all_sessions = _get_sorted_sessions(sample)
    session_map = {s[0]: (s[1], s[2]) for s in all_sessions}
    for sess_idx in session_indices:
        if sess_idx not in session_map:
            continue
        date_time, turns = session_map[sess_idx]
        ref = _normalize_reference_time(str(date_time))
        if not ref:
            ref = datetime.now().strftime("%Y-%m-%d %H:%M")
        parts.append(f"\n[Session {sess_idx}] (Reference Time: {ref})")
        for t in turns:
            dia_id = t.get("dia_id", "")
            speaker = t.get("speaker", "")
            text = t.get("text", "")
            if t.get("blip_caption"):
                text = f"{text} [Image: {t['blip_caption']}]"
            parts.append(f"{dia_id} {speaker}: {text}")
    return "\n".join(parts)


def _build_turns_from_sample(sample: Dict[str, Any], session_indices: List[int]) -> List[Dict[str, Any]]:
    conv = sample.get("conversation") or {}
    assert isinstance(conv, dict)
    turns: List[Dict[str, Any]] = []
    for idx in session_indices:
        key = f"session_{int(idx)}"
        dt = str(conv.get(f"{key}_date_time", "") or "")
        sess_turns = conv.get(key) or []
        assert isinstance(sess_turns, list)
        for t in sess_turns:
            if not isinstance(t, dict):
                continue
            turns.append(
                {
                    "dia_id": t.get("dia_id"),
                    "speaker": t.get("speaker"),
                    "text": t.get("text"),
                    "blip_caption": t.get("blip_caption"),
                    "session_idx": int(idx),
                    "session_date_time": dt,
                }
            )
    return turns


def test_dialog_tkg_context_matches_benchmark_v2_for_two_sessions() -> None:
    sample = _load_sample("conv-26")
    expected = _expected_benchmark_v2_build_context(sample, [1, 2])
    turns = _build_turns_from_sample(sample, [1, 2])
    got = build_dialogue_context(session_id="conv-26", turns=turns, reference_time_iso=None)
    assert got == expected
