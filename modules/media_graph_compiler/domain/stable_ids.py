from __future__ import annotations

import hashlib
from typing import Any


def _canon_part(part: Any) -> str:
    if part is None:
        return "<none>"
    if isinstance(part, str):
        return part
    if isinstance(part, bool):
        return "1" if part else "0"
    if isinstance(part, int):
        return str(part)
    if isinstance(part, float):
        return f"{part:.3f}"
    if isinstance(part, (list, tuple)):
        return "[" + ",".join(_canon_part(x) for x in part) + "]"
    return str(part)


def stable_hash_id(prefix: str, *parts: Any, digest_len: int = 16) -> str:
    raw = "|".join(_canon_part(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[: int(digest_len)]
    return f"{prefix}_{digest}"


def stable_window_id(source_id: str, t_start_s: float, t_end_s: float, modality: str) -> str:
    return stable_hash_id("window", source_id, t_start_s, t_end_s, modality)


def stable_utterance_id(
    source_id: str,
    speaker_track_id: str | None,
    t_start_s: float | None,
    t_end_s: float | None,
    text: str,
) -> str:
    return stable_hash_id("utt", source_id, speaker_track_id, t_start_s, t_end_s, text)


def stable_visual_entity_id(source_id: str, track_id: str) -> str:
    return stable_hash_id("personv", source_id, track_id)


def stable_speaker_entity_id(source_id: str, track_id: str) -> str:
    return stable_hash_id("persons", source_id, track_id)


def stable_character_candidate_id(source_id: str, visual_track_id: str | None, speaker_track_id: str | None) -> str:
    return stable_hash_id("character", source_id, visual_track_id, speaker_track_id)


def stable_event_id(source_id: str, window_id: str, summary: str) -> str:
    return stable_hash_id("evt", source_id, window_id, summary)
