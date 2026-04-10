from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Mapping, Sequence


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))


def _merge_spans(
    spans: Sequence[Sequence[float]],
    *,
    max_gap_s: float,
    min_duration_s: float,
) -> List[Dict[str, float]]:
    normalized = sorted(
        (
            (
                max(0.0, float(item[0])),
                max(max(0.0, float(item[0])), float(item[1])),
            )
            for item in spans
            if len(item) >= 2
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not normalized:
        return []
    merged: List[List[float]] = [[normalized[0][0], normalized[0][1]]]
    for start_s, end_s in normalized[1:]:
        last = merged[-1]
        if start_s - last[1] <= float(max_gap_s):
            last[1] = max(last[1], end_s)
            continue
        merged.append([start_s, end_s])
    return [
        {
            "t_start_s": start_s,
            "t_end_s": end_s,
            "duration_s": max(0.0, end_s - start_s),
        }
        for start_s, end_s in merged
        if (end_s - start_s) >= float(min_duration_s)
    ]


def detect_speech_spans_b64(
    audio_b64: bytes,
    *,
    min_speech_s: float = 0.35,
    min_silence_s: float = 0.45,
    pad_s: float = 0.12,
    seek_step_ms: int = 20,
    merge_gap_s: float = 0.18,
) -> List[Dict[str, float]]:
    try:
        from pydub import AudioSegment, silence  # type: ignore
    except Exception:
        return []

    try:
        audio = AudioSegment.from_wav(io.BytesIO(base64.b64decode(audio_b64)))
    except Exception:
        return []
    if len(audio) <= 0:
        return []

    dbfs = float(audio.dBFS) if audio.dBFS != float("-inf") else -60.0
    silence_thresh = max(-48.0, min(-24.0, dbfs - 14.0))
    raw_spans_ms = silence.detect_nonsilent(
        audio,
        min_silence_len=max(120, int(float(min_silence_s) * 1000.0)),
        silence_thresh=silence_thresh,
        seek_step=max(5, int(seek_step_ms)),
    )
    if not raw_spans_ms:
        return []

    padded_spans = []
    audio_duration_s = float(len(audio)) / 1000.0
    for start_ms, end_ms in raw_spans_ms:
        start_s = max(0.0, (float(start_ms) / 1000.0) - float(pad_s))
        end_s = min(audio_duration_s, (float(end_ms) / 1000.0) + float(pad_s))
        padded_spans.append((start_s, end_s))
    return _merge_spans(
        padded_spans,
        max_gap_s=merge_gap_s,
        min_duration_s=min_speech_s,
    )


def compact_audio_b64_to_speech_islands(
    audio_b64: bytes,
    spans: Sequence[Mapping[str, Any]],
    *,
    join_gap_s: float = 0.12,
) -> Dict[str, Any]:
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception:
        return {
            "audio_b64": audio_b64,
            "timeline_map": [],
            "original_duration_s": 0.0,
            "compacted_duration_s": 0.0,
            "speech_ratio": 1.0,
            "applied": False,
        }

    try:
        audio = AudioSegment.from_wav(io.BytesIO(base64.b64decode(audio_b64)))
    except Exception:
        return {
            "audio_b64": audio_b64,
            "timeline_map": [],
            "original_duration_s": 0.0,
            "compacted_duration_s": 0.0,
            "speech_ratio": 1.0,
            "applied": False,
        }

    original_duration_s = float(len(audio)) / 1000.0
    if len(audio) <= 0 or not spans:
        return {
            "audio_b64": audio_b64,
            "timeline_map": [],
            "original_duration_s": original_duration_s,
            "compacted_duration_s": original_duration_s,
            "speech_ratio": 1.0,
            "applied": False,
        }

    output = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    timeline_map: List[Dict[str, float]] = []
    cursor_ms = 0.0
    join_gap_ms = max(0.0, float(join_gap_s) * 1000.0)
    total_speech_ms = 0.0

    for index, item in enumerate(spans):
        start_ms = max(0.0, float(item.get("t_start_s") or 0.0) * 1000.0)
        end_ms = max(start_ms, float(item.get("t_end_s") or 0.0) * 1000.0)
        if end_ms <= start_ms:
            continue
        segment = audio[int(start_ms) : int(end_ms)]
        if len(segment) <= 0:
            continue
        segment_ms = float(len(segment))
        timeline_map.append(
            {
                "compact_t_start_s": cursor_ms / 1000.0,
                "compact_t_end_s": (cursor_ms + segment_ms) / 1000.0,
                "original_t_start_s": start_ms / 1000.0,
                "original_t_end_s": end_ms / 1000.0,
                "speech_island_index": float(index),
            }
        )
        output += segment
        cursor_ms += segment_ms
        total_speech_ms += segment_ms
        if join_gap_ms > 0.0 and index < len(spans) - 1:
            output += AudioSegment.silent(duration=join_gap_ms, frame_rate=audio.frame_rate)
            cursor_ms += join_gap_ms

    if not timeline_map or total_speech_ms <= 0.0:
        return {
            "audio_b64": audio_b64,
            "timeline_map": [],
            "original_duration_s": original_duration_s,
            "compacted_duration_s": original_duration_s,
            "speech_ratio": 1.0,
            "applied": False,
        }

    buffer = io.BytesIO()
    output.export(buffer, format="wav")
    buffer.seek(0)
    compacted_duration_s = float(len(output)) / 1000.0
    return {
        "audio_b64": base64.b64encode(buffer.read()),
        "timeline_map": timeline_map,
        "original_duration_s": original_duration_s,
        "compacted_duration_s": compacted_duration_s,
        "speech_ratio": max(0.0, min(1.0, total_speech_ms / max(1.0, float(len(audio))))),
        "applied": compacted_duration_s > 0.0 and compacted_duration_s < original_duration_s,
    }


def remap_compacted_segments_to_original(
    segments: Sequence[Mapping[str, Any]],
    timeline_map: Sequence[Mapping[str, Any]],
    *,
    min_duration_s: float = 0.4,
) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    if not timeline_map:
        return [dict(item) for item in segments]

    for item in segments:
        track_id = str(item.get("track_id") or "").strip()
        seg_start = float(item.get("t_start_s") or 0.0)
        seg_end = max(seg_start, float(item.get("t_end_s") or seg_start))
        metadata = dict(item.get("metadata") or {})
        for span in timeline_map:
            compact_start = float(span.get("compact_t_start_s") or 0.0)
            compact_end = float(span.get("compact_t_end_s") or compact_start)
            overlap_s = _overlap(seg_start, seg_end, compact_start, compact_end)
            if overlap_s <= 0.0:
                continue
            original_start = float(span.get("original_t_start_s") or 0.0) + max(
                0.0, seg_start - compact_start
            )
            original_end = float(span.get("original_t_start_s") or 0.0) + min(
                compact_end, seg_end
            ) - compact_start
            if (original_end - original_start) < float(min_duration_s):
                continue
            remapped.append(
                {
                    "track_id": track_id,
                    "t_start_s": original_start,
                    "t_end_s": original_end,
                    "metadata": {
                        **metadata,
                        "speech_island_index": int(span.get("speech_island_index") or 0),
                        "timeline_remapped": True,
                    },
                }
            )
    return remapped


__all__ = [
    "compact_audio_b64_to_speech_islands",
    "detect_speech_spans_b64",
    "remap_compacted_segments_to_original",
]
