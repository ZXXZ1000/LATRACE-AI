from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from modules.media_graph_compiler.adapters.ops.audio_segments import (
    audio_duration_seconds,
)

_DIARIZATION_PIPELINE = None
_LAST_DIARIZATION_STATUS: Dict[str, Any] = {}
_LOG = logging.getLogger(__name__)


def _model_id() -> str:
    return str(
        os.getenv("MGC_DIARIZATION_MODEL_ID") or "pyannote/speaker-diarization-3.1"
    ).strip()


def _resolve_device() -> str:
    override = str(os.getenv("MGC_SPEAKER_DEVICE", "")).strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        return override
    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _ensure_pipeline():
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE

    from pyannote.audio import Pipeline  # type: ignore

    model_id = _model_id()
    token = str(
        os.getenv("PYANNOTE_AUTH_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    kwargs = {"use_auth_token": token} if token else {}
    pipeline = Pipeline.from_pretrained(model_id, **kwargs)
    try:
        import torch  # type: ignore

        device = _resolve_device()
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device(device))
    except Exception as exc:
        _LOG.warning("speaker diarization set_device failed: %s", exc)
    _DIARIZATION_PIPELINE = pipeline
    return pipeline


def _fallback_segments(
    audio_b64: bytes,
    min_turn_length_s: float,
    *,
    reason: str,
    error: str | None = None,
) -> List[Dict[str, Any]]:
    duration_s = audio_duration_seconds(audio_b64)
    if duration_s <= 0.0:
        return []
    if duration_s < float(min_turn_length_s):
        return []
    metadata: Dict[str, Any] = {
        "runtime": "fallback_single_speaker",
        "reason": str(reason or "fallback"),
    }
    if error:
        metadata["error"] = str(error)
    return [
        {
            "track_id": "voice_1",
            "t_start_s": 0.0,
            "t_end_s": duration_s,
            "metadata": metadata,
        }
    ]


def _merge_adjacent(segments: List[Dict[str, Any]], *, max_gap_s: float = 0.15) -> List[Dict[str, Any]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (float(item["t_start_s"]), float(item["t_end_s"])))
    merged: List[Dict[str, Any]] = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        same_track = str(last["track_id"]) == str(item["track_id"])
        gap = float(item["t_start_s"]) - float(last["t_end_s"])
        if same_track and gap <= float(max_gap_s):
            last["t_end_s"] = max(float(last["t_end_s"]), float(item["t_end_s"]))
            continue
        merged.append(dict(item))
    return merged


def _extract_annotation(diarization_obj: Any) -> Any:
    """Normalize pyannote diarization outputs across versions.

    - pyannote <= 3.x: pipeline(...) returns Annotation directly (has itertracks)
    - pyannote >= 4.x: pipeline(...) returns DiarizeOutput with .speaker_diarization
    """
    if hasattr(diarization_obj, "itertracks"):
        return diarization_obj
    for attr in ("speaker_diarization", "exclusive_speaker_diarization"):
        candidate = getattr(diarization_obj, attr, None)
        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate
    raise AttributeError("unsupported diarization output: missing itertracks-compatible annotation")


def diarize_audio_b64(
    audio_b64: bytes,
    *,
    min_turn_length_s: float = 0.4,
) -> List[Dict[str, Any]]:
    global _LAST_DIARIZATION_STATUS
    try:
        raw = base64.b64decode(audio_b64)
    except Exception as exc:
        _LAST_DIARIZATION_STATUS = {
            "ok": False,
            "runtime": "pyannote_diarization",
            "model_id": _model_id(),
            "device": _resolve_device(),
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "invalid_base64",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return []

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        pipeline = _ensure_pipeline()
        diarization = pipeline(str(tmp_path))
        annotation = _extract_annotation(diarization)

        label_map: dict[str, str] = {}
        segments: List[Dict[str, Any]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start_s = max(0.0, float(getattr(turn, "start", 0.0) or 0.0))
            end_s = max(start_s, float(getattr(turn, "end", start_s) or start_s))
            if (end_s - start_s) < float(min_turn_length_s):
                continue
            speaker_key = str(speaker or "speaker").strip() or "speaker"
            if speaker_key not in label_map:
                label_map[speaker_key] = f"voice_{len(label_map) + 1}"
            segments.append(
                {
                    "track_id": label_map[speaker_key],
                    "t_start_s": start_s,
                    "t_end_s": end_s,
                    "metadata": {
                        "runtime": "pyannote_diarization",
                        "speaker_label": speaker_key,
                    },
                }
            )
        merged = _merge_adjacent(segments)
        if merged:
            _LAST_DIARIZATION_STATUS = {
                "ok": True,
                "runtime": "pyannote_diarization",
                "model_id": _model_id(),
                "device": _resolve_device(),
                "segment_count": len(merged),
                "speaker_count": len(label_map),
                "fallback_used": False,
                "reason": None,
                "error": None,
            }
            return merged

        fallback = _fallback_segments(
            audio_b64,
            min_turn_length_s,
            reason="empty_diarization",
        )
        _LAST_DIARIZATION_STATUS = {
            "ok": False,
            "runtime": "pyannote_diarization",
            "model_id": _model_id(),
            "device": _resolve_device(),
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "empty_diarization",
            "error": None,
        }
        return fallback
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _LAST_DIARIZATION_STATUS = {
            "ok": False,
            "runtime": "pyannote_diarization",
            "model_id": _model_id(),
            "device": _resolve_device(),
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "pipeline_error",
            "error": error,
        }
        return _fallback_segments(
            audio_b64,
            min_turn_length_s,
            reason="pipeline_error",
            error=error,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def get_last_diarization_status() -> Dict[str, Any]:
    return dict(_LAST_DIARIZATION_STATUS)


__all__ = ["diarize_audio_b64", "get_last_diarization_status"]
