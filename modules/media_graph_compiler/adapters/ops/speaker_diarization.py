from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from modules.media_graph_compiler.adapters.ops.audio_vad import (
    compact_audio_b64_to_speech_islands,
    detect_speech_spans_b64,
    remap_compacted_segments_to_original,
)
from modules.media_graph_compiler.adapters.ops.audio_segments import (
    audio_duration_seconds,
)

_DIARIZATION_PIPELINE = None
_LAST_DIARIZATION_STATUS: Dict[str, Any] = {}
_LOG = logging.getLogger(__name__)
_DEFAULT_SEGMENTATION_STEP = 0.18
_LONG_AUDIO_SEGMENTATION_STEPS: tuple[tuple[float, float], ...] = ()


def _model_id() -> str:
    return str(
        os.getenv("MGC_DIARIZATION_MODEL_ID") or "pyannote/speaker-diarization-3.1"
    ).strip()


def _resolve_device() -> str:
    override = str(os.getenv("MGC_SPEAKER_DEVICE", "")).strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        return override
    if sys.platform == "darwin":
        return "cpu"
    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _adaptive_segmentation_step(duration_s: float | None) -> float:
    duration_value = max(0.0, float(duration_s or 0.0))
    for threshold_s, step in _LONG_AUDIO_SEGMENTATION_STEPS:
        if duration_value >= float(threshold_s):
            return float(step)
    return _DEFAULT_SEGMENTATION_STEP


def _resolve_segmentation_step(duration_s: float | None = None) -> float:
    raw = str(os.getenv("MGC_DIARIZATION_SEGMENTATION_STEP") or "").strip()
    if not raw:
        return _adaptive_segmentation_step(duration_s)
    try:
        value = float(raw)
    except ValueError:
        _LOG.warning(
            "invalid MGC_DIARIZATION_SEGMENTATION_STEP=%r, fallback=%s",
            raw,
            _adaptive_segmentation_step(duration_s),
        )
        return _adaptive_segmentation_step(duration_s)
    if 0.0 < value <= 1.0:
        return value
    _LOG.warning(
        "out-of-range MGC_DIARIZATION_SEGMENTATION_STEP=%r, fallback=%s",
        raw,
        _adaptive_segmentation_step(duration_s),
    )
    return _adaptive_segmentation_step(duration_s)


def _resolve_batch_size(name: str, default: int) -> int:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, int(default))


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _apply_runtime_config(
    pipeline: Any,
    *,
    processing_duration_s: float | None = None,
) -> Dict[str, Any]:
    segmentation_step = _resolve_segmentation_step(processing_duration_s)
    segmentation_duration_s = None
    device = _resolve_device()
    segmentation_batch_size = _resolve_batch_size(
        "MGC_DIARIZATION_SEGMENTATION_BATCH_SIZE",
        1 if device == "cpu" else 4,
    )
    embedding_batch_size = _resolve_batch_size(
        "MGC_DIARIZATION_EMBEDDING_BATCH_SIZE",
        1 if device == "cpu" else 8,
    )
    try:
        if hasattr(pipeline, "segmentation_step"):
            pipeline.segmentation_step = segmentation_step
        if hasattr(pipeline, "segmentation_batch_size"):
            pipeline.segmentation_batch_size = segmentation_batch_size
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = embedding_batch_size
        segmentation_model = getattr(pipeline, "_segmentation", None)
        if segmentation_model is not None:
            duration = float(
                getattr(segmentation_model, "duration", 0.0) or 0.0
            )
            if duration > 0.0 and hasattr(segmentation_model, "step"):
                segmentation_model.step = segmentation_step * duration
                segmentation_duration_s = duration
    except Exception as exc:
        _LOG.warning("speaker diarization runtime config failed: %s", exc)
    return {
        "segmentation_step": segmentation_step,
        "segmentation_window_duration_s": segmentation_duration_s,
        "segmentation_batch_size": segmentation_batch_size,
        "embedding_batch_size": embedding_batch_size,
    }


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
    enable_vad: bool = True,
) -> List[Dict[str, Any]]:
    global _LAST_DIARIZATION_STATUS
    original_duration_s = audio_duration_seconds(audio_b64)
    runtime_config = {
        "segmentation_step": _resolve_segmentation_step(original_duration_s),
        "segmentation_window_duration_s": None,
        "segmentation_batch_size": None,
        "embedding_batch_size": None,
    }
    timeline_map: List[Dict[str, Any]] = []
    processing_audio_b64 = audio_b64
    vad_status: Dict[str, Any] = {
        "enabled": bool(enable_vad),
        "applied": False,
        "speech_span_count": 0,
        "speech_ratio": 1.0,
        "original_duration_s": original_duration_s,
        "processing_duration_s": original_duration_s,
        "compaction_allowed": bool(
            enable_vad
            and _as_bool(
                os.getenv("MGC_DIARIZATION_ENABLE_VAD_COMPACTION"),
                False,
            )
        ),
    }
    if bool(vad_status.get("compaction_allowed")):
        try:
            speech_spans = detect_speech_spans_b64(
                audio_b64,
                min_speech_s=max(float(min_turn_length_s), 0.35),
                min_silence_s=float(
                    os.getenv("MGC_DIARIZATION_VAD_MIN_SILENCE_S") or 0.45
                ),
                pad_s=float(os.getenv("MGC_DIARIZATION_VAD_PAD_S") or 0.12),
                seek_step_ms=int(os.getenv("MGC_DIARIZATION_VAD_SEEK_STEP_MS") or 20),
                merge_gap_s=float(os.getenv("MGC_DIARIZATION_VAD_MERGE_GAP_S") or 0.18),
            )
            vad_status["speech_span_count"] = len(speech_spans)
            compacted = compact_audio_b64_to_speech_islands(
                audio_b64,
                speech_spans,
                join_gap_s=float(os.getenv("MGC_DIARIZATION_VAD_JOIN_GAP_S") or 0.12),
            )
            speech_ratio = float(compacted.get("speech_ratio") or 1.0)
            processing_duration_s = float(
                compacted.get("compacted_duration_s") or original_duration_s
            )
            vad_status.update(
                {
                    "speech_ratio": speech_ratio,
                    "original_duration_s": float(
                        compacted.get("original_duration_s") or original_duration_s
                    ),
                    "processing_duration_s": processing_duration_s,
                }
            )
            if (
                bool(compacted.get("applied"))
                and len(speech_spans) >= 2
                and speech_ratio < 0.96
            ):
                processing_audio_b64 = compacted.get("audio_b64") or audio_b64
                timeline_map = list(compacted.get("timeline_map") or [])
                vad_status["applied"] = True
                runtime_config["segmentation_step"] = _resolve_segmentation_step(
                    processing_duration_s
                )
        except Exception as exc:
            vad_status["error"] = f"{type(exc).__name__}: {exc}"
    try:
        raw = base64.b64decode(processing_audio_b64)
    except Exception as exc:
        _LAST_DIARIZATION_STATUS = {
            "ok": False,
            "runtime": "pyannote_diarization",
            "model_id": _model_id(),
            "device": _resolve_device(),
            "segmentation_step": runtime_config["segmentation_step"],
            "segmentation_window_duration_s": runtime_config[
                "segmentation_window_duration_s"
            ],
            "segmentation_batch_size": runtime_config["segmentation_batch_size"],
            "embedding_batch_size": runtime_config["embedding_batch_size"],
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "invalid_base64",
            "error": f"{type(exc).__name__}: {exc}",
            "vad": vad_status,
        }
        return []

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        pipeline = _ensure_pipeline()
        runtime_config = _apply_runtime_config(
            pipeline,
            processing_duration_s=float(
                vad_status.get("processing_duration_s") or original_duration_s
            ),
        )
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
                        "vad_compacted": bool(vad_status.get("applied")),
                    },
                }
            )
        if timeline_map:
            segments = remap_compacted_segments_to_original(
                segments,
                timeline_map,
                min_duration_s=min_turn_length_s,
            )
        merged = _merge_adjacent(segments)
        if merged:
            _LAST_DIARIZATION_STATUS = {
                "ok": True,
                "runtime": "pyannote_diarization",
                "model_id": _model_id(),
                "device": _resolve_device(),
                "segmentation_step": runtime_config["segmentation_step"],
                "segmentation_window_duration_s": runtime_config[
                    "segmentation_window_duration_s"
                ],
                "segmentation_batch_size": runtime_config["segmentation_batch_size"],
                "embedding_batch_size": runtime_config["embedding_batch_size"],
                "segment_count": len(merged),
                "speaker_count": len(label_map),
                "fallback_used": False,
                "reason": None,
                "error": None,
                "vad": vad_status,
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
            "segmentation_step": runtime_config["segmentation_step"],
            "segmentation_window_duration_s": runtime_config[
                "segmentation_window_duration_s"
            ],
            "segmentation_batch_size": runtime_config["segmentation_batch_size"],
            "embedding_batch_size": runtime_config["embedding_batch_size"],
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "empty_diarization",
            "error": None,
            "vad": vad_status,
        }
        return fallback
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _LAST_DIARIZATION_STATUS = {
            "ok": False,
            "runtime": "pyannote_diarization",
            "model_id": _model_id(),
            "device": _resolve_device(),
            "segmentation_step": runtime_config["segmentation_step"],
            "segmentation_window_duration_s": runtime_config[
                "segmentation_window_duration_s"
            ],
            "segmentation_batch_size": runtime_config["segmentation_batch_size"],
            "embedding_batch_size": runtime_config["embedding_batch_size"],
            "segment_count": 0,
            "speaker_count": 0,
            "fallback_used": True,
            "reason": "pipeline_error",
            "error": error,
            "vad": vad_status,
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
