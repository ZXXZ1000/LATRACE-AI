from __future__ import annotations

"""Copied local ASR operator with safe optional-dependency fallbacks."""

import base64
import io
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple


_ASR_CACHE_LOCK = threading.Lock()
_ASR_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_LAST_ASR_STATUS: Dict[str, Any] = {}
_LOG = logging.getLogger(__name__)


def _select_device(device: str) -> str:
    requested = str(device or "").strip().lower()
    if requested == "mps":
        # faster-whisper/ctranslate2 does not support MPS.
        return "cpu"
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_whisper_model(model_size: str, device: str):
    key = (model_size, device)
    with _ASR_CACHE_LOCK:
        if key in _ASR_MODEL_CACHE:
            return _ASR_MODEL_CACHE[key]
        from faster_whisper import WhisperModel  # type: ignore

        compute_type = "int8" if device == "cpu" else "float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _ASR_MODEL_CACHE[key] = model
        return model


def _fmt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = int(sec - m * 60)
    return f"{m:02d}:{s:02d}"


def transcribe_audio_b64(
    audio_b64: bytes,
    *,
    language: Optional[str] = None,
    model_size: str = "small",
    device: str = "auto",
    max_segment_s: float = 15.0,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    global _LAST_ASR_STATUS
    try:
        raw = base64.b64decode(audio_b64)
        bio = io.BytesIO(raw)
        dev = _select_device(device)
        model = _get_whisper_model(model_size, dev)
        segments, _info = model.transcribe(
            bio,
            language=language,
            vad_filter=True,
            beam_size=1,
        )
        out: List[Dict[str, Any]] = []
        for seg in segments:
            st = float(getattr(seg, "start", 0.0) or 0.0)
            et = float(getattr(seg, "end", st) or st)
            txt = str(getattr(seg, "text", "")).strip()
            if not txt:
                continue
            out.append(
                {
                    "start_time": _fmt_ts(st),
                    "end_time": _fmt_ts(et),
                    "t_start_s": st,
                    "t_end_s": et,
                    "asr": txt,
                }
            )
        if out:
            _LAST_ASR_STATUS = {
                "ok": True,
                "runtime": "faster_whisper",
                "device": dev,
                "model_size": model_size,
                "segment_count": len(out),
                "fallback_used": False,
                "error": None,
            }
            return out
    except Exception as exc:
        _LAST_ASR_STATUS = {
            "ok": False,
            "runtime": "faster_whisper",
            "device": _select_device(device),
            "model_size": model_size,
            "segment_count": 0,
            "fallback_used": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _LOG.warning("local ASR failed, falling back to placeholder transcript: %s", exc)
        if strict:
            raise

    try:
        from pydub import AudioSegment  # type: ignore

        raw = base64.b64decode(audio_b64)
        bio = io.BytesIO(raw)
        audio = AudioSegment.from_wav(bio)
        dur_s = max(0.0, float(len(audio)) / 1000.0)
        _LAST_ASR_STATUS = {
            **_LAST_ASR_STATUS,
            "fallback_runtime": "pydub_placeholder",
        }
        return [
            {
                "start_time": _fmt_ts(0.0),
                "end_time": _fmt_ts(dur_s),
                "t_start_s": 0.0,
                "t_end_s": dur_s,
                "asr": "(audio)",
            }
        ]
    except Exception:
        _LAST_ASR_STATUS = {
            **_LAST_ASR_STATUS,
            "fallback_runtime": "hard_placeholder",
        }
        return [
            {
                "start_time": "00:00",
                "end_time": "00:01",
                "t_start_s": 0.0,
                "t_end_s": 1.0,
                "asr": "(audio)",
            }
        ]


def get_last_asr_status() -> Dict[str, Any]:
    return dict(_LAST_ASR_STATUS)


__all__ = ["get_last_asr_status", "transcribe_audio_b64"]
