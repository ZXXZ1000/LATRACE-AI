from __future__ import annotations

import logging
import os
from typing import Dict, List, Sequence

import numpy as np

from modules.media_graph_compiler.adapters.ops.audio_segments import mean_embedding

_RUNTIME = None
_RUNTIME_ERROR = None
_LOG = logging.getLogger(__name__)


def _ensure_runtime():
    global _RUNTIME, _RUNTIME_ERROR
    if _RUNTIME is not None:
        return _RUNTIME
    if _RUNTIME_ERROR is not None:
        raise RuntimeError(_RUNTIME_ERROR)

    try:
        import wespeakerruntime as wespeaker_runtime  # type: ignore

        lang = str(os.getenv("MGC_SPEAKER_EMBED_LANG") or "en").strip()
        _RUNTIME = ("wespeakerruntime", wespeaker_runtime.Speaker(lang=lang))
        return _RUNTIME
    except Exception as exc:
        _LOG.warning("speaker embedding runtime init failed for wespeakerruntime: %s", exc)

    try:
        import wespeaker  # type: ignore

    except Exception:
        _RUNTIME_ERROR = "no speaker embedding runtime available"
        raise RuntimeError(_RUNTIME_ERROR)

    model_name = str(os.getenv("MGC_SPEAKER_EMBED_MODEL") or "english").strip()
    model = wespeaker.load_model(model_name)
    try:
        device = str(os.getenv("MGC_SPEAKER_DEVICE") or "cpu").strip()
        if hasattr(model, "set_device"):
            model.set_device(device)
    except Exception as exc:
        _LOG.warning("speaker embedding runtime set_device failed: %s", exc)
    _RUNTIME = ("wespeaker", model)
    return _RUNTIME


def extract_audio_embedding(audio_file: str) -> List[float]:
    try:
        runtime_kind, runtime = _ensure_runtime()
        if runtime_kind == "wespeakerruntime":
            vector = runtime.extract_embedding(audio_file)
        else:
            vector = runtime.extract_embedding(audio_file)
        flat = np.asarray(vector, dtype=np.float32).reshape(-1)
        return [float(item) for item in flat.tolist()]
    except Exception as exc:
        _LOG.warning("speaker embedding extract failed: %s", exc)
        return []


def build_track_embedding(audio_files: Sequence[str]) -> List[float]:
    return mean_embedding(extract_audio_embedding(path) for path in audio_files)


def speaker_embedding_runtime() -> str:
    try:
        runtime_kind, _ = _ensure_runtime()
        return str(runtime_kind)
    except Exception:
        return "unavailable"


def warmup_speaker_embedding_model() -> Dict[str, object]:
    try:
        runtime_kind, _ = _ensure_runtime()
        return {"ready": True, "runtime": str(runtime_kind), "error": None}
    except Exception as exc:
        return {"ready": False, "runtime": "unavailable", "error": str(exc)}


__all__ = [
    "build_track_embedding",
    "extract_audio_embedding",
    "speaker_embedding_runtime",
    "warmup_speaker_embedding_model",
]
