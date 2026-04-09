from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_RUNTIME = None
_RUNTIME_ERROR = None


def _resolve_workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_repo_path() -> Path:
    configured = str(os.getenv("MGC_LIGHT_ASD_REPO") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (_resolve_workspace_root() / "reference" / "Light-ASD").resolve()


def _resolve_weight_path(repo_path: Path) -> Path:
    configured = str(os.getenv("MGC_LIGHT_ASD_WEIGHT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (repo_path / "weight" / "pretrain_AVA_CVPR.model").resolve()


def _resolve_device(torch_module) -> str:
    override = str(os.getenv("MGC_LIGHT_ASD_DEVICE") or "").strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        return override
    try:
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if torch_module.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _LightAsdRuntime:
    def __init__(self, *, repo_path: Path, weight_path: Path) -> None:
        if not repo_path.exists():
            raise RuntimeError(f"Light-ASD repo not found: {repo_path}")
        if not weight_path.exists():
            raise RuntimeError(f"Light-ASD weight not found: {weight_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        import cv2  # type: ignore
        import python_speech_features  # type: ignore
        import torch  # type: ignore
        from loss import lossAV  # type: ignore
        from model.Model import ASD_Model  # type: ignore

        self._cv2 = cv2
        self._mfcc = python_speech_features.mfcc
        self._torch = torch
        self._device = torch.device(_resolve_device(torch))
        self._model = ASD_Model().to(self._device).eval()
        self._head = lossAV().to(self._device).eval()
        self._duration_set = _parse_duration_set(
            str(os.getenv("MGC_LIGHT_ASD_DURATION_SET") or "1,2,3")
        )
        self._load_weight(weight_path)

    def _load_weight(self, weight_path: Path) -> None:
        checkpoint = self._torch.load(str(weight_path), map_location="cpu")
        model_state = self._model.state_dict()
        head_state = self._head.state_dict()
        for key, value in checkpoint.items():
            if key.startswith("model."):
                dst_key = key[len("model.") :]
                if dst_key in model_state and model_state[dst_key].shape == value.shape:
                    model_state[dst_key].copy_(value)
            elif key.startswith("lossAV."):
                dst_key = key[len("lossAV.") :]
                if dst_key in head_state and head_state[dst_key].shape == value.shape:
                    head_state[dst_key].copy_(value)

    def score(
        self,
        *,
        audio_file: str,
        face_frame_paths: Sequence[str],
        face_frame_timestamps_s: Sequence[float],
    ) -> Dict[str, Any] | None:
        audio = _load_wav_16k(audio_file)
        if audio is None or audio.size < 1600:
            return None
        max_audio_seconds = float(os.getenv("MGC_LIGHT_ASD_MAX_AUDIO_SECONDS") or 12.0)
        if max_audio_seconds > 0:
            max_samples = int(16000 * max_audio_seconds)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
        mfcc = self._mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        if mfcc is None or len(mfcc) < 40:
            return None

        duration_s = float(len(audio)) / 16000.0
        video_feature = self._build_video_feature(
            face_frame_paths=face_frame_paths,
            face_frame_timestamps_s=face_frame_timestamps_s,
            duration_s=duration_s,
        )
        if video_feature is None or video_feature.shape[0] < 8:
            return None

        valid_audio_frames = int(mfcc.shape[0] - (mfcc.shape[0] % 4))
        length_s = min(float(valid_audio_frames) / 100.0, float(video_feature.shape[0]) / 25.0)
        if length_s < 0.4:
            return None

        audio_feature = np.asarray(mfcc[: int(round(length_s * 100)), :], dtype=np.float32)
        video_feature = np.asarray(video_feature[: int(round(length_s * 25)), :, :], dtype=np.float32)
        if audio_feature.size == 0 or video_feature.size == 0:
            return None

        all_scores: List[np.ndarray] = []
        with self._torch.no_grad():
            for duration_s in self._duration_set:
                scores = self._score_with_duration(
                    audio_feature=audio_feature,
                    video_feature=video_feature,
                    duration_s=duration_s,
                )
                if scores.size > 0:
                    all_scores.append(scores)
        if not all_scores:
            return None

        stacked = np.stack(
            [np.pad(s, (0, max(0, max(map(len, all_scores)) - len(s))), mode="edge") for s in all_scores],
            axis=0,
        )
        raw_score = float(np.mean(stacked))
        norm_score = float(1.0 / (1.0 + math.exp(-raw_score)))
        return {
            "raw_score": raw_score,
            "norm_score": norm_score,
            "duration_s": length_s,
            "frame_count": int(video_feature.shape[0]),
            "device": str(self._device),
            "duration_set": list(self._duration_set),
        }

    def _score_with_duration(
        self,
        *,
        audio_feature: np.ndarray,
        video_feature: np.ndarray,
        duration_s: int,
    ) -> np.ndarray:
        batch_count = int(math.ceil((len(video_feature) / 25.0) / float(duration_s)))
        outputs: List[np.ndarray] = []
        for index in range(batch_count):
            a_start = int(index * duration_s * 100)
            a_end = int((index + 1) * duration_s * 100)
            v_start = int(index * duration_s * 25)
            v_end = int((index + 1) * duration_s * 25)
            a_chunk = audio_feature[a_start:a_end, :]
            v_chunk = video_feature[v_start:v_end, :, :]
            if a_chunk.size == 0 or v_chunk.size == 0:
                continue
            input_a = self._torch.FloatTensor(a_chunk).unsqueeze(0).to(self._device)
            input_v = self._torch.FloatTensor(v_chunk).unsqueeze(0).to(self._device)
            embed_a = self._model.forward_audio_frontend(input_a)
            embed_v = self._model.forward_visual_frontend(input_v)
            out = self._model.forward_audio_visual_backend(embed_a, embed_v)
            score = self._head.forward(out, labels=None)
            if score is not None and len(score) > 0:
                outputs.append(np.asarray(score, dtype=np.float32))
        if not outputs:
            return np.array([], dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _build_video_feature(
        self,
        *,
        face_frame_paths: Sequence[str],
        face_frame_timestamps_s: Sequence[float],
        duration_s: float,
    ) -> np.ndarray | None:
        if not face_frame_paths:
            return None
        images: List[np.ndarray] = []
        for path in face_frame_paths:
            frame = self._load_face_frame(path)
            if frame is not None:
                images.append(frame)
        if not images:
            return None

        target_count = max(8, int(round(duration_s * 25.0)))
        if target_count <= len(images):
            return np.stack(images[:target_count], axis=0)

        stamps = [float(item) for item in face_frame_timestamps_s[: len(images)]]
        if len(stamps) < len(images):
            stamps = [float(index) / 2.0 for index in range(len(images))]

        start_s = 0.0
        end_s = max(duration_s, 0.04)
        step = (end_s - start_s) / float(max(1, target_count - 1))
        expanded: List[np.ndarray] = []
        for i in range(target_count):
            target_t = start_s + step * i
            nearest_index = min(
                range(len(stamps)),
                key=lambda idx: abs(stamps[idx] - target_t),
            )
            expanded.append(images[nearest_index])
        return np.stack(expanded, axis=0)

    def _load_face_frame(self, path: str) -> np.ndarray | None:
        try:
            image = self._cv2.imread(str(path))
            if image is None:
                return None
            gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
            resized = self._cv2.resize(gray, (224, 224))
            center = resized[56:168, 56:168]
            return center
        except Exception:
            return None


def _parse_duration_set(raw: str) -> List[int]:
    values: List[int] = []
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            values.append(value)
    return values or [1, 2, 3]


def _load_wav_16k(path: str) -> np.ndarray | None:
    try:
        from scipy.io import wavfile  # type: ignore
        from scipy.signal import resample_poly  # type: ignore
    except Exception:
        return None
    try:
        sr, audio = wavfile.read(path)
    except Exception:
        return None
    if audio is None:
        return None
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    arr = arr.astype(np.float32)
    if sr <= 0:
        return None
    if sr != 16000:
        gcd = math.gcd(int(sr), 16000)
        up = 16000 // gcd
        down = int(sr) // gcd
        arr = resample_poly(arr, up, down).astype(np.float32)
    arr = np.clip(arr, -32768.0, 32767.0).astype(np.int16)
    return arr


def _ensure_runtime() -> _LightAsdRuntime | None:
    global _RUNTIME, _RUNTIME_ERROR
    if _RUNTIME is not None:
        return _RUNTIME
    if _RUNTIME_ERROR is not None:
        return None
    try:
        repo_path = _resolve_repo_path()
        weight_path = _resolve_weight_path(repo_path)
        _RUNTIME = _LightAsdRuntime(repo_path=repo_path, weight_path=weight_path)
        return _RUNTIME
    except Exception as exc:
        _RUNTIME_ERROR = f"{type(exc).__name__}:{exc}"
        return None


def score_light_asd(
    *,
    audio_file: str,
    face_frame_paths: Sequence[str],
    face_frame_timestamps_s: Sequence[float],
) -> Dict[str, Any] | None:
    runtime = _ensure_runtime()
    if runtime is None:
        return None
    return runtime.score(
        audio_file=audio_file,
        face_frame_paths=face_frame_paths,
        face_frame_timestamps_s=face_frame_timestamps_s,
    )


def light_asd_status() -> Dict[str, Any]:
    runtime = _ensure_runtime()
    return {
        "ready": runtime is not None,
        "error": _RUNTIME_ERROR,
    }


__all__ = ["light_asd_status", "score_light_asd"]
