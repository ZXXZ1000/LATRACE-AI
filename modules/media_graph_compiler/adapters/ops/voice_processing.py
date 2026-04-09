from __future__ import annotations

import base64
import io
import json
import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from modules.media_graph_compiler.adapters.ops.asr_local import transcribe_audio_b64
from modules.media_graph_compiler.adapters.ops.chat_api import generate_messages, get_response
from modules.media_graph_compiler.adapters.ops.config import get_processing_config
from modules.media_graph_compiler.adapters.ops.prompts import prompt_audio_segmentation

logger = logging.getLogger(__name__)
processing_config = get_processing_config()
MAX_RETRIES = int(processing_config.get("max_retries", 10))

_AUDIO_STATE = {
    "ready": False,
    "device": None,
    "model": None,
    "feature_extractor": None,
    "torch": None,
    "torchaudio": None,
}


def _ensure_audio_model():
    global _AUDIO_STATE
    if _AUDIO_STATE["ready"]:
        return True
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore
        from pathlib import Path as _Path
        from speakerlab.process.processor import FBank  # type: ignore
        from speakerlab.utils.builder import dynamic_import  # type: ignore

        mdl_path = os.getenv("MEMA_VOICE_MODEL_PATH")
        if not mdl_path:
            mdl_path = str((_Path(__file__).parent / "models" / "pretrained_eres2netv2w24s4ep4.ckpt"))
        pretrained_state = torch.load(mdl_path, map_location="cpu")
        model = {
            "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
            "args": {
                "feat_dim": 80,
                "embedding_size": 192,
                "baseWidth": 24,
                "scale": 4,
                "expansion": 4,
            },
        }
        embedding_model = dynamic_import(model["obj"])(**model["args"])
        embedding_model.load_state_dict(pretrained_state)
        device = torch.device("cpu")
        override = str(os.getenv("MEMA_VOICE_DEVICE", "")).strip().lower()
        try:
            if override in {"cpu", "mps", "cuda"}:
                if override == "cuda" and not torch.cuda.is_available():
                    device = torch.device("cpu")
                else:
                    device = torch.device(override)
            else:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
        embedding_model.to(device)
        embedding_model.eval()
        feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        _AUDIO_STATE.update(
            {
                "ready": True,
                "device": device,
                "model": embedding_model,
                "feature_extractor": feature_extractor,
                "torch": torch,
                "torchaudio": torchaudio,
            }
        )
        return True
    except Exception as exc:
        logger.warning("voice_processing: failed to init audio model, fallback to zeros. err=%s", exc)
        _AUDIO_STATE.update({"ready": False})
        return False


def validate_and_fix_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        left = text.find("[")
        right = text.rfind("]")
        if left != -1 and right != -1 and right > left:
            return json.loads(text[left : right + 1])
    except Exception:
        return None
    return None


def normalize_embedding(embedding_bytes):
    if embedding_bytes is None:
        return []
    if isinstance(embedding_bytes, (bytes, bytearray)):
        buf = bytes(embedding_bytes)
        if len(buf) == 0 or len(buf) % 4 != 0:
            return []
        count = len(buf) // 4
        try:
            return list(struct.unpack(f"<{count}f", buf))
        except Exception:
            try:
                return list(struct.unpack(f">{count}f", buf))
            except Exception:
                return []
    if isinstance(embedding_bytes, (list, tuple)):
        try:
            return [float(v) for v in embedding_bytes]
        except Exception:
            return []
    try:
        import numpy as _np  # type: ignore

        if isinstance(embedding_bytes, _np.ndarray):
            return [float(v) for v in embedding_bytes.astype(_np.float32).ravel().tolist()]
    except Exception:
        pass
    try:
        return [float(embedding_bytes)]
    except Exception:
        return []


def get_embedding(wav):
    def load_wav(wav_file, obj_fs=16000):
        if not _ensure_audio_model():
            return None
        torchaudio = _AUDIO_STATE["torchaudio"]
        wav_tensor, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav_tensor, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav_tensor, fs, effects=[["rate", str(obj_fs)]]
            )
        if wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor[0, :].unsqueeze(0)
        return wav_tensor

    def compute_embedding(wav_file):
        if not _ensure_audio_model():
            return [0.0] * 192
        torch = _AUDIO_STATE["torch"]
        model = _AUDIO_STATE["model"]
        device = _AUDIO_STATE["device"]
        feat_ex = _AUDIO_STATE["feature_extractor"]
        wav_tensor = load_wav(wav_file)
        if wav_tensor is None:
            return [0.0] * 192
        feat = feat_ex(wav_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(feat).detach().squeeze(0).cpu().numpy()
        return embedding

    return compute_embedding(wav)


def generate(wav):
    if not _ensure_audio_model():
        return [0.0] * 192
    wav_bytes = base64.b64decode(wav)
    wav_file = BytesIO(wav_bytes)
    return get_embedding(wav_file)


def get_audio_embeddings(audio_segments):
    if not audio_segments:
        return []

    def _run(segment):
        payload = segment.decode("utf-8") if isinstance(segment, (bytes, bytearray)) else str(segment)
        vec = generate(payload)
        return struct.pack("f" * len(vec), *vec)

    max_workers = min(len(audio_segments), os.cpu_count() or 4)
    if max_workers <= 1:
        return [_run(segment) for segment in audio_segments]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, audio_segments))


def _audio_duration_seconds(b64_audio: bytes) -> float:
    try:
        from pydub import AudioSegment  # type: ignore

        raw = base64.b64decode(b64_audio)
        audio = AudioSegment.from_wav(io.BytesIO(raw))
        return max(0.0, float(len(audio)) / 1000.0)
    except Exception:
        return 0.0


def asr_transcribe_with_adapt(base64_audio: bytes, *, rctx: dict | None = None) -> tuple[list[dict], str]:
    mode = str(((rctx or {}).get("asr") or {}).get("mode", "local")).lower()
    model_size = str(((rctx or {}).get("asr") or {}).get("model_size", "small"))
    device = str(((rctx or {}).get("asr") or {}).get("device", "auto"))
    max_segment_s = float(((rctx or {}).get("asr") or {}).get("max_segment_s", 15.0))
    rtf_threshold = float(((rctx or {}).get("asr") or {}).get("rtf_threshold", 0.0))
    fallback_model = ((rctx or {}).get("asr") or {}).get("fallback_model")

    import time as _time

    dur_s = _audio_duration_seconds(base64_audio)
    t0 = _time.perf_counter()
    segments = transcribe_audio_b64(
        base64_audio,
        language=None,
        model_size=model_size,
        device=device,
        max_segment_s=max_segment_s,
        strict=(mode == "local"),
    )
    dt = _time.perf_counter() - t0

    if (
        mode == "hybrid"
        and rtf_threshold
        and dur_s > 0
        and (dt / dur_s) > rtf_threshold
        and fallback_model
        and str(fallback_model) != model_size
    ):
        segments = transcribe_audio_b64(
            base64_audio,
            language=None,
            model_size=str(fallback_model),
            device=device,
            max_segment_s=max_segment_s,
            strict=False,
        )
        return segments, str(fallback_model)
    return segments, model_size


def process_voices(video_graph, base64_audio, base64_video, save_path, preprocessing=None, rctx=None):
    preprocessing = preprocessing or []

    def get_audio_segments(encoded_audio, dialogs):
        from pydub import AudioSegment  # type: ignore

        audio_data = base64.b64decode(encoded_audio)
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))

        audio_segments = []
        for start_time, end_time in dialogs:
            try:
                start_min, start_sec = map(int, start_time.split(":"))
                end_min, end_sec = map(int, end_time.split(":"))
            except ValueError:
                audio_segments.append(None)
                continue
            if (start_min < 0 or start_sec < 0 or start_sec >= 60) or (end_min < 0 or end_sec < 0 or end_sec >= 60):
                audio_segments.append(None)
                continue
            start_msec = (start_min * 60 + start_sec) * 1000
            end_msec = (end_min * 60 + end_sec) * 1000
            if start_msec >= end_msec or end_msec > len(audio):
                audio_segments.append(None)
                continue
            segment = audio[start_msec:end_msec]
            segment_buffer = io.BytesIO()
            segment.export(segment_buffer, format="wav")
            segment_buffer.seek(0)
            audio_segments.append(base64.b64encode(segment_buffer.getvalue()))
        return audio_segments

    def diarize_audio(encoded_video, filter=None, *, encoded_audio=None, rctx=None):
        mode = str(((rctx or {}).get("asr") or {}).get("mode", "local")).lower()
        if mode in ("local", "hybrid") and encoded_audio is not None:
            segments, _model_used = asr_transcribe_with_adapt(encoded_audio, rctx=rctx)
            asrs = []
            for segment in segments:
                try:
                    st = segment.get("start_time")
                    et = segment.get("end_time")
                    txt = segment.get("asr")
                    if not st or not et:
                        continue
                    sm, ss = map(int, str(st).split(":"))
                    em, es = map(int, str(et).split(":"))
                    dur = (em * 60 + es) - (sm * 60 + ss)
                    item = {"start_time": st, "end_time": et, "asr": txt or "", "duration": dur}
                    if not filter or filter(item):
                        asrs.append(item)
                except Exception:
                    continue
            return asrs

        if encoded_video is None:  # pragma: no cover - legacy fallback only
            raise RuntimeError("LLM diarization fallback requires base64 video input")

        inputs = [
            {"type": "video_base64/mp4", "content": encoded_video.decode("utf-8")},
            {"type": "text", "content": prompt_audio_segmentation},
        ]
        messages = generate_messages(inputs)
        model = "multimodal"
        asrs = None
        for _ in range(MAX_RETRIES):
            response, _ = get_response(model, messages, timeout=30)
            asrs = validate_and_fix_json(response)
            if asrs is not None:
                break
        if asrs is None:
            raise RuntimeError("Failed to diarize audio via LLM")
        for asr in asrs:
            start_min, start_sec = map(int, asr["start_time"].split(":"))
            end_min, end_sec = map(int, asr["end_time"].split(":"))
            asr["duration"] = (end_min * 60 + end_sec) - (start_min * 60 + start_sec)
        return [asr for asr in asrs if not filter or filter(asr)]

    def get_normed_audio_embeddings(audios):
        audio_segments = [audio["audio_segment"] for audio in audios]
        embeddings = get_audio_embeddings(audio_segments)
        normed_embeddings = [normalize_embedding(embedding) for embedding in embeddings]
        for audio, embedding in zip(audios, normed_embeddings):
            audio["embedding"] = embedding
        return audios

    def create_audio_segments(encoded_audio, asrs):
        try:
            dialogs = [(asr["start_time"], asr["end_time"]) for asr in asrs]
            audio_segments = get_audio_segments(encoded_audio, dialogs)
            for asr, audio_segment in zip(asrs, audio_segments):
                asr["audio_segment"] = audio_segment
            return asrs
        except Exception:
            for asr in asrs:
                asr["audio_segment"] = encoded_audio
            return asrs

    def filter_duration_based(audio):
        threshold = float(
            ((rctx or {}).get("audio") or {}).get(
                "min_duration_for_audio",
                processing_config["min_duration_for_audio"],
            )
        )
        return float(audio["duration"]) >= threshold

    def update_videograph(vg, audios):
        if vg is None:
            if not audios:
                return {}
            mapped = []
            for audio in audios:
                audio["matched_node"] = "voice_1"
                mapped.append(audio)
            return {"voice_1": mapped}

        id2audios = {}
        for audio in audios:
            audio_info = {
                "embeddings": [audio["embedding"]],
                "contents": [audio["asr"]],
            }
            matched_nodes = vg.search_voice_nodes(audio_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                vg.update_node(matched_node, audio_info)
                audio["matched_node"] = matched_node
            else:
                matched_node = vg.add_voice_node(audio_info)
                audio["matched_node"] = matched_node
            id2audios.setdefault(matched_node, []).append(audio)
        return id2audios

    if not base64_audio:
        return {}

    audios = None
    try:
        with open(save_path, "r") as f:
            audios = json.load(f)
        for audio in audios:
            audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
    except Exception:
        audios = None

    if audios is None:
        try:
            asrs = diarize_audio(base64_video, filter=filter_duration_based, encoded_audio=base64_audio, rctx=rctx)
        except Exception:
            mode_try = str(((rctx or {}).get("asr") or {}).get("mode", "local")).lower()
            if mode_try == "local":
                raise
            asrs = diarize_audio(base64_video, filter=filter_duration_based, rctx=rctx)
        audios = create_audio_segments(base64_audio, asrs)
        audios = [audio for audio in audios if audio["audio_segment"] is not None]
        if audios:
            audios = get_normed_audio_embeddings(audios)
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(save_path, "w") as f:
            serializable = []
            for audio in audios:
                item = dict(audio)
                item["audio_segment"] = item["audio_segment"].decode("utf-8")
                serializable.append(item)
            json.dump(serializable, f)
            for audio in audios:
                audio["audio_segment"] = audio["audio_segment"].encode("utf-8")
        logger.info("Write voice detection results to %s", save_path)

    if "voice" in preprocessing or len(audios) == 0:
        return {}

    return update_videograph(video_graph, audios)


__all__ = [
    "asr_transcribe_with_adapt",
    "get_audio_embeddings",
    "normalize_embedding",
    "process_voices",
]
