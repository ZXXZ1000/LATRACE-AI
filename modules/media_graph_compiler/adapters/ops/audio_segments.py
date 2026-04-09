from __future__ import annotations

import base64
import io
import wave
from typing import Iterable, List, Sequence


def audio_duration_seconds(audio_b64: bytes) -> float:
    raw = b""
    try:
        raw = base64.b64decode(audio_b64)
        with wave.open(io.BytesIO(raw), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        if rate > 0:
            return max(0.0, float(frames) / float(rate))
    except Exception:
        pass
    try:
        from pydub import AudioSegment  # type: ignore

        audio = AudioSegment.from_wav(io.BytesIO(raw))
        return max(0.0, float(len(audio)) / 1000.0)
    except Exception:
        return 0.0


def slice_audio_b64_segments(
    audio_b64: bytes,
    spans_s: Sequence[Sequence[float]],
) -> List[bytes | None]:
    raw = b""
    try:
        raw = base64.b64decode(audio_b64)
        with wave.open(io.BytesIO(raw), "rb") as wav:
            nchannels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if framerate <= 0 or sampwidth <= 0 or nchannels <= 0:
            raise RuntimeError("invalid_wav_header")

        frame_size = nchannels * sampwidth
        output: List[bytes | None] = []
        for span in spans_s:
            try:
                start_s = max(0.0, float(span[0]))
                end_s = max(start_s, float(span[1]))
                start_frame = int(start_s * framerate)
                end_frame = int(end_s * framerate)
                if end_frame <= start_frame:
                    output.append(None)
                    continue
                start_byte = start_frame * frame_size
                end_byte = end_frame * frame_size
                chunk = frames[start_byte:end_byte]
                if not chunk:
                    output.append(None)
                    continue
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as out_wav:
                    out_wav.setnchannels(nchannels)
                    out_wav.setsampwidth(sampwidth)
                    out_wav.setframerate(framerate)
                    out_wav.writeframes(chunk)
                output.append(base64.b64encode(buffer.getvalue()))
            except Exception:
                output.append(None)
        return output
    except Exception:
        pass

    try:
        from pydub import AudioSegment  # type: ignore

        audio = AudioSegment.from_wav(io.BytesIO(raw))
    except Exception:
        return [None for _ in spans_s]

    out: List[bytes | None] = []
    for span in spans_s:
        try:
            start_s = max(0.0, float(span[0]))
            end_s = max(start_s, float(span[1]))
            start_ms = int(start_s * 1000.0)
            end_ms = int(end_s * 1000.0)
            if start_ms >= end_ms or end_ms > len(audio):
                out.append(None)
                continue
            segment = audio[start_ms:end_ms]
            buffer = io.BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)
            out.append(base64.b64encode(buffer.getvalue()))
        except Exception:
            out.append(None)
    if any(item is not None for item in out):
        return out

    return out


def mean_embedding(vectors: Iterable[Sequence[float]]) -> List[float]:
    vectors = [list(map(float, item)) for item in vectors if item]
    if not vectors:
        return []
    dim = len(vectors[0])
    if dim <= 0:
        return []
    acc = [0.0] * dim
    used = 0
    for vec in vectors:
        if len(vec) != dim:
            continue
        used += 1
        for index, value in enumerate(vec):
            acc[index] += float(value)
    if used <= 0:
        return []
    inv = 1.0 / float(used)
    return [value * inv for value in acc]


__all__ = ["audio_duration_seconds", "slice_audio_b64_segments", "mean_embedding"]
