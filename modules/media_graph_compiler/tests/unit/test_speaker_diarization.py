from pathlib import Path
import sys
import base64
import io
import wave

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.ops.speaker_diarization import (
    diarize_audio_b64,
    get_last_diarization_status,
)


def _wav_bytes(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    frame_count = int(sample_rate * (duration_ms / 1000.0))
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
    return bio.getvalue()


def test_diarize_audio_b64_reports_pipeline_error_status(monkeypatch) -> None:
    wav_b64 = base64.b64encode(_wav_bytes())

    def _raise_pipeline():
        raise RuntimeError("auth missing")

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization._ensure_pipeline",
        _raise_pipeline,
    )

    segments = diarize_audio_b64(wav_b64, min_turn_length_s=0.2)
    status = get_last_diarization_status()

    assert len(segments) == 1
    assert segments[0]["track_id"] == "voice_1"
    assert segments[0]["metadata"]["reason"] == "pipeline_error"
    assert status["ok"] is False
    assert status["fallback_used"] is True
    assert status["reason"] == "pipeline_error"


def test_diarize_audio_b64_supports_v4_diarize_output_wrapper(monkeypatch) -> None:
    wav_b64 = base64.b64encode(_wav_bytes())

    class _Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            yield _Turn(0.0, 0.7), None, "SPEAKER_00"
            yield _Turn(0.9, 1.4), None, "SPEAKER_01"

    class _DiarizeOutput:
        def __init__(self) -> None:
            self.speaker_diarization = _Annotation()

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization._ensure_pipeline",
        lambda: (lambda _path: _DiarizeOutput()),
    )

    segments = diarize_audio_b64(wav_b64, min_turn_length_s=0.2)
    status = get_last_diarization_status()

    assert len(segments) == 2
    assert segments[0]["track_id"] == "voice_1"
    assert segments[1]["track_id"] == "voice_2"
    assert status["ok"] is True
    assert status["fallback_used"] is False
    assert status["speaker_count"] == 2
