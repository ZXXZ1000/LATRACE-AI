from pathlib import Path
import sys
import base64
import io
import wave

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.ops.speaker_diarization import (
    _adaptive_segmentation_step,
    _apply_runtime_config,
    _resolve_device,
    _resolve_segmentation_step,
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
    assert status["segmentation_step"] == 0.18


def test_resolve_segmentation_step_defaults_to_optimized_value(monkeypatch) -> None:
    monkeypatch.delenv("MGC_DIARIZATION_SEGMENTATION_STEP", raising=False)

    assert _resolve_segmentation_step() == 0.18


def test_resolve_segmentation_step_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("MGC_DIARIZATION_SEGMENTATION_STEP", "0.15")

    assert _resolve_segmentation_step() == 0.15


def test_adaptive_segmentation_step_increases_for_long_audio() -> None:
    assert _adaptive_segmentation_step(60.0) == 0.18
    assert _adaptive_segmentation_step(180.0) == 0.18
    assert _adaptive_segmentation_step(300.0) == 0.18


def test_apply_runtime_config_updates_pipeline_segmentation_step() -> None:
    class _Segmentation:
        def __init__(self) -> None:
            self.duration = 10.0
            self.step = 1.0

    class _Pipeline:
        def __init__(self) -> None:
            self.segmentation_step = 0.1
            self._segmentation = _Segmentation()

    pipeline = _Pipeline()
    runtime_config = _apply_runtime_config(pipeline)

    assert pipeline.segmentation_step == 0.18
    assert round(pipeline._segmentation.step, 6) == 1.8
    assert runtime_config["segmentation_step"] == 0.18
    assert runtime_config["segmentation_window_duration_s"] == 10.0


def test_apply_runtime_config_uses_long_audio_segmentation_step() -> None:
    class _Segmentation:
        def __init__(self) -> None:
            self.duration = 10.0
            self.step = 1.0

    class _Pipeline:
        def __init__(self) -> None:
            self.segmentation_step = 0.1
            self._segmentation = _Segmentation()

    pipeline = _Pipeline()
    runtime_config = _apply_runtime_config(pipeline, processing_duration_s=300.0)

    assert pipeline.segmentation_step == 0.18
    assert round(pipeline._segmentation.step, 6) == 1.8
    assert runtime_config["segmentation_step"] == 0.18


def test_diarize_audio_b64_reports_vad_compaction_status(monkeypatch) -> None:
    wav_b64 = base64.b64encode(_wav_bytes(duration_ms=2000))
    monkeypatch.setenv("MGC_DIARIZATION_ENABLE_VAD_COMPACTION", "1")

    class _Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            yield _Turn(0.0, 0.6), None, "SPEAKER_00"

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization.detect_speech_spans_b64",
        lambda *_args, **_kwargs: [
            {"t_start_s": 0.2, "t_end_s": 0.7},
            {"t_start_s": 1.0, "t_end_s": 1.4},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization.compact_audio_b64_to_speech_islands",
        lambda *_args, **_kwargs: {
            "audio_b64": wav_b64,
            "timeline_map": [
                {
                    "compact_t_start_s": 0.0,
                    "compact_t_end_s": 0.6,
                    "original_t_start_s": 0.2,
                    "original_t_end_s": 0.8,
                    "speech_island_index": 0,
                }
            ],
            "original_duration_s": 2.0,
            "compacted_duration_s": 0.72,
            "speech_ratio": 0.45,
            "applied": True,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization.remap_compacted_segments_to_original",
        lambda segments, *_args, **_kwargs: [
            {
                **dict(segments[0]),
                "t_start_s": 0.2,
                "t_end_s": 0.8,
            }
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization._ensure_pipeline",
        lambda: (lambda _path: _Annotation()),
    )

    segments = diarize_audio_b64(wav_b64, min_turn_length_s=0.2, enable_vad=True)
    status = get_last_diarization_status()

    assert len(segments) == 1
    assert segments[0]["t_start_s"] == 0.2
    assert segments[0]["t_end_s"] == 0.8
    assert status["ok"] is True
    assert status["vad"]["enabled"] is True
    assert status["vad"]["applied"] is True
    assert status["vad"]["compaction_allowed"] is True
    assert status["vad"]["speech_span_count"] == 2


def test_diarize_audio_b64_keeps_vad_compaction_disabled_by_default(monkeypatch) -> None:
    wav_b64 = base64.b64encode(_wav_bytes(duration_ms=2000))

    class _Turn:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class _Annotation:
        def itertracks(self, yield_label: bool = False):
            yield _Turn(0.0, 0.6), None, "SPEAKER_00"

    monkeypatch.delenv("MGC_DIARIZATION_ENABLE_VAD_COMPACTION", raising=False)
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.ops.speaker_diarization._ensure_pipeline",
        lambda: (lambda _path: _Annotation()),
    )

    segments = diarize_audio_b64(wav_b64, min_turn_length_s=0.2, enable_vad=True)
    status = get_last_diarization_status()

    assert len(segments) == 1
    assert segments[0]["t_start_s"] == 0.0
    assert segments[0]["t_end_s"] == 0.6
    assert status["vad"]["enabled"] is True
    assert status["vad"]["applied"] is False
    assert status["vad"]["compaction_allowed"] is False
    assert status["vad"]["speech_span_count"] == 0


def test_apply_runtime_config_respects_batch_size_env(monkeypatch) -> None:
    class _Segmentation:
        def __init__(self) -> None:
            self.duration = 10.0
            self.step = 1.0

    class _Pipeline:
        def __init__(self) -> None:
            self.segmentation_step = 0.1
            self._segmentation = _Segmentation()
            self.segmentation_batch_size = 1
            self.embedding_batch_size = 1

    monkeypatch.setenv("MGC_DIARIZATION_SEGMENTATION_BATCH_SIZE", "3")
    monkeypatch.setenv("MGC_DIARIZATION_EMBEDDING_BATCH_SIZE", "5")

    pipeline = _Pipeline()
    runtime_config = _apply_runtime_config(pipeline, processing_duration_s=300.0)

    assert pipeline.segmentation_batch_size == 3
    assert pipeline.embedding_batch_size == 5
    assert runtime_config["segmentation_batch_size"] == 3
    assert runtime_config["embedding_batch_size"] == 5


def test_resolve_device_defaults_to_cpu_on_darwin(monkeypatch) -> None:
    monkeypatch.delenv("MGC_SPEAKER_DEVICE", raising=False)
    monkeypatch.setattr("modules.media_graph_compiler.adapters.ops.speaker_diarization.sys.platform", "darwin")

    assert _resolve_device() == "cpu"
