from pathlib import Path
import sys
import base64
import io
import wave

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.local_speaker_stage import LocalSpeakerStage
from modules.media_graph_compiler.types import CompileAudioRequest, MediaRoutingContext, MediaSourceRef


def _wav_bytes(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    frame_count = int(sample_rate * (duration_ms / 1000.0))
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
    return bio.getvalue()


def test_local_speaker_stage_builds_tracks_from_diarization_asr_and_embedding(tmp_path: Path, monkeypatch) -> None:
    wav_bytes = _wav_bytes()
    wav_b64 = base64.b64encode(wav_bytes)

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.diarize_audio_b64",
        lambda *args, **kwargs: [
            {"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 1.0},
            {"track_id": "voice_2", "t_start_s": 1.0, "t_end_s": 2.0},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.slice_audio_b64_segments",
        lambda *args, **kwargs: [wav_b64, wav_b64],
    )
    transcripts = [
        {"t_start_s": 0.0, "t_end_s": 1.0, "asr": "hello"},
        {"t_start_s": 1.0, "t_end_s": 2.0, "asr": "world"},
    ]
    calls = {"count": 0}

    def _transcribe(*args, **kwargs):
        calls["count"] += 1
        return list(transcripts)

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.transcribe_audio_b64",
        _transcribe,
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.get_last_asr_status",
        lambda: {
            "ok": True,
            "runtime": "faster_whisper",
            "fallback_used": False,
            "error": None,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.get_last_diarization_status",
        lambda: {
            "ok": True,
            "runtime": "pyannote_diarization",
            "fallback_used": False,
            "reason": None,
            "error": None,
            "speaker_count": 2,
            "segment_count": 2,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.build_track_embedding",
        lambda paths: [0.1, 0.2, 0.3] if paths else [],
    )

    request = CompileAudioRequest(
        routing=MediaRoutingContext(tenant_id="tenant", user_id=["u1"], memory_domain="media"),
        source=MediaSourceRef(source_id="audio_src", file_path=str(tmp_path / "demo.wav")),
        metadata={"artifacts_dir": str(tmp_path)},
    )
    stage = LocalSpeakerStage()
    out = stage.run(
        {
            "source_id": "audio_src",
            "request": request,
            "audio_b64": wav_b64,
            "optimization_plan": {"audio": {"min_turn_length_s": 0.4}},
        }
    )

    assert len(out["speaker_tracks"]) == 2
    assert len(out["utterances"]) == 2
    assert out["speaker_tracks"][0]["metadata"]["embedding_dim"] == 3
    assert out["utterances"][0]["text"] == "hello"
    assert Path(out["evidence"][0]["file_path"]).exists()
    assert calls["count"] == 1


def test_local_speaker_stage_exposes_diarization_and_asr_runtime_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wav_bytes = _wav_bytes()
    wav_b64 = base64.b64encode(wav_bytes)

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.diarize_audio_b64",
        lambda *args, **kwargs: [
            {
                "track_id": "voice_1",
                "t_start_s": 0.0,
                "t_end_s": 1.0,
                "metadata": {
                    "runtime": "fallback_single_speaker",
                    "reason": "pipeline_error",
                    "error": "GatedRepoError",
                },
            },
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.get_last_diarization_status",
        lambda: {
            "ok": False,
            "runtime": "pyannote_diarization",
            "fallback_used": True,
            "reason": "pipeline_error",
            "error": "GatedRepoError",
            "speaker_count": 0,
            "segment_count": 0,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.slice_audio_b64_segments",
        lambda *args, **kwargs: [wav_b64],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.transcribe_audio_b64",
        lambda *args, **kwargs: [
            {"t_start_s": 0.0, "t_end_s": 0.8, "asr": "well well well"}
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.get_last_asr_status",
        lambda: {
            "ok": True,
            "runtime": "faster_whisper",
            "fallback_used": False,
            "error": None,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.build_track_embedding",
        lambda *_: [0.1, 0.2, 0.3],
    )

    request = CompileAudioRequest(
        routing=MediaRoutingContext(tenant_id="tenant", user_id=["u1"], memory_domain="media"),
        source=MediaSourceRef(source_id="audio_src", file_path=str(tmp_path / "demo.wav")),
        metadata={"artifacts_dir": str(tmp_path)},
    )
    stage = LocalSpeakerStage()
    out = stage.run(
        {
            "source_id": "audio_src",
            "request": request,
            "audio_b64": wav_b64,
            "optimization_plan": {"audio": {"min_turn_length_s": 0.4}},
        }
    )

    assert out["utterances"][0]["t_end_s"] == 0.8
    assert out["utterances"][0]["metadata"]["asr_runtime"] == "faster_whisper"
    assert out["utterances"][0]["metadata"]["diarization_reason"] == "pipeline_error"
    assert out["evidence"][0]["metadata"]["diarization_reason"] == "pipeline_error"
    assert out["speaker_tracks"][0]["metadata"]["diarization_status"]["fallback_used"] is True
