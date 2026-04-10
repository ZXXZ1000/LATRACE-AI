from pathlib import Path
import sys
import base64
import io
import wave

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.scripts.benchmark_dry_run import (
    run_audio_benchmark,
    run_video_benchmark,
)


def test_run_video_benchmark_smoke(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    wav_bio = io.BytesIO()
    with wave.open(wav_bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    wav_bytes = wav_bio.getvalue()

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.media_normalizer.MediaNormalizer.probe_media",
        lambda self, path: {
            "path": str(path),
            "duration_seconds": 12.0,
            "frame_rate": 1.0,
            "has_audio": True,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_media_pipeline.process_video_to_fs",
        lambda video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000, clip_start_s=0.0, clip_end_s=None: {
            "frames_clip": [str((tmp_path / f"clip_{i:06d}.jpg").resolve()) for i in range(24)],
            "frames_face": [str((tmp_path / f"face_{i:06d}.jpg").resolve()) for i in range(24)],
            "audio_b64": base64.b64encode(wav_bytes),
            "audio_path": str((tmp_path / "demo.wav").resolve()),
            "duration": 12.0,
        },
    )
    for i in range(24):
        (tmp_path / f"clip_{i:06d}.jpg").write_bytes(f"clip{i // 2}".encode("utf-8"))
        (tmp_path / f"face_{i:06d}.jpg").write_bytes(f"face{i}".encode("utf-8"))
    (tmp_path / "demo.wav").write_bytes(wav_bytes)

    report = run_video_benchmark(str(video_path), tmp_path / "out", repeat=1)
    assert report["mode"] == "video"
    assert len(report["cold_runs"]) == 1
    assert len(report["replay_runs"]) == 1
    assert report["latest_result"]["graph_event_count"] >= 1
    assert report["latest_result"]["visual_selected_frame_count"] == 12
    assert report["latest_result"]["visual_dropped_frame_count"] > 0
    assert report["latest_result"]["optimization_plan"]["visual"]["enable_frame_dedup"] is True


def test_run_audio_benchmark_smoke(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.media_normalizer.MediaNormalizer.probe_media",
        lambda self, path: {
            "path": str(path),
            "duration_seconds": 6.0,
            "frame_rate": 0.0,
            "has_audio": True,
        },
    )

    report = run_audio_benchmark(str(audio_path), tmp_path / "out", repeat=1)
    assert report["mode"] == "audio"
    assert len(report["cold_runs"]) == 1
    assert len(report["replay_runs"]) == 1
    assert report["latest_result"]["speaker_track_count"] >= 1
    assert report["latest_result"]["audio_dropped_short_turn_count"] > 0
    assert report["latest_result"]["audio_asr_model"] == "dry_asr_compact"
    assert report["latest_result"]["optimization_plan"]["audio"]["enable_vad"] is True
