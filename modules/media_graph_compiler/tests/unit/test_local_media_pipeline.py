from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.local_media_pipeline import LocalMediaPipeline


class _Normalizer:
    def ensure_video_path(self, path):
        return str(path)

    def probe_media(self, path):
        return {
            "path": str(path),
            "duration_seconds": 120.0,
            "frame_rate": 2.0,
            "width": 1280,
            "height": 720,
            "has_audio": True,
        }


def test_prepare_video_inputs_respects_clip_bounds_and_offsets_timestamps(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def _fake_process_video_to_fs(
        video_path,
        *,
        fps=0.5,
        clip_px=256,
        face_px=640,
        out_base="",
        audio_fps=16000,
        clip_start_s=0.0,
        clip_end_s=None,
    ):
        captured["clip_start_s"] = clip_start_s
        captured["clip_end_s"] = clip_end_s
        return {
            "frames_clip": [str(tmp_path / f"clip_{i}.jpg") for i in range(4)],
            "frames_face": [str(tmp_path / f"face_{i}.jpg") for i in range(4)],
            "audio_b64": None,
            "audio_path": None,
            "duration": 12.0,
        }

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_media_pipeline.process_video_to_fs",
        _fake_process_video_to_fs,
    )

    pipeline = LocalMediaPipeline(normalizer=_Normalizer())
    result = pipeline.prepare_video_inputs(
        file_path=str(tmp_path / "demo.mp4"),
        artifacts_dir=tmp_path / "artifacts",
        clip_start_s=30.0,
        clip_end_s=None,
        requested_duration_s=12.0,
        sample_fps=2.0,
        clip_px=256,
        face_px=640,
        enable_dedup=False,
        similarity_threshold=5,
        max_frames_per_source=0,
    )

    assert captured["clip_start_s"] == 30.0
    assert captured["clip_end_s"] == 42.0
    assert result["clip_start_s"] == 30.0
    assert result["clip_end_s"] == 42.0
    assert result["duration_seconds"] == 12.0
    assert result["frame_timestamps_s"] == [30.0, 33.0, 36.0, 39.0]
