from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.local_visual_stage import LocalVisualStage
from modules.media_graph_compiler.types import CompileVideoRequest, MediaRoutingContext, MediaSourceRef


class _Predictor:
    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "sess_1"}
        return {"frame_index": request.get("frame_index", 0), "outputs": {"out_obj_ids": [1]}}

    def handle_stream_request(self, request):
        yield {
            "frame_index": 0,
            "outputs": {
                "out_obj_ids": [1],
                "out_boxes_xywh": [[0, 0, 10, 10]],
                "out_binary_masks": ["mask_0"],
            },
        }
        yield {
            "frame_index": 1,
            "outputs": {
                "out_obj_ids": [1],
                "out_boxes_xywh": [[1, 1, 10, 10]],
                "out_binary_masks": ["mask_1"],
            },
        }


class _BrokenPredictor:
    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "sess_1"}
        return {"frame_index": request.get("frame_index", 0)}

    def handle_stream_request(self, request):
        raise RuntimeError("tracking unavailable")


def _request(tmp_path: Path) -> CompileVideoRequest:
    return CompileVideoRequest(
        routing=MediaRoutingContext(tenant_id="t1", user_id=["u1"]),
        source=MediaSourceRef(source_id="video_1", file_path=str(tmp_path / "demo.mp4")),
        metadata={"artifacts_dir": str(tmp_path)},
    )


def _seed_face_payload() -> dict:
    return {
        "face_1": [
            {
                "frame_id": 0,
                "bounding_box": [10, 20, 40, 60],
                "face_emb": [1.0, 0.0],
                "cluster_id": 0,
                "extra_data": {
                    "face_base64": base64.b64encode(b"jpeg-bytes").decode("utf-8"),
                    "face_detection_score": "0.99",
                    "face_quality_score": "42.0",
                },
            },
            {
                "frame_id": 1,
                "bounding_box": [12, 20, 42, 60],
                "face_emb": [0.99, 0.01],
                "cluster_id": 0,
                "extra_data": {
                    "face_base64": base64.b64encode(b"jpeg-bytes-2").decode("utf-8"),
                    "face_detection_score": "0.98",
                    "face_quality_score": "39.0",
                },
            },
        ]
    }


def _ctx(tmp_path: Path, request: CompileVideoRequest, **extra):
    return {
        "request": request,
        "source_id": request.source.source_id,
        "face_frame_paths": ["frame_0.jpg", "frame_1.jpg"],
        "frame_timestamps_s": [0.0, 0.5],
        "backbone_segments": [
            SimpleNamespace(id="seg_1", index=0, t_media_start=0.0, t_media_end=1.0),
        ],
        "normalized_video_path": str(tmp_path / "demo.mp4"),
        "optimization_plan": {"visual": {"max_frames_per_window": 0}},
        **extra,
    }


def test_local_visual_stage_uses_seed_tracklets_when_predictor_missing(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    stage = LocalVisualStage()
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        lambda *args, **kwargs: _seed_face_payload(),
    )

    out = stage.run(_ctx(tmp_path, request))

    assert len(out["visual_tracks"]) == 1
    assert out["visual_tracks"][0]["track_id"] == "face_1"
    assert out["visual_tracks"][0]["metadata"]["tracking_mode"] == "seed_tracklets"
    assert out["visual_stats"]["tracking_runtime_used"] is False
    assert any(item["kind"] == "frame_crop" for item in out["evidence"])
    assert Path(out["evidence"][0]["file_path"]).exists()


def test_local_visual_stage_prefers_tracking_runtime_when_predictor_injected(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    stage = LocalVisualStage()
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        lambda *args, **kwargs: _seed_face_payload(),
    )

    out = stage.run(_ctx(tmp_path, request, visual_tracking_predictor=_Predictor()))

    assert len(out["visual_tracks"]) == 1
    assert out["visual_tracks"][0]["metadata"]["tracking_mode"] == "session_box_prompt"
    assert out["visual_stats"]["tracking_runtime_success_count"] == 1
    assert any(item["kind"] == "mask" for item in out["evidence"])
    assert any(item["kind"] == "frame_crop" for item in out["evidence"])


def test_local_visual_stage_falls_back_to_seed_tracklets_on_tracking_failure(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    stage = LocalVisualStage()
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        lambda *args, **kwargs: _seed_face_payload(),
    )

    out = stage.run(_ctx(tmp_path, request, visual_tracking_predictor=_BrokenPredictor()))

    assert len(out["visual_tracks"]) == 1
    assert out["visual_tracks"][0]["metadata"]["tracking_mode"] == "seed_tracklets"
    assert out["visual_stats"]["tracking_runtime_failure_count"] == 1
    assert out["visual_stats"]["tracking_mode"] == "seed_tracklets"


def test_local_visual_stage_caps_detection_frames_and_remaps_indices(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    stage = LocalVisualStage()
    seen = {}

    def _fake_process_faces(_graph, frames, cache_path, **kwargs):
        seen["frames"] = list(frames)
        seen["cache_path"] = cache_path
        seen["segments"] = list(kwargs.get("segments") or [])
        return {
            "face_1": [
                {
                    "frame_id": 0,
                    "bounding_box": [10, 20, 40, 60],
                    "face_emb": [1.0, 0.0],
                    "cluster_id": 0,
                    "extra_data": {
                        "face_base64": base64.b64encode(b"jpeg-bytes").decode("utf-8"),
                        "face_detection_score": "0.99",
                        "face_quality_score": "42.0",
                    },
                },
                {
                    "frame_id": 1,
                    "bounding_box": [12, 20, 42, 60],
                    "face_emb": [0.99, 0.01],
                    "cluster_id": 0,
                    "extra_data": {
                        "face_base64": base64.b64encode(b"jpeg-bytes-2").decode("utf-8"),
                        "face_detection_score": "0.98",
                        "face_quality_score": "39.0",
                    },
                },
            ]
        }

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        _fake_process_faces,
    )

    out = stage.run(
        {
            **_ctx(tmp_path, request),
            "face_frame_paths": [f"frame_{i}.jpg" for i in range(5)],
            "frame_timestamps_s": [0.0, 0.5, 1.0, 1.5, 2.0],
            "optimization_plan": {
                "system": {"max_detection_frames_per_source": 2},
                "visual": {"max_frames_per_window": 0},
            },
        }
    )

    assert seen["frames"] == ["frame_0.jpg", "frame_2.jpg"]
    assert seen["segments"] == [{"index": 0, "segment_id": "seg_1", "frame_start": 0, "frame_end": 1}]
    assert "faces_" in Path(seen["cache_path"]).name
    assert out["visual_tracks"][0]["frame_start"] == 0
    assert out["visual_tracks"][0]["frame_end"] == 2
