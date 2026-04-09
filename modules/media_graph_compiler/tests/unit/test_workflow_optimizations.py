from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler import (
    CompileAudioRequest,
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
)
from modules.media_graph_compiler.adapters import FrameSelector
from modules.media_graph_compiler.application import OptimizationPlanBuilder


def _routing() -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="tenant_local",
        user_id=["u:test"],
        memory_domain="media",
        run_id="run_opt",
    )


def test_frame_selector_reuses_legacy_dedup_and_cap_logic(tmp_path: Path) -> None:
    frames = []
    payloads = [b"red", b"red", b"green", b"green", b"blue"]
    for index, payload in enumerate(payloads):
        path = tmp_path / f"frame_{index}.bin"
        path.write_bytes(payload)
        frames.append(str(path))

    selector = FrameSelector()
    result = selector.select_indices(
        frames,
        enable_dedup=True,
        similarity_threshold=5,
        max_frames=2,
    )

    assert result.kept_indices == [0, 2]
    assert set(result.dropped_indices) == {1, 3, 4}
    assert result.hashes[0] == result.hashes[1]
    assert result.hashes[2] == result.hashes[3]


def test_optimization_plan_builder_exposes_video_stage_knobs() -> None:
    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_1", file_path="/tmp/demo.mp4"),
    )
    plan = OptimizationPlanBuilder().build_video_plan(
        request=request,
        runtime_inputs={
            "normalized_video_path": "/tmp/demo.mp4",
            "extracted_audio_path": "/tmp/demo.wav",
            "duration_seconds": 12.0,
            "frame_rate": 1.0,
        },
    )

    assert plan["system"]["prefer_asset_replay"] is True
    assert plan["visual"]["enable_frame_dedup"] is True
    assert plan["visual"]["max_frames_per_window"] == 12
    assert plan["audio"]["enable_asr_rtf_adaptation"] is True
    assert plan["semantic"]["allow_frame_bundle"] is True


def test_optimization_plan_builder_exposes_audio_stage_knobs() -> None:
    request = CompileAudioRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="audio_1", file_path="/tmp/demo.wav"),
    )
    plan = OptimizationPlanBuilder().build_audio_plan(
        request=request,
        runtime_inputs={"normalized_audio_path": "/tmp/demo.wav", "duration_seconds": 6.0},
    )

    assert plan["system"]["drop_full_video_payload"] is True
    assert plan["audio"]["enable_vad"] is True
    assert plan["audio"]["min_turn_length_s"] == 0.4
    assert plan["semantic"]["audio_chunk_seconds"] == 5.0
