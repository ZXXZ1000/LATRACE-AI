from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler import (
    CompileAudioRequest,
    CompileResult,
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
    OperatorAssetRef,
)
from modules.media_graph_compiler.types.contracts import (
    VisualContinuityServiceRequest,
    VisualContinuityServiceResponse,
    VisualPrompt,
)
from modules.memory.contracts.graph_models import Event, GraphUpsertRequest


def _routing() -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="tenant_local",
        user_id=["u:test"],
        memory_domain="video",
        run_id="run_001",
    )


def test_source_requires_location() -> None:
    try:
        MediaSourceRef(source_id="src_1")
    except ValueError as exc:
        assert "file_path or blob_ref" in str(exc)
    else:
        raise AssertionError("expected MediaSourceRef validation to fail")


def test_video_request_defaults() -> None:
    req = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="src_1", file_path="/tmp/demo.mp4"),
    )
    assert req.windowing.video_window_seconds == 8.0
    assert req.enable_visual_operator is True
    assert req.enable_audio_operator is True
    assert req.visual_policy.include_face_crops is True
    assert req.speaker_policy.enable_diarization is True
    assert req.optimization.enable_visual_dedup is True
    assert req.optimization.max_visual_frames_per_window == 12


def test_audio_request_defaults() -> None:
    req = CompileAudioRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="src_2", blob_ref="blob://audio/demo.wav"),
    )
    assert req.windowing.audio_window_seconds == 15.0
    assert req.enable_audio_operator is True
    assert req.identity.enable_identity_resolution is True
    assert req.identity.auto_create_provisional_person is True
    assert req.speaker_policy.enable_voice_features is True
    assert req.optimization.enable_audio_vad is True


def test_compile_request_accepts_precomputed_track_assets() -> None:
    req = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="src_3", file_path="/tmp/movie.mp4"),
        asset_inputs={
            "visual_tracks": {
                "asset_id": "asset_visual_1",
                "asset_type": "visual_tracks",
                "file_path": "/tmp/visual_tracks.json",
            },
            "speaker_tracks": {
                "asset_id": "asset_speaker_1",
                "asset_type": "speaker_tracks",
                "blob_ref": "blob://tracks/speaker.json",
            },
        },
    )
    assert req.asset_inputs.visual_tracks is not None
    assert req.asset_inputs.visual_tracks.asset_type == "visual_tracks"
    assert req.asset_inputs.speaker_tracks is not None
    assert req.asset_inputs.speaker_tracks.asset_type == "speaker_tracks"


def test_compile_result_accepts_graph_request_and_assets() -> None:
    result = CompileResult(
        status="compiled",
        source_id="src_1",
        asset_outputs=[
            OperatorAssetRef(
                asset_id="asset_visual_out",
                asset_type="visual_tracks",
                file_path="/tmp/out_visual_tracks.json",
            )
        ],
        graph_request=GraphUpsertRequest(
            events=[
                Event(
                    id="evt_1",
                    summary="Alice enters the kitchen.",
                    tenant_id="tenant_local",
                )
            ]
        ),
    )
    assert result.graph_request is not None
    assert result.graph_request.events[0].id == "evt_1"
    assert result.asset_outputs[0].asset_type == "visual_tracks"


def test_compile_result_capture_tracks_and_utterances() -> None:
    result = CompileResult(
        status="compiled",
        source_id="src_7",
        visual_tracks=[
            {
                "track_id": "face_1",
                "t_start_s": 0.0,
                "t_end_s": 5.0,
            }
        ],
        speaker_tracks=[
            {
                "track_id": "voice_1",
                "t_start_s": 0.0,
                "t_end_s": 5.0,
            }
        ],
        utterances=[
            {
                "utterance_id": "utt_1",
                "speaker_track_id": "voice_1",
                "t_start_s": 0.2,
                "t_end_s": 1.3,
                "text": "hello world",
            }
        ],
    )
    assert result.visual_tracks[0].track_id == "face_1"
    assert result.speaker_tracks[0].track_id == "voice_1"
    assert result.utterances[0].speaker_track_id == "voice_1"


def test_visual_prompt_requires_shape_by_kind() -> None:
    prompt = VisualPrompt(
        prompt_id="person_default",
        kind="text",
        text="person",
    )
    assert prompt.text == "person"

    try:
        VisualPrompt(prompt_id="bad_box", kind="box")
    except ValueError as exc:
        assert "box_xyxy" in str(exc)
    else:
        raise AssertionError("expected box prompt validation to fail")

    try:
        VisualPrompt(prompt_id="bad_point", kind="point", points_xy=[[10.0, 12.0]], point_labels=[1, 0])
    except ValueError as exc:
        assert "point_labels length" in str(exc)
    else:
        raise AssertionError("expected point prompt validation to fail")


def test_visual_continuity_service_request_requires_prompt() -> None:
    try:
        VisualContinuityServiceRequest(
            source=MediaSourceRef(source_id="src_v", file_path="/tmp/demo.mp4"),
            prompts=[],
        )
    except ValueError as exc:
        assert "at least one visual prompt" in str(exc)
    else:
        raise AssertionError("expected visual continuity request validation to fail")


def test_visual_continuity_service_response_matches_stage_shape() -> None:
    response = VisualContinuityServiceResponse(
        source_id="src_v",
        visual_tracks=[
            {
                "track_id": "face_1",
                "category": "person",
                "t_start_s": 0.0,
                "t_end_s": 10.0,
                "frame_start": 0,
                "frame_end": 200,
                "evidence_refs": ["ev_1"],
            }
        ],
        evidence=[
            {
                "evidence_id": "ev_1",
                "kind": "frame_crop",
                "file_path": "/tmp/face_1_000120.jpg",
            }
        ],
        stats={"frames_processed": 200},
    )
    assert response.visual_tracks[0].track_id == "face_1"
    assert response.evidence[0].kind == "frame_crop"
    assert response.stats["frames_processed"] == 200
