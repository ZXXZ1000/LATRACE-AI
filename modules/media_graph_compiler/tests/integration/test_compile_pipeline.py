from pathlib import Path
import sys
import base64
import io
import wave

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler import (
    CompileAssetInputs,
    CompileAudioRequest,
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
    compile_audio,
    compile_video,
)
from modules.media_graph_compiler.adapters import LocalOperatorAssetStore, OperatorBus
from modules.media_graph_compiler.application import (
    FACE_VOICE_ASSOCIATION_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)


def _routing() -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="tenant_local",
        user_id=["u:test"],
        memory_domain="media",
        run_id="run_001",
    )


def _visual_stage(_ctx):
    return {
        "visual_tracks": [
            {
                "track_id": "face_1",
                "category": "person",
                "t_start_s": 0.0,
                "t_end_s": 4.0,
                "evidence_refs": ["ev_face_1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev_face_1",
                "kind": "frame_crop",
                "t_start_s": 0.0,
                "t_end_s": 0.5,
                "metadata": {"algorithm": "sam3"},
            }
        ],
    }


def _speaker_stage(_ctx):
    return {
        "speaker_tracks": [
            {
                "track_id": "voice_1",
                "t_start_s": 0.0,
                "t_end_s": 4.0,
                "utterance_ids": ["utt_1"],
                "evidence_refs": ["ev_audio_1"],
            }
        ],
        "utterances": [
            {
                "utterance_id": "utt_1",
                "speaker_track_id": "voice_1",
                "t_start_s": 0.2,
                "t_end_s": 1.4,
                "text": "hello from the kitchen",
                "evidence_refs": ["ev_audio_1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev_audio_1",
                "kind": "audio_chunk",
                "t_start_s": 0.2,
                "t_end_s": 1.4,
                "metadata": {"algorithm": "diarization", "transcript": "hello from the kitchen"},
            }
        ],
    }


def _assert_runtime_ctx_visual_stage(ctx):
    assert ctx["normalized_video_path"].endswith(".mp4")
    assert ctx["extracted_audio_path"].endswith(".wav")
    assert ctx["duration_seconds"] == 12.0
    assert ctx["optimization_plan"]["visual"]["enable_frame_dedup"] is True
    return _visual_stage(ctx)


def _semantic_stage(ctx):
    assert ctx["optimization_plan"]["system"]["prefer_asset_replay"] is True
    digests = []
    for index, payload in enumerate(ctx["window_payloads"]):
        summary = " ".join(item["text"] for item in payload.get("utterances", [])) or f"window_{index}"
        participant_refs = [item["track_id"] for item in payload.get("visual_tracks", [])]
        participant_refs.extend(item["track_id"] for item in payload.get("speaker_tracks", []))
        evidence_refs = [item["evidence_id"] for item in payload.get("evidence", [])]
        digests.append(
            {
                "window_id": payload["window_id"],
                "modality": payload["modality"],
                "t_start_s": payload["t_start_s"],
                "t_end_s": payload["t_end_s"],
                "summary": summary,
                "participant_refs": participant_refs,
                "evidence_refs": evidence_refs,
                "semantic_payload": {
                    "stage": "semantic_compile_stage",
                    "window_index": index,
                },
            }
        )
    return {"window_digests": digests}


def _association_stage(_ctx):
    return {
        "face_voice_links": [
            {
                "link_id": "fvlink_demo_1",
                "speaker_track_id": "voice_1",
                "visual_track_id": "face_1",
                "t_start_s": 0.2,
                "t_end_s": 1.4,
                "confidence": 0.91,
                "overlap_s": 1.2,
                "support_evidence_refs": ["ev_face_1", "ev_audio_1"],
                "support_utterance_ids": ["utt_1"],
                "metadata": {"method": "light_asd_temporal_fusion"},
            }
        ]
    }


def test_compile_video_runs_end_to_end_with_assets_and_graph(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-a-real-video")
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

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
        lambda video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000: {
            "frames_clip": [str((assets_dir / "clip_000001.jpg").resolve()), str((assets_dir / "clip_000002.jpg").resolve())],
            "frames_face": [str((assets_dir / "face_000001.jpg").resolve()), str((assets_dir / "face_000002.jpg").resolve())],
            "audio_b64": None,
            "audio_path": str((assets_dir / "demo.wav").resolve()),
            "duration": 12.0,
        },
    )
    (assets_dir / "clip_000001.jpg").write_bytes(b"clip1")
    (assets_dir / "clip_000002.jpg").write_bytes(b"clip2")
    (assets_dir / "face_000001.jpg").write_bytes(b"face1")
    (assets_dir / "face_000002.jpg").write_bytes(b"face2")
    (assets_dir / "demo.wav").write_bytes(b"wav")

    bus = OperatorBus()
    bus.register(VISUAL_TRACK_STAGE, _assert_runtime_ctx_visual_stage)
    bus.register(SPEAKER_TRACK_STAGE, _speaker_stage)
    bus.register(FACE_VOICE_ASSOCIATION_STAGE, _association_stage)
    bus.register(SEMANTIC_COMPILE_STAGE, _semantic_stage)

    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_1", file_path=str(video_path)),
        metadata={
            "duration_seconds": 12.0,
            "artifacts_dir": str(assets_dir),
        },
    )
    result = compile_video(
        request,
        operator_bus=bus,
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    assert result.status == "compiled"
    assert len(result.visual_tracks) == 1
    assert len(result.speaker_tracks) == 1
    assert len(result.face_voice_links) == 1
    assert len(result.window_digests) == 2
    assert len(result.asset_outputs) == 2
    assert result.graph_request is not None
    assert result.trace.optimization_plan["visual"]["max_frames_per_window"] == 12
    occurs_in = [edge for edge in result.graph_request.edges if edge.rel_type == "OCCURS_IN"]
    aligned_with = [edge for edge in result.graph_request.edges if edge.rel_type == "ALIGNED_WITH"]
    assert len(occurs_in) == len(result.window_digests)
    assert len(aligned_with) == 1
    assert aligned_with[0].confidence == 0.91
    assert all(Path(asset.file_path).exists() for asset in result.asset_outputs if asset.file_path)
    assert result.stats["stage_timings_ms"]["visual_stage_ms"] >= 0.0
    assert result.stats["stage_timings_ms"]["speaker_stage_ms"] >= 0.0
    assert result.stats["stage_timings_ms"]["total_ms"] >= result.stats["stage_timings_ms"]["graph_compile_ms"]

    cached_request = request.model_copy(
        update={
            "asset_inputs": CompileAssetInputs(
                visual_tracks=result.asset_outputs[0],
                speaker_tracks=result.asset_outputs[1],
            ),
            "enable_visual_operator": False,
            "enable_audio_operator": False,
        }
    )
    replay_result = compile_video(
        cached_request,
        operator_bus=OperatorBus(),
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    assert replay_result.visual_tracks[0].track_id == "face_1"
    assert replay_result.speaker_tracks[0].track_id == "voice_1"
    assert replay_result.asset_outputs == []


def test_compile_audio_can_write_graph_with_injected_writer(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"not-a-real-audio")
    bus = OperatorBus()
    bus.register(SPEAKER_TRACK_STAGE, _speaker_stage)
    bus.register(SEMANTIC_COMPILE_STAGE, _semantic_stage)
    written = []

    request = CompileAudioRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="audio_1", file_path=str(audio_path)),
        metadata={"duration_seconds": 6.0},
        write_graph=True,
    )

    result = compile_audio(
        request,
        operator_bus=bus,
        asset_store=LocalOperatorAssetStore(tmp_path / "audio_assets"),
        graph_writer=written.append,
    )

    assert result.status == "written"
    assert len(written) == 1
    assert result.graph_request is written[0]
    assert len(result.window_digests) == 1
    assert result.trace.warnings == []
    assert result.trace.optimization_plan["audio"]["enable_vad"] is True
    assert result.stats["stage_timings_ms"]["speaker_stage_ms"] >= 0.0
    assert result.stats["stage_timings_ms"]["total_ms"] >= result.stats["stage_timings_ms"]["graph_compile_ms"]


def _wav_bytes(duration_ms: int = 1200, sample_rate: int = 16000) -> bytes:
    frame_count = int(sample_rate * (duration_ms / 1000.0))
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)
    return bio.getvalue()


def test_compile_video_uses_default_local_stages_when_bus_is_empty(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-a-real-video")
    assets_dir = tmp_path / "assets_default"
    assets_dir.mkdir(parents=True, exist_ok=True)
    wav_bytes = _wav_bytes()
    wav_b64 = base64.b64encode(wav_bytes)

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.media_normalizer.MediaNormalizer.probe_media",
        lambda self, path: {
            "path": str(path),
            "duration_seconds": 4.0,
            "frame_rate": 1.0,
            "has_audio": True,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_media_pipeline.process_video_to_fs",
        lambda video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000: {
            "frames_clip": [str((assets_dir / "clip_000001.jpg").resolve()), str((assets_dir / "clip_000002.jpg").resolve())],
            "frames_face": [str((assets_dir / "face_000001.jpg").resolve()), str((assets_dir / "face_000002.jpg").resolve())],
            "audio_b64": wav_b64,
            "audio_path": str((assets_dir / "demo.wav").resolve()),
            "duration": 4.0,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        lambda *args, **kwargs: {
            "face_1": [
                {
                    "frame_id": 0,
                    "bounding_box": [1, 2, 30, 40],
                    "face_emb": [0.1, 0.2],
                    "cluster_id": 0,
                    "matched_node": "face_1",
                    "extra_data": {
                        "face_base64": base64.b64encode(b"jpg").decode("utf-8"),
                        "face_detection_score": "0.99",
                        "face_quality_score": "40.0",
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.diarize_audio_b64",
        lambda *args, **kwargs: [
            {"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 1.0},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.slice_audio_b64_segments",
        lambda *args, **kwargs: [wav_b64],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.transcribe_audio_b64",
        lambda *args, **kwargs: [
            {"start_time": "00:00", "end_time": "00:01", "asr": "hello world"},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.build_track_embedding",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "0")

    (assets_dir / "clip_000001.jpg").write_bytes(b"clip1")
    (assets_dir / "clip_000002.jpg").write_bytes(b"clip2")
    (assets_dir / "face_000001.jpg").write_bytes(b"face1")
    (assets_dir / "face_000002.jpg").write_bytes(b"face2")
    (assets_dir / "demo.wav").write_bytes(wav_bytes)

    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_default_1", file_path=str(video_path)),
        metadata={
            "duration_seconds": 4.0,
            "artifacts_dir": str(assets_dir),
        },
    )

    result = compile_video(
        request,
        operator_bus=OperatorBus(),
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    assert result.status == "compiled"
    assert result.visual_tracks[0].track_id == "face_1"
    assert result.speaker_tracks[0].track_id == "voice_1"
    assert result.window_digests[0].semantic_payload["frame_batch"] >= 1
    assert result.window_digests[0].semantic_payload["speaker_batch"] == 1


def test_compile_video_default_semantic_stage_uses_resolved_provider(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "provider_demo.mp4"
    video_path.write_bytes(b"not-a-real-video")
    assets_dir = tmp_path / "assets_provider"
    assets_dir.mkdir(parents=True, exist_ok=True)
    wav_bytes = _wav_bytes()
    wav_b64 = base64.b64encode(wav_bytes)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.media_normalizer.MediaNormalizer.probe_media",
        lambda self, path: {
            "path": str(path),
            "duration_seconds": 4.0,
            "frame_rate": 1.0,
            "has_audio": True,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_media_pipeline.process_video_to_fs",
        lambda video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000: {
            "frames_clip": [str((assets_dir / "clip_000001.jpg").resolve())],
            "frames_face": [str((assets_dir / "face_000001.jpg").resolve())],
            "audio_b64": wav_b64,
            "audio_path": str((assets_dir / "demo.wav").resolve()),
            "duration": 4.0,
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_visual_stage.process_faces",
        lambda *args, **kwargs: {
            "face_1": [
                {
                    "frame_id": 0,
                    "bounding_box": [1, 2, 30, 40],
                    "face_emb": [0.1, 0.2],
                    "cluster_id": 0,
                    "matched_node": "face_1",
                    "extra_data": {
                        "face_base64": base64.b64encode(b"jpg").decode("utf-8"),
                        "face_detection_score": "0.99",
                        "face_quality_score": "40.0",
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.diarize_audio_b64",
        lambda *args, **kwargs: [
            {"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 1.0},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.slice_audio_b64_segments",
        lambda *args, **kwargs: [wav_b64],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.transcribe_audio_b64",
        lambda *args, **kwargs: [
            {"start_time": "00:00", "end_time": "00:01", "asr": "hello world"},
        ],
    )
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_speaker_stage.build_track_embedding",
        lambda *args, **kwargs: [0.1, 0.2],
    )
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "0")

    class _ResolvedAdapter:
        kind = "openrouter_http"

        def generate(self, messages, response_format=None):
            return (
                '{"semantic_timeline":[{"text":"face_1 在厨房里说 hello world。",'
                '"actor_tag":"face_1","images":["img1"]}],"semantic":["厨房里有人说话"]}'
            )

    monkeypatch.setattr(
        "modules.memory.build_llm_from_byok",
        lambda **_: _ResolvedAdapter(),
    )

    (assets_dir / "clip_000001.jpg").write_bytes(b"clip1")
    (assets_dir / "face_000001.jpg").write_bytes(b"face1")
    (assets_dir / "demo.wav").write_bytes(wav_bytes)

    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_provider_1", file_path=str(video_path)),
        provider={
            "provider": "openrouter",
            "model": "google/gemini-2.5-flash",
        },
        metadata={
            "duration_seconds": 4.0,
            "artifacts_dir": str(assets_dir),
            "prompt_profile": "strict_json",
        },
    )

    result = compile_video(
        request,
        operator_bus=OperatorBus(),
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    digest = result.window_digests[0]
    assert "face_1 在厨房里说 hello world" in (digest.summary or "")
    assert digest.warnings == []
    assert digest.semantic_payload["provider_response"]["semantic"][0] == "厨房里有人说话"


def test_compile_video_ignores_zero_duration_metadata_and_uses_probe_duration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "duration_demo.mp4"
    video_path.write_bytes(b"not-a-real-video")
    assets_dir = tmp_path / "assets_duration"
    assets_dir.mkdir(parents=True, exist_ok=True)

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
        lambda video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000: {
            "frames_clip": [str((assets_dir / f"clip_{i:06d}.jpg").resolve()) for i in range(16)],
            "frames_face": [str((assets_dir / f"face_{i:06d}.jpg").resolve()) for i in range(16)],
            "audio_b64": None,
            "audio_path": str((assets_dir / "demo.wav").resolve()),
            "duration": 12.0,
        },
    )
    for i in range(16):
        (assets_dir / f"clip_{i:06d}.jpg").write_bytes(f"clip{i}".encode("utf-8"))
        (assets_dir / f"face_{i:06d}.jpg").write_bytes(f"face{i}".encode("utf-8"))
    (assets_dir / "demo.wav").write_bytes(b"wav")

    bus = OperatorBus()
    bus.register(VISUAL_TRACK_STAGE, _visual_stage)
    bus.register(SPEAKER_TRACK_STAGE, _speaker_stage)
    bus.register(SEMANTIC_COMPILE_STAGE, _semantic_stage)

    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_duration_zero", file_path=str(video_path)),
        metadata={
            "duration_seconds": 0.0,
            "artifacts_dir": str(assets_dir),
        },
    )
    result = compile_video(
        request,
        operator_bus=bus,
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    assert len(result.window_digests) == 2
    assert result.window_digests[0].t_start_s == 0.0
    assert result.window_digests[-1].t_end_s == 12.0


def test_compile_video_uses_visual_policy_fps_for_local_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "fps_demo.mp4"
    video_path.write_bytes(b"not-a-real-video")
    assets_dir = tmp_path / "assets_fps"
    assets_dir.mkdir(parents=True, exist_ok=True)
    captured_fps = {"value": None}

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.media_normalizer.MediaNormalizer.probe_media",
        lambda self, path: {
            "path": str(path),
            "duration_seconds": 4.0,
            "frame_rate": 1.0,
            "has_audio": False,
        },
    )

    def _fake_process_video_to_fs(video_path, fps=0.5, clip_px=256, face_px=640, out_base="", audio_fps=16000):
        captured_fps["value"] = float(fps)
        return {
            "frames_clip": [str((assets_dir / "clip_000001.jpg").resolve())],
            "frames_face": [str((assets_dir / "face_000001.jpg").resolve())],
            "audio_b64": None,
            "audio_path": None,
            "duration": 4.0,
        }

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_media_pipeline.process_video_to_fs",
        _fake_process_video_to_fs,
    )
    (assets_dir / "clip_000001.jpg").write_bytes(b"clip")
    (assets_dir / "face_000001.jpg").write_bytes(b"face")

    bus = OperatorBus()
    bus.register(VISUAL_TRACK_STAGE, _visual_stage)
    bus.register(SPEAKER_TRACK_STAGE, lambda _ctx: {"speaker_tracks": [], "utterances": [], "evidence": []})
    bus.register(SEMANTIC_COMPILE_STAGE, _semantic_stage)

    request = CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(source_id="video_fps_default", file_path=str(video_path)),
        metadata={"artifacts_dir": str(assets_dir)},
    )
    compile_video(
        request,
        operator_bus=bus,
        asset_store=LocalOperatorAssetStore(assets_dir),
    )

    assert captured_fps["value"] == 2.0
