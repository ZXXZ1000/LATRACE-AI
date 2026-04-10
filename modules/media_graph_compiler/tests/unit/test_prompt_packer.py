from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler import (
    EvidencePointer,
    SpeakerTrackRecord,
    UtteranceRecord,
    VisualTrackRecord,
)
from modules.media_graph_compiler.adapters.media_probe import MediaProbeAdapter
from modules.media_graph_compiler.application.prompt_packer import PromptPacker


def test_prompt_packer_builds_representative_frames_and_segment_visual_profile(
    tmp_path: Path,
) -> None:
    clip_frames = []
    face_frames = []
    frame_timestamps_s = []
    for index in range(5):
        clip_path = tmp_path / f"clip_{index}.jpg"
        face_path = tmp_path / f"face_{index}.jpg"
        clip_path.write_bytes(f"clip-{index}".encode("utf-8"))
        face_path.write_bytes(f"face-{index}".encode("utf-8"))
        clip_frames.append(str(clip_path))
        face_frames.append(str(face_path))
        frame_timestamps_s.append(float(index))

    backbone = MediaProbeAdapter().build_backbone(
        source_id="video_1",
        probe_meta={"frame_rate": 1.0},
        scenes=[{"start": 0.0, "end": 8.0, "modality": "video"}],
        default_modality="video",
    )
    packer = PromptPacker(image_embedder=lambda _content: [1.0, 2.0, 3.0])

    payloads = packer.build_video_window_payloads(
        backbone=backbone,
        visual_tracks=[
            VisualTrackRecord(
                track_id="face_1",
                category="person",
                t_start_s=0.0,
                t_end_s=4.0,
                evidence_refs=["ev_face_1"],
            )
        ],
        speaker_tracks=[
            SpeakerTrackRecord(
                track_id="voice_1",
                t_start_s=0.0,
                t_end_s=4.0,
                utterance_ids=["utt_1"],
                evidence_refs=["ev_audio_1"],
            )
        ],
        face_voice_links=[],
        utterances=[
            UtteranceRecord(
                utterance_id="utt_1",
                speaker_track_id="voice_1",
                t_start_s=0.5,
                t_end_s=1.0,
                text="hello there",
            )
        ],
        evidence=[
            EvidencePointer(
                evidence_id="ev_face_1",
                kind="frame_crop",
                t_start_s=0.5,
                t_end_s=0.6,
                metadata={"track_id": "face_1"},
            )
        ],
        clip_frames=clip_frames,
        face_frames=face_frames,
        frame_timestamps_s=frame_timestamps_s,
        max_frames_per_window=2,
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert len(payload["clip_frames"]) == 2
    assert len(payload["face_frames"]) == 2
    assert len(payload["representative_frames"]) == 3
    assert payload["window_stats"]["clip_frames_total"] == 5
    assert payload["window_stats"]["clip_frames_selected"] == 2
    assert payload["window_stats"]["representative_frames"] == 3
    profile = payload["segment_visual_profile"]
    assert len(profile["representative_thumbnails"]) == 3
    assert profile["thumbnail_evidence_refs"] == ["ev_face_1"]
    assert profile["vector"] == [1.0, 2.0, 3.0]
    assert profile["vector_summary"]["dim"] == 3
    assert profile["vector_summary"]["sample_count"] == 3
    assert profile["vector_summary"]["preview"] == [1.0, 2.0, 3.0]
