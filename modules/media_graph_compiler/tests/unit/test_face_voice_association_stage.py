from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.media_graph_compiler.adapters.local_face_voice_association_stage import (
    LocalFaceVoiceAssociationStage,
)


def test_local_face_voice_association_uses_light_asd_score(monkeypatch) -> None:
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")
    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_face_voice_association_stage.score_light_asd",
        lambda **_: {
            "raw_score": 1.7,
            "norm_score": 0.85,
            "frame_count": 50,
            "duration_s": 2.0,
        },
    )

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_1",
            "visual_tracks": [{"track_id": "face_1", "t_start_s": 0.0, "t_end_s": 3.0}],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.5, "t_end_s": 2.5}],
            "utterances": [
                {
                    "utterance_id": "utt_1",
                    "speaker_track_id": "voice_1",
                    "t_start_s": 1.0,
                    "t_end_s": 1.6,
                }
            ],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_1",
                    "file_path": "/tmp/f1.jpg",
                    "t_start_s": 0.8,
                    "t_end_s": 0.8,
                    "metadata": {"track_id": "face_1"},
                },
                {
                    "evidence_id": "ev_face_2",
                    "file_path": "/tmp/f2.jpg",
                    "t_start_s": 1.2,
                    "t_end_s": 1.2,
                    "metadata": {"track_id": "face_1"},
                },
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_1",
                    "file_path": "/tmp/a1.wav",
                    "t_start_s": 0.5,
                    "t_end_s": 2.5,
                    "metadata": {"speaker_track_id": "voice_1"},
                }
            ],
        }
    )

    links = result.get("face_voice_links") or []
    assert len(links) == 1
    assert links[0]["speaker_track_id"] == "voice_1"
    assert links[0]["visual_track_id"] == "face_1"
    assert float(links[0]["confidence"]) > 0.7
    assert links[0]["metadata"]["method"] == "light_asd_temporal_fusion"


def test_local_face_voice_association_falls_back_to_temporal(monkeypatch) -> None:
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "0")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_2",
            "visual_tracks": [{"track_id": "face_1", "t_start_s": 0.0, "t_end_s": 2.0}],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.5, "t_end_s": 1.5}],
            "utterances": [],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_1",
                    "file_path": "/tmp/f1.jpg",
                    "t_start_s": 1.0,
                    "t_end_s": 1.0,
                    "metadata": {"track_id": "face_1"},
                }
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_1",
                    "file_path": "/tmp/a1.wav",
                    "t_start_s": 0.5,
                    "t_end_s": 1.5,
                    "metadata": {"speaker_track_id": "voice_1"},
                }
            ],
        }
    )

    links = result.get("face_voice_links") or []
    assert len(links) == 1
    assert links[0]["metadata"]["method"] == "temporal_overlap_fallback"


def test_local_face_voice_association_caps_asd_calls_per_speaker(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")
    monkeypatch.setenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER", "1")

    def _fake_score(**kwargs):
        calls.append(kwargs)
        return {
            "raw_score": 1.0,
            "norm_score": 0.7,
            "frame_count": 24,
            "duration_s": 1.0,
        }

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_face_voice_association_stage.score_light_asd",
        _fake_score,
    )

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_3",
            "visual_tracks": [
                {"track_id": "face_1", "t_start_s": 0.0, "t_end_s": 3.0},
                {"track_id": "face_2", "t_start_s": 0.0, "t_end_s": 3.0},
            ],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.5, "t_end_s": 2.5}],
            "utterances": [
                {
                    "utterance_id": "utt_1",
                    "speaker_track_id": "voice_1",
                    "t_start_s": 1.0,
                    "t_end_s": 1.4,
                }
            ],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_1",
                    "file_path": "/tmp/f1.jpg",
                    "t_start_s": 0.8,
                    "t_end_s": 0.8,
                    "metadata": {"track_id": "face_1"},
                },
                {
                    "evidence_id": "ev_face_2",
                    "file_path": "/tmp/f2.jpg",
                    "t_start_s": 0.9,
                    "t_end_s": 0.9,
                    "metadata": {"track_id": "face_2"},
                },
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_1",
                    "file_path": "/tmp/a1.wav",
                    "t_start_s": 0.5,
                    "t_end_s": 2.5,
                    "metadata": {"speaker_track_id": "voice_1"},
                }
            ],
        }
    )

    assert len(calls) == 1
    assert result["association_stats"]["light_asd_success_count"] == 1
