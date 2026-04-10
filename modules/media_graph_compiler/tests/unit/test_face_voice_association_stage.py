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


def test_local_face_voice_association_prefers_verified_asd_over_temporal_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")
    monkeypatch.setenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER", "2")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_ASD_PRIMARY_SCORE", "0.55")

    def _fake_score(**kwargs):
        face_paths = kwargs.get("face_frame_paths") or []
        if face_paths and "asd_win" in str(face_paths[0]):
            return {
                "raw_score": 1.9,
                "norm_score": 0.88,
                "frame_count": 30,
                "duration_s": 1.5,
            }
        return None

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_face_voice_association_stage.score_light_asd",
        _fake_score,
    )

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_4",
            "visual_tracks": [
                {"track_id": "face_1", "t_start_s": 0.5, "t_end_s": 2.5},
                {"track_id": "face_2", "t_start_s": 0.0, "t_end_s": 3.0},
            ],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 3.0}],
            "utterances": [],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_asd",
                    "file_path": "/tmp/asd_win.jpg",
                    "t_start_s": 1.0,
                    "t_end_s": 1.0,
                    "metadata": {"track_id": "face_1"},
                },
                {
                    "evidence_id": "ev_face_fallback",
                    "file_path": "/tmp/fallback_only.jpg",
                    "t_start_s": 1.0,
                    "t_end_s": 1.0,
                    "metadata": {"track_id": "face_2"},
                },
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_1",
                    "file_path": "/tmp/a1.wav",
                    "t_start_s": 0.0,
                    "t_end_s": 3.0,
                    "metadata": {"speaker_track_id": "voice_1"},
                }
            ],
        }
    )

    links = result.get("face_voice_links") or []
    assert len(links) == 1
    assert links[0]["visual_track_id"] == "face_1"
    assert links[0]["metadata"]["method"] == "light_asd_temporal_fusion"
    assert links[0]["metadata"]["asd_verified"] is True
    assert links[0]["metadata"]["provisional"] is False


def test_local_face_voice_association_prefers_scored_asd_over_fallback_only_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")
    monkeypatch.setenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER", "2")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_ASD_PRIMARY_SCORE", "0.55")

    def _fake_score(**kwargs):
        face_paths = kwargs.get("face_frame_paths") or []
        if face_paths and "asd_probe" in str(face_paths[0]):
            return {
                "raw_score": 0.32,
                "norm_score": 0.12,
                "frame_count": 18,
                "duration_s": 1.0,
            }
        return None

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_face_voice_association_stage.score_light_asd",
        _fake_score,
    )

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_4b",
            "visual_tracks": [
                {"track_id": "face_1", "t_start_s": 0.5, "t_end_s": 2.5},
                {"track_id": "face_2", "t_start_s": 0.0, "t_end_s": 3.0},
            ],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 3.0}],
            "utterances": [],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_asd",
                    "file_path": "/tmp/asd_probe.jpg",
                    "t_start_s": 1.0,
                    "t_end_s": 1.0,
                    "metadata": {"track_id": "face_1"},
                },
                {
                    "evidence_id": "ev_face_fallback",
                    "file_path": "/tmp/fallback_only.jpg",
                    "t_start_s": 1.0,
                    "t_end_s": 1.0,
                    "metadata": {"track_id": "face_2"},
                },
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_1",
                    "file_path": "/tmp/a1.wav",
                    "t_start_s": 0.0,
                    "t_end_s": 3.0,
                    "metadata": {"speaker_track_id": "voice_1"},
                }
            ],
        }
    )

    links = result.get("face_voice_links") or []
    assert len(links) == 1
    assert links[0]["visual_track_id"] == "face_1"
    assert links[0]["metadata"]["method"] == "light_asd_temporal_fusion"
    assert links[0]["metadata"]["asd_verified"] is False
    assert float(links[0]["confidence"]) >= 0.4


def test_local_face_voice_association_scores_asd_on_utterance_windows(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MIN_CONF", "0.1")
    monkeypatch.setenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER", "1")
    monkeypatch.setenv("MGC_ASSOCIATION_MAX_ASD_WINDOWS_PER_CANDIDATE", "2")
    monkeypatch.setenv("MGC_ASSOCIATION_ASD_WINDOW_PAD_S", "0.05")

    def _fake_score(**kwargs):
        calls.append(kwargs)
        if kwargs.get("audio_file") == "/tmp/a_late.wav":
            return {
                "raw_score": 2.1,
                "norm_score": 0.9,
                "frame_count": 18,
                "duration_s": 0.5,
            }
        return None

    monkeypatch.setattr(
        "modules.media_graph_compiler.adapters.local_face_voice_association_stage.score_light_asd",
        _fake_score,
    )

    stage = LocalFaceVoiceAssociationStage()
    result = stage.run(
        {
            "source_id": "video_5",
            "visual_tracks": [{"track_id": "face_1", "t_start_s": 0.0, "t_end_s": 10.0}],
            "speaker_tracks": [{"track_id": "voice_1", "t_start_s": 0.0, "t_end_s": 10.0}],
            "utterances": [
                {
                    "utterance_id": "utt_early",
                    "speaker_track_id": "voice_1",
                    "t_start_s": 1.0,
                    "t_end_s": 1.3,
                },
                {
                    "utterance_id": "utt_late",
                    "speaker_track_id": "voice_1",
                    "t_start_s": 6.0,
                    "t_end_s": 6.4,
                },
            ],
            "visual_evidence": [
                {
                    "evidence_id": "ev_face_early",
                    "file_path": "/tmp/f_early.jpg",
                    "t_start_s": 1.1,
                    "t_end_s": 1.1,
                    "metadata": {"track_id": "face_1"},
                },
                {
                    "evidence_id": "ev_face_late",
                    "file_path": "/tmp/f_late.jpg",
                    "t_start_s": 6.1,
                    "t_end_s": 6.1,
                    "metadata": {"track_id": "face_1"},
                },
            ],
            "audio_evidence": [
                {
                    "evidence_id": "ev_audio_early",
                    "file_path": "/tmp/a_early.wav",
                    "t_start_s": 0.8,
                    "t_end_s": 1.5,
                    "metadata": {"speaker_track_id": "voice_1"},
                },
                {
                    "evidence_id": "ev_audio_late",
                    "file_path": "/tmp/a_late.wav",
                    "t_start_s": 5.9,
                    "t_end_s": 6.5,
                    "metadata": {"speaker_track_id": "voice_1"},
                },
            ],
        }
    )

    links = result.get("face_voice_links") or []
    assert len(links) == 1
    assert [call.get("audio_file") for call in calls] == [
        "/tmp/a_early.wav",
        "/tmp/a_late.wav",
    ]
    assert links[0]["metadata"]["asd_verified"] is True
    assert links[0]["metadata"]["asd_window_count"] == 1
    assert links[0]["metadata"]["asd_detail"]["audio_file"] == "/tmp/a_late.wav"
    assert links[0]["metadata"]["asd_detail"]["window_utterance_ids"] == ["utt_late"]
