from __future__ import annotations

from typing import Dict, List, Sequence

from modules.media_graph_compiler.adapters.media_probe import MediaBackbone
from modules.media_graph_compiler.types import (
    EvidencePointer,
    FaceVoiceLinkRecord,
    SpeakerTrackRecord,
    UtteranceRecord,
    VisualTrackRecord,
)


class PromptPacker:
    """Pack deterministic stage outputs into per-window semantic payloads."""

    def build_video_window_payloads(
        self,
        *,
        backbone: MediaBackbone,
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
        face_voice_links: Sequence[FaceVoiceLinkRecord],
        utterances: Sequence[UtteranceRecord],
        evidence: Sequence[EvidencePointer],
        clip_frames: Sequence[str] | None = None,
        face_frames: Sequence[str] | None = None,
        frame_timestamps_s: Sequence[float] | None = None,
        max_frames_per_window: int = 0,
    ) -> List[Dict[str, object]]:
        payloads: List[Dict[str, object]] = []
        clip_frames = list(clip_frames or [])
        face_frames = list(face_frames or [])
        frame_timestamps_s = list(frame_timestamps_s or [])
        dumped_visual_tracks = [(item, item.model_dump()) for item in visual_tracks]
        dumped_speaker_tracks = [(item, item.model_dump()) for item in speaker_tracks]
        dumped_face_voice_links = [(item, item.model_dump()) for item in face_voice_links]
        dumped_utterances = [(item, item.model_dump()) for item in utterances]
        dumped_evidence = [(item, item.model_dump()) for item in evidence]
        for segment in backbone.segments:
            clip_bundle = self._select_frames_for_segment(
                frame_paths=clip_frames,
                frame_timestamps_s=frame_timestamps_s,
                segment=segment,
                cap=max_frames_per_window,
            )
            face_bundle = self._select_frames_for_segment(
                frame_paths=face_frames,
                frame_timestamps_s=frame_timestamps_s,
                segment=segment,
                cap=max_frames_per_window,
            )
            payloads.append(
                {
                    "window_id": segment.id,
                    "modality": "video",
                    "t_start_s": segment.t_media_start,
                    "t_end_s": segment.t_media_end,
                    "clip_frames": clip_bundle,
                    "face_frames": face_bundle,
                    "visual_tracks": [
                        dumped
                        for item, dumped in dumped_visual_tracks
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "speaker_tracks": [
                        dumped
                        for item, dumped in dumped_speaker_tracks
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "face_voice_links": [
                        dumped
                        for item, dumped in dumped_face_voice_links
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "utterances": [
                        dumped
                        for item, dumped in dumped_utterances
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "evidence": [
                        dumped
                        for item, dumped in dumped_evidence
                        if item.t_start_s is None
                        or (item.t_start_s <= segment.t_media_end and (item.t_end_s or item.t_start_s) >= segment.t_media_start)
                    ],
                }
            )
        return payloads

    @staticmethod
    def _select_frames_for_segment(
        *,
        frame_paths: Sequence[str],
        frame_timestamps_s: Sequence[float],
        segment,
        cap: int,
    ) -> List[Dict[str, object]]:
        selected: List[Dict[str, object]] = []
        for index, path in enumerate(frame_paths):
            timestamp = float(frame_timestamps_s[index]) if index < len(frame_timestamps_s) else 0.0
            if segment.t_media_start <= timestamp <= segment.t_media_end:
                selected.append(
                    {
                        "frame_index": index,
                        "file_path": path,
                        "t_media_s": timestamp,
                    }
                )
        if cap > 0 and len(selected) > cap:
            if cap == 1:
                return [selected[0]]
            step = len(selected) / float(cap)
            return [selected[int(i * step)] for i in range(cap)]
        return selected

    def build_audio_window_payloads(
        self,
        *,
        backbone: MediaBackbone,
        speaker_tracks: Sequence[SpeakerTrackRecord],
        utterances: Sequence[UtteranceRecord],
        evidence: Sequence[EvidencePointer],
    ) -> List[Dict[str, object]]:
        payloads: List[Dict[str, object]] = []
        dumped_speaker_tracks = [(item, item.model_dump()) for item in speaker_tracks]
        dumped_utterances = [(item, item.model_dump()) for item in utterances]
        dumped_evidence = [(item, item.model_dump()) for item in evidence]
        for segment in backbone.segments:
            payloads.append(
                {
                    "window_id": segment.id,
                    "modality": "audio",
                    "t_start_s": segment.t_media_start,
                    "t_end_s": segment.t_media_end,
                    "speaker_tracks": [
                        dumped
                        for item, dumped in dumped_speaker_tracks
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "utterances": [
                        dumped
                        for item, dumped in dumped_utterances
                        if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
                    ],
                    "evidence": [
                        dumped
                        for item, dumped in dumped_evidence
                        if item.t_start_s is None
                        or (item.t_start_s <= segment.t_media_end and (item.t_end_s or item.t_start_s) >= segment.t_media_start)
                    ],
                }
            )
        return payloads


__all__ = ["PromptPacker"]
