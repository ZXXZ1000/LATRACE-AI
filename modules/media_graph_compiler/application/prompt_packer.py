from __future__ import annotations

import base64
import os
from math import sqrt
from typing import Any, Callable, Dict, List, Sequence

from modules.media_graph_compiler.adapters.media_probe import MediaBackbone
from modules.media_graph_compiler.types import (
    EvidencePointer,
    FaceVoiceLinkRecord,
    SpeakerTrackRecord,
    UtteranceRecord,
    VisualTrackRecord,
)
from modules.memory.application.embedding_adapter import (
    build_image_embedding_from_settings,
)


class PromptPacker:
    """Pack deterministic stage outputs into per-window semantic payloads."""

    def __init__(
        self,
        *,
        image_embedder: Callable[[str], List[float]] | None = None,
    ) -> None:
        self._image_embedder = image_embedder or build_image_embedding_from_settings(
            {
                "provider": str(
                    os.getenv("MGC_SEGMENT_IMAGE_EMBEDDING_PROVIDER") or ""
                ).strip(),
                "model": str(os.getenv("MGC_SEGMENT_IMAGE_EMBEDDING_MODEL") or "").strip()
                or None,
                "pretrained": str(
                    os.getenv("MGC_SEGMENT_IMAGE_EMBEDDING_PRETRAINED") or ""
                ).strip()
                or None,
                "dim": int(os.getenv("MGC_SEGMENT_IMAGE_EMBEDDING_DIM") or 512),
            }
        )
        self._data_url_cache: Dict[str, str | None] = {}
        self._image_vector_cache: Dict[str, List[float]] = {}

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
            clip_frames_all = self._frames_for_segment(
                frame_paths=clip_frames,
                frame_timestamps_s=frame_timestamps_s,
                segment=segment,
            )
            face_frames_all = self._frames_for_segment(
                frame_paths=face_frames,
                frame_timestamps_s=frame_timestamps_s,
                segment=segment,
            )
            clip_bundle = self._cap_frame_bundle(
                clip_frames_all,
                cap=max_frames_per_window,
            )
            face_bundle = self._cap_frame_bundle(
                face_frames_all,
                cap=max_frames_per_window,
            )
            representative_frames = self._cap_frame_bundle(
                clip_frames_all,
                cap=min(3, len(clip_frames_all)),
            )
            visual_tracks_in_window = [
                dumped
                for item, dumped in dumped_visual_tracks
                if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
            ]
            speaker_tracks_in_window = [
                dumped
                for item, dumped in dumped_speaker_tracks
                if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
            ]
            face_voice_links_in_window = [
                dumped
                for item, dumped in dumped_face_voice_links
                if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
            ]
            utterances_in_window = [
                dumped
                for item, dumped in dumped_utterances
                if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
            ]
            evidence_in_window = [
                dumped
                for item, dumped in dumped_evidence
                if item.t_start_s is None
                or (item.t_start_s <= segment.t_media_end and (item.t_end_s or item.t_start_s) >= segment.t_media_start)
            ]
            payloads.append(
                {
                    "window_id": segment.id,
                    "modality": "video",
                    "t_start_s": segment.t_media_start,
                    "t_end_s": segment.t_media_end,
                    "clip_frames": clip_bundle,
                    "face_frames": face_bundle,
                    "representative_frames": representative_frames,
                    "visual_tracks": visual_tracks_in_window,
                    "speaker_tracks": speaker_tracks_in_window,
                    "face_voice_links": face_voice_links_in_window,
                    "utterances": utterances_in_window,
                    "evidence": evidence_in_window,
                    "window_stats": {
                        "clip_frames_total": len(clip_frames_all),
                        "clip_frames_selected": len(clip_bundle),
                        "face_frames_total": len(face_frames_all),
                        "face_frames_selected": len(face_bundle),
                        "representative_frames": len(representative_frames),
                        "visual_tracks": len(visual_tracks_in_window),
                        "speaker_tracks": len(speaker_tracks_in_window),
                        "face_voice_links": len(face_voice_links_in_window),
                        "utterances": len(utterances_in_window),
                        "evidence": len(evidence_in_window),
                    },
                    "segment_visual_profile": self._build_segment_visual_profile(
                        frames=representative_frames,
                        evidence=evidence_in_window,
                    ),
                }
            )
        return payloads

    @staticmethod
    def _frames_for_segment(
        *,
        frame_paths: Sequence[str],
        frame_timestamps_s: Sequence[float],
        segment,
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
        return selected

    @staticmethod
    def _cap_frame_bundle(
        frames: Sequence[Dict[str, object]],
        *,
        cap: int,
    ) -> List[Dict[str, object]]:
        selected = list(frames or [])
        if cap <= 0 or len(selected) <= cap:
            return selected
        if cap == 1:
            return [selected[0]]
        step = len(selected) / float(cap)
        return [selected[min(len(selected) - 1, int(i * step + step / 2.0))] for i in range(cap)]

    def _build_segment_visual_profile(
        self,
        *,
        frames: Sequence[Dict[str, object]],
        evidence: Sequence[Dict[str, object]],
    ) -> Dict[str, Any]:
        vector = self._mean_vectors(
            self._embed_frame(item.get("file_path")) for item in frames
        )
        vector_provider = str(
            os.getenv("MGC_SEGMENT_IMAGE_EMBEDDING_PROVIDER") or "hash"
        ).strip() or "hash"
        thumbnail_refs = [
            str(item.get("evidence_id"))
            for item in evidence
            if str(item.get("kind") or "") in {"thumbnail", "frame_crop", "mask"}
            and item.get("evidence_id")
        ][:4]
        return {
            "representative_thumbnails": [
                {
                    "frame_index": item.get("frame_index"),
                    "t_media_s": item.get("t_media_s"),
                    "file_path": item.get("file_path"),
                }
                for item in frames
            ],
            "thumbnail_evidence_refs": thumbnail_refs,
            "vector": vector,
            "vector_summary": {
                "provider": vector_provider,
                "strategy": "mean_pool",
                "sample_count": len(frames),
                "dim": len(vector),
                "preview": [round(float(value), 6) for value in vector[:8]],
                "l2_norm": round(
                    sqrt(sum(float(value) * float(value) for value in vector)),
                    6,
                )
                if vector
                else 0.0,
            },
        }

    def _embed_frame(self, file_path: object) -> List[float]:
        if not isinstance(file_path, str) or not file_path:
            return []
        cached = self._image_vector_cache.get(file_path)
        if cached is not None:
            return list(cached)
        data_url = self._to_data_url(file_path)
        if not data_url:
            return []
        try:
            vector = [float(item) for item in (self._image_embedder(data_url) or [])]
        except Exception:
            vector = []
        self._image_vector_cache[file_path] = vector
        return list(vector)

    def _to_data_url(self, file_path: str) -> str | None:
        if file_path in self._data_url_cache:
            return self._data_url_cache[file_path]
        if not os.path.exists(file_path):
            self._data_url_cache[file_path] = None
            return None
        try:
            raw = open(file_path, "rb").read()
            data_url = f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
            self._data_url_cache[file_path] = data_url
            return data_url
        except Exception:
            self._data_url_cache[file_path] = None
            return None

    @staticmethod
    def _mean_vectors(vectors: Sequence[Sequence[float]]) -> List[float]:
        normalized = [list(map(float, item)) for item in vectors if item]
        if not normalized:
            return []
        dim = max((len(item) for item in normalized), default=0)
        if dim <= 0:
            return []
        acc = [0.0] * dim
        count = 0
        for vec in normalized:
            if len(vec) != dim:
                continue
            count += 1
            for index, value in enumerate(vec):
                acc[index] += float(value)
        if count <= 0:
            return []
        return [value / float(count) for value in acc]

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
