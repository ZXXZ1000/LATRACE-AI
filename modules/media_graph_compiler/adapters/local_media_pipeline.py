from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from modules.media_graph_compiler.adapters.frame_selector import FrameSelector
from modules.media_graph_compiler.adapters.media_normalizer import MediaNormalizer
from modules.media_graph_compiler.adapters.ops.video_processing import (
    process_video_to_fs,
)


class LocalMediaPipeline:
    """Prepare deterministic local media assets for the default video path.

    This adapter is the migrated "single source of truth" for local video
    preprocessing in `media_graph_compiler`:
    - normalize container
    - probe once
    - decode once into two frame streams
    - extract audio once
    - apply dedup/cap once

    The semantic provider should consume the resulting rich batch instead of
    re-owning raw media preprocessing by default.
    """

    def __init__(
        self,
        *,
        normalizer: MediaNormalizer | None = None,
        frame_selector: FrameSelector | None = None,
    ) -> None:
        self._normalizer = normalizer or MediaNormalizer()
        self._frame_selector = frame_selector or FrameSelector()

    def prepare_video_inputs(
        self,
        *,
        file_path: str,
        artifacts_dir: str | Path,
        sample_fps: float,
        clip_px: int,
        face_px: int,
        enable_dedup: bool,
        similarity_threshold: int,
        max_frames_per_source: int,
    ) -> Dict[str, Any]:
        normalized_path = self._normalizer.ensure_video_path(file_path)
        probe = self._normalizer.probe_media(normalized_path)
        media_root = Path(artifacts_dir) / "media_pipeline"
        prepared = process_video_to_fs(
            normalized_path,
            fps=sample_fps,
            clip_px=clip_px,
            face_px=face_px,
            out_base=str(media_root),
            audio_fps=16000,
        )
        frames_clip = list(prepared.get("frames_clip") or [])
        frames_face = list(prepared.get("frames_face") or [])

        selection = self._frame_selector.select_indices(
            frames_clip,
            enable_dedup=enable_dedup,
            similarity_threshold=similarity_threshold,
            max_frames=max_frames_per_source,
        )
        kept_indices = selection.kept_indices or list(range(len(frames_clip)))
        selected_clip = [frames_clip[index] for index in kept_indices]
        selected_face = [frames_face[index] for index in kept_indices] if frames_face else []

        duration_seconds = float(
            probe.get("duration_seconds")
            or prepared.get("duration")
            or 0.0
        )
        effective_frame_rate = (
            float(len(selected_face or selected_clip)) / duration_seconds
            if duration_seconds > 0 and (selected_face or selected_clip)
            else float(sample_fps or 0.0)
        )
        frame_timestamps_s = self._build_frame_timestamps(
            count=len(selected_clip),
            effective_fps=effective_frame_rate,
            duration_seconds=duration_seconds,
        )

        return {
            "normalized_video_path": normalized_path,
            "duration_seconds": duration_seconds,
            "frame_rate": probe.get("frame_rate"),
            "width": probe.get("width"),
            "height": probe.get("height"),
            "has_audio": bool(probe.get("has_audio")),
            "clip_frame_paths": selected_clip,
            "face_frame_paths": selected_face,
            "frame_timestamps_s": frame_timestamps_s,
            "selected_frame_indices": kept_indices,
            "dropped_frame_indices": selection.dropped_indices,
            "source_sample_fps": float(sample_fps or 0.0),
            "effective_frame_rate": effective_frame_rate,
            "audio_b64": prepared.get("audio_b64"),
            "extracted_audio_path": prepared.get("audio_path"),
            "media_mode": "frame_bundle",
        }

    @staticmethod
    def _build_frame_timestamps(
        *,
        count: int,
        effective_fps: float,
        duration_seconds: float,
    ) -> List[float]:
        if count <= 0:
            return []
        if effective_fps > 0:
            timestamps = [round(index / effective_fps, 3) for index in range(count)]
            if duration_seconds > 0:
                return [min(duration_seconds, item) for item in timestamps]
            return timestamps
        if duration_seconds <= 0:
            return [0.0 for _ in range(count)]
        if count == 1:
            return [0.0]
        step = duration_seconds / float(count)
        return [round(index * step, 3) for index in range(count)]


__all__ = ["LocalMediaPipeline"]
