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
        clip_start_s: float = 0.0,
        clip_end_s: float | None = None,
        requested_duration_s: float | None = None,
        sample_fps: float,
        clip_px: int,
        face_px: int,
        enable_dedup: bool,
        similarity_threshold: int,
        max_frames_per_source: int,
    ) -> Dict[str, Any]:
        normalized_path = self._normalizer.ensure_video_path(file_path)
        probe = self._normalizer.probe_media(normalized_path)
        effective_clip_start_s, effective_clip_end_s, effective_duration_s = self._resolve_clip_bounds(
            probe_duration_s=probe.get("duration_seconds"),
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            requested_duration_s=requested_duration_s,
        )
        media_root = Path(artifacts_dir) / "media_pipeline"
        prepared = process_video_to_fs(
            normalized_path,
            fps=sample_fps,
            clip_px=clip_px,
            face_px=face_px,
            out_base=str(media_root),
            audio_fps=16000,
            clip_start_s=effective_clip_start_s,
            clip_end_s=effective_clip_end_s,
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
            effective_duration_s
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
            start_offset_s=effective_clip_start_s,
        )

        return {
            "normalized_video_path": normalized_path,
            "duration_seconds": duration_seconds,
            "clip_start_s": effective_clip_start_s,
            "clip_end_s": effective_clip_end_s,
            "media_time_offset_s": effective_clip_start_s,
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
        start_offset_s: float = 0.0,
    ) -> List[float]:
        if count <= 0:
            return []
        if effective_fps > 0:
            timestamps = [round(start_offset_s + (index / effective_fps), 3) for index in range(count)]
            if duration_seconds > 0:
                clip_end_s = start_offset_s + duration_seconds
                return [min(clip_end_s, item) for item in timestamps]
            return timestamps
        if duration_seconds <= 0:
            return [round(start_offset_s, 3) for _ in range(count)]
        if count == 1:
            return [round(start_offset_s, 3)]
        step = duration_seconds / float(count)
        return [round(start_offset_s + (index * step), 3) for index in range(count)]

    @staticmethod
    def _resolve_clip_bounds(
        *,
        probe_duration_s: Any,
        clip_start_s: float,
        clip_end_s: float | None,
        requested_duration_s: float | None,
    ) -> tuple[float, float | None, float]:
        start_s = max(0.0, float(clip_start_s or 0.0))
        probe_duration = 0.0
        try:
            if probe_duration_s is not None:
                probe_duration = max(0.0, float(probe_duration_s))
        except Exception:
            probe_duration = 0.0

        end_s: float | None = None
        if clip_end_s is not None:
            try:
                end_candidate = float(clip_end_s)
                if end_candidate > start_s:
                    end_s = end_candidate
            except Exception:
                end_s = None
        if end_s is None and requested_duration_s is not None:
            try:
                requested_duration = float(requested_duration_s)
                if requested_duration > 0.0:
                    end_s = start_s + requested_duration
            except Exception:
                end_s = None
        if end_s is None and probe_duration > 0.0:
            end_s = probe_duration

        if probe_duration > 0.0 and end_s is not None:
            end_s = min(end_s, probe_duration)
        if end_s is not None and end_s < start_s:
            end_s = start_s

        duration_s = max(0.0, (end_s - start_s) if end_s is not None else probe_duration)
        return start_s, end_s, duration_s


__all__ = ["LocalMediaPipeline"]
