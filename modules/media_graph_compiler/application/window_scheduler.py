from __future__ import annotations

from typing import Dict, List, Mapping

from modules.media_graph_compiler.types import WindowingPolicy


class WindowScheduler:
    """Builds deterministic rolling windows for video or audio inputs."""

    _MAX_VIDEO_SEMANTIC_WINDOWS = 32
    _MAX_VIDEO_WINDOW_SECONDS = 20.0

    def build_windows(
        self,
        *,
        modality: str,
        clip_start_s: float,
        clip_end_s: float,
        policy: WindowingPolicy,
        resolved_settings: Mapping[str, float | bool] | None = None,
    ) -> List[Dict[str, float | str]]:
        settings = dict(
            resolved_settings
            or self.resolve_settings(
                modality=modality,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                policy=policy,
            )
        )
        window_size = float(settings["window_size_seconds"])
        overlap_seconds = float(settings["overlap_seconds"])
        if clip_end_s <= clip_start_s:
            clip_end_s = clip_start_s + window_size

        step = max(window_size - overlap_seconds, 0.1)
        windows: List[Dict[str, float | str]] = []
        cursor = clip_start_s
        while cursor < clip_end_s:
            end = min(cursor + window_size, clip_end_s)
            windows.append(
                {
                    "start": cursor,
                    "end": end,
                    "modality": modality,
                }
            )
            if end >= clip_end_s:
                break
            cursor += step
        if not windows:
            windows.append(
                {
                    "start": clip_start_s,
                    "end": clip_start_s + window_size,
                    "modality": modality,
                }
            )
        return windows

    def resolve_settings(
        self,
        *,
        modality: str,
        clip_start_s: float,
        clip_end_s: float,
        policy: WindowingPolicy,
    ) -> Dict[str, float | bool]:
        window_size = float(
            policy.video_window_seconds if modality == "video" else policy.audio_window_seconds
        )
        overlap_seconds = float(policy.overlap_seconds)
        if clip_end_s <= clip_start_s:
            clip_end_s = clip_start_s + window_size
        duration_seconds = max(0.0, float(clip_end_s) - float(clip_start_s))
        adaptive = False

        estimated_window_count = self._estimate_window_count(
            duration_seconds=duration_seconds,
            window_size_seconds=window_size,
            overlap_seconds=overlap_seconds,
        )
        if modality == "video" and estimated_window_count > self._MAX_VIDEO_SEMANTIC_WINDOWS:
            adaptive = True
            overlap_seconds = min(overlap_seconds, 1.0)
            estimated_window_count = self._estimate_window_count(
                duration_seconds=duration_seconds,
                window_size_seconds=window_size,
                overlap_seconds=overlap_seconds,
            )
            while (
                estimated_window_count > self._MAX_VIDEO_SEMANTIC_WINDOWS
                and window_size < self._MAX_VIDEO_WINDOW_SECONDS
            ):
                window_size += 1.0
                estimated_window_count = self._estimate_window_count(
                    duration_seconds=duration_seconds,
                    window_size_seconds=window_size,
                    overlap_seconds=overlap_seconds,
                )

        return {
            "window_size_seconds": float(window_size),
            "overlap_seconds": float(overlap_seconds),
            "step_seconds": max(float(window_size) - float(overlap_seconds), 0.1),
            "duration_seconds": duration_seconds,
            "estimated_window_count": int(estimated_window_count),
            "adaptive": adaptive,
        }

    @staticmethod
    def _estimate_window_count(
        *,
        duration_seconds: float,
        window_size_seconds: float,
        overlap_seconds: float,
    ) -> int:
        if duration_seconds <= 0.0:
            return 1
        step_seconds = max(float(window_size_seconds) - float(overlap_seconds), 0.1)
        cursor = 0.0
        count = 0
        while cursor < duration_seconds:
            count += 1
            end = min(cursor + float(window_size_seconds), duration_seconds)
            if end >= duration_seconds:
                break
            cursor += step_seconds
        return max(1, count)
