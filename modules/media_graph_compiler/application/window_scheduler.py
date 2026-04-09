from __future__ import annotations

from typing import Dict, List

from modules.media_graph_compiler.types import WindowingPolicy


class WindowScheduler:
    """Builds deterministic rolling windows for video or audio inputs."""

    def build_windows(
        self,
        *,
        modality: str,
        clip_start_s: float,
        clip_end_s: float,
        policy: WindowingPolicy,
    ) -> List[Dict[str, float | str]]:
        window_size = (
            policy.video_window_seconds if modality == "video" else policy.audio_window_seconds
        )
        if clip_end_s <= clip_start_s:
            clip_end_s = clip_start_s + window_size

        step = max(window_size - policy.overlap_seconds, 0.1)
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
