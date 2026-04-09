from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class MediaSegmentRecord:
    """Normalized media segment derived from probe + scene detection."""

    id: str
    index: int
    source_id: str
    t_media_start: float
    t_media_end: float
    duration_seconds: float
    modality: str
    frame_rate: float
    has_physical_time: bool = False
    recorded_at: Optional[datetime] = None


@dataclass
class BackboneEdge:
    src_id: str
    dst_id: str
    rel_type: str = "NEXT_SEGMENT"


@dataclass
class MediaBackbone:
    segments: List[MediaSegmentRecord]
    edges: List[BackboneEdge]
    frame_rate: float


class MediaProbeAdapter:
    """Builds a segment backbone from probe metadata and scheduled windows."""

    def build_backbone(
        self,
        source_id: str,
        probe_meta: dict,
        scenes: Sequence[dict] | Iterable[dict],
        default_modality: str = "video",
    ) -> MediaBackbone:
        frame_rate = self._coerce_frame_rate(probe_meta.get("frame_rate"))
        normalized = self._normalize_scenes(scenes, default_modality)
        segments = self._build_segments(source_id, normalized, frame_rate)
        edges = self._build_edges(segments)
        return MediaBackbone(segments=segments, edges=edges, frame_rate=frame_rate)

    @staticmethod
    def _coerce_frame_rate(value: object) -> float:
        try:
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _normalize_scenes(
        self,
        scenes: Sequence[dict] | Iterable[dict],
        default_modality: str,
    ) -> List[tuple[float, float, str]]:
        normalized: List[tuple[float, float, str]] = []
        seen_keys: set[tuple[float, float, str]] = set()
        for scene in scenes:
            start = self._coerce_time(scene.get("start"))
            end = self._coerce_time(scene.get("end"))
            modality = str(scene.get("modality") or default_modality)
            if start is None or end is None or end <= start:
                continue
            key = (round(start, 3), round(end, 3), modality)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            normalized.append((start, end, modality))
        normalized.sort(key=lambda item: item[0])
        return normalized

    @staticmethod
    def _coerce_time(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            parts = value.split(":")
            try:
                if len(parts) == 3:
                    hours, minutes, seconds = (float(p) for p in parts)
                    return hours * 3600.0 + minutes * 60.0 + seconds
                if len(parts) == 2:
                    minutes, seconds = (float(p) for p in parts)
                    return minutes * 60.0 + seconds
                return float(value)
            except Exception:
                return None
        return None

    def _build_segments(
        self,
        source_id: str,
        normalized: Sequence[tuple[float, float, str]],
        frame_rate: float,
    ) -> List[MediaSegmentRecord]:
        segments: List[MediaSegmentRecord] = []
        for index, (start, end, modality) in enumerate(normalized):
            seg_id = self._stable_segment_id(source_id, start, end, modality)
            segments.append(
                MediaSegmentRecord(
                    id=seg_id,
                    index=index,
                    source_id=source_id,
                    t_media_start=start,
                    t_media_end=end,
                    duration_seconds=end - start,
                    modality=modality,
                    frame_rate=frame_rate,
                )
            )
        return segments

    @staticmethod
    def _build_edges(segments: Sequence[MediaSegmentRecord]) -> List[BackboneEdge]:
        edges: List[BackboneEdge] = []
        for idx in range(len(segments) - 1):
            edges.append(BackboneEdge(src_id=segments[idx].id, dst_id=segments[idx + 1].id))
        return edges

    @staticmethod
    def _stable_segment_id(source_id: str, start: float, end: float, modality: str) -> str:
        raw = f"{source_id}|{start:.3f}|{end:.3f}|{modality}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"segment_{digest}"
