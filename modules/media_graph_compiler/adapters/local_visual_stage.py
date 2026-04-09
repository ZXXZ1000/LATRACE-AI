from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from modules.media_graph_compiler.adapters.frame_selector import FrameSelector
from modules.media_graph_compiler.adapters.ops.face_processing import process_faces
from modules.media_graph_compiler.adapters.visual_continuity_pipeline import (
    SeedDetection,
    VisualContinuityPipeline,
)
from modules.media_graph_compiler.domain.stable_ids import stable_hash_id


class LocalVisualStage:
    """Default visual stage with a single seed-track-stitch mainline."""

    def __init__(self, *, continuity: VisualContinuityPipeline | None = None) -> None:
        self._continuity = continuity or VisualContinuityPipeline()
        self._frame_selector = FrameSelector()

    def run(self, ctx: Mapping[str, Any]) -> Dict[str, Any]:
        face_frame_paths = list(ctx.get("face_frame_paths") or [])
        frame_timestamps_s = list(ctx.get("frame_timestamps_s") or [])
        backbone_segments = list(ctx.get("backbone_segments") or [])
        source_id = str(ctx.get("source_id") or "unknown")
        artifacts_dir = str(
            (ctx.get("request").metadata.get("artifacts_dir") if ctx.get("request") else None)
            or ".artifacts/media_graph_compiler"
        )
        if not face_frame_paths:
            return {"visual_tracks": [], "evidence": []}

        visual_plan = dict((ctx.get("optimization_plan") or {}).get("visual") or {})
        system_plan = dict((ctx.get("optimization_plan") or {}).get("system") or {})
        detection_bundle = self._select_detection_frames(
            face_frame_paths=face_frame_paths,
            frame_timestamps_s=frame_timestamps_s,
            max_detection_frames=int(system_plan.get("max_detection_frames_per_source") or 0),
        )
        detection_frame_paths = detection_bundle["paths"]
        detection_frame_timestamps_s = detection_bundle["timestamps"]
        detection_index_map = detection_bundle["index_map"]
        if not detection_frame_paths:
            return {"visual_tracks": [], "evidence": []}

        cache_path = self._build_cache_path(
            artifacts_dir=artifacts_dir,
            source_id=source_id,
            detection_index_map=detection_index_map,
        )
        segments = self._segments_with_frame_bounds(
            backbone_segments,
            frame_timestamps_s=detection_frame_timestamps_s,
        )
        id2faces = process_faces(
            None,
            detection_frame_paths,
            cache_path,
            preprocessing=[],
            segments=segments,
            stride=1,
            max_frames_per_segment=int(visual_plan.get("max_frames_per_window") or 0),
        ) or {}
        if not id2faces:
            return {"visual_tracks": [], "evidence": []}
        id2faces = self._remap_detection_faces(
            detections_by_track=id2faces,
            detection_index_map=detection_index_map,
        )

        seed_tracks, seed_evidence = self._build_seed_tracks_and_evidence(
            source_id=source_id,
            artifacts_dir=artifacts_dir,
            detections_by_track=id2faces,
            frame_timestamps_s=frame_timestamps_s,
            backbone_segments=backbone_segments,
        )
        return self._continuity.run(
            ctx=ctx,
            seed_tracks=seed_tracks,
            seed_evidence=seed_evidence,
        )

    def _select_detection_frames(
        self,
        *,
        face_frame_paths: Sequence[str],
        frame_timestamps_s: Sequence[float],
        max_detection_frames: int,
    ) -> Dict[str, Any]:
        count = len(face_frame_paths)
        if count <= 0:
            return {"paths": [], "timestamps": [], "index_map": []}
        if max_detection_frames <= 0 or count <= max_detection_frames:
            return {
                "paths": list(face_frame_paths),
                "timestamps": list(frame_timestamps_s[:count]),
                "index_map": list(range(count)),
            }

        kept_indices = self._frame_selector.sample_indices(count, max_detection_frames)
        return {
            "paths": [face_frame_paths[index] for index in kept_indices],
            "timestamps": [
                float(frame_timestamps_s[index]) if index < len(frame_timestamps_s) else 0.0
                for index in kept_indices
            ],
            "index_map": kept_indices,
        }

    @staticmethod
    def _build_cache_path(
        *,
        artifacts_dir: str,
        source_id: str,
        detection_index_map: Sequence[int],
    ) -> str:
        raw = ",".join(str(index) for index in detection_index_map) or "full"
        suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return str(Path(artifacts_dir) / "cache" / f"{source_id}_faces_{suffix}.json")

    @staticmethod
    def _remap_detection_faces(
        *,
        detections_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
        detection_index_map: Sequence[int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        remapped: Dict[str, List[Dict[str, Any]]] = {}
        for track_id, detections in detections_by_track.items():
            for detection in detections:
                detection_dict = dict(detection)
                frame_id = int(detection_dict.get("frame_id", -1))
                if 0 <= frame_id < len(detection_index_map):
                    detection_dict["frame_id"] = int(detection_index_map[frame_id])
                remapped.setdefault(str(track_id), []).append(detection_dict)
        return remapped

    @staticmethod
    def _frame_timestamp(frame_index: int, frame_timestamps_s: Sequence[float]) -> float:
        if 0 <= frame_index < len(frame_timestamps_s):
            return float(frame_timestamps_s[frame_index])
        return 0.0

    @staticmethod
    def _segment_id_for_timestamp(segments: Sequence[Any], timestamp: float) -> str | None:
        for segment in segments:
            if segment.t_media_start <= timestamp <= segment.t_media_end:
                return segment.id
        return None

    def _segments_with_frame_bounds(
        self,
        segments: Sequence[Any],
        *,
        frame_timestamps_s: Sequence[float],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for segment in segments:
            start_index = None
            end_index = None
            for index, timestamp in enumerate(frame_timestamps_s):
                if segment.t_media_start <= timestamp <= segment.t_media_end:
                    if start_index is None:
                        start_index = index
                    end_index = index
            if start_index is None or end_index is None:
                continue
            items.append(
                {
                    "index": segment.index,
                    "segment_id": segment.id,
                    "frame_start": start_index,
                    "frame_end": end_index,
                }
            )
        return items

    def _build_seed_tracks_and_evidence(
        self,
        *,
        source_id: str,
        artifacts_dir: str,
        detections_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
        frame_timestamps_s: Sequence[float],
        backbone_segments: Sequence[Any],
    ) -> tuple[Dict[str, List[SeedDetection]], List[Dict[str, Any]]]:
        seed_tracks: Dict[str, List[SeedDetection]] = {}
        evidence: List[Dict[str, Any]] = []
        for track_id, detections in detections_by_track.items():
            for det in detections:
                frame_index = int(det.get("frame_id", -1))
                timestamp = self._frame_timestamp(frame_index, frame_timestamps_s)
                segment_id = self._segment_id_for_timestamp(backbone_segments, timestamp)
                bbox = list(det.get("bounding_box") or [])
                extra_data = dict(det.get("extra_data") or {})
                evidence_id = stable_hash_id(
                    "evface",
                    source_id,
                    track_id,
                    frame_index,
                    segment_id,
                    bbox,
                )
                image_ref = self._write_face_crop(
                    artifacts_dir=artifacts_dir,
                    evidence_id=evidence_id,
                    face_base64=extra_data.get("face_base64"),
                )
                evidence.append(
                    {
                        "evidence_id": evidence_id,
                        "kind": "frame_crop",
                        "file_path": image_ref,
                        "t_start_s": timestamp,
                        "t_end_s": timestamp,
                        "metadata": {
                            "algorithm": "seed_face_detection",
                            "algorithm_version": "v2",
                            "segment_id": segment_id,
                            "bbox": bbox,
                            "track_id": track_id,
                            "confidence": float(extra_data.get("face_detection_score") or 0.0),
                            "quality_score": float(extra_data.get("face_quality_score") or 0.0),
                        },
                    }
                )
                seed_tracks.setdefault(str(track_id), []).append(
                    SeedDetection(
                        seed_track_id=str(track_id),
                        frame_index=frame_index,
                        timestamp_s=timestamp,
                        bbox_xyxy=[float(item) for item in bbox],
                        embedding=[float(item) for item in (det.get("face_emb") or [])],
                        evidence_id=evidence_id,
                        segment_id=segment_id,
                        detection_confidence=float(extra_data.get("face_detection_score") or 0.0),
                        quality_score=float(extra_data.get("face_quality_score") or 0.0),
                    )
                )
        return seed_tracks, evidence

    @staticmethod
    def _write_face_crop(
        *,
        artifacts_dir: str,
        evidence_id: str,
        face_base64: str | None,
    ) -> str | None:
        if not face_base64:
            return None
        try:
            raw = str(face_base64)
            if raw.startswith("data:") and "base64," in raw:
                raw = raw.split("base64,", 1)[1]
            out_dir = Path(artifacts_dir) / "evidence_media" / "faces"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{evidence_id}.jpg"
            out_path.write_bytes(base64.b64decode(raw))
            return str(out_path)
        except Exception:
            return None


__all__ = ["LocalVisualStage"]
