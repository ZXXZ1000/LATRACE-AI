from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from modules.media_graph_compiler.adapters.visual_track_runtime import VisualTrackRuntime
from modules.media_graph_compiler.domain.stable_ids import stable_hash_id


@dataclass(frozen=True)
class SeedDetection:
    seed_track_id: str
    frame_index: int
    timestamp_s: float
    bbox_xyxy: List[float]
    embedding: List[float]
    evidence_id: str
    segment_id: str | None = None
    detection_confidence: float = 0.0
    quality_score: float = 0.0


@dataclass
class Tracklet:
    seed_track_id: str
    tracklet_id: str
    t_start_s: float
    t_end_s: float
    frame_start: int | None
    frame_end: int | None
    evidence_refs: List[str] = field(default_factory=list)
    detections: List[SeedDetection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _bbox_xywh(bbox_xyxy: Sequence[float]) -> List[float] | None:
    if len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return None
    return [x1, y1, width, height]


def _mean_embedding(detections: Sequence[SeedDetection]) -> List[float]:
    if not detections:
        return []
    dim = max((len(item.embedding) for item in detections), default=0)
    if dim <= 0:
        return []
    sums = [0.0 for _ in range(dim)]
    count = 0
    for item in detections:
        if len(item.embedding) != dim:
            continue
        for index, value in enumerate(item.embedding):
            sums[index] += float(value)
        count += 1
    if count <= 0:
        return []
    return [value / float(count) for value in sums]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for value_a, value_b in zip(vec_a, vec_b):
        a = float(value_a)
        b = float(value_b)
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class VisualContinuityPipeline:
    """Track-first visual continuity on top of local seed detections.

    The single visual mainline is:
    seed detections -> tracking tracklets -> stitch -> final visual tracks.

    When a tracking predictor is not available, we keep the same contract by
    falling back to seed-anchored sparse tracklets instead of switching to a
    different external route.
    """

    def run(
        self,
        *,
        ctx: Mapping[str, Any],
        seed_tracks: Mapping[str, Sequence[SeedDetection]],
        seed_evidence: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        request = ctx.get("request")
        tracking_enabled = bool(request.visual_policy.enable_tracking) if request is not None else True
        tracking_predictor = self._resolve_tracking_predictor(ctx)
        resource_path = str(
            ctx.get("normalized_video_path")
            or (request.source.file_path if request is not None and request.source.file_path else "")
        ).strip()
        frame_timestamps_s = list(ctx.get("frame_timestamps_s") or [])

        tracklets: List[Tracklet] = []
        extra_evidence: List[Dict[str, Any]] = []
        runtime_success = 0
        runtime_failures = 0
        runtime_errors: List[str] = []
        runtime_used = False

        for seed_track_id, detections in seed_tracks.items():
            seed_list = sorted(detections, key=lambda item: (item.frame_index, item.timestamp_s))
            if not seed_list:
                continue
            runtime_tracklet: Tracklet | None = None
            if tracking_enabled and tracking_predictor is not None and resource_path:
                runtime_used = True
                try:
                    runtime_output = self._run_tracking_runtime(
                        predictor=tracking_predictor,
                        resource_path=resource_path,
                        seed_track_id=seed_track_id,
                        detections=seed_list,
                        frame_timestamps_s=frame_timestamps_s,
                        include_masks=bool(request.visual_policy.include_masks) if request is not None else False,
                    )
                    runtime_tracklet = runtime_output["tracklet"]
                    extra_evidence.extend(runtime_output["evidence"])
                    runtime_success += 1
                except Exception as exc:
                    runtime_failures += 1
                    runtime_errors.append(f"{seed_track_id}:{type(exc).__name__}:{exc}")

            tracklets.append(runtime_tracklet or self._build_seed_tracklet(seed_track_id=seed_track_id, detections=seed_list))

        stitched_tracks = self._stitch_tracklets(tracklets)
        evidence = self._merge_evidence(seed_evidence, extra_evidence)
        visual_tracks = self._to_visual_tracks(stitched_tracks, request=request, runtime_used=runtime_used)

        tracking_mode = "session_box_prompt" if runtime_success > 0 else "seed_tracklets"
        if runtime_failures > 0 and runtime_success > 0:
            tracking_mode = "session_box_prompt_with_seed_fallback"
        elif runtime_failures > 0 and runtime_success == 0:
            tracking_mode = "seed_tracklets"

        return {
            "visual_tracks": visual_tracks,
            "evidence": evidence,
            "visual_stats": {
                "seed_track_count": len(seed_tracks),
                "tracklet_count": len(tracklets),
                "stitched_track_count": len(stitched_tracks),
                "tracking_enabled": tracking_enabled,
                "tracking_runtime_used": runtime_used,
                "tracking_runtime_success_count": runtime_success,
                "tracking_runtime_failure_count": runtime_failures,
                "tracking_runtime_errors": runtime_errors,
                "tracking_mode": tracking_mode,
                "stitch_strategy": "seed_anchor_merge",
            },
        }

    @staticmethod
    def _resolve_tracking_predictor(ctx: Mapping[str, Any]) -> Any | None:
        predictor = ctx.get("visual_tracking_predictor")
        if predictor is not None:
            return predictor
        request = ctx.get("request")
        if request is not None:
            predictor = request.metadata.get("visual_tracking_predictor")
            if predictor is not None:
                return predictor
        return None

    def _run_tracking_runtime(
        self,
        *,
        predictor: Any,
        resource_path: str,
        seed_track_id: str,
        detections: Sequence[SeedDetection],
        frame_timestamps_s: Sequence[float],
        include_masks: bool,
    ) -> Dict[str, Any]:
        prompt = self._select_prompt_detection(detections)
        bbox_xywh = _bbox_xywh(prompt.bbox_xyxy)
        if bbox_xywh is None:
            raise RuntimeError("invalid prompt bbox")
        runtime = VisualTrackRuntime(predictor)
        payload = runtime.run_box_prompt_tracking(
            resource_path=resource_path,
            boxes_xywh=[bbox_xywh],
            prompt_frame_index=prompt.frame_index,
            track_id=seed_track_id,
            frame_timestamps_s=frame_timestamps_s,
            include_masks=include_masks,
        )
        tracks = list(payload.get("visual_tracks") or [])
        if not tracks:
            raise RuntimeError("tracking runtime returned no track")
        evidence = [dict(item) for item in (payload.get("evidence") or [])]
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            metadata.setdefault("track_id", seed_track_id)
            metadata.setdefault("seed_track_id", seed_track_id)
            item["metadata"] = metadata
        track = dict(tracks[0])
        tracklet = Tracklet(
            seed_track_id=seed_track_id,
            tracklet_id=str(track.get("track_id") or seed_track_id),
            t_start_s=float(track.get("t_start_s") or prompt.timestamp_s),
            t_end_s=float(track.get("t_end_s") or prompt.timestamp_s),
            frame_start=track.get("frame_start"),
            frame_end=track.get("frame_end"),
            evidence_refs=list(track.get("evidence_refs") or []),
            detections=list(detections),
            metadata={
                "tracking_mode": "session_box_prompt",
                "prompt_frame_index": prompt.frame_index,
                "prompt_bbox_xyxy": list(prompt.bbox_xyxy),
                "seed_detection_count": len(detections),
            },
        )
        return {"tracklet": tracklet, "evidence": evidence}

    @staticmethod
    def _select_prompt_detection(detections: Sequence[SeedDetection]) -> SeedDetection:
        return max(
            detections,
            key=lambda item: (item.detection_confidence, item.quality_score, -item.frame_index),
        )

    @staticmethod
    def _build_seed_tracklet(*, seed_track_id: str, detections: Sequence[SeedDetection]) -> Tracklet:
        timestamps = [item.timestamp_s for item in detections]
        frames = [item.frame_index for item in detections]
        return Tracklet(
            seed_track_id=seed_track_id,
            tracklet_id=stable_hash_id("seedtracklet", seed_track_id, frames[0] if frames else 0, frames[-1] if frames else 0),
            t_start_s=min(timestamps) if timestamps else 0.0,
            t_end_s=max(timestamps) if timestamps else 0.0,
            frame_start=min(frames) if frames else None,
            frame_end=max(frames) if frames else None,
            evidence_refs=[item.evidence_id for item in detections],
            detections=list(detections),
            metadata={
                "tracking_mode": "seed_tracklets",
                "seed_detection_count": len(detections),
            },
        )

    def _stitch_tracklets(self, tracklets: Sequence[Tracklet]) -> List[Tracklet]:
        grouped: Dict[str, List[Tracklet]] = {}
        for item in tracklets:
            grouped.setdefault(item.seed_track_id, []).append(item)

        stitched: List[Tracklet] = []
        for seed_track_id, items in grouped.items():
            ordered = sorted(items, key=lambda item: (item.t_start_s, item.frame_start or -1))
            current: Tracklet | None = None
            for item in ordered:
                if current is None:
                    current = item
                    continue
                if self._can_merge(current, item):
                    current = self._merge_tracklets(seed_track_id=seed_track_id, left=current, right=item)
                else:
                    stitched.append(current)
                    current = item
            if current is not None:
                stitched.append(current)
        return stitched

    def _can_merge(self, left: Tracklet, right: Tracklet) -> bool:
        if left.seed_track_id != right.seed_track_id:
            return False
        left_emb = _mean_embedding(left.detections)
        right_emb = _mean_embedding(right.detections)
        similarity = _cosine_similarity(left_emb, right_emb)
        temporal_gap = max(0.0, float(right.t_start_s) - float(left.t_end_s))
        return similarity >= 0.75 or temporal_gap <= 2.0

    @staticmethod
    def _merge_tracklets(*, seed_track_id: str, left: Tracklet, right: Tracklet) -> Tracklet:
        metadata = dict(left.metadata)
        metadata.update(
            {
                "stitch_strategy": "seed_anchor_merge",
                "merged_tracklet_count": int(left.metadata.get("merged_tracklet_count") or 1) + int(right.metadata.get("merged_tracklet_count") or 1),
            }
        )
        return Tracklet(
            seed_track_id=seed_track_id,
            tracklet_id=seed_track_id,
            t_start_s=min(left.t_start_s, right.t_start_s),
            t_end_s=max(left.t_end_s, right.t_end_s),
            frame_start=min(item for item in [left.frame_start, right.frame_start] if item is not None) if (left.frame_start is not None or right.frame_start is not None) else None,
            frame_end=max(item for item in [left.frame_end, right.frame_end] if item is not None) if (left.frame_end is not None or right.frame_end is not None) else None,
            evidence_refs=list(dict.fromkeys([*left.evidence_refs, *right.evidence_refs])),
            detections=[*left.detections, *right.detections],
            metadata=metadata,
        )

    @staticmethod
    def _merge_evidence(
        seed_evidence: Sequence[Mapping[str, Any]],
        extra_evidence: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in [*seed_evidence, *extra_evidence]:
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            payload = dict(item)
            payload["metadata"] = dict(item.get("metadata") or {})
            merged[evidence_id] = payload
        return list(merged.values())

    @staticmethod
    def _to_visual_tracks(
        tracklets: Sequence[Tracklet],
        *,
        request: Any | None,
        runtime_used: bool,
    ) -> List[Dict[str, Any]]:
        visual_tracks: List[Dict[str, Any]] = []
        for item in tracklets:
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "runtime": "track_first_visual_continuity",
                    "tracking_runtime_used": runtime_used,
                    "evidence_count": len(item.evidence_refs),
                    "seed_detection_count": len(item.detections),
                }
            )
            if request is not None:
                metadata["include_masks"] = bool(request.visual_policy.include_masks)
                metadata["include_face_crops"] = bool(request.visual_policy.include_face_crops)
            visual_tracks.append(
                {
                    "track_id": item.seed_track_id,
                    "category": "person",
                    "t_start_s": item.t_start_s,
                    "t_end_s": item.t_end_s,
                    "frame_start": item.frame_start,
                    "frame_end": item.frame_end,
                    "evidence_refs": list(item.evidence_refs),
                    "metadata": metadata,
                }
            )
        return visual_tracks


__all__ = ["SeedDetection", "VisualContinuityPipeline"]
