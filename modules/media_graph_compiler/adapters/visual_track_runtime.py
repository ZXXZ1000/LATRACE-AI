from __future__ import annotations

from typing import Any, Dict, List, Sequence

from modules.media_graph_compiler.adapters.visual_tracking_session import (
    VisualTrackingSessionRequestBuilder,
)


class VisualTrackRuntime:
    """Thin runtime wrapper around a predictor that follows the reference session API."""

    def __init__(self, predictor: Any) -> None:
        self._predictor = predictor
        self._requests = VisualTrackingSessionRequestBuilder()

    def run_text_prompt_tracking(
        self,
        *,
        resource_path: str,
        prompt_text: str = "person",
        prompt_frame_index: int = 0,
        track_id: str = "face_1",
        frame_timestamps_s: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        result = self._requests.run_text_tracking(
            predictor=self._predictor,
            resource_path=resource_path,
            prompt_text=prompt_text,
            prompt_frame_index=prompt_frame_index,
        )
        return self._to_stage_output(
            track_id=track_id,
            outputs=result.get("outputs") or [],
            frame_timestamps_s=frame_timestamps_s or [],
            runtime_name="visual_tracking_text_prompt",
        )

    def run_box_prompt_tracking(
        self,
        *,
        resource_path: str,
        boxes_xywh: Sequence[Sequence[float]],
        prompt_frame_index: int = 0,
        track_id: str = "face_1",
        labels: Sequence[int] | None = None,
        frame_timestamps_s: Sequence[float] | None = None,
        include_masks: bool = False,
    ) -> Dict[str, Any]:
        result = self._requests.run_box_tracking(
            predictor=self._predictor,
            resource_path=resource_path,
            boxes_xywh=boxes_xywh,
            prompt_frame_index=prompt_frame_index,
            labels=labels,
        )
        return self._to_stage_output(
            track_id=track_id,
            outputs=result.get("outputs") or [],
            frame_timestamps_s=frame_timestamps_s or [],
            runtime_name="visual_tracking_box_prompt",
            include_masks=include_masks,
        )

    def _to_stage_output(
        self,
        *,
        track_id: str,
        outputs: Sequence[Dict[str, Any]],
        frame_timestamps_s: Sequence[float],
        runtime_name: str,
        include_masks: bool = False,
    ) -> Dict[str, Any]:
        evidence: List[Dict[str, Any]] = []
        frame_indices: List[int] = []
        for item in outputs:
            frame_index = int(item.get("frame_index", 0))
            frame_indices.append(frame_index)
            payload = item.get("outputs") or {}
            evidence_id = f"{track_id}_frame_{frame_index:06d}"
            timestamp = self._frame_timestamp(frame_index, frame_timestamps_s)
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "kind": "mask",
                    "t_start_s": timestamp,
                    "t_end_s": timestamp,
                    "metadata": {
                        "track_id": track_id,
                        "runtime": runtime_name,
                        "frame_index": frame_index,
                        "out_obj_ids": payload.get("out_obj_ids"),
                        "out_boxes_xywh": payload.get("out_boxes_xywh"),
                        "out_binary_masks": payload.get("out_binary_masks") if include_masks else None,
                    },
                }
            )
        if not frame_indices:
            return {"visual_tracks": [], "evidence": []}
        return {
            "visual_tracks": [
                {
                    "track_id": track_id,
                    "category": "person",
                    "t_start_s": self._frame_timestamp(min(frame_indices), frame_timestamps_s),
                    "t_end_s": self._frame_timestamp(max(frame_indices), frame_timestamps_s),
                    "frame_start": min(frame_indices),
                    "frame_end": max(frame_indices),
                    "evidence_refs": [item["evidence_id"] for item in evidence],
                    "metadata": {"runtime": runtime_name},
                }
            ],
            "evidence": evidence,
        }

    @staticmethod
    def _frame_timestamp(frame_index: int, frame_timestamps_s: Sequence[float]) -> float:
        if 0 <= frame_index < len(frame_timestamps_s):
            return float(frame_timestamps_s[frame_index])
        return float(frame_index)


__all__ = ["VisualTrackRuntime"]
