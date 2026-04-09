from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence


class VisualTrackingSessionRequestBuilder:
    """Borrow the request/session protocol shape from SAM3's predictor API."""

    def build_start_session(
        self,
        *,
        resource_path: str,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "type": "start_session",
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if session_id is not None:
            request["session_id"] = session_id
        return request

    def build_add_text_prompt(
        self,
        *,
        session_id: str,
        frame_index: int,
        text: str,
        obj_id: int | None = None,
        output_prob_thresh: float | None = None,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": text,
        }
        if obj_id is not None:
            request["obj_id"] = obj_id
        if output_prob_thresh is not None:
            request["output_prob_thresh"] = output_prob_thresh
        return request

    def build_add_box_prompt(
        self,
        *,
        session_id: str,
        frame_index: int,
        boxes_xywh: Sequence[Sequence[float]],
        labels: Sequence[int] | None = None,
        obj_id: int | None = None,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "bounding_boxes": [list(item) for item in boxes_xywh],
        }
        if labels is not None:
            request["bounding_box_labels"] = list(labels)
        if obj_id is not None:
            request["obj_id"] = obj_id
        return request

    def build_propagate_request(
        self,
        *,
        session_id: str,
        propagation_direction: str = "both",
        start_frame_index: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
        }
        if start_frame_index is not None:
            request["start_frame_index"] = start_frame_index
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = max_frame_num_to_track
        if output_prob_thresh is not None:
            request["output_prob_thresh"] = output_prob_thresh
        return request

    def run_text_tracking(
        self,
        *,
        predictor: Any,
        resource_path: str,
        prompt_text: str,
        prompt_frame_index: int = 0,
        session_id: str | None = None,
        propagation_direction: str = "both",
    ) -> Dict[str, Any]:
        start_response = predictor.handle_request(
            self.build_start_session(
                resource_path=resource_path,
                session_id=session_id,
            )
        )
        current_session_id = str(start_response["session_id"])
        prompt_response = predictor.handle_request(
            self.build_add_text_prompt(
                session_id=current_session_id,
                frame_index=prompt_frame_index,
                text=prompt_text,
            )
        )
        outputs = list(
            predictor.handle_stream_request(
                self.build_propagate_request(
                    session_id=current_session_id,
                    propagation_direction=propagation_direction,
                )
            )
        )
        return {
            "session_id": current_session_id,
            "prompt_response": prompt_response,
            "outputs": outputs,
        }

    def run_box_tracking(
        self,
        *,
        predictor: Any,
        resource_path: str,
        boxes_xywh: Sequence[Sequence[float]],
        prompt_frame_index: int = 0,
        labels: Sequence[int] | None = None,
        obj_id: int | None = None,
        session_id: str | None = None,
        propagation_direction: str = "both",
    ) -> Dict[str, Any]:
        start_response = predictor.handle_request(
            self.build_start_session(
                resource_path=resource_path,
                session_id=session_id,
            )
        )
        current_session_id = str(start_response["session_id"])
        prompt_response = predictor.handle_request(
            self.build_add_box_prompt(
                session_id=current_session_id,
                frame_index=prompt_frame_index,
                boxes_xywh=boxes_xywh,
                labels=labels,
                obj_id=obj_id,
            )
        )
        outputs = list(
            predictor.handle_stream_request(
                self.build_propagate_request(
                    session_id=current_session_id,
                    propagation_direction=propagation_direction,
                )
            )
        )
        return {
            "session_id": current_session_id,
            "prompt_response": prompt_response,
            "outputs": outputs,
        }


__all__ = ["VisualTrackingSessionRequestBuilder"]
