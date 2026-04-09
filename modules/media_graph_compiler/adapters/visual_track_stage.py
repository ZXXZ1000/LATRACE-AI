from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from modules.media_graph_compiler.types import EvidencePointer, VisualTrackRecord


class VisualTrackStageAdapter:
    """Normalizes visual stage outputs into visual track and evidence contracts."""

    def normalize(
        self,
        *,
        source_id: str,
        stage_output: Mapping[str, Any],
    ) -> Tuple[List[VisualTrackRecord], List[EvidencePointer]]:
        if "visual_tracks" in stage_output:
            tracks = [VisualTrackRecord.model_validate(item) for item in stage_output.get("visual_tracks", [])]
            evidence = [EvidencePointer.model_validate(item) for item in stage_output.get("evidence", [])]
            return tracks, evidence
        if "clusters" in stage_output or "detections" in stage_output:
            return self._normalize_legacy_shape(stage_output)
        return [], []

    def _normalize_legacy_shape(
        self,
        stage_output: Mapping[str, Any],
    ) -> Tuple[List[VisualTrackRecord], List[EvidencePointer]]:
        tracks: Dict[str, VisualTrackRecord] = {}
        evidence: List[EvidencePointer] = []

        for cluster in stage_output.get("clusters", []) or []:
            track_id = str(cluster.get("id") or "face_unknown")
            tracks[track_id] = VisualTrackRecord(
                track_id=track_id,
                category="person",
                t_start_s=0.0,
                t_end_s=0.0,
                evidence_refs=list(cluster.get("evidence_ids") or []),
                metadata={"legacy_cluster": True},
            )

        face_detections = ((stage_output.get("detections") or {}).get("face") or [])
        for det in face_detections:
            evidence_id = str(det.get("id"))
            timestamp = det.get("timestamp")
            evidence.append(
                EvidencePointer(
                    evidence_id=evidence_id,
                    kind="frame_crop",
                    t_start_s=float(timestamp) if timestamp is not None else None,
                    t_end_s=float(timestamp) if timestamp is not None else None,
                    metadata={
                        "segment_id": det.get("segment_id"),
                        "bbox": det.get("bbox"),
                        "model": det.get("model"),
                    },
                )
            )
            extras = det.get("extras") or {}
            track_id = str(extras.get("cluster_id") or f"face_{evidence_id}")
            if track_id not in tracks:
                tracks[track_id] = VisualTrackRecord(
                    track_id=track_id,
                    category="person",
                    t_start_s=float(timestamp) if timestamp is not None else 0.0,
                    t_end_s=float(timestamp) if timestamp is not None else 0.0,
                    evidence_refs=[],
                    metadata={"legacy_face_detection": True},
                )
            current = tracks[track_id]
            new_start = current.t_start_s if current.evidence_refs else (float(timestamp) if timestamp is not None else current.t_start_s)
            new_end = current.t_end_s
            if timestamp is not None:
                ts = float(timestamp)
                new_start = min(current.t_start_s, ts) if current.evidence_refs else ts
                new_end = max(current.t_end_s, ts)
            tracks[track_id] = current.model_copy(
                update={
                    "t_start_s": new_start,
                    "t_end_s": new_end,
                    "evidence_refs": [*current.evidence_refs, evidence_id],
                }
            )

        object_detections = ((stage_output.get("detections") or {}).get("object") or [])
        for det in object_detections:
            evidence_id = str(det.get("id"))
            timestamp = det.get("timestamp")
            evidence.append(
                EvidencePointer(
                    evidence_id=evidence_id,
                    kind="frame_crop",
                    t_start_s=float(timestamp) if timestamp is not None else None,
                    t_end_s=float(timestamp) if timestamp is not None else None,
                    metadata={
                        "segment_id": det.get("segment_id"),
                        "label": det.get("label"),
                        "bbox": det.get("bbox"),
                        "model": det.get("model"),
                    },
                )
            )
        return list(tracks.values()), evidence
