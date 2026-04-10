from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from modules.media_graph_compiler.adapters.ops.audio_segments import (
    slice_audio_b64_segments,
)
from modules.media_graph_compiler.adapters.ops.light_asd_scoring import (
    light_asd_status,
    score_light_asd,
)
from modules.media_graph_compiler.domain.stable_ids import stable_hash_id


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))


def _intersects_window(item_start: float, item_end: float, win_start: float, win_end: float) -> bool:
    start = float(item_start)
    end = float(item_end)
    if end < start:
        end = start
    if abs(end - start) < 1e-6:
        return float(win_start) <= start <= float(win_end)
    return _overlap(start, end, win_start, win_end) > 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class LocalFaceVoiceAssociationStage:
    """Associate speaker tracks and face tracks with local Light-ASD scoring.

    Primary path:
    - Light-ASD local scoring on (audio chunk, face frame sequence)

    Fallback path:
    - temporal overlap + utterance co-occurrence heuristic
    """

    def run(self, ctx: Mapping[str, Any]) -> Dict[str, Any]:
        request = ctx.get("request")
        source_id = str(ctx.get("source_id") or "unknown")
        if request is not None and not bool(request.identity.enable_cross_modal_association):
            return {"face_voice_links": [], "association_stats": {"disabled": True}}

        visual_tracks = list(ctx.get("visual_tracks") or [])
        speaker_tracks = list(ctx.get("speaker_tracks") or [])
        utterances = list(ctx.get("utterances") or [])
        visual_evidence = list(ctx.get("visual_evidence") or [])
        audio_evidence = list(ctx.get("audio_evidence") or [])
        if not visual_tracks or not speaker_tracks:
            return {"face_voice_links": [], "association_stats": {"skipped": "missing_tracks"}}

        visual_frames_by_track = self._collect_visual_frames(visual_evidence)
        audio_chunks_by_track = self._collect_audio_chunks(audio_evidence)

        min_overlap_s = float(os.getenv("MGC_ASSOCIATION_MIN_OVERLAP_S") or 0.4)
        min_conf = float(os.getenv("MGC_ASSOCIATION_MIN_CONF") or 0.25)
        min_asd_primary_score = float(
            os.getenv("MGC_ASSOCIATION_MIN_ASD_PRIMARY_SCORE") or 0.55
        )
        fallback_conf_cap = float(
            os.getenv("MGC_ASSOCIATION_FALLBACK_CONF_CAP") or 0.59
        )
        max_asd_candidates_per_speaker = max(
            1,
            int(os.getenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER") or 2),
        )
        max_asd_windows_per_candidate = max(
            1,
            int(os.getenv("MGC_ASSOCIATION_MAX_ASD_WINDOWS_PER_CANDIDATE") or 3),
        )
        min_face_frames_for_asd = max(
            1,
            int(os.getenv("MGC_ASSOCIATION_MIN_FACE_FRAMES_FOR_ASD") or 4),
        )
        asd_window_pad_s = max(
            0.0,
            float(os.getenv("MGC_ASSOCIATION_ASD_WINDOW_PAD_S") or 0.2),
        )
        raw_face_window_pad_s = max(
            0.0,
            float(os.getenv("MGC_ASSOCIATION_RAW_FACE_WINDOW_PAD_S") or 0.35),
        )
        enable_light_asd = str(os.getenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD") or "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        artifacts_dir = str(
            (request.metadata.get("artifacts_dir") if request else None)
            or ".artifacts/media_graph_compiler"
        )
        raw_face_frame_paths = list(ctx.get("face_frame_paths") or [])
        raw_face_frame_timestamps_s = list(ctx.get("frame_timestamps_s") or [])

        candidates: List[Dict[str, Any]] = []
        asd_ok = 0
        for speaker in speaker_tracks:
            speaker_id = str(speaker.get("track_id") or "")
            speaker_start = float(speaker.get("t_start_s") or 0.0)
            speaker_end = float(speaker.get("t_end_s") or speaker_start)
            speaker_duration = max(0.0, speaker_end - speaker_start)
            speaker_candidates: List[Dict[str, Any]] = []
            for visual in visual_tracks:
                visual_id = str(visual.get("track_id") or "")
                visual_start = float(visual.get("t_start_s") or 0.0)
                visual_end = float(visual.get("t_end_s") or visual_start)
                visual_duration = max(0.0, visual_end - visual_start)
                overlap_s = _overlap(speaker_start, speaker_end, visual_start, visual_end)
                if overlap_s < min_overlap_s:
                    continue

                t_start = max(speaker_start, visual_start)
                t_end = min(speaker_end, visual_end)
                support_utterances = self._utterances_for_window(
                    utterances=utterances,
                    speaker_track_id=speaker_id,
                    t_start=t_start,
                    t_end=t_end,
                )
                overlap_ratio = overlap_s / max(0.1, min(speaker_duration, visual_duration))
                utterance_score = min(1.0, float(len(support_utterances)) / 2.0)
                temporal_score = _clip01(0.75 * overlap_ratio + 0.25 * utterance_score)

                support_evidence_refs = self._support_evidence_refs(
                    visual_frames_by_track.get(visual_id) or [],
                    audio_chunks_by_track.get(speaker_id) or [],
                    t_start=t_start,
                    t_end=t_end,
                )
                speaker_candidates.append(
                    {
                        "speaker_track_id": speaker_id,
                        "visual_track_id": visual_id,
                        "t_start_s": t_start,
                        "t_end_s": t_end,
                        "overlap_s": overlap_s,
                        "support_evidence_refs": support_evidence_refs,
                        "support_utterance_ids": support_utterances,
                        "temporal_score": temporal_score,
                        "asd_norm_score": None,
                        "asd_detail": {},
                    }
                )
            if enable_light_asd and speaker_candidates:
                ranked_candidates = sorted(
                    speaker_candidates,
                    key=lambda item: (
                        float(item.get("temporal_score") or 0.0),
                        float(item.get("overlap_s") or 0.0),
                    ),
                    reverse=True,
                )[:max_asd_candidates_per_speaker]
                for item in ranked_candidates:
                    asd_results = self._score_candidate_asd_windows(
                        utterances=utterances,
                        speaker_track_id=speaker_id,
                        audio_chunks=audio_chunks_by_track.get(speaker_id) or [],
                        face_frames=visual_frames_by_track.get(
                            str(item.get("visual_track_id") or "")
                        )
                        or [],
                        t_start=float(item.get("t_start_s") or 0.0),
                        t_end=float(item.get("t_end_s") or 0.0),
                        max_windows=max_asd_windows_per_candidate,
                        pad_s=asd_window_pad_s,
                        source_id=source_id,
                        visual_track_id=str(item.get("visual_track_id") or ""),
                        artifacts_dir=artifacts_dir,
                        raw_face_frame_paths=raw_face_frame_paths,
                        raw_face_frame_timestamps_s=raw_face_frame_timestamps_s,
                        min_face_frames=min_face_frames_for_asd,
                        raw_face_window_pad_s=raw_face_window_pad_s,
                    )
                    if asd_results:
                        best_asd = max(
                            asd_results,
                            key=lambda row: float(row.get("norm_score") or 0.0),
                        )
                        item["asd_norm_score"] = float(best_asd.get("norm_score") or 0.0)
                        item["asd_detail"] = dict(best_asd)
                        item["asd_window_count"] = len(asd_results)
                        item["asd_scored_windows"] = asd_results
                        asd_ok += len(asd_results)

            for item in speaker_candidates:
                temporal_score = float(item.get("temporal_score") or 0.0)
                asd_norm = item.get("asd_norm_score")
                asd_verified = bool(
                    asd_norm is not None and float(asd_norm) >= min_asd_primary_score
                )
                temporal_confidence = min(
                    fallback_conf_cap,
                    _clip01(0.35 + 0.35 * temporal_score),
                )
                confidence = temporal_confidence
                method = "temporal_overlap_fallback"
                if asd_norm is not None:
                    confidence = _clip01(
                        max(
                            temporal_confidence * 0.92,
                            0.25 + 0.25 * temporal_score + 0.50 * float(asd_norm),
                        )
                    )
                    method = "light_asd_temporal_fusion"
                candidates.append(
                    {
                        **item,
                        "confidence": confidence,
                        "metadata": {
                            "method": method,
                            "temporal_score": temporal_score,
                            "temporal_confidence": temporal_confidence,
                            "asd_norm_score": asd_norm,
                            "asd_verified": asd_verified,
                            "provisional": not asd_verified,
                            "asd_window_count": int(item.get("asd_window_count") or 0),
                            "asd_detail": dict(item.get("asd_detail") or {}),
                        },
                    }
                )

        links = self._select_links(
            source_id=source_id,
            candidates=candidates,
            min_conf=min_conf,
        )
        return {
            "face_voice_links": links,
            "association_stats": {
                "candidate_count": len(candidates),
                "link_count": len(links),
                "min_conf_threshold": min_conf,
                "top_candidates": [
                    {
                        "speaker_track_id": item.get("speaker_track_id"),
                        "visual_track_id": item.get("visual_track_id"),
                        "confidence": item.get("confidence"),
                        "method": (item.get("metadata") or {}).get("method"),
                        "asd_norm_score": (item.get("metadata") or {}).get("asd_norm_score"),
                        "asd_verified": (item.get("metadata") or {}).get("asd_verified"),
                    }
                    for item in sorted(
                        candidates,
                        key=lambda row: (
                            bool(((row.get("metadata") or {}).get("asd_verified"))),
                            ((row.get("metadata") or {}).get("asd_norm_score") is not None),
                            float(
                                ((row.get("metadata") or {}).get("asd_norm_score") or -1.0)
                            ),
                            float(row.get("confidence") or 0.0),
                        ),
                        reverse=True,
                    )[:5]
                ],
                "light_asd_enabled": enable_light_asd,
                "light_asd_success_count": asd_ok,
                "light_asd_status": light_asd_status(),
            },
        }

    @staticmethod
    def _collect_visual_frames(evidence: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            track_id = str(metadata.get("track_id") or "").strip()
            if not track_id:
                continue
            t = float(item.get("t_start_s") or 0.0)
            out.setdefault(track_id, []).append(
                {
                    "t_start_s": t,
                    "t_end_s": float(item.get("t_end_s") or t),
                    "file_path": item.get("file_path"),
                    "evidence_id": item.get("evidence_id"),
                    "bbox": list(metadata.get("bbox") or []),
                }
            )
        for key in out:
            out[key].sort(key=lambda x: float(x.get("t_start_s") or 0.0))
        return out

    @staticmethod
    def _collect_audio_chunks(evidence: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            metadata = dict(item.get("metadata") or {})
            track_id = str(metadata.get("speaker_track_id") or "").strip()
            if not track_id:
                continue
            t_start = float(item.get("t_start_s") or 0.0)
            t_end = float(item.get("t_end_s") or t_start)
            out.setdefault(track_id, []).append(
                {
                    "t_start_s": t_start,
                    "t_end_s": t_end,
                    "file_path": item.get("file_path"),
                    "evidence_id": item.get("evidence_id"),
                }
            )
        for key in out:
            out[key].sort(key=lambda x: float(x.get("t_start_s") or 0.0))
        return out

    @staticmethod
    def _utterances_for_window(
        *,
        utterances: Sequence[Mapping[str, Any]],
        speaker_track_id: str,
        t_start: float,
        t_end: float,
    ) -> List[str]:
        out: List[str] = []
        for item in utterances:
            if str(item.get("speaker_track_id") or "") != speaker_track_id:
                continue
            if _overlap(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            ) <= 0:
                continue
            utterance_id = str(item.get("utterance_id") or "").strip()
            if utterance_id:
                out.append(utterance_id)
        return out

    @staticmethod
    def _utterance_windows_for_candidate(
        *,
        utterances: Sequence[Mapping[str, Any]],
        speaker_track_id: str,
        t_start: float,
        t_end: float,
        max_windows: int,
        pad_s: float,
    ) -> List[Dict[str, Any]]:
        windows: List[Dict[str, Any]] = []
        for item in utterances:
            if str(item.get("speaker_track_id") or "") != speaker_track_id:
                continue
            utt_start = float(item.get("t_start_s") or 0.0)
            utt_end = float(item.get("t_end_s") or utt_start)
            overlap_s = _overlap(utt_start, utt_end, t_start, t_end)
            if overlap_s <= 0.0:
                continue
            utt_id = str(item.get("utterance_id") or "").strip()
            win_start = max(float(t_start), utt_start - float(pad_s))
            win_end = min(float(t_end), utt_end + float(pad_s))
            if win_end <= win_start:
                win_end = min(float(t_end), win_start + 0.4)
            windows.append(
                {
                    "t_start_s": win_start,
                    "t_end_s": win_end,
                    "utterance_ids": [utt_id] if utt_id else [],
                    "duration_s": max(0.0, win_end - win_start),
                }
            )

        if not windows:
            return [
                {
                    "t_start_s": float(t_start),
                    "t_end_s": float(t_end),
                    "utterance_ids": [],
                    "duration_s": max(0.0, float(t_end) - float(t_start)),
                }
            ]

        ranked = sorted(
            windows,
            key=lambda item: (
                float(item.get("duration_s") or 0.0),
                -float(item.get("t_start_s") or 0.0),
            ),
            reverse=True,
        )[:max_windows]
        ranked.sort(key=lambda item: float(item.get("t_start_s") or 0.0))
        return ranked

    @classmethod
    def _score_candidate_asd_windows(
        cls,
        *,
        utterances: Sequence[Mapping[str, Any]],
        speaker_track_id: str,
        audio_chunks: Sequence[Mapping[str, Any]],
        face_frames: Sequence[Mapping[str, Any]],
        t_start: float,
        t_end: float,
        max_windows: int,
        pad_s: float,
        source_id: str,
        visual_track_id: str,
        artifacts_dir: str,
        raw_face_frame_paths: Sequence[str],
        raw_face_frame_timestamps_s: Sequence[float],
        min_face_frames: int,
        raw_face_window_pad_s: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for window in cls._utterance_windows_for_candidate(
            utterances=utterances,
            speaker_track_id=speaker_track_id,
            t_start=t_start,
            t_end=t_end,
            max_windows=max_windows,
            pad_s=pad_s,
        ):
            win_start = float(window.get("t_start_s") or t_start)
            win_end = float(window.get("t_end_s") or win_start)
            audio_chunk = cls._pick_audio_chunk_item(
                audio_chunks,
                t_start=win_start,
                t_end=win_end,
            )
            selected_face_frames = cls._pick_face_frame_records(
                face_frames,
                t_start=win_start,
                t_end=win_end,
            )
            selected_face_frames = cls._augment_face_frames_with_raw(
                selected_face_frames,
                track_frames=face_frames,
                raw_face_frame_paths=raw_face_frame_paths,
                raw_face_frame_timestamps_s=raw_face_frame_timestamps_s,
                t_start=win_start,
                t_end=win_end,
                source_id=source_id,
                track_id=visual_track_id,
                artifacts_dir=artifacts_dir,
                min_face_frames=min_face_frames,
                pad_s=raw_face_window_pad_s,
            )
            audio_file = cls._materialize_audio_window(
                audio_chunk,
                t_start=win_start,
                t_end=win_end,
                source_id=source_id,
                speaker_track_id=speaker_track_id,
                artifacts_dir=artifacts_dir,
            )
            if not audio_file or not selected_face_frames:
                continue
            face_frame_payload = {
                "paths": [str(item["file_path"]) for item in selected_face_frames],
                "timestamps": [float(item.get("t_start_s") or 0.0) for item in selected_face_frames],
            }
            asd_result = score_light_asd(
                audio_file=audio_file,
                face_frame_paths=face_frame_payload["paths"],
                face_frame_timestamps_s=face_frame_payload["timestamps"],
            )
            if asd_result is None:
                continue
            results.append(
                {
                    **dict(asd_result),
                    "window_t_start_s": win_start,
                    "window_t_end_s": win_end,
                    "window_utterance_ids": list(window.get("utterance_ids") or []),
                    "audio_file": audio_file,
                    "face_frame_count": len(face_frame_payload["paths"]),
                }
            )
        return results

    @staticmethod
    def _pick_audio_chunk_item(
        chunks: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> Dict[str, Any] | None:
        best = None
        best_overlap = -1.0
        for item in chunks:
            overlap_s = _overlap(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            )
            if overlap_s > best_overlap and item.get("file_path"):
                best = dict(item)
                best_overlap = overlap_s
        return best

    @classmethod
    def _materialize_audio_window(
        cls,
        chunk: Mapping[str, Any] | None,
        *,
        t_start: float,
        t_end: float,
        source_id: str,
        speaker_track_id: str,
        artifacts_dir: str,
    ) -> str | None:
        if not chunk:
            return None
        audio_file = str(chunk.get("file_path") or "").strip()
        if not audio_file:
            return None
        chunk_start = float(chunk.get("t_start_s") or 0.0)
        chunk_end = float(chunk.get("t_end_s") or chunk_start)
        if t_start <= chunk_start and t_end >= chunk_end:
            return audio_file
        rel_start = max(0.0, float(t_start) - chunk_start)
        rel_end = max(rel_start, min(float(t_end), chunk_end) - chunk_start)
        if rel_end <= rel_start:
            return audio_file
        try:
            audio_b64 = base64.b64encode(Path(audio_file).read_bytes())
            sliced = slice_audio_b64_segments(audio_b64, [(rel_start, rel_end)])[0]
            if not sliced:
                return audio_file
            out_dir = Path(artifacts_dir) / "evidence_media" / "association_voices"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_id = stable_hash_id(
                "evasd",
                source_id,
                speaker_track_id,
                t_start,
                t_end,
            )
            out_path = out_dir / f"{out_id}.wav"
            if not out_path.exists():
                out_path.write_bytes(base64.b64decode(sliced))
            return str(out_path)
        except Exception:
            return audio_file

    @staticmethod
    def _pick_audio_chunk(
        chunks: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> str | None:
        best = LocalFaceVoiceAssociationStage._pick_audio_chunk_item(
            chunks,
            t_start=t_start,
            t_end=t_end,
        )
        if best is None:
            return None
        return str(best.get("file_path") or "")

    @staticmethod
    def _pick_face_frame_records(
        frames: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> List[Dict[str, Any]]:
        selected = [
            dict(item)
            for item in frames
            if _intersects_window(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            )
            and item.get("file_path")
        ]
        return LocalFaceVoiceAssociationStage._cap_face_frames(selected)

    @staticmethod
    def _pick_face_frames(
        frames: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> Dict[str, List[Any]]:
        capped = LocalFaceVoiceAssociationStage._pick_face_frame_records(
            frames,
            t_start=t_start,
            t_end=t_end,
        )
        return {
            "paths": [str(item["file_path"]) for item in capped],
            "timestamps": [float(item.get("t_start_s") or 0.0) for item in capped],
        }

    @classmethod
    def _augment_face_frames_with_raw(
        cls,
        selected_frames: Sequence[Mapping[str, Any]],
        *,
        track_frames: Sequence[Mapping[str, Any]],
        raw_face_frame_paths: Sequence[str],
        raw_face_frame_timestamps_s: Sequence[float],
        t_start: float,
        t_end: float,
        source_id: str,
        track_id: str,
        artifacts_dir: str,
        min_face_frames: int,
        pad_s: float,
    ) -> List[Dict[str, Any]]:
        selected = [dict(item) for item in selected_frames]
        if len(selected) >= min_face_frames:
            return selected
        anchor_bbox = cls._nearest_bbox(
            track_frames or selected,
            center_s=(float(t_start) + float(t_end)) / 2.0,
        )
        if not anchor_bbox or not raw_face_frame_paths or not raw_face_frame_timestamps_s:
            return selected
        target_count = max(min_face_frames, len(selected))
        window_start = max(0.0, float(t_start) - float(pad_s))
        window_end = float(t_end) + float(pad_s)
        candidate_indices = [
            index
            for index, timestamp in enumerate(raw_face_frame_timestamps_s)
            if window_start <= float(timestamp) <= window_end
        ]
        if len(candidate_indices) < target_count:
            center_s = (float(t_start) + float(t_end)) / 2.0
            ranked = sorted(
                range(min(len(raw_face_frame_paths), len(raw_face_frame_timestamps_s))),
                key=lambda idx: abs(float(raw_face_frame_timestamps_s[idx]) - center_s),
            )
            for index in ranked:
                if index not in candidate_indices:
                    candidate_indices.append(index)
                if len(candidate_indices) >= target_count:
                    break
        augmented: List[Dict[str, Any]] = list(selected)
        existing_paths = {str(item.get("file_path") or "") for item in selected}
        for index in candidate_indices:
            frame_path = str(raw_face_frame_paths[index])
            timestamp = float(raw_face_frame_timestamps_s[index])
            crop_path = cls._write_association_face_crop(
                frame_path=frame_path,
                bbox=anchor_bbox,
                artifacts_dir=artifacts_dir,
                source_id=source_id,
                track_id=track_id,
                timestamp_s=timestamp,
            )
            if not crop_path or crop_path in existing_paths:
                continue
            augmented.append(
                {
                    "file_path": crop_path,
                    "t_start_s": timestamp,
                    "t_end_s": timestamp,
                    "evidence_id": "",
                    "bbox": list(anchor_bbox),
                }
            )
            existing_paths.add(crop_path)
        augmented.sort(key=lambda item: float(item.get("t_start_s") or 0.0))
        return cls._cap_face_frames(augmented)

    @staticmethod
    def _nearest_bbox(
        frames: Sequence[Mapping[str, Any]],
        *,
        center_s: float,
    ) -> List[int] | None:
        best = None
        best_distance = None
        for item in frames:
            bbox = list(item.get("bbox") or [])
            if len(bbox) != 4:
                continue
            distance = abs(float(item.get("t_start_s") or 0.0) - float(center_s))
            if best is None or best_distance is None or distance < best_distance:
                best = [int(value) for value in bbox]
                best_distance = distance
        return best

    @staticmethod
    def _write_association_face_crop(
        *,
        frame_path: str,
        bbox: Sequence[int],
        artifacts_dir: str,
        source_id: str,
        track_id: str,
        timestamp_s: float,
    ) -> str | None:
        try:
            import cv2  # type: ignore

            image = cv2.imread(str(frame_path))
            if image is None:
                return None
            height, width = image.shape[:2]
            x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            pad_x = int(box_w * 0.12)
            pad_y = int(box_h * 0.12)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            out_dir = Path(artifacts_dir) / "evidence_media" / "association_faces"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_id = stable_hash_id(
                "evasdface",
                source_id,
                track_id,
                Path(frame_path).name,
                timestamp_s,
                [x1, y1, x2, y2],
            )
            out_path = out_dir / f"{out_id}.jpg"
            if not out_path.exists():
                cv2.imwrite(str(out_path), crop)
            return str(out_path)
        except Exception:
            return None

    @staticmethod
    def _cap_face_frames(frames: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        cap = max(1, int(os.getenv("MGC_ASSOCIATION_MAX_FACE_FRAMES") or 24))
        selected = list(frames)
        if len(selected) <= cap:
            return selected
        step = len(selected) / float(cap)
        return [selected[min(len(selected) - 1, int(index * step))] for index in range(cap)]

    @staticmethod
    def _support_evidence_refs(
        face_frames: Sequence[Mapping[str, Any]],
        audio_chunks: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> List[str]:
        refs: List[str] = []
        for item in face_frames:
            if _intersects_window(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            ):
                ev = str(item.get("evidence_id") or "").strip()
                if ev:
                    refs.append(ev)
        for item in audio_chunks:
            if _intersects_window(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            ):
                ev = str(item.get("evidence_id") or "").strip()
                if ev:
                    refs.append(ev)
        return refs

    @staticmethod
    def _select_links(
        *,
        source_id: str,
        candidates: Sequence[Mapping[str, Any]],
        min_conf: float,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                bool(((item.get("metadata") or {}).get("asd_verified"))),
                ((item.get("metadata") or {}).get("asd_norm_score") is not None),
                float(((item.get("metadata") or {}).get("asd_norm_score") or -1.0)),
                float(item.get("confidence") or 0.0),
                float(item.get("temporal_score") or 0.0),
            ),
            reverse=True,
        )
        used_speakers: set[str] = set()
        used_visuals: set[str] = set()
        links: List[Dict[str, Any]] = []
        for item in ordered:
            speaker_id = str(item.get("speaker_track_id") or "")
            visual_id = str(item.get("visual_track_id") or "")
            confidence = float(item.get("confidence") or 0.0)
            if not speaker_id or not visual_id:
                continue
            if confidence < min_conf:
                continue
            if speaker_id in used_speakers or visual_id in used_visuals:
                continue
            used_speakers.add(speaker_id)
            used_visuals.add(visual_id)
            t_start = float(item.get("t_start_s") or 0.0)
            t_end = float(item.get("t_end_s") or t_start)
            links.append(
                {
                    "link_id": stable_hash_id(
                        "fvlink",
                        source_id,
                        speaker_id,
                        visual_id,
                        t_start,
                        t_end,
                    ),
                    "speaker_track_id": speaker_id,
                    "visual_track_id": visual_id,
                    "t_start_s": t_start,
                    "t_end_s": t_end,
                    "confidence": confidence,
                    "overlap_s": float(item.get("overlap_s") or 0.0),
                    "support_evidence_refs": list(item.get("support_evidence_refs") or []),
                    "support_utterance_ids": list(item.get("support_utterance_ids") or []),
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
        return links


__all__ = ["LocalFaceVoiceAssociationStage"]
