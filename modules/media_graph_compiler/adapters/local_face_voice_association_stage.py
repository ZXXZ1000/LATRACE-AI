from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Sequence

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
        max_asd_candidates_per_speaker = max(
            1,
            int(os.getenv("MGC_ASSOCIATION_MAX_ASD_CANDIDATES_PER_SPEAKER") or 2),
        )
        enable_light_asd = str(os.getenv("MGC_ASSOCIATION_ENABLE_LIGHT_ASD") or "1").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

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
                    audio_file = self._pick_audio_chunk(
                        audio_chunks_by_track.get(speaker_id) or [],
                        t_start=float(item.get("t_start_s") or 0.0),
                        t_end=float(item.get("t_end_s") or 0.0),
                    )
                    face_frames = self._pick_face_frames(
                        visual_frames_by_track.get(str(item.get("visual_track_id") or "")) or [],
                        t_start=float(item.get("t_start_s") or 0.0),
                        t_end=float(item.get("t_end_s") or 0.0),
                    )
                    if audio_file and face_frames["paths"]:
                        asd_result = score_light_asd(
                            audio_file=audio_file,
                            face_frame_paths=face_frames["paths"],
                            face_frame_timestamps_s=face_frames["timestamps"],
                        )
                        if asd_result is not None:
                            item["asd_norm_score"] = float(asd_result.get("norm_score") or 0.0)
                            item["asd_detail"] = dict(asd_result)
                            asd_ok += 1

            for item in speaker_candidates:
                temporal_score = float(item.get("temporal_score") or 0.0)
                asd_norm = item.get("asd_norm_score")
                confidence = temporal_score
                method = "temporal_overlap_fallback"
                if asd_norm is not None:
                    confidence = _clip01(0.7 * float(asd_norm) + 0.3 * temporal_score)
                    method = "light_asd_temporal_fusion"
                candidates.append(
                    {
                        **item,
                        "confidence": confidence,
                        "metadata": {
                            "method": method,
                            "temporal_score": temporal_score,
                            "asd_norm_score": asd_norm,
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
                    }
                    for item in sorted(
                        candidates,
                        key=lambda row: float(row.get("confidence") or 0.0),
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
    def _pick_audio_chunk(
        chunks: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> str | None:
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
                best = str(item.get("file_path"))
                best_overlap = overlap_s
        return best

    @staticmethod
    def _pick_face_frames(
        frames: Sequence[Mapping[str, Any]],
        *,
        t_start: float,
        t_end: float,
    ) -> Dict[str, List[Any]]:
        selected = [
            item
            for item in frames
            if _intersects_window(
                float(item.get("t_start_s") or 0.0),
                float(item.get("t_end_s") or 0.0),
                t_start,
                t_end,
            )
            and item.get("file_path")
        ]
        capped = LocalFaceVoiceAssociationStage._cap_face_frames(selected)
        return {
            "paths": [str(item["file_path"]) for item in capped],
            "timestamps": [float(item.get("t_start_s") or 0.0) for item in capped],
        }

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
        ordered = sorted(candidates, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
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
