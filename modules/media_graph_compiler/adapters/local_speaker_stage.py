from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Mapping

from modules.media_graph_compiler.adapters.ops.asr_local import (
    get_last_asr_status,
    transcribe_audio_b64,
)
from modules.media_graph_compiler.adapters.ops.audio_segments import (
    slice_audio_b64_segments,
)
from modules.media_graph_compiler.adapters.ops.speaker_diarization import (
    diarize_audio_b64,
    get_last_diarization_status,
)
from modules.media_graph_compiler.adapters.ops.speaker_embedding import (
    build_track_embedding,
    speaker_embedding_runtime,
)
from modules.media_graph_compiler.domain.stable_ids import stable_hash_id, stable_utterance_id


def _parse_timecode(tc: str | None) -> float | None:
    if not tc:
        return None
    try:
        parts = tc.split(":")
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        return None
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))


def _normalize_asr_segments(transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for transcript in transcript_segments or []:
        start_s = _as_float(transcript.get("t_start_s"))
        if start_s is None:
            start_s = _parse_timecode(transcript.get("start_time")) or 0.0
        end_s = _as_float(transcript.get("t_end_s"))
        if end_s is None:
            end_s = _parse_timecode(transcript.get("end_time")) or start_s
        if end_s < start_s:
            end_s = start_s
        text = str(transcript.get("asr") or "").strip() or "(audio)"
        out.append(
            {
                "t_start_s": float(start_s),
                "t_end_s": float(end_s),
                "asr": text,
            }
        )
    return out


class LocalSpeakerStage:
    """Default speaker stage using diarization + ASR + embedding.

    This keeps the public stage contract stable while replacing the legacy
    one-pot voice operator with a clearer three-step pipeline:
    1. speaker diarization
    2. full-audio ASR + diarization-aligned assignment
    3. speaker embedding aggregation
    """

    def run(self, ctx: Mapping[str, Any]) -> Dict[str, Any]:
        source_id = str(ctx.get("source_id") or "unknown")
        request = ctx.get("request")
        artifacts_dir = str(
            (request.metadata.get("artifacts_dir") if request else None)
            or ".artifacts/media_graph_compiler"
        )
        audio_b64 = ctx.get("audio_b64")
        if not audio_b64:
            extracted_audio_path = ctx.get("extracted_audio_path") or ctx.get("normalized_audio_path")
            if extracted_audio_path:
                audio_b64 = base64.b64encode(Path(str(extracted_audio_path)).read_bytes())
        if not audio_b64:
            return {"speaker_tracks": [], "utterances": [], "evidence": []}

        audio_plan = dict((ctx.get("optimization_plan") or {}).get("audio") or {})
        utterances: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        speaker_tracks: List[Dict[str, Any]] = []
        min_turn_length_s = float(audio_plan.get("min_turn_length_s") or 0.4)
        diarized = diarize_audio_b64(audio_b64, min_turn_length_s=min_turn_length_s)
        diarization_status = get_last_diarization_status()
        chunk_payloads = slice_audio_b64_segments(
            audio_b64,
            [(float(item["t_start_s"]), float(item["t_end_s"])) for item in diarized],
        )

        track_files: Dict[str, List[str]] = {}
        track_utterances: Dict[str, List[str]] = {}
        track_evidence: Dict[str, List[str]] = {}
        track_starts: Dict[str, List[float]] = {}
        track_ends: Dict[str, List[float]] = {}
        track_asr_fallback_chunks: Dict[str, int] = {}
        track_asr_errors: Dict[str, List[str]] = {}
        track_asr_segments: Dict[str, int] = {}
        full_audio_transcripts = transcribe_audio_b64(
            audio_b64,
            language=None,
            model_size=str(audio_plan.get("fallback_asr_model") or "small"),
            device="auto",
            strict=False,
        )
        asr_status = get_last_asr_status()
        asr_runtime = str(
            asr_status.get("runtime")
            or asr_status.get("fallback_runtime")
            or "unknown"
        )
        normalized_full_transcripts = _normalize_asr_segments(full_audio_transcripts)
        if bool(asr_status.get("fallback_used")) and normalized_full_transcripts:
            placeholder_only = all(str(item.get("asr") or "").strip() == "(audio)" for item in normalized_full_transcripts)
            if placeholder_only:
                normalized_full_transcripts = []

        for diar_item, chunk_b64 in zip(diarized, chunk_payloads):
            if not chunk_b64:
                continue
            track_id = str(diar_item["track_id"])
            diar_start_s = float(diar_item["t_start_s"])
            diar_end_s = float(diar_item["t_end_s"])
            evidence_id = stable_hash_id(
                "evaudio",
                source_id,
                track_id,
                diar_start_s,
                diar_end_s,
            )
            file_path = self._write_audio_chunk(
                artifacts_dir=artifacts_dir,
                evidence_id=evidence_id,
                encoded_audio_segment=chunk_b64,
            )
            if file_path:
                track_files.setdefault(track_id, []).append(file_path)
            chunk_transcripts: List[Dict[str, Any]] = []
            for transcript in normalized_full_transcripts:
                seg_start = float(transcript.get("t_start_s") or 0.0)
                seg_end = float(transcript.get("t_end_s") or seg_start)
                if _overlap(seg_start, seg_end, diar_start_s, diar_end_s) <= 0.0:
                    continue
                chunk_transcripts.append(
                    {
                        "t_start_s": max(diar_start_s, seg_start),
                        "t_end_s": min(diar_end_s, seg_end),
                        "asr": str(transcript.get("asr") or "").strip() or "(audio)",
                    }
                )
            if not chunk_transcripts:
                track_asr_fallback_chunks[track_id] = int(track_asr_fallback_chunks.get(track_id, 0)) + 1
                chunk_transcripts = [
                    {
                        "t_start_s": diar_start_s,
                        "t_end_s": max(diar_start_s + 1.0, diar_end_s),
                        "asr": "(audio)",
                    }
                ]
            if asr_status.get("error"):
                track_err_list = track_asr_errors.setdefault(track_id, [])
                err_text = str(asr_status["error"])
                if err_text not in track_err_list:
                    track_err_list.append(err_text)
            track_asr_segments[track_id] = int(track_asr_segments.get(track_id, 0)) + len(chunk_transcripts or [])

            chunk_utterance_ids: List[str] = []
            transcript_texts: List[str] = []
            diar_metadata = dict(diar_item.get("metadata") or {})
            for transcript in chunk_transcripts:
                start_s = float(transcript.get("t_start_s") or diar_start_s)
                end_s = float(transcript.get("t_end_s") or start_s)
                if end_s < start_s:
                    end_s = start_s
                text = str(transcript.get("asr") or "").strip() or "(audio)"
                utterance_id = stable_utterance_id(
                    source_id,
                    track_id,
                    start_s,
                    end_s,
                    text,
                )
                utterances.append(
                    {
                        "utterance_id": utterance_id,
                        "speaker_track_id": track_id,
                        "t_start_s": start_s,
                        "t_end_s": end_s,
                        "text": text,
                        "evidence_refs": [evidence_id],
                        "metadata": {
                            "runtime": "diarization_global_asr_align",
                            "asr_runtime": asr_runtime,
                            "asr_fallback_used": bool(asr_status.get("fallback_used")) or text == "(audio)",
                            "asr_error": asr_status.get("error"),
                            "diarization_runtime": diar_metadata.get("runtime")
                            or diarization_status.get("runtime"),
                            "diarization_reason": diar_metadata.get("reason")
                            or diarization_status.get("reason"),
                        },
                    }
                )
                chunk_utterance_ids.append(utterance_id)
                transcript_texts.append(text)
                track_starts.setdefault(track_id, []).append(start_s)
                track_ends.setdefault(track_id, []).append(end_s)

            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "kind": "audio_chunk",
                    "file_path": file_path,
                    "t_start_s": diar_start_s,
                    "t_end_s": diar_end_s,
                    "metadata": {
                        "algorithm": "speaker_diarization",
                        "algorithm_version": "v2",
                        "speaker_track_id": track_id,
                        "transcript": " ".join(transcript_texts[:2]),
                        "diarization_runtime": diar_metadata.get("runtime")
                        or diarization_status.get("runtime"),
                        "diarization_reason": diar_metadata.get("reason")
                        or diarization_status.get("reason"),
                        "asr_runtime": asr_runtime,
                        "asr_fallback_used": bool(asr_status.get("fallback_used")),
                        "asr_error": asr_status.get("error"),
                    },
                }
            )
            track_utterances.setdefault(track_id, []).extend(chunk_utterance_ids)
            track_evidence.setdefault(track_id, []).append(evidence_id)

        for track_id, utterance_ids in track_utterances.items():
            starts = track_starts.get(track_id) or []
            ends = track_ends.get(track_id) or []
            if not starts:
                continue
            embedding = build_track_embedding(track_files.get(track_id) or [])
            track_runtime_counts = {asr_runtime: 1}
            track_errors = list(track_asr_errors.get(track_id) or [])
            speaker_tracks.append(
                {
                    "track_id": track_id,
                    "t_start_s": min(starts),
                    "t_end_s": max(ends),
                    "utterance_ids": utterance_ids,
                    "evidence_refs": track_evidence.get(track_id) or [],
                    "metadata": {
                        "runtime": "pyannote_wespeaker_style",
                        "embedding_runtime": speaker_embedding_runtime(),
                        "speaker_strategy": "anonymous_local_tracks",
                        "utterance_count": len(utterance_ids),
                        "embedding_dim": len(embedding),
                        "diarization_status": {
                            "ok": bool(diarization_status.get("ok")),
                            "runtime": diarization_status.get("runtime"),
                            "fallback_used": bool(diarization_status.get("fallback_used")),
                            "reason": diarization_status.get("reason"),
                            "error": diarization_status.get("error"),
                            "speaker_count": diarization_status.get("speaker_count"),
                            "segment_count": diarization_status.get("segment_count"),
                        },
                        "asr_status": {
                            "chunk_count": len(track_files.get(track_id) or []),
                            "segment_count": int(track_asr_segments.get(track_id, 0)),
                            "fallback_chunks": int(track_asr_fallback_chunks.get(track_id, 0)),
                            "runtime_counts": track_runtime_counts,
                            "errors": track_errors,
                            "mode": "full_audio_asr_align",
                        },
                    },
                }
            )

        if not speaker_tracks:
            return {"speaker_tracks": [], "utterances": [], "evidence": []}
        return {
            "speaker_tracks": speaker_tracks,
            "utterances": utterances,
            "evidence": evidence,
        }

    @staticmethod
    def _write_audio_chunk(
        *,
        artifacts_dir: str,
        evidence_id: str,
        encoded_audio_segment: Any,
    ) -> str | None:
        try:
            out_dir = Path(artifacts_dir) / "evidence_media" / "voices"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{evidence_id}.wav"
            if isinstance(encoded_audio_segment, str):
                raw = base64.b64decode(encoded_audio_segment)
            else:
                raw = base64.b64decode(encoded_audio_segment or b"")
            out_path.write_bytes(raw)
            return str(out_path)
        except Exception:
            return None


__all__ = ["LocalSpeakerStage"]
