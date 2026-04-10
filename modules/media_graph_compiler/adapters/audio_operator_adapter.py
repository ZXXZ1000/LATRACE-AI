from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from modules.media_graph_compiler.domain.stable_ids import stable_utterance_id
from modules.media_graph_compiler.types import (
    EvidencePointer,
    SpeakerTrackRecord,
    UtteranceRecord,
)


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


class AudioOperatorAdapter:
    """Normalizes legacy or provider-specific speaker outputs into module contracts."""

    def normalize(
        self,
        *,
        source_id: str,
        stage_output: Mapping[str, Any],
    ) -> Tuple[List[SpeakerTrackRecord], List[UtteranceRecord], List[EvidencePointer]]:
        if "speaker_tracks" in stage_output or "utterances" in stage_output:
            return self._normalize_current_shape(stage_output)
        if "id2audios" in stage_output:
            return self._normalize_legacy_shape(source_id, stage_output.get("id2audios", {}))
        return [], [], []

    def extract_stage_stats(
        self,
        *,
        stage_output: Mapping[str, Any],
    ) -> Dict[str, Any]:
        stats = stage_output.get("stage_stats")
        if isinstance(stats, Mapping):
            return dict(stats)
        return {}

    def _normalize_current_shape(
        self,
        stage_output: Mapping[str, Any],
    ) -> Tuple[List[SpeakerTrackRecord], List[UtteranceRecord], List[EvidencePointer]]:
        speaker_tracks = [SpeakerTrackRecord.model_validate(item) for item in stage_output.get("speaker_tracks", [])]
        utterances = [UtteranceRecord.model_validate(item) for item in stage_output.get("utterances", [])]
        evidence = [EvidencePointer.model_validate(item) for item in stage_output.get("evidence", [])]
        return speaker_tracks, utterances, evidence

    def _normalize_legacy_shape(
        self,
        source_id: str,
        id2audios: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> Tuple[List[SpeakerTrackRecord], List[UtteranceRecord], List[EvidencePointer]]:
        speaker_tracks: List[SpeakerTrackRecord] = []
        utterances: List[UtteranceRecord] = []
        evidence: List[EvidencePointer] = []

        for raw_track_id, entries in (id2audios or {}).items():
            track_id = str(raw_track_id)
            if not track_id.startswith("voice_"):
                track_id = f"voice_{track_id}"
            starts: List[float] = []
            ends: List[float] = []
            utterance_ids: List[str] = []
            evidence_ids: List[str] = []
            for index, item in enumerate(entries or []):
                start_s = _parse_timecode(item.get("start_time")) or 0.0
                end_s = _parse_timecode(item.get("end_time")) or start_s
                starts.append(start_s)
                ends.append(end_s)
                utterance_id = stable_utterance_id(
                    source_id=source_id,
                    speaker_track_id=track_id,
                    t_start_s=start_s,
                    t_end_s=end_s,
                    text=str(item.get("asr") or ""),
                )
                evidence_id = f"{track_id}_audio_{index}"
                utterances.append(
                    UtteranceRecord(
                        utterance_id=utterance_id,
                        speaker_track_id=track_id,
                        t_start_s=start_s,
                        t_end_s=end_s,
                        text=str(item.get("asr") or ""),
                        confidence=float(item.get("score") or 0.0),
                        evidence_refs=[evidence_id],
                    )
                )
                evidence.append(
                    EvidencePointer(
                        evidence_id=evidence_id,
                        kind="audio_chunk",
                        t_start_s=start_s,
                        t_end_s=end_s,
                        metadata={
                            "speaker_track_id": track_id,
                            "transcript": item.get("asr"),
                        },
                    )
                )
                utterance_ids.append(utterance_id)
                evidence_ids.append(evidence_id)
            if starts:
                speaker_tracks.append(
                    SpeakerTrackRecord(
                        track_id=track_id,
                        t_start_s=min(starts),
                        t_end_s=max(ends),
                        utterance_ids=utterance_ids,
                        evidence_refs=evidence_ids,
                    )
                )
        return speaker_tracks, utterances, evidence
