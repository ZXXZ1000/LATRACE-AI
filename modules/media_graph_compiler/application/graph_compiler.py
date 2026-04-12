from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Sequence

from modules.media_graph_compiler.adapters.media_probe import MediaBackbone, MediaSegmentRecord
from modules.media_graph_compiler.domain import (
    stable_event_id,
    stable_speaker_entity_id,
    stable_visual_entity_id,
    stable_window_id,
)
from modules.media_graph_compiler.types import (
    CompileTrace,
    EvidencePointer,
    FaceVoiceLinkRecord,
    MediaRoutingContext,
    SpeakerTrackRecord,
    UtteranceRecord,
    VisualTrackRecord,
    WindowDigest,
)
from modules.memory.contracts.graph_models import (
    Entity,
    Event,
    Evidence,
    GraphEdge,
    GraphUpsertRequest,
    MediaSegment,
    PendingEquiv,
    TimeSlice,
    UtteranceEvidence,
)


class GraphCompiler:
    """Compile stage outputs into the canonical graph upsert contract."""

    def compile(
        self,
        *,
        routing: MediaRoutingContext,
        backbone: MediaBackbone,
        window_digests: Sequence[WindowDigest],
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
        face_voice_links: Sequence[FaceVoiceLinkRecord],
        utterances: Sequence[UtteranceRecord],
        evidence: Sequence[EvidencePointer],
        trace: CompileTrace,
        source_recorded_at: str | None = None,
    ) -> GraphUpsertRequest:
        _ = trace
        source_id = backbone.segments[0].source_id if backbone.segments else "unknown"
        recorded_at = self._parse_recorded_at(source_recorded_at)
        segment_lookup = {segment.id: segment for segment in backbone.segments}
        utterance_segment_ids = {
            item.utterance_id: self._segment_id_for_span(
                segments=backbone.segments,
                t_start_s=item.t_start_s,
                t_end_s=item.t_end_s,
            )
            for item in utterances
        }
        evidence_segment_ids = {
            item.evidence_id: self._segment_id_for_span(
                segments=backbone.segments,
                t_start_s=item.t_start_s,
                t_end_s=item.t_end_s,
            )
            for item in evidence
        }

        segments = [
            self._segment_to_graph(
                routing,
                segment,
                recorded_at=recorded_at,
            )
            for segment in backbone.segments
        ]
        entities = self._build_entities(routing, source_id, visual_tracks, speaker_tracks)
        events = self._build_events(
            routing,
            source_id,
            window_digests,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
            recorded_at=recorded_at,
        )
        utterance_nodes = [
            self._utterance_to_graph(
                routing,
                item,
                segment_id=utterance_segment_ids.get(item.utterance_id),
            )
            for item in utterances
        ]
        evidence_nodes = [
            self._evidence_to_graph(
                routing,
                source_id,
                item,
                segment=segment_lookup.get(evidence_segment_ids.get(item.evidence_id) or ""),
            )
            for item in evidence
        ]
        time_slices = self._build_time_slices(
            routing=routing,
            source_id=source_id,
            window_digests=window_digests,
            recorded_at=recorded_at,
        )
        pending_equivs = self._build_pending_equivs(
            routing=routing,
            source_id=source_id,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
            face_voice_links=face_voice_links,
        )
        edges = self._build_edges(
            routing=routing,
            backbone=backbone,
            window_digests=window_digests,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
            utterances=utterances,
            evidence=evidence,
            events=events,
            time_slices=time_slices,
            evidence_segment_ids=evidence_segment_ids,
        )
        return GraphUpsertRequest(
            segments=segments,
            evidences=evidence_nodes,
            utterances=utterance_nodes,
            entities=entities,
            events=events,
            time_slices=time_slices,
            edges=edges,
            pending_equivs=pending_equivs,
        )

    def _segment_to_graph(
        self,
        routing: MediaRoutingContext,
        segment: MediaSegmentRecord,
        *,
        recorded_at: datetime | None,
    ) -> MediaSegment:
        return MediaSegment(
            id=segment.id,
            source_id=segment.source_id,
            t_media_start=segment.t_media_start,
            t_media_end=segment.t_media_end,
            recorded_at=recorded_at or segment.recorded_at,
            has_physical_time=bool(recorded_at or segment.recorded_at or segment.has_physical_time),
            duration_seconds=segment.duration_seconds,
            modality=segment.modality,
            time_origin="media",
            tenant_id=routing.tenant_id,
            user_id=routing.user_id,
            memory_domain=routing.memory_domain,
        )

    def _build_entities(
        self,
        routing: MediaRoutingContext,
        source_id: str,
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
    ) -> List[Entity]:
        entities: List[Entity] = []
        for track in visual_tracks:
            entities.append(
                Entity(
                    id=stable_visual_entity_id(source_id, track.track_id),
                    type="PERSON",
                    cluster_label=track.track_id,
                    name=track.track_id,
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        for track in speaker_tracks:
            entities.append(
                Entity(
                    id=stable_speaker_entity_id(source_id, track.track_id),
                    type="PERSON",
                    cluster_label=track.track_id,
                    name=track.track_id,
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        return entities

    def _build_events(
        self,
        routing: MediaRoutingContext,
        source_id: str,
        window_digests: Sequence[WindowDigest],
        *,
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
        recorded_at: datetime | None,
    ) -> List[Event]:
        events: List[Event] = []
        entity_lookup = self._entity_lookup(
            source_id=source_id,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
        )
        for digest in window_digests:
            provider_response = dict((digest.semantic_payload or {}).get("provider_response") or {})
            event_payload = self._primary_event_payload(provider_response)
            summary = digest.summary or f"{digest.modality} activity between {digest.t_start_s:.1f}s and {digest.t_end_s:.1f}s"
            desc = self._normalize_optional_text(event_payload.get("text") or event_payload.get("desc"))
            if desc == summary:
                desc = None
            keywords = self._normalize_text_list(
                provider_response.get("episodic") or provider_response.get("keywords") or []
            )
            tags = self._normalize_text_list(
                provider_response.get("semantic") or event_payload.get("tags") or []
            )
            if "media_compile" not in tags:
                tags.insert(0, "media_compile")
            actor_id = self._actor_entity_id(
                source_id=source_id,
                actor_tag=event_payload.get("actor_tag"),
                entity_lookup=entity_lookup,
            )
            events.append(
                Event(
                    id=stable_event_id(source_id, digest.window_id, summary),
                    summary=summary,
                    desc=desc,
                    t_abs_start=self._absolute_time(recorded_at, digest.t_start_s),
                    t_abs_end=self._absolute_time(recorded_at, digest.t_end_s),
                    source=source_id,
                    logical_event_id=digest.window_id,
                    evidence_count=len(digest.evidence_refs),
                    tags=tags,
                    keywords=keywords or None,
                    event_type=self._normalize_optional_text(event_payload.get("event_type")),
                    action=self._normalize_optional_text(event_payload.get("action")),
                    actor_id=actor_id,
                    event_confidence=self._safe_float(event_payload.get("event_confidence")),
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        return events

    def _utterance_to_graph(
        self,
        routing: MediaRoutingContext,
        item: UtteranceRecord,
        *,
        segment_id: str | None,
    ) -> UtteranceEvidence:
        return UtteranceEvidence(
            id=item.utterance_id,
            raw_text=item.text,
            t_media_start=item.t_start_s,
            t_media_end=item.t_end_s,
            speaker_track_id=item.speaker_track_id,
            lang=item.language,
            segment_id=segment_id,
            time_origin="media",
            tenant_id=routing.tenant_id,
            user_id=routing.user_id,
            memory_domain=routing.memory_domain,
        )

    def _evidence_to_graph(
        self,
        routing: MediaRoutingContext,
        source_id: str,
        item: EvidencePointer,
        *,
        segment: MediaSegmentRecord | None,
    ) -> Evidence:
        offset_in_segment = item.t_start_s
        if segment is not None and item.t_start_s is not None:
            offset_in_segment = max(0.0, item.t_start_s - segment.t_media_start)
        return Evidence(
            id=item.evidence_id,
            source_id=source_id,
            algorithm=str(item.metadata.get("algorithm") or item.kind),
            algorithm_version=str(item.metadata.get("algorithm_version") or "v1"),
            confidence=float(item.metadata.get("confidence") or 1.0),
            offset_in_segment=offset_in_segment,
            text=str(item.metadata.get("transcript")) if item.metadata.get("transcript") else None,
            subtype=item.kind,
            extras=dict(item.metadata),
            time_origin="media",
            tenant_id=routing.tenant_id,
            user_id=routing.user_id,
            memory_domain=routing.memory_domain,
        )

    def _build_time_slices(
        self,
        *,
        routing: MediaRoutingContext,
        source_id: str,
        window_digests: Sequence[WindowDigest],
        recorded_at: datetime | None,
    ) -> List[TimeSlice]:
        time_slices: List[TimeSlice] = []
        for digest in window_digests:
            duration_seconds = max(1, int(round(digest.t_end_s - digest.t_start_s)))
            time_slices.append(
                TimeSlice(
                    id=self._time_slice_id(source_id, digest.window_id),
                    kind="media_window",
                    t_abs_start=self._absolute_time(recorded_at, digest.t_start_s),
                    t_abs_end=self._absolute_time(recorded_at, digest.t_end_s),
                    t_media_start=digest.t_start_s,
                    t_media_end=digest.t_end_s,
                    granularity_level=duration_seconds,
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        return time_slices

    def _build_pending_equivs(
        self,
        *,
        routing: MediaRoutingContext,
        source_id: str,
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
        face_voice_links: Sequence[FaceVoiceLinkRecord],
    ) -> List[PendingEquiv]:
        entity_lookup = self._entity_lookup(
            source_id=source_id,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
        )
        pending: List[PendingEquiv] = []
        for link in face_voice_links:
            speaker_entity_id = entity_lookup["speaker"].get(link.speaker_track_id)
            visual_entity_id = entity_lookup["visual"].get(link.visual_track_id)
            if not speaker_entity_id or not visual_entity_id:
                continue
            pending.append(
                PendingEquiv(
                    id=link.link_id,
                    entity_id=speaker_entity_id,
                    candidate_id=visual_entity_id,
                    evidence_id=(link.support_evidence_refs[0] if link.support_evidence_refs else None),
                    confidence=float(link.confidence),
                    status="pending",
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        return pending

    def _build_edges(
        self,
        *,
        routing: MediaRoutingContext,
        backbone: MediaBackbone,
        window_digests: Sequence[WindowDigest],
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
        utterances: Sequence[UtteranceRecord],
        evidence: Sequence[EvidencePointer],
        events: Sequence[Event],
        time_slices: Sequence[TimeSlice],
        evidence_segment_ids: Mapping[str, str | None],
    ) -> List[GraphEdge]:
        source_id = backbone.segments[0].source_id if backbone.segments else "unknown"
        entity_lookup = self._entity_lookup(
            source_id=source_id,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
        )
        visual_entity_by_track = dict(entity_lookup["visual"])
        speaker_entity_by_track = dict(entity_lookup["speaker"])
        event_by_window = {digest.window_id: event.id for digest, event in zip(window_digests, events)}
        event_actor_by_id = {event.id: event.actor_id for event in events}
        timeslice_by_window = {
            digest.window_id: timeslice.id
            for digest, timeslice in zip(window_digests, time_slices)
        }
        segment_by_window = {
            key: segment.id
            for segment in backbone.segments
            for key in (
                segment.id,
                stable_window_id(
                    source_id,
                    segment.t_media_start,
                    segment.t_media_end,
                    segment.modality,
                ),
            )
        }

        edges: List[GraphEdge] = []
        seen_edges: set[tuple[str, str, str, str | None, str | None]] = set()

        def append_edge(edge: GraphEdge) -> None:
            key = (edge.src_id, edge.dst_id, edge.rel_type, edge.src_type, edge.dst_type)
            if key in seen_edges:
                return
            seen_edges.add(key)
            edges.append(edge)

        for edge in backbone.edges:
            append_edge(
                GraphEdge(
                    src_id=edge.src_id,
                    dst_id=edge.dst_id,
                    rel_type=edge.rel_type,
                    src_type="MediaSegment",
                    dst_type="MediaSegment",
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )

        for digest in window_digests:
            event_id = event_by_window.get(digest.window_id)
            segment_id = segment_by_window.get(digest.window_id)
            timeslice_id = timeslice_by_window.get(digest.window_id)
            if event_id is None or segment_id is None:
                continue
            append_edge(
                GraphEdge(
                    src_id=event_id,
                    dst_id=segment_id,
                    rel_type="SUMMARIZES",
                    src_type="Event",
                    dst_type="MediaSegment",
                    time_origin="media",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
            if timeslice_id:
                append_edge(
                    GraphEdge(
                        src_id=timeslice_id,
                        dst_id=segment_id,
                        rel_type="COVERS_SEGMENT",
                        src_type="TimeSlice",
                        dst_type="MediaSegment",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
                append_edge(
                    GraphEdge(
                        src_id=timeslice_id,
                        dst_id=event_id,
                        rel_type="COVERS_EVENT",
                        src_type="TimeSlice",
                        dst_type="Event",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
            for participant_ref in digest.participant_refs:
                entity_id = self._resolve_entity_id(
                    source_id=source_id,
                    ref=participant_ref,
                    entity_lookup=entity_lookup,
                )
                if entity_id is None:
                    continue
                append_edge(
                    GraphEdge(
                        src_id=event_id,
                        dst_id=entity_id,
                        rel_type="INVOLVES",
                        src_type="Event",
                        dst_type="Entity",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
            actor_id = event_actor_by_id.get(event_id)
            if actor_id:
                append_edge(
                    GraphEdge(
                        src_id=event_id,
                        dst_id=actor_id,
                        rel_type="INVOLVES",
                        src_type="Event",
                        dst_type="Entity",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
            for evidence_id in digest.evidence_refs:
                append_edge(
                    GraphEdge(
                        src_id=event_id,
                        dst_id=evidence_id,
                        rel_type="SUPPORTED_BY",
                        src_type="Event",
                        dst_type="Evidence",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        for utterance in utterances:
            speaker_entity_id = speaker_entity_by_track.get(utterance.speaker_track_id)
            if speaker_entity_id:
                append_edge(
                    GraphEdge(
                        src_id=utterance.utterance_id,
                        dst_id=speaker_entity_id,
                        rel_type="SPOKEN_BY",
                        src_type="UtteranceEvidence",
                        dst_type="Entity",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
            for evidence_id in utterance.evidence_refs:
                append_edge(
                    GraphEdge(
                        src_id=utterance.utterance_id,
                        dst_id=evidence_id,
                        rel_type="SUPPORTED_BY",
                        src_type="UtteranceEvidence",
                        dst_type="Evidence",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        for item in evidence:
            segment_id = evidence_segment_ids.get(item.evidence_id)
            if segment_id:
                append_edge(
                    GraphEdge(
                        src_id=segment_id,
                        dst_id=item.evidence_id,
                        rel_type="CONTAINS_EVIDENCE",
                        src_type="MediaSegment",
                        dst_type="Evidence",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        for track in visual_tracks:
            entity_id = visual_entity_by_track.get(track.track_id)
            if not entity_id:
                continue
            for evidence_id in track.evidence_refs:
                append_edge(
                    GraphEdge(
                        src_id=evidence_id,
                        dst_id=entity_id,
                        rel_type="BELONGS_TO_ENTITY",
                        src_type="Evidence",
                        dst_type="Entity",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        for track in speaker_tracks:
            entity_id = speaker_entity_by_track.get(track.track_id)
            if not entity_id:
                continue
            for evidence_id in track.evidence_refs:
                append_edge(
                    GraphEdge(
                        src_id=evidence_id,
                        dst_id=entity_id,
                        rel_type="BELONGS_TO_ENTITY",
                        src_type="Evidence",
                        dst_type="Entity",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        for digest in window_digests:
            event_id = event_by_window.get(digest.window_id)
            if event_id is None:
                continue
            for utterance in utterances:
                if not self._ranges_overlap(
                    digest.t_start_s,
                    digest.t_end_s,
                    utterance.t_start_s,
                    utterance.t_end_s,
                ):
                    continue
                append_edge(
                    GraphEdge(
                        src_id=event_id,
                        dst_id=utterance.utterance_id,
                        rel_type="SUPPORTED_BY",
                        src_type="Event",
                        dst_type="UtteranceEvidence",
                        time_origin="media",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )

        return edges

    @staticmethod
    def _entity_lookup(
        *,
        source_id: str,
        visual_tracks: Sequence[VisualTrackRecord],
        speaker_tracks: Sequence[SpeakerTrackRecord],
    ) -> Dict[str, Dict[str, str]]:
        return {
            "visual": {
                item.track_id: stable_visual_entity_id(source_id, item.track_id)
                for item in visual_tracks
            },
            "speaker": {
                item.track_id: stable_speaker_entity_id(source_id, item.track_id)
                for item in speaker_tracks
            },
        }

    @staticmethod
    def _primary_event_payload(provider_response: Mapping[str, Any]) -> Dict[str, Any]:
        events = provider_response.get("events") or []
        if isinstance(events, list):
            for item in events:
                if isinstance(item, Mapping):
                    participants = item.get("participants") or []
                    actor_tag = None
                    if isinstance(participants, list) and participants:
                        first = participants[0]
                        if isinstance(first, str) and first.strip():
                            actor_tag = first.strip()
                    return {
                        "text": item.get("summary"),
                        "desc": item.get("desc"),
                        "event_type": item.get("event_type"),
                        "action": item.get("action"),
                        "actor_tag": actor_tag,
                        "event_confidence": item.get("event_confidence"),
                        "tags": item.get("tags") or [],
                    }
        timeline = provider_response.get("semantic_timeline") or []
        if isinstance(timeline, list):
            for item in timeline:
                if isinstance(item, Mapping):
                    return dict(item)
        return {}

    @classmethod
    def _actor_entity_id(
        cls,
        *,
        source_id: str,
        actor_tag: Any,
        entity_lookup: Mapping[str, Mapping[str, str]],
    ) -> str | None:
        if not isinstance(actor_tag, str) or not actor_tag.strip():
            return None
        return cls._resolve_entity_id(
            source_id=source_id,
            ref=actor_tag.strip(),
            entity_lookup=entity_lookup,
        )

    @staticmethod
    def _resolve_entity_id(
        *,
        source_id: str,
        ref: str,
        entity_lookup: Mapping[str, Mapping[str, str]],
    ) -> str | None:
        cleaned = str(ref or "").strip()
        if not cleaned:
            return None
        if cleaned in entity_lookup["visual"]:
            return entity_lookup["visual"][cleaned]
        if cleaned in entity_lookup["speaker"]:
            return entity_lookup["speaker"][cleaned]
        if cleaned.startswith("face_"):
            return stable_visual_entity_id(source_id, cleaned)
        if cleaned.startswith("voice_"):
            return stable_speaker_entity_id(source_id, cleaned)
        return None

    @staticmethod
    def _normalize_text_list(values: Any) -> List[str]:
        out: List[str] = []
        if not isinstance(values, list):
            return out
        for item in values:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text and text not in out:
                out.append(text)
        return out

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _segment_id_for_span(
        *,
        segments: Sequence[MediaSegmentRecord],
        t_start_s: float | None,
        t_end_s: float | None,
    ) -> str | None:
        if t_start_s is None and t_end_s is None:
            return None
        start = float(t_start_s if t_start_s is not None else t_end_s or 0.0)
        end = float(t_end_s if t_end_s is not None else t_start_s or 0.0)
        for segment in segments:
            if GraphCompiler._ranges_overlap(
                segment.t_media_start,
                segment.t_media_end,
                start,
                end,
            ):
                return segment.id
        return None

    @staticmethod
    def _ranges_overlap(
        first_start: float,
        first_end: float,
        second_start: float,
        second_end: float,
    ) -> bool:
        return second_end > first_start and second_start < first_end

    @staticmethod
    def _time_slice_id(source_id: str, window_id: str) -> str:
        return f"{source_id}#timeslice::{window_id}"

    @staticmethod
    def _parse_recorded_at(value: str | None) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _absolute_time(recorded_at: datetime | None, offset_seconds: float) -> datetime | None:
        if recorded_at is None:
            return None
        return recorded_at + timedelta(seconds=max(0.0, float(offset_seconds)))
