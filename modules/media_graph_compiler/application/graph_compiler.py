from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

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
    ) -> GraphUpsertRequest:
        segments = [self._segment_to_graph(routing, segment) for segment in backbone.segments]
        entities = self._build_entities(routing, backbone.segments[0].source_id if backbone.segments else "unknown", visual_tracks, speaker_tracks)
        events = self._build_events(routing, backbone.segments[0].source_id if backbone.segments else "unknown", window_digests)
        utterance_nodes = [self._utterance_to_graph(routing, item) for item in utterances]
        evidence_nodes = [self._evidence_to_graph(routing, backbone.segments[0].source_id if backbone.segments else "unknown", item) for item in evidence]
        edges = self._build_edges(
            routing=routing,
            backbone=backbone,
            window_digests=window_digests,
            visual_tracks=visual_tracks,
            speaker_tracks=speaker_tracks,
            face_voice_links=face_voice_links,
            utterances=utterances,
            evidence=evidence,
            events=events,
        )
        return GraphUpsertRequest(
            segments=segments,
            evidences=evidence_nodes,
            utterances=utterance_nodes,
            entities=entities,
            events=events,
            edges=edges,
        )

    def _segment_to_graph(self, routing: MediaRoutingContext, segment: MediaSegmentRecord) -> MediaSegment:
        return MediaSegment(
            id=segment.id,
            source_id=segment.source_id,
            t_media_start=segment.t_media_start,
            t_media_end=segment.t_media_end,
            duration_seconds=segment.duration_seconds,
            modality=segment.modality,
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
    ) -> List[Event]:
        events: List[Event] = []
        for digest in window_digests:
            summary = digest.summary or f"{digest.modality} activity between {digest.t_start_s:.1f}s and {digest.t_end_s:.1f}s"
            events.append(
                Event(
                    id=stable_event_id(source_id, digest.window_id, summary),
                    summary=summary,
                    tags=["media_compile"],
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        return events

    def _utterance_to_graph(self, routing: MediaRoutingContext, item: UtteranceRecord) -> UtteranceEvidence:
        return UtteranceEvidence(
            id=item.utterance_id,
            raw_text=item.text,
            t_media_start=item.t_start_s,
            t_media_end=item.t_end_s,
            speaker_track_id=item.speaker_track_id,
            lang=item.language,
            tenant_id=routing.tenant_id,
            user_id=routing.user_id,
            memory_domain=routing.memory_domain,
        )

    def _evidence_to_graph(self, routing: MediaRoutingContext, source_id: str, item: EvidencePointer) -> Evidence:
        return Evidence(
            id=item.evidence_id,
            source_id=source_id,
            algorithm=str(item.metadata.get("algorithm") or item.kind),
            algorithm_version=str(item.metadata.get("algorithm_version") or "v1"),
            confidence=float(item.metadata.get("confidence") or 1.0),
            offset_in_segment=item.t_start_s,
            text=str(item.metadata.get("transcript")) if item.metadata.get("transcript") else None,
            subtype=item.kind,
            extras=dict(item.metadata),
            tenant_id=routing.tenant_id,
            user_id=routing.user_id,
            memory_domain=routing.memory_domain,
        )

    def _build_edges(
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
        events: Sequence[Event],
    ) -> List[GraphEdge]:
        source_id = backbone.segments[0].source_id if backbone.segments else "unknown"
        visual_entity_by_track = {
            item.track_id: stable_visual_entity_id(source_id, item.track_id) for item in visual_tracks
        }
        speaker_entity_by_track = {
            item.track_id: stable_speaker_entity_id(source_id, item.track_id) for item in speaker_tracks
        }
        event_by_window = {digest.window_id: event.id for digest, event in zip(window_digests, events)}
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
        utterances_by_evidence: Dict[str, List[str]] = {}
        for utterance in utterances:
            for evidence_id in utterance.evidence_refs:
                utterances_by_evidence.setdefault(evidence_id, []).append(utterance.utterance_id)

        edges: List[GraphEdge] = []
        for digest in window_digests:
            event_id = event_by_window.get(digest.window_id)
            segment_id = segment_by_window.get(digest.window_id)
            if event_id is None or segment_id is None:
                continue
            edges.append(
                GraphEdge(
                    src_id=event_id,
                    dst_id=segment_id,
                    rel_type="OCCURS_IN",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        for utterance in utterances:
            speaker_entity_id = speaker_entity_by_track.get(utterance.speaker_track_id)
            if speaker_entity_id:
                edges.append(
                    GraphEdge(
                        src_id=utterance.utterance_id,
                        dst_id=speaker_entity_id,
                        rel_type="SAID_BY",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                        )
                    )
        for link in face_voice_links:
            speaker_entity_id = speaker_entity_by_track.get(link.speaker_track_id)
            visual_entity_id = visual_entity_by_track.get(link.visual_track_id)
            if not speaker_entity_id or not visual_entity_id:
                continue
            edges.append(
                GraphEdge(
                    src_id=speaker_entity_id,
                    dst_id=visual_entity_id,
                    rel_type="ALIGNED_WITH",
                    confidence=float(link.confidence),
                    weight=float(link.overlap_s or 0.0),
                    role="face_voice_association",
                    tenant_id=routing.tenant_id,
                    user_id=routing.user_id,
                    memory_domain=routing.memory_domain,
                )
            )
        for item in evidence:
            for utterance_id in utterances_by_evidence.get(item.evidence_id, []):
                edges.append(
                    GraphEdge(
                        src_id=item.evidence_id,
                        dst_id=utterance_id,
                        rel_type="SUPPORTS_UTTERANCE",
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
                if digest.t_start_s <= utterance.t_start_s <= digest.t_end_s:
                    edges.append(
                        GraphEdge(
                            src_id=event_id,
                            dst_id=utterance.utterance_id,
                            rel_type="HAS_UTTERANCE",
                            tenant_id=routing.tenant_id,
                            user_id=routing.user_id,
                            memory_domain=routing.memory_domain,
                        )
                    )
            for ref in digest.participant_refs:
                entity_id = visual_entity_by_track.get(ref) or speaker_entity_by_track.get(ref)
                if entity_id:
                    edges.append(
                        GraphEdge(
                            src_id=event_id,
                            dst_id=entity_id,
                            rel_type="INVOLVES",
                            tenant_id=routing.tenant_id,
                            user_id=routing.user_id,
                            memory_domain=routing.memory_domain,
                        )
                    )
            for evidence_id in digest.evidence_refs:
                edges.append(
                    GraphEdge(
                        src_id=event_id,
                        dst_id=evidence_id,
                        rel_type="SUPPORTED_BY",
                        tenant_id=routing.tenant_id,
                        user_id=routing.user_id,
                        memory_domain=routing.memory_domain,
                    )
                )
        return edges
