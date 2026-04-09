"""TKG (Typed Knowledge Graph) Data Models - PRIMARY memory storage system.

This module defines the canonical data models for the memory system:
- Event: semantic events with summary, actors, timestamps
- Entity: persons, objects, places (with face/voice/text vector_ids)
- Evidence: raw perceptual evidence (face detections, voice segments, etc.)
- MediaSegment: video/audio clips with timestamps
- GraphEdge: typed relationships between nodes

Usage:
    from modules.memory.contracts.graph_models import GraphUpsertRequest, Event, Entity
    
    req = GraphUpsertRequest(
        events=[Event(id="evt_1", summary="Richard cooks dinner", tenant_id="t")],
        entities=[Entity(id="person::richard", type="PERSON", name="Richard")],
        edges=[GraphEdge(src_id="evt_1", dst_id="person::richard", rel_type="INVOLVES")],
    )
    await graph_service.upsert(req)

Note: TKG replaces the deprecated MemoryEntry/MemoryNode system.
      TKG nodes are automatically indexed in Qdrant with node_type/node_id payload.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    confidence: Optional[float] = None


class Provenanced(BaseModel):
    tenant_id: Optional[str] = None
    time_origin: Optional[str] = None
    provenance: Optional[Provenance] = None
    ttl: Optional[float] = None
    importance: Optional[float] = None
    published: Optional[bool] = None
    memory_strength: Optional[float] = None
    last_accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    forgetting_policy: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Scoping metadata for vector search filtering (used by _upsert_tkg_vectors)
    user_id: Optional[List[str]] = None  # List of user IDs that own this node
    memory_domain: Optional[str] = None  # Domain/namespace for memory isolation


class MediaSegment(Provenanced):
    id: str
    source_id: str
    t_media_start: float
    t_media_end: float
    recorded_at: Optional[datetime] = None
    has_physical_time: bool = False
    duration_seconds: Optional[float] = None
    vector_id: Optional[str] = None
    thumbnail_ref: Optional[str] = None
    modality: Optional[str] = None


class Evidence(Provenanced):
    id: str
    source_id: str
    algorithm: str
    algorithm_version: str
    confidence: float
    embedding_ref: Optional[str] = None
    bbox: Optional[str] = None
    offset_in_segment: Optional[float] = None
    text: Optional[str] = None
    utterance_id: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None
    subtype: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


class UtteranceEvidence(Provenanced):
    id: str
    raw_text: str
    t_media_start: float
    t_media_end: float
    speaker_track_id: Optional[str] = None
    asr_model_version: Optional[str] = None
    lang: Optional[str] = None
    segment_id: Optional[str] = None


class Entity(Provenanced):
    id: str
    type: str
    name: Optional[str] = None
    cluster_label: Optional[str] = None
    manual_name: Optional[str] = None
    # Multi-vector references (point to Qdrant vector IDs, set by GraphService)
    face_vector_id: Optional[str] = None   # -> memory_face collection
    voice_vector_id: Optional[str] = None  # -> memory_audio collection
    text_vector_id: Optional[str] = None   # -> memory_text collection
    # Transient embeddings for vector upsert (not persisted to Neo4j)
    face_embedding: Optional[List[float]] = None
    voice_embedding: Optional[List[float]] = None


class SpatioTemporalRegion(Provenanced):
    id: str
    name: Optional[str] = None
    polygon: Optional[str] = None
    room: Optional[str] = None
    region_type: Optional[str] = None
    parent_id: Optional[str] = None


class State(Provenanced):
    id: str
    subject_id: str
    property: str
    value: str
    raw_value: Optional[str] = None
    confidence: Optional[float] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    source_event_id: Optional[str] = None
    extractor_version: Optional[str] = None
    status: Optional[str] = None


class PendingState(Provenanced):
    id: str
    subject_id: str
    property: str
    value: str
    raw_value: Optional[str] = None
    confidence: Optional[float] = None
    valid_from: Optional[datetime] = None
    source_event_id: Optional[str] = None
    extractor_version: Optional[str] = None
    status: Optional[str] = None
    pending_reason: Optional[str] = None


class Knowledge(Provenanced):
    id: str
    schema_version: Optional[str] = None
    summary: Optional[str] = None
    buckets_meta: Optional[Dict[str, str]] = None
    data: Optional[Dict[str, object]] = None
    # Optional identity registry metadata for facts
    registry_status: Optional[str] = None


class Event(Provenanced):
    id: str
    summary: str
    desc: Optional[str] = None
    t_abs_start: Optional[datetime] = None
    t_abs_end: Optional[datetime] = None
    source: Optional[str] = None
    logical_event_id: Optional[str] = None
    source_turn_ids: Optional[List[str]] = None
    evidence_status: Optional[str] = None
    evidence_confidence: Optional[float] = None
    event_confidence: Optional[float] = None
    evidence_count: Optional[int] = None
    topic_id: Optional[str] = None
    topic_path: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    time_bucket: Optional[List[str]] = None
    tags_vocab_version: Optional[str] = None
    # P0-2: VLM structured output fields
    event_type: Optional[str] = None  # e.g., "meeting", "call", "walk", "eat"
    action: Optional[str] = None  # e.g., "lock", "unlock", "use_phone", "talk"
    actor_id: Optional[str] = None  # Entity ID of the actor
    # Multi-vector references (point to Qdrant vector IDs)
    text_vector_id: Optional[str] = None   # -> memory_text collection (summary embedding)
    clip_vector_id: Optional[str] = None   # -> memory_image collection (CLIP scene embedding)


class Place(Provenanced):
    id: str
    name: str
    geo_location: Optional[str] = None
    floor: Optional[str] = None
    area_type: Optional[str] = None


class TimeSlice(Provenanced):
    id: str
    kind: str
    t_abs_start: Optional[datetime] = None
    t_abs_end: Optional[datetime] = None
    t_media_start: Optional[float] = None
    t_media_end: Optional[float] = None
    granularity_level: Optional[int] = None
    parent_id: Optional[str] = None


class GraphEdge(Provenanced):
    src_id: str
    dst_id: str
    rel_type: str
    confidence: Optional[float] = None
    role: Optional[str] = None
    layer: Optional[str] = None
    kind: Optional[str] = None
    source: Optional[str] = None
    status: Optional[str] = None
    weight: Optional[float] = None
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    src_type: Optional[str] = None
    dst_type: Optional[str] = None


class GraphUpsertRequest(BaseModel):
    segments: List[MediaSegment] = Field(default_factory=list)
    evidences: List[Evidence] = Field(default_factory=list)
    utterances: List[UtteranceEvidence] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    places: List[Place] = Field(default_factory=list)
    time_slices: List[TimeSlice] = Field(default_factory=list)
    regions: List[SpatioTemporalRegion] = Field(default_factory=list)
    states: List[State] = Field(default_factory=list)
    knowledge: List[Knowledge] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    pending_equivs: List["PendingEquiv"] = Field(default_factory=list)


class PendingEquiv(Provenanced):
    """Identity registry pending record; to be approved before merging entities."""

    id: str
    entity_id: str
    candidate_id: str
    evidence_id: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "pending"  # pending|approved|rejected
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None


__all__ = [
    "Provenance",
    "Provenanced",
    "MediaSegment",
    "Evidence",
    "UtteranceEvidence",
    "Entity",
    "SpatioTemporalRegion",
    "State",
    "Knowledge",
    "PendingEquiv",
    "Event",
    "Place",
    "TimeSlice",
    "GraphEdge",
    "GraphUpsertRequest",
]
