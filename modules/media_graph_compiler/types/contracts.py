from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from modules.memory.contracts.graph_models import GraphUpsertRequest


class MediaRoutingContext(BaseModel):
    tenant_id: str
    user_id: List[str] = Field(default_factory=list)
    memory_domain: str = "media"
    run_id: Optional[str] = None
    trace_id: Optional[str] = None


class MediaSourceRef(BaseModel):
    source_id: str
    file_path: Optional[str] = None
    blob_ref: Optional[str] = None
    mime_type: Optional[str] = None
    recorded_at: Optional[str] = None

    @model_validator(mode="after")
    def _validate_location(self) -> "MediaSourceRef":
        if not self.file_path and not self.blob_ref:
            raise ValueError("either file_path or blob_ref must be provided")
        return self


class WindowingPolicy(BaseModel):
    video_window_seconds: float = 8.0
    audio_window_seconds: float = 15.0
    overlap_seconds: float = 2.0
    video_fps: float = 1.0
    audio_chunk_seconds: float = 5.0


class ProviderSelection(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 4096


class OperatorSelection(BaseModel):
    engine: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None


class VisualTrackPolicy(BaseModel):
    enable_tracking: bool = True
    sample_fps: float = 2.0
    min_track_length_s: float = 1.0
    include_masks: bool = False
    include_face_crops: bool = True
    persist_artifacts: bool = True


class SpeakerTrackPolicy(BaseModel):
    enable_asr: bool = True
    enable_diarization: bool = True
    enable_voice_features: bool = True
    min_turn_length_s: float = 0.4
    persist_artifacts: bool = True


class IdentityPolicy(BaseModel):
    enable_identity_resolution: bool = True
    enable_cross_modal_association: bool = True
    auto_create_provisional_person: bool = True
    require_manual_merge_approval: bool = True
    face_match_threshold: float = 0.75
    voice_match_threshold: float = 0.85
    cross_modal_max_offset_s: float = 2.0


class OptimizationPolicy(BaseModel):
    enable_visual_dedup: bool = True
    visual_similarity_threshold: int = 5
    max_visual_frames_per_window: int = 12
    max_visual_frames_per_source: int = 1200
    max_detection_frames_per_source: int = 200
    prefer_clip_frames: bool = True
    drop_full_video_payload: bool = True
    prefer_asset_replay: bool = True
    enable_audio_vad: bool = True
    min_audio_turn_length_s: float = 0.4
    enable_asr_rtf_adaptation: bool = True
    asr_rtf_threshold: float = 1.25
    fallback_asr_model: Optional[str] = None
    prefer_native_video_mode: bool = False
    allow_frame_bundle_mode: bool = True
    allow_realtime_stream_mode: bool = True


class OperatorAssetRef(BaseModel):
    asset_id: str
    asset_type: Literal["visual_tracks", "speaker_tracks", "semantic_windows"]
    source_id: Optional[str] = None
    file_path: Optional[str] = None
    blob_ref: Optional[str] = None
    created_by_run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_location(self) -> "OperatorAssetRef":
        if not self.file_path and not self.blob_ref:
            raise ValueError("either file_path or blob_ref must be provided")
        return self


class CompileAssetInputs(BaseModel):
    visual_tracks: Optional[OperatorAssetRef] = None
    speaker_tracks: Optional[OperatorAssetRef] = None


class EvidencePointer(BaseModel):
    evidence_id: str
    kind: Literal[
        "frame_crop",
        "mask",
        "thumbnail",
        "audio_chunk",
        "transcript",
        "other",
    ]
    file_path: Optional[str] = None
    blob_ref: Optional[str] = None
    t_start_s: Optional[float] = None
    t_end_s: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VisualTrackRecord(BaseModel):
    track_id: str
    category: Literal["person", "object", "unknown"] = "person"
    t_start_s: float
    t_end_s: float
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VisualPrompt(BaseModel):
    prompt_id: str
    kind: Literal["text", "box", "point"]
    frame_index: int = 0
    target_category: Literal["person", "object", "unknown"] = "person"
    text: Optional[str] = None
    box_xyxy: Optional[List[float]] = None
    points_xy: List[List[float]] = Field(default_factory=list)
    point_labels: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prompt_shape(self) -> "VisualPrompt":
        if self.kind == "text" and not self.text:
            raise ValueError("text prompt requires text")
        if self.kind == "box":
            if self.box_xyxy is None or len(self.box_xyxy) != 4:
                raise ValueError("box prompt requires box_xyxy with 4 coordinates")
        if self.kind == "point":
            if not self.points_xy:
                raise ValueError("point prompt requires at least one point")
            if self.point_labels and len(self.point_labels) != len(self.points_xy):
                raise ValueError("point_labels length must match points_xy length")
        return self


class VisualContinuityServiceRequest(BaseModel):
    source: MediaSourceRef
    prompts: List[VisualPrompt]
    clip_start_s: Optional[float] = None
    clip_end_s: Optional[float] = None
    sample_fps: float = 2.0
    max_frames: Optional[int] = None
    include_masks: bool = False
    include_boxes: bool = True
    include_crops: bool = True
    return_dense_frame_outputs: bool = False
    persist_artifacts: bool = False
    session_mode: Literal["batch", "session"] = "batch"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_prompts(self) -> "VisualContinuityServiceRequest":
        if not self.prompts:
            raise ValueError("at least one visual prompt must be provided")
        return self


class VisualContinuityServiceResponse(BaseModel):
    source_id: str
    visual_tracks: List[VisualTrackRecord] = Field(default_factory=list)
    evidence: List[EvidencePointer] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class UtteranceRecord(BaseModel):
    utterance_id: str
    speaker_track_id: str
    t_start_s: float
    t_end_s: float
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeakerTrackRecord(BaseModel):
    track_id: str
    t_start_s: float
    t_end_s: float
    utterance_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FaceVoiceLinkRecord(BaseModel):
    link_id: str
    speaker_track_id: str
    visual_track_id: str
    t_start_s: float
    t_end_s: float
    confidence: float
    overlap_s: float = 0.0
    support_evidence_refs: List[str] = Field(default_factory=list)
    support_utterance_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompileVideoRequest(BaseModel):
    routing: MediaRoutingContext
    source: MediaSourceRef
    clip_start_s: Optional[float] = None
    clip_end_s: Optional[float] = None
    windowing: WindowingPolicy = Field(default_factory=WindowingPolicy)
    provider: ProviderSelection = Field(default_factory=ProviderSelection)
    visual_operator: OperatorSelection = Field(default_factory=OperatorSelection)
    speaker_operator: OperatorSelection = Field(default_factory=OperatorSelection)
    visual_policy: VisualTrackPolicy = Field(default_factory=VisualTrackPolicy)
    speaker_policy: SpeakerTrackPolicy = Field(default_factory=SpeakerTrackPolicy)
    identity: IdentityPolicy = Field(default_factory=IdentityPolicy)
    optimization: OptimizationPolicy = Field(default_factory=OptimizationPolicy)
    asset_inputs: CompileAssetInputs = Field(default_factory=CompileAssetInputs)
    enable_visual_operator: bool = True
    enable_audio_operator: bool = True
    write_graph: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompileAudioRequest(BaseModel):
    routing: MediaRoutingContext
    source: MediaSourceRef
    clip_start_s: Optional[float] = None
    clip_end_s: Optional[float] = None
    windowing: WindowingPolicy = Field(default_factory=WindowingPolicy)
    provider: ProviderSelection = Field(default_factory=ProviderSelection)
    speaker_operator: OperatorSelection = Field(default_factory=OperatorSelection)
    speaker_policy: SpeakerTrackPolicy = Field(default_factory=SpeakerTrackPolicy)
    identity: IdentityPolicy = Field(default_factory=IdentityPolicy)
    optimization: OptimizationPolicy = Field(default_factory=OptimizationPolicy)
    asset_inputs: CompileAssetInputs = Field(default_factory=CompileAssetInputs)
    enable_audio_operator: bool = True
    write_graph: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WindowDigest(BaseModel):
    window_id: str
    modality: Literal["video", "audio"]
    t_start_s: float
    t_end_s: float
    summary: Optional[str] = None
    participant_refs: List[str] = Field(default_factory=list)
    semantic_payload: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CompileTrace(BaseModel):
    scheduler_version: str = "v1"
    prompt_version: str = "media_tkg_unified_extractor_system_prompt_v1"
    provider: Optional[str] = None
    model: Optional[str] = None
    operator_versions: Dict[str, str] = Field(default_factory=dict)
    optimization_plan: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class CompileResult(BaseModel):
    status: Literal["planned", "compiled", "written"]
    source_id: str
    window_digests: List[WindowDigest] = Field(default_factory=list)
    visual_tracks: List[VisualTrackRecord] = Field(default_factory=list)
    speaker_tracks: List[SpeakerTrackRecord] = Field(default_factory=list)
    face_voice_links: List[FaceVoiceLinkRecord] = Field(default_factory=list)
    utterances: List[UtteranceRecord] = Field(default_factory=list)
    evidence: List[EvidencePointer] = Field(default_factory=list)
    asset_outputs: List[OperatorAssetRef] = Field(default_factory=list)
    graph_request: Optional[GraphUpsertRequest] = None
    trace: CompileTrace = Field(default_factory=CompileTrace)
    stats: Dict[str, Any] = Field(default_factory=dict)
