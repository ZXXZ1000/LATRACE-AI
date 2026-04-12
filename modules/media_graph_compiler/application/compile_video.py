from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List

from modules.media_graph_compiler.adapters import (
    AudioOperatorAdapter,
    LocalMediaPipeline,
    LocalOperatorAssetStore,
    MediaNormalizer,
    MediaProbeAdapter,
    VisualTrackStageAdapter,
    default_operator_bus,
)
from modules.media_graph_compiler.adapters.default_operator_stages import (
    ensure_default_operator_stages,
)
from modules.media_graph_compiler.application.graph_compiler import GraphCompiler
from modules.media_graph_compiler.application.optimization_plan import OptimizationPlanBuilder
from modules.media_graph_compiler.application.prompt_packer import PromptPacker
from modules.media_graph_compiler.application.stage_names import (
    FACE_VOICE_ASSOCIATION_STAGE,
    LEGACY_ASSOCIATION_STAGE,
    LEGACY_SEMANTIC_STAGE,
    LEGACY_SPEAKER_STAGE,
    LEGACY_VISUAL_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)
from modules.media_graph_compiler.application.window_scheduler import WindowScheduler
from modules.media_graph_compiler.domain import stable_window_id
from modules.media_graph_compiler.types import (
    CompileResult,
    CompileTrace,
    CompileVideoRequest,
    FaceVoiceLinkRecord,
    WindowDigest,
)


def compile_video(
    request: CompileVideoRequest,
    *,
    operator_bus=default_operator_bus,
    asset_store: LocalOperatorAssetStore | None = None,
    graph_writer: Callable[[Any], None] | None = None,
) -> CompileResult:
    total_started_at = perf_counter()
    stage_timings_ms: Dict[str, float] = {}
    asset_store = asset_store or LocalOperatorAssetStore(
        request.metadata.get("artifacts_dir")
    )
    normalizer = MediaNormalizer()
    local_media = LocalMediaPipeline(normalizer=normalizer)
    scheduler = WindowScheduler()
    probe = MediaProbeAdapter()
    packer = PromptPacker()
    compiler = GraphCompiler()
    optimization_builder = OptimizationPlanBuilder()

    ensure_default_operator_stages(operator_bus)
    started_at = perf_counter()
    runtime_inputs = _prepare_video_runtime_inputs(
        request=request,
        local_media=local_media,
    )
    stage_timings_ms["prepare_inputs_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    clip_start = float(runtime_inputs.get("clip_start_s") or request.clip_start_s or 0.0)
    started_at = perf_counter()
    runtime_clip_end = runtime_inputs.get("clip_end_s")
    if runtime_clip_end is not None:
        clip_end = float(runtime_clip_end)
    else:
        clip_end = _resolve_clip_end(
            request,
            clip_start,
            request.windowing.video_window_seconds,
            runtime_duration=runtime_inputs.get("duration_seconds"),
        )
    optimization_plan = optimization_builder.build_video_plan(
        request=request,
        runtime_inputs=runtime_inputs,
    )
    semantic_windowing = scheduler.resolve_settings(
        modality="video",
        clip_start_s=clip_start,
        clip_end_s=clip_end,
        policy=request.windowing,
    )
    optimization_plan["semantic"].update(
        {
            "effective_video_window_seconds": semantic_windowing["window_size_seconds"],
            "effective_overlap_seconds": semantic_windowing["overlap_seconds"],
            "effective_step_seconds": semantic_windowing["step_seconds"],
            "estimated_window_count": semantic_windowing["estimated_window_count"],
            "adaptive_windowing": bool(semantic_windowing["adaptive"]),
        }
    )
    windows = scheduler.build_windows(
        modality="video",
        clip_start_s=clip_start,
        clip_end_s=clip_end,
        policy=request.windowing,
        resolved_settings=semantic_windowing,
    )
    probe_meta = dict(request.metadata.get("probe_meta") or {})
    if request.windowing.video_fps:
        probe_meta.setdefault("frame_rate", request.windowing.video_fps)
    if runtime_inputs.get("frame_rate"):
        probe_meta.setdefault("frame_rate", runtime_inputs.get("frame_rate"))
    backbone = probe.build_backbone(
        source_id=request.source.source_id,
        probe_meta=probe_meta,
        scenes=windows,
        default_modality="video",
    )
    stage_timings_ms["planning_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    started_at = perf_counter()
    visual_tracks, visual_evidence = _load_or_run_visual_stage(
        request=request,
        operator_bus=operator_bus,
        asset_store=asset_store,
        source_id=request.source.source_id,
        runtime_inputs=runtime_inputs,
        optimization_plan=optimization_plan,
        backbone=backbone,
    )
    stage_timings_ms["visual_stage_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    started_at = perf_counter()
    speaker_tracks, utterances, audio_evidence, speaker_stage_stats = _load_or_run_speaker_stage(
        request=request,
        operator_bus=operator_bus,
        asset_store=asset_store,
        source_id=request.source.source_id,
        runtime_inputs=runtime_inputs,
        optimization_plan=optimization_plan,
        backbone=backbone,
    )
    stage_timings_ms["speaker_stage_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    started_at = perf_counter()
    face_voice_links = _load_or_run_association_stage(
        request=request,
        operator_bus=operator_bus,
        source_id=request.source.source_id,
        optimization_plan=optimization_plan,
        backbone=backbone,
        visual_tracks=visual_tracks,
        speaker_tracks=speaker_tracks,
        utterances=utterances,
        visual_evidence=visual_evidence,
        audio_evidence=audio_evidence,
        runtime_inputs=runtime_inputs,
    )
    stage_timings_ms["association_stage_ms"] = round((perf_counter() - started_at) * 1000.0, 3)
    evidence = [*visual_evidence, *audio_evidence]
    asset_outputs = []
    started_at = perf_counter()
    if visual_tracks and request.visual_policy.persist_artifacts and request.asset_inputs.visual_tracks is None:
        asset_outputs.append(
            asset_store.save_asset(
                asset_id=f"{request.source.source_id}_visual_tracks",
                asset_type="visual_tracks",
                payload={
                    "visual_tracks": [item.model_dump() for item in visual_tracks],
                    "evidence": [item.model_dump() for item in visual_evidence],
                },
                source_id=request.source.source_id,
                created_by_run_id=request.routing.run_id,
                metadata={"stage": VISUAL_TRACK_STAGE},
            )
        )
    if speaker_tracks and request.speaker_policy.persist_artifacts and request.asset_inputs.speaker_tracks is None:
        asset_outputs.append(
            asset_store.save_asset(
                asset_id=f"{request.source.source_id}_speaker_tracks",
                asset_type="speaker_tracks",
                payload={
                    "speaker_tracks": [item.model_dump() for item in speaker_tracks],
                    "utterances": [item.model_dump() for item in utterances],
                    "evidence": [item.model_dump() for item in audio_evidence],
                },
                source_id=request.source.source_id,
                created_by_run_id=request.routing.run_id,
                metadata={"stage": SPEAKER_TRACK_STAGE},
            )
        )
    stage_timings_ms["asset_persist_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    window_payloads = packer.build_video_window_payloads(
        backbone=backbone,
        visual_tracks=visual_tracks,
        speaker_tracks=speaker_tracks,
        face_voice_links=face_voice_links,
        utterances=utterances,
        evidence=evidence,
        clip_frames=runtime_inputs.get("clip_frame_paths") or [],
        face_frames=runtime_inputs.get("face_frame_paths") or [],
        frame_timestamps_s=runtime_inputs.get("frame_timestamps_s") or [],
        max_frames_per_window=int(
            optimization_plan.get("visual", {}).get("max_frames_per_window")
            or 0
        ),
    )
    window_payload_stats = _summarize_window_payloads(window_payloads)
    started_at = perf_counter()
    window_digests = _build_window_digests(
        request=request,
        backbone=backbone,
        window_payloads=window_payloads,
        utterances=utterances,
        visual_tracks=visual_tracks,
        speaker_tracks=speaker_tracks,
        face_voice_links=face_voice_links,
        evidence=evidence,
        operator_bus=operator_bus,
        optimization_plan=optimization_plan,
    )
    stage_timings_ms["semantic_compile_ms"] = round((perf_counter() - started_at) * 1000.0, 3)
    trace = CompileTrace(
        provider=request.provider.provider,
        model=request.provider.model,
        operator_versions={
            "visual_stage": request.visual_operator.version or request.visual_operator.engine or "",
            "speaker_stage": request.speaker_operator.version or request.speaker_operator.engine or "",
            "association_stage": "local_face_voice_association_v1",
        },
        optimization_plan=optimization_plan,
    )
    started_at = perf_counter()
    graph_request = compiler.compile(
        routing=request.routing,
        backbone=backbone,
        window_digests=window_digests,
        visual_tracks=visual_tracks,
        speaker_tracks=speaker_tracks,
        face_voice_links=face_voice_links,
        utterances=utterances,
        evidence=evidence,
        trace=trace,
        source_recorded_at=request.source.recorded_at,
    )
    stage_timings_ms["graph_compile_ms"] = round((perf_counter() - started_at) * 1000.0, 3)
    status = "compiled"
    if request.write_graph:
        if graph_writer is not None:
            graph_writer(graph_request)
            status = "written"
        else:
            trace.warnings.append(
                "write_graph requested but no graph_writer provided; returning compiled graph only"
            )
    stage_timings_ms["total_ms"] = round((perf_counter() - total_started_at) * 1000.0, 3)
    return CompileResult(
        status=status,
        source_id=request.source.source_id,
        window_digests=window_digests,
        visual_tracks=visual_tracks,
        speaker_tracks=speaker_tracks,
        face_voice_links=face_voice_links,
        utterances=utterances,
        evidence=evidence,
        asset_outputs=asset_outputs,
        graph_request=graph_request,
        trace=trace,
        stats={
            "segments": len(backbone.segments),
            "visual_tracks": len(visual_tracks),
            "speaker_tracks": len(speaker_tracks),
            "face_voice_links": len(face_voice_links),
            "utterances": len(utterances),
            "events": len(graph_request.events) if graph_request else 0,
            "stage_timings_ms": stage_timings_ms,
            "speaker_stage": speaker_stage_stats,
            "optimization": optimization_plan,
            "semantic_windowing": semantic_windowing,
            "semantic_windows": window_payload_stats,
        },
    )


def _resolve_clip_end(
    request: CompileVideoRequest,
    clip_start: float,
    default_window: float,
    runtime_duration: float | None = None,
) -> float:
    if request.clip_end_s is not None:
        return float(request.clip_end_s)
    duration = request.metadata.get("duration_seconds")
    if duration is not None:
        duration_value = float(duration)
        if duration_value > 0.0:
            return clip_start + duration_value
    if runtime_duration is not None:
        return clip_start + float(runtime_duration)
    return clip_start + default_window


def _summarize_window_payloads(window_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    vector_dims = [
        int(
            ((payload.get("segment_visual_profile") or {}).get("vector_summary") or {}).get("dim")
            or 0
        )
        for payload in window_payloads
    ]
    representative_counts = [len(payload.get("representative_frames") or []) for payload in window_payloads]
    return {
        "windows": len(window_payloads),
        "representative_frames_total": sum(representative_counts),
        "representative_frames_max": max(representative_counts, default=0),
        "image_vector_windows": sum(1 for dim in vector_dims if dim > 0),
        "image_vector_dim_max": max(vector_dims, default=0),
    }


def _load_or_run_visual_stage(*, request: CompileVideoRequest, operator_bus, asset_store: LocalOperatorAssetStore, source_id: str, runtime_inputs: Dict[str, Any], optimization_plan: Dict[str, Any], backbone):
    adapter = VisualTrackStageAdapter()
    if request.asset_inputs.visual_tracks is not None:
        payload = asset_store.load_asset(request.asset_inputs.visual_tracks)
        return adapter.normalize(source_id=source_id, stage_output=payload)
    if not request.enable_visual_operator:
        return [], []
    operator = _resolve_operator(operator_bus, VISUAL_TRACK_STAGE, LEGACY_VISUAL_STAGE)
    if operator is None:
        return [], []
    payload = operator(
        {
            "request": request,
            "source_id": source_id,
            "stage": VISUAL_TRACK_STAGE,
            "optimization_plan": optimization_plan,
            "backbone_segments": backbone.segments,
            **runtime_inputs,
        }
    ) or {}
    return adapter.normalize(source_id=source_id, stage_output=payload)


def _load_or_run_speaker_stage(*, request: CompileVideoRequest, operator_bus, asset_store: LocalOperatorAssetStore, source_id: str, runtime_inputs: Dict[str, Any], optimization_plan: Dict[str, Any], backbone):
    adapter = AudioOperatorAdapter()
    if request.asset_inputs.speaker_tracks is not None:
        payload = asset_store.load_asset(request.asset_inputs.speaker_tracks)
        return (*adapter.normalize(source_id=source_id, stage_output=payload), adapter.extract_stage_stats(stage_output=payload))
    if not request.enable_audio_operator:
        return [], [], [], {}
    operator = _resolve_operator(operator_bus, SPEAKER_TRACK_STAGE, LEGACY_SPEAKER_STAGE)
    if operator is None:
        return [], [], [], {}
    payload = operator(
        {
            "request": request,
            "source_id": source_id,
            "stage": SPEAKER_TRACK_STAGE,
            "optimization_plan": optimization_plan,
            "backbone_segments": backbone.segments,
            **runtime_inputs,
        }
    ) or {}
    return (*adapter.normalize(source_id=source_id, stage_output=payload), adapter.extract_stage_stats(stage_output=payload))


def _build_window_digests(
    *,
    request: CompileVideoRequest,
    backbone,
    window_payloads,
    utterances,
    visual_tracks,
    speaker_tracks,
    face_voice_links,
    evidence,
    operator_bus,
    optimization_plan,
) -> List[WindowDigest]:
    semantic_operator = _resolve_operator(
        operator_bus,
        SEMANTIC_COMPILE_STAGE,
        LEGACY_SEMANTIC_STAGE,
    )
    if semantic_operator is not None:
        payload = semantic_operator(
            {
                "request": request,
                "backbone": backbone,
                "window_payloads": window_payloads,
                "utterances": utterances,
                "visual_tracks": visual_tracks,
                "speaker_tracks": speaker_tracks,
                "face_voice_links": face_voice_links,
                "evidence": evidence,
                "optimization_plan": optimization_plan,
            }
        ) or {}
        digests = payload.get("window_digests") or []
        if digests:
            return [WindowDigest.model_validate(item) for item in digests]

    utterances_by_window: Dict[str, List[str]] = {}
    for segment in backbone.segments:
        utterances_by_window[segment.id] = [
            item.text
            for item in utterances
            if segment.t_media_start <= item.t_start_s <= segment.t_media_end
        ]
    digests: List[WindowDigest] = []
    for segment in backbone.segments:
        participant_refs = [
            item.track_id
            for item in visual_tracks
            if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
        ]
        participant_refs.extend(
            item.track_id
            for item in speaker_tracks
            if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
        )
        evidence_refs = [
            item.evidence_id
            for item in evidence
            if item.t_start_s is None
            or (segment.t_media_start <= item.t_start_s <= segment.t_media_end)
        ]
        summary_bits = utterances_by_window.get(segment.id) or []
        summary = " ".join(summary_bits[:2]) if summary_bits else f"video activity in {segment.t_media_start:.1f}-{segment.t_media_end:.1f}s"
        digests.append(
            WindowDigest(
                window_id=stable_window_id(
                    request.source.source_id,
                    segment.t_media_start,
                    segment.t_media_end,
                    "video",
                ),
                modality="video",
                t_start_s=segment.t_media_start,
                t_end_s=segment.t_media_end,
                summary=summary,
                participant_refs=participant_refs,
                evidence_refs=evidence_refs,
            )
        )
    return digests


def _resolve_operator(operator_bus, canonical_name: str, legacy_name: str):
    return operator_bus.get(canonical_name) or operator_bus.get(legacy_name)


def _load_or_run_association_stage(
    *,
    request: CompileVideoRequest,
    operator_bus,
    source_id: str,
    optimization_plan: Dict[str, Any],
    backbone,
    visual_tracks,
    speaker_tracks,
    utterances,
    visual_evidence,
    audio_evidence,
    runtime_inputs: Dict[str, Any],
) -> List[FaceVoiceLinkRecord]:
    if not bool(request.identity.enable_cross_modal_association):
        return []
    operator = _resolve_operator(
        operator_bus,
        FACE_VOICE_ASSOCIATION_STAGE,
        LEGACY_ASSOCIATION_STAGE,
    )
    if operator is None:
        return []
    payload = operator(
        {
            "request": request,
            "source_id": source_id,
            "stage": FACE_VOICE_ASSOCIATION_STAGE,
            "optimization_plan": optimization_plan,
            "backbone_segments": backbone.segments,
            "visual_tracks": [item.model_dump() for item in visual_tracks],
            "speaker_tracks": [item.model_dump() for item in speaker_tracks],
            "utterances": [item.model_dump() for item in utterances],
            "visual_evidence": [item.model_dump() for item in visual_evidence],
            "audio_evidence": [item.model_dump() for item in audio_evidence],
            **runtime_inputs,
        }
    ) or {}
    links = payload.get("face_voice_links") or []
    if not links:
        return []
    return [FaceVoiceLinkRecord.model_validate(item) for item in links]


def _prepare_video_runtime_inputs(
    *,
    request: CompileVideoRequest,
    local_media: LocalMediaPipeline,
) -> Dict[str, Any]:
    runtime_inputs: Dict[str, Any] = {}
    duration = request.metadata.get("duration_seconds")
    if duration is not None:
        duration_value = float(duration)
        if duration_value > 0.0:
            runtime_inputs["duration_seconds"] = duration_value
    file_path = request.source.file_path
    if not file_path:
        return runtime_inputs
    artifacts_dir = request.metadata.get("artifacts_dir") or ".artifacts/media_graph_compiler"
    sample_fps = float(request.visual_policy.sample_fps or 0.0)
    if sample_fps <= 0.0:
        sample_fps = float(request.windowing.video_fps or 1.0)
    prepared = local_media.prepare_video_inputs(
        file_path=file_path,
        artifacts_dir=artifacts_dir,
        clip_start_s=float(request.clip_start_s or 0.0),
        clip_end_s=(float(request.clip_end_s) if request.clip_end_s is not None else None),
        requested_duration_s=runtime_inputs.get("duration_seconds"),
        sample_fps=sample_fps,
        clip_px=int(request.metadata.get("clip_resize_px") or 256),
        face_px=int(request.metadata.get("face_resize_px") or 640),
        enable_dedup=bool(request.optimization.enable_visual_dedup),
        similarity_threshold=int(request.optimization.visual_similarity_threshold),
        max_frames_per_source=int(request.optimization.max_visual_frames_per_source),
    )
    runtime_inputs.update(prepared)
    return runtime_inputs
