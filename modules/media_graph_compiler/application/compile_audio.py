from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, List

from modules.media_graph_compiler.adapters import (
    AudioOperatorAdapter,
    LocalOperatorAssetStore,
    MediaNormalizer,
    MediaProbeAdapter,
    default_operator_bus,
)
from modules.media_graph_compiler.adapters.default_operator_stages import (
    ensure_default_operator_stages,
)
from modules.media_graph_compiler.application.graph_compiler import GraphCompiler
from modules.media_graph_compiler.application.optimization_plan import OptimizationPlanBuilder
from modules.media_graph_compiler.application.prompt_packer import PromptPacker
from modules.media_graph_compiler.application.stage_names import (
    LEGACY_SEMANTIC_STAGE,
    LEGACY_SPEAKER_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
)
from modules.media_graph_compiler.application.window_scheduler import WindowScheduler
from modules.media_graph_compiler.domain import stable_window_id
from modules.media_graph_compiler.types import (
    CompileAudioRequest,
    CompileResult,
    CompileTrace,
    WindowDigest,
)


def compile_audio(
    request: CompileAudioRequest,
    *,
    operator_bus=default_operator_bus,
    asset_store: LocalOperatorAssetStore | None = None,
    graph_writer: Callable[[object], None] | None = None,
) -> CompileResult:
    total_started_at = perf_counter()
    stage_timings_ms: Dict[str, float] = {}
    asset_store = asset_store or LocalOperatorAssetStore(
        request.metadata.get("artifacts_dir")
    )
    normalizer = MediaNormalizer()
    scheduler = WindowScheduler()
    probe = MediaProbeAdapter()
    packer = PromptPacker()
    compiler = GraphCompiler()
    optimization_builder = OptimizationPlanBuilder()

    ensure_default_operator_stages(operator_bus)
    clip_start = float(request.clip_start_s or 0.0)
    started_at = perf_counter()
    runtime_inputs = _prepare_audio_runtime_inputs(
        request=request,
        normalizer=normalizer,
    )
    stage_timings_ms["prepare_inputs_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    started_at = perf_counter()
    clip_end = _resolve_clip_end(
        request,
        clip_start,
        request.windowing.audio_window_seconds,
        runtime_duration=runtime_inputs.get("duration_seconds"),
    )
    windows = scheduler.build_windows(
        modality="audio",
        clip_start_s=clip_start,
        clip_end_s=clip_end,
        policy=request.windowing,
    )
    probe_meta = dict(request.metadata.get("probe_meta") or {})
    backbone = probe.build_backbone(
        source_id=request.source.source_id,
        probe_meta=probe_meta,
        scenes=windows,
        default_modality="audio",
    )
    optimization_plan = optimization_builder.build_audio_plan(
        request=request,
        runtime_inputs=runtime_inputs,
    )
    stage_timings_ms["planning_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    started_at = perf_counter()
    speaker_tracks, utterances, audio_evidence, speaker_stage_stats = _load_or_run_speaker_stage(
        request=request,
        operator_bus=operator_bus,
        asset_store=asset_store,
        source_id=request.source.source_id,
        runtime_inputs=runtime_inputs,
        optimization_plan=optimization_plan,
    )
    stage_timings_ms["speaker_stage_ms"] = round((perf_counter() - started_at) * 1000.0, 3)
    asset_outputs = []
    started_at = perf_counter()
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

    started_at = perf_counter()
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
                "window_payloads": packer.build_audio_window_payloads(
                    backbone=backbone,
                    speaker_tracks=speaker_tracks,
                    utterances=utterances,
                    evidence=audio_evidence,
                ),
                "utterances": utterances,
                "speaker_tracks": speaker_tracks,
                "evidence": audio_evidence,
                "optimization_plan": optimization_plan,
            }
        ) or {}
        digests = payload.get("window_digests") or []
        window_digests = [WindowDigest.model_validate(item) for item in digests]
    else:
        window_digests = _build_window_digests(request, backbone, utterances, speaker_tracks, audio_evidence)
    stage_timings_ms["semantic_compile_ms"] = round((perf_counter() - started_at) * 1000.0, 3)

    trace = CompileTrace(
        provider=request.provider.provider,
        model=request.provider.model,
        operator_versions={
            "speaker_stage": request.speaker_operator.version or request.speaker_operator.engine or "",
        },
        optimization_plan=optimization_plan,
    )
    started_at = perf_counter()
    graph_request = compiler.compile(
        routing=request.routing,
        backbone=backbone,
        window_digests=window_digests,
        visual_tracks=[],
        speaker_tracks=speaker_tracks,
        face_voice_links=[],
        utterances=utterances,
        evidence=audio_evidence,
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
        speaker_tracks=speaker_tracks,
        utterances=utterances,
        evidence=audio_evidence,
        asset_outputs=asset_outputs,
        graph_request=graph_request,
        trace=trace,
        stats={
            "segments": len(backbone.segments),
            "speaker_tracks": len(speaker_tracks),
            "utterances": len(utterances),
            "events": len(graph_request.events) if graph_request else 0,
            "stage_timings_ms": stage_timings_ms,
            "speaker_stage": speaker_stage_stats,
            "optimization": optimization_plan,
        },
    )


def _resolve_clip_end(
    request: CompileAudioRequest,
    clip_start: float,
    default_window: float,
    runtime_duration: float | None = None,
) -> float:
    if request.clip_end_s is not None:
        return float(request.clip_end_s)
    duration = request.metadata.get("duration_seconds")
    if duration is not None:
        return clip_start + float(duration)
    if runtime_duration is not None:
        return clip_start + float(runtime_duration)
    return clip_start + default_window


def _load_or_run_speaker_stage(*, request: CompileAudioRequest, operator_bus, asset_store: LocalOperatorAssetStore, source_id: str, runtime_inputs: Dict[str, object], optimization_plan: Dict[str, object]):
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
            **runtime_inputs,
        }
    ) or {}
    return (*adapter.normalize(source_id=source_id, stage_output=payload), adapter.extract_stage_stats(stage_output=payload))


def _build_window_digests(request: CompileAudioRequest, backbone, utterances, speaker_tracks, evidence):
    digests: List[WindowDigest] = []
    for segment in backbone.segments:
        digest_utterances = [
            item.text for item in utterances if segment.t_media_start <= item.t_start_s <= segment.t_media_end
        ]
        summary = " ".join(digest_utterances[:2]) if digest_utterances else f"audio activity in {segment.t_media_start:.1f}-{segment.t_media_end:.1f}s"
        participant_refs = [
            item.track_id
            for item in speaker_tracks
            if item.t_start_s <= segment.t_media_end and item.t_end_s >= segment.t_media_start
        ]
        evidence_refs = [
            item.evidence_id
            for item in evidence
            if item.t_start_s is None
            or (segment.t_media_start <= item.t_start_s <= segment.t_media_end)
        ]
        digests.append(
            WindowDigest(
                window_id=stable_window_id(
                    request.source.source_id,
                    segment.t_media_start,
                    segment.t_media_end,
                    "audio",
                ),
                modality="audio",
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


def _prepare_audio_runtime_inputs(
    *,
    request: CompileAudioRequest,
    normalizer: MediaNormalizer,
) -> Dict[str, object]:
    runtime_inputs: Dict[str, object] = {}
    if request.metadata.get("duration_seconds") is not None:
        runtime_inputs["duration_seconds"] = float(request.metadata["duration_seconds"])
    file_path = request.source.file_path
    if not file_path:
        return runtime_inputs
    runtime_inputs["normalized_audio_path"] = file_path
    try:
        probe = normalizer.probe_media(file_path)
        runtime_inputs.update({k: v for k, v in probe.items() if v is not None})
    except Exception:
        pass
    return runtime_inputs
