from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

from modules.media_graph_compiler import (
    CompileAssetInputs,
    CompileAudioRequest,
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
    compile_audio,
    compile_video,
)
from modules.media_graph_compiler.adapters import (
    AudioFromVideoAdapter,
    FrameSelector,
    LocalOperatorAssetStore,
    OperatorBus,
)
from modules.media_graph_compiler.application import (
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)


def _routing(run_id: str, memory_domain: str) -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="local_bench",
        user_id=["u:dry_run"],
        memory_domain=memory_domain,
        run_id=run_id,
    )


def _visual_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    duration = float(ctx.get("duration_seconds") or 8.0)
    optimization = dict(ctx.get("optimization_plan") or {})
    visual_plan = dict(optimization.get("visual") or {})
    selector = FrameSelector()
    sample_fps = float(visual_plan.get("sample_fps") or 2.0)
    raw_frame_count = max(int(duration * max(sample_fps, 1.0) * 2), 6)
    synthetic_frames = [f"frame_group_{index // 2}" for index in range(raw_frame_count)]
    selection = selector.select_indices(
        synthetic_frames,
        enable_dedup=bool(visual_plan.get("enable_frame_dedup", True)),
        similarity_threshold=int(visual_plan.get("similarity_threshold", 5)),
        max_frames=int(visual_plan.get("max_frames_per_window", 12)),
    )
    selected_frame_count = max(len(selection.kept_indices), 1)
    track_count = min(2 if duration >= 10.0 else 1, selected_frame_count)
    window = max(duration / max(track_count, 1), 1.0)
    visual_tracks: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for index in range(track_count):
        start = float(index) * window
        end = min(duration, start + window * 0.85)
        track_id = f"face_{index + 1}"
        evidence_id = f"{track_id}_mask_0001"
        visual_tracks.append(
            {
                "track_id": track_id,
                "category": "person",
                "t_start_s": start,
                "t_end_s": end,
                "frame_start": int(start),
                "frame_end": int(end),
                "evidence_refs": [evidence_id],
                "metadata": {
                    "normalized_video_path": ctx.get("normalized_video_path"),
                    "raw_frame_count": raw_frame_count,
                    "selected_frame_count": selected_frame_count,
                    "dropped_frame_count": len(selection.dropped_indices),
                    "selected_frame_indices": selection.kept_indices,
                },
            }
        )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "kind": "mask",
                "t_start_s": start,
                "t_end_s": min(duration, start + 0.5),
                "metadata": {
                    "runtime": "dry_visual_stage",
                    "frame_index": int(start),
                },
            }
        )
    return {"visual_tracks": visual_tracks, "evidence": evidence}


def _speaker_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    duration = float(ctx.get("duration_seconds") or 8.0)
    optimization = dict(ctx.get("optimization_plan") or {})
    audio_plan = dict(optimization.get("audio") or {})
    track_count = 2 if duration >= 10.0 else 1
    turn_step = max(duration / max(track_count * 2, 1), 0.5)
    raw_turn_templates = [0.2, 0.7, 1.1, 0.3]
    min_turn_length = float(audio_plan.get("min_turn_length_s") or 0.4)
    synthetic_rtf = 1.5
    fallback_model = audio_plan.get("fallback_asr_model") or "dry_asr_compact"
    model_used = "dry_asr_default"
    if bool(audio_plan.get("enable_asr_rtf_adaptation", True)) and synthetic_rtf > float(
        audio_plan.get("asr_rtf_threshold") or 1.25
    ):
        model_used = str(fallback_model)
    speaker_tracks: List[Dict[str, Any]] = []
    utterances: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []
    for track_index in range(track_count):
        track_id = f"voice_{track_index + 1}"
        utterance_ids: List[str] = []
        evidence_refs: List[str] = []
        kept_turn_count = 0
        dropped_short_turns = 0
        for turn_index, nominal_duration in enumerate(raw_turn_templates):
            start = float(track_index * 2 + turn_index) * turn_step
            end = min(duration, start + nominal_duration)
            actual_duration = max(end - start, 0.0)
            if bool(audio_plan.get("enable_vad", True)) and actual_duration < min_turn_length:
                dropped_short_turns += 1
                continue
            utt_id = f"{track_id}_utt_{turn_index + 1}"
            ev_id = f"{track_id}_audio_{turn_index + 1}"
            utterance_ids.append(utt_id)
            evidence_refs.append(ev_id)
            kept_turn_count += 1
            utterances.append(
                {
                    "utterance_id": utt_id,
                    "speaker_track_id": track_id,
                    "t_start_s": start,
                    "t_end_s": end,
                    "text": f"{track_id} says turn {turn_index + 1}",
                    "evidence_refs": [ev_id],
                    "metadata": {
                        "normalized_audio_path": ctx.get("normalized_audio_path") or ctx.get("extracted_audio_path"),
                        "asr_model": model_used,
                        "synthetic_rtf": synthetic_rtf,
                    },
                }
            )
            evidence.append(
                {
                    "evidence_id": ev_id,
                    "kind": "audio_chunk",
                    "t_start_s": start,
                    "t_end_s": end,
                    "metadata": {
                        "runtime": "dry_speaker_stage",
                        "transcript": f"{track_id} says turn {turn_index + 1}",
                        "asr_model": model_used,
                    },
                }
            )
        if not utterance_ids:
            continue
        speaker_tracks.append(
            {
                "track_id": track_id,
                "t_start_s": min(item["t_start_s"] for item in utterances if item["speaker_track_id"] == track_id),
                "t_end_s": max(item["t_end_s"] for item in utterances if item["speaker_track_id"] == track_id),
                "utterance_ids": utterance_ids,
                "evidence_refs": evidence_refs,
                "metadata": {
                    "raw_turn_count": len(raw_turn_templates),
                    "kept_turn_count": kept_turn_count,
                    "dropped_short_turn_count": dropped_short_turns,
                    "asr_model": model_used,
                    "synthetic_rtf": synthetic_rtf,
                },
            }
        )
    return {
        "speaker_tracks": speaker_tracks,
        "utterances": utterances,
        "evidence": evidence,
    }


def _semantic_stage(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    optimization = dict(ctx.get("optimization_plan") or {})
    digests: List[Dict[str, Any]] = []
    for index, payload in enumerate(ctx.get("window_payloads") or []):
        utterances = payload.get("utterances") or []
        visual_tracks = payload.get("visual_tracks") or []
        speaker_tracks = payload.get("speaker_tracks") or []
        evidence = payload.get("evidence") or []
        summary_parts: List[str] = []
        if utterances:
            summary_parts.append("; ".join(item["text"] for item in utterances[:2]))
        if visual_tracks:
            summary_parts.append(f"visual={len(visual_tracks)}")
        if speaker_tracks:
            summary_parts.append(f"speaker={len(speaker_tracks)}")
        summary = " | ".join(summary_parts) if summary_parts else f"window_{index}"
        participant_refs = [item["track_id"] for item in visual_tracks]
        participant_refs.extend(item["track_id"] for item in speaker_tracks)
        digests.append(
            {
                "window_id": payload["window_id"],
                "modality": payload["modality"],
                "t_start_s": payload["t_start_s"],
                "t_end_s": payload["t_end_s"],
                "summary": summary,
                "participant_refs": participant_refs,
                "evidence_refs": [item["evidence_id"] for item in evidence],
                "semantic_payload": {
                    "window_index": index,
                    "utterance_count": len(utterances),
                    "visual_count": len(visual_tracks),
                    "speaker_count": len(speaker_tracks),
                    "optimization": {
                        "allow_frame_bundle": bool(
                            (optimization.get("semantic") or {}).get("allow_frame_bundle", False)
                        ),
                        "allow_realtime_stream": bool(
                            (optimization.get("semantic") or {}).get("allow_realtime_stream", False)
                        ),
                    },
                },
            }
        )
    return {"window_digests": digests}


def _timed(name: str, sink: Dict[str, List[float]], func: Callable[[Mapping[str, Any]], Dict[str, Any]]):
    def wrapped(ctx: Mapping[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            return func(ctx)
        finally:
            sink.setdefault(name, []).append((time.perf_counter() - start) * 1000.0)

    return wrapped


def _summarize_result(result) -> Dict[str, Any]:
    graph = result.graph_request
    visual_meta = result.visual_tracks[0].metadata if result.visual_tracks else {}
    speaker_meta = result.speaker_tracks[0].metadata if result.speaker_tracks else {}
    return {
        "status": result.status,
        "window_count": len(result.window_digests),
        "visual_track_count": len(result.visual_tracks),
        "speaker_track_count": len(result.speaker_tracks),
        "utterance_count": len(result.utterances),
        "evidence_count": len(result.evidence),
        "asset_output_count": len(result.asset_outputs),
        "graph_event_count": len(graph.events) if graph else 0,
        "graph_entity_count": len(graph.entities) if graph else 0,
        "graph_edge_count": len(graph.edges) if graph else 0,
        "visual_raw_frame_count": int(visual_meta.get("raw_frame_count") or 0),
        "visual_selected_frame_count": int(visual_meta.get("selected_frame_count") or 0),
        "visual_dropped_frame_count": int(visual_meta.get("dropped_frame_count") or 0),
        "audio_raw_turn_count": int(speaker_meta.get("raw_turn_count") or 0),
        "audio_kept_turn_count": int(speaker_meta.get("kept_turn_count") or 0),
        "audio_dropped_short_turn_count": int(speaker_meta.get("dropped_short_turn_count") or 0),
        "audio_asr_model": speaker_meta.get("asr_model"),
        "optimization_plan": dict(result.trace.optimization_plan or {}),
        "stats": dict(result.stats),
    }


def _mean(values: Sequence[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def _extract_audio_fixture(video_path: str, output_dir: Path) -> str | None:
    adapter = AudioFromVideoAdapter()
    return adapter.extract_wav(video_path, output_dir=output_dir / "audio_fixture")


def run_video_benchmark(video_path: str, output_dir: Path, repeat: int) -> Dict[str, Any]:
    cold_runs: List[Dict[str, Any]] = []
    replay_runs: List[Dict[str, Any]] = []
    latest_assets = None
    latest_request = None
    latest_result = None

    for index in range(repeat):
        timing: Dict[str, List[float]] = {}
        bus = OperatorBus()
        bus.register(VISUAL_TRACK_STAGE, _timed(VISUAL_TRACK_STAGE, timing, _visual_stage))
        bus.register(SPEAKER_TRACK_STAGE, _timed(SPEAKER_TRACK_STAGE, timing, _speaker_stage))
        bus.register(SEMANTIC_COMPILE_STAGE, _timed(SEMANTIC_COMPILE_STAGE, timing, _semantic_stage))
        artifacts_dir = output_dir / f"video_run_{index + 1}"
        request = CompileVideoRequest(
            routing=_routing(run_id=f"video-bench-{index + 1}", memory_domain="media_video_bench"),
            source=MediaSourceRef(source_id=f"video_src_{index + 1}", file_path=video_path),
        )
        start = time.perf_counter()
        result = compile_video(
            request,
            operator_bus=bus,
            asset_store=LocalOperatorAssetStore(artifacts_dir),
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        stage_ms = sum(_mean(values) for values in timing.values())
        cold_runs.append(
            {
                "total_ms": round(total_ms, 3),
                "stage_ms": {name: _mean(values) for name, values in timing.items()},
                "orchestration_ms": round(max(total_ms - stage_ms, 0.0), 3),
                "result": _summarize_result(result),
            }
        )
        latest_assets = result.asset_outputs
        latest_request = request
        latest_result = result

    if latest_assets and latest_request:
        for index in range(repeat):
            timing: Dict[str, List[float]] = {}
            bus = OperatorBus()
            bus.register(SEMANTIC_COMPILE_STAGE, _timed(SEMANTIC_COMPILE_STAGE, timing, _semantic_stage))
            replay_request = latest_request.model_copy(
                update={
                    "routing": _routing(run_id=f"video-replay-{index + 1}", memory_domain="media_video_bench"),
                    "asset_inputs": CompileAssetInputs(
                        visual_tracks=latest_assets[0] if len(latest_assets) > 0 else None,
                        speaker_tracks=latest_assets[1] if len(latest_assets) > 1 else None,
                    ),
                    "enable_visual_operator": False,
                    "enable_audio_operator": False,
                }
            )
            start = time.perf_counter()
            replay_result = compile_video(
                replay_request,
                operator_bus=bus,
                asset_store=LocalOperatorAssetStore(output_dir / f"video_replay_{index + 1}"),
            )
            total_ms = (time.perf_counter() - start) * 1000.0
            stage_ms = sum(_mean(values) for values in timing.values())
            replay_runs.append(
                {
                    "total_ms": round(total_ms, 3),
                    "stage_ms": {name: _mean(values) for name, values in timing.items()},
                    "orchestration_ms": round(max(total_ms - stage_ms, 0.0), 3),
                    "result": _summarize_result(replay_result),
                }
            )

    return {
        "mode": "video",
        "input_path": video_path,
        "repeat": repeat,
        "cold_runs": cold_runs,
        "replay_runs": replay_runs,
        "latest_result": _summarize_result(latest_result) if latest_result else {},
    }


def run_audio_benchmark(audio_path: str, output_dir: Path, repeat: int) -> Dict[str, Any]:
    cold_runs: List[Dict[str, Any]] = []
    replay_runs: List[Dict[str, Any]] = []
    latest_assets = None
    latest_request = None
    latest_result = None

    for index in range(repeat):
        timing: Dict[str, List[float]] = {}
        bus = OperatorBus()
        bus.register(SPEAKER_TRACK_STAGE, _timed(SPEAKER_TRACK_STAGE, timing, _speaker_stage))
        bus.register(SEMANTIC_COMPILE_STAGE, _timed(SEMANTIC_COMPILE_STAGE, timing, _semantic_stage))
        artifacts_dir = output_dir / f"audio_run_{index + 1}"
        request = CompileAudioRequest(
            routing=_routing(run_id=f"audio-bench-{index + 1}", memory_domain="media_audio_bench"),
            source=MediaSourceRef(source_id=f"audio_src_{index + 1}", file_path=audio_path),
        )
        start = time.perf_counter()
        result = compile_audio(
            request,
            operator_bus=bus,
            asset_store=LocalOperatorAssetStore(artifacts_dir),
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        stage_ms = sum(_mean(values) for values in timing.values())
        cold_runs.append(
            {
                "total_ms": round(total_ms, 3),
                "stage_ms": {name: _mean(values) for name, values in timing.items()},
                "orchestration_ms": round(max(total_ms - stage_ms, 0.0), 3),
                "result": _summarize_result(result),
            }
        )
        latest_assets = result.asset_outputs
        latest_request = request
        latest_result = result

    if latest_assets and latest_request:
        for index in range(repeat):
            timing: Dict[str, List[float]] = {}
            bus = OperatorBus()
            bus.register(SEMANTIC_COMPILE_STAGE, _timed(SEMANTIC_COMPILE_STAGE, timing, _semantic_stage))
            replay_request = latest_request.model_copy(
                update={
                    "routing": _routing(run_id=f"audio-replay-{index + 1}", memory_domain="media_audio_bench"),
                    "asset_inputs": CompileAssetInputs(
                        speaker_tracks=latest_assets[0] if latest_assets else None,
                    ),
                    "enable_audio_operator": False,
                }
            )
            start = time.perf_counter()
            replay_result = compile_audio(
                replay_request,
                operator_bus=bus,
                asset_store=LocalOperatorAssetStore(output_dir / f"audio_replay_{index + 1}"),
            )
            total_ms = (time.perf_counter() - start) * 1000.0
            stage_ms = sum(_mean(values) for values in timing.values())
            replay_runs.append(
                {
                    "total_ms": round(total_ms, 3),
                    "stage_ms": {name: _mean(values) for name, values in timing.items()},
                    "orchestration_ms": round(max(total_ms - stage_ms, 0.0), 3),
                    "result": _summarize_result(replay_result),
                }
            )

    return {
        "mode": "audio",
        "input_path": audio_path,
        "repeat": repeat,
        "cold_runs": cold_runs,
        "replay_runs": replay_runs,
        "latest_result": _summarize_result(latest_result) if latest_result else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run benchmark for media_graph_compiler.")
    parser.add_argument("--video", default="reference/sam3/assets/videos/bedroom.mp4")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--mode", choices=["video", "audio", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="modules/media_graph_compiler/outputs/dry_run_benchmark",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "repeat": args.repeat,
        "reports": [],
    }

    if args.mode in {"video", "both"}:
        report["reports"].append(run_video_benchmark(args.video, output_dir, args.repeat))

    audio_path = args.audio
    if args.mode in {"audio", "both"} and audio_path is None and args.video:
        audio_path = _extract_audio_fixture(args.video, output_dir)

    if args.mode in {"audio", "both"} and audio_path:
        report["reports"].append(run_audio_benchmark(audio_path, output_dir, args.repeat))

    out_path = output_dir / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark report written to {out_path}")
    for item in report["reports"]:
        cold_ms = [run["total_ms"] for run in item["cold_runs"]]
        replay_ms = [run["total_ms"] for run in item["replay_runs"]]
        print(
            f"[{item['mode']}] cold_mean_ms={_mean(cold_ms)} replay_mean_ms={_mean(replay_ms)} "
            f"windows={item['latest_result'].get('window_count', 0)} "
            f"events={item['latest_result'].get('graph_event_count', 0)} "
            f"selected_frames={item['latest_result'].get('visual_selected_frame_count', 0)} "
            f"dropped_turns={item['latest_result'].get('audio_dropped_short_turn_count', 0)}"
        )


if __name__ == "__main__":
    main()
