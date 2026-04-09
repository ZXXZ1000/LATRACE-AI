from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from modules.media_graph_compiler import (
    CompileAudioRequest,
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
    compile_audio,
    compile_video,
)
from modules.media_graph_compiler.adapters import (
    LocalOperatorAssetStore,
    OperatorBus,
)
from modules.media_graph_compiler.adapters.default_operator_stages import (
    ensure_default_operator_stages,
)
from modules.media_graph_compiler.application import (
    FACE_VOICE_ASSOCIATION_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)


def _routing() -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="tenant_local",
        user_id=["u:demo"],
        memory_domain="media",
        run_id="run_showcase",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def _window_payload_summary(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for payload in payloads:
        summary.append(
            {
                "window_id": payload.get("window_id"),
                "modality": payload.get("modality"),
                "t_start_s": payload.get("t_start_s"),
                "t_end_s": payload.get("t_end_s"),
                "clip_frames": len(payload.get("clip_frames") or []),
                "face_frames": len(payload.get("face_frames") or []),
                "visual_tracks": len(payload.get("visual_tracks") or []),
                "speaker_tracks": len(payload.get("speaker_tracks") or []),
                "face_voice_links": len(payload.get("face_voice_links") or []),
                "utterances": len(payload.get("utterances") or []),
                "evidence": len(payload.get("evidence") or []),
            }
        )
    return summary


def _ctx_summary(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": ctx.get("source_id"),
        "stage": ctx.get("stage"),
        "normalized_video_path": ctx.get("normalized_video_path"),
        "normalized_audio_path": ctx.get("normalized_audio_path"),
        "extracted_audio_path": ctx.get("extracted_audio_path"),
        "duration_seconds": ctx.get("duration_seconds"),
        "clip_frames": len(ctx.get("clip_frame_paths") or []),
        "face_frames": len(ctx.get("face_frame_paths") or []),
        "frame_timestamps_s": len(ctx.get("frame_timestamps_s") or []),
        "backbone_segments": len(ctx.get("backbone_segments") or []),
        "window_payloads": _window_payload_summary(list(ctx.get("window_payloads") or [])),
        "speaker_tracks": len(ctx.get("speaker_tracks") or []),
        "face_voice_links": len(ctx.get("face_voice_links") or []),
        "visual_tracks": len(ctx.get("visual_tracks") or []),
        "utterances": len(ctx.get("utterances") or []),
        "evidence": len(ctx.get("evidence") or []),
        "optimization_plan": ctx.get("optimization_plan") or {},
        "provider": {
            "provider": getattr(getattr(ctx.get("request"), "provider", None), "provider", None),
            "model": getattr(getattr(ctx.get("request"), "provider", None), "model", None),
        },
    }


def _wrap_stage(name: str, fn, dump_dir: Path):
    def _wrapped(ctx: Mapping[str, Any]) -> Dict[str, Any]:
        result = fn(ctx)
        out_path = dump_dir / f"{name}.json"
        out_path.write_text(
            json.dumps(
                {
                    "input_summary": _ctx_summary(ctx),
                    "output": result,
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        print(f"[stage] {name}")
        print(
            json.dumps(
                {
                    "input": _ctx_summary(ctx),
                    "output_keys": sorted((result or {}).keys()),
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
        )
        return result

    return _wrapped


def _build_bus(output_dir: Path) -> OperatorBus:
    bus = OperatorBus()
    ensure_default_operator_stages(bus)
    dump_dir = output_dir / "stages"
    dump_dir.mkdir(parents=True, exist_ok=True)
    for stage_name in (
        VISUAL_TRACK_STAGE,
        SPEAKER_TRACK_STAGE,
        FACE_VOICE_ASSOCIATION_STAGE,
        SEMANTIC_COMPILE_STAGE,
    ):
        fn = bus.get(stage_name)
        if fn is not None:
            bus.register(stage_name, _wrap_stage(stage_name, fn, dump_dir))
    return bus


def _build_video_request(args: argparse.Namespace, output_dir: Path) -> CompileVideoRequest:
    metadata = {
        "artifacts_dir": str(output_dir / "artifacts"),
        "prompt_profile": args.prompt_profile,
        "attach_frames": args.attach_frames,
    }
    if args.duration_seconds is not None and float(args.duration_seconds) > 0.0:
        metadata["duration_seconds"] = float(args.duration_seconds)
    return CompileVideoRequest(
        routing=_routing(),
        source=MediaSourceRef(
            source_id=args.source_id or Path(args.source).stem,
            file_path=str(Path(args.source).resolve()),
        ),
        provider={
            "provider": args.provider,
            "model": args.model,
        },
        metadata=metadata,
    )


def _build_audio_request(args: argparse.Namespace, output_dir: Path) -> CompileAudioRequest:
    metadata = {
        "artifacts_dir": str(output_dir / "artifacts"),
        "prompt_profile": args.prompt_profile,
    }
    if args.duration_seconds is not None and float(args.duration_seconds) > 0.0:
        metadata["duration_seconds"] = float(args.duration_seconds)
    return CompileAudioRequest(
        routing=_routing(),
        source=MediaSourceRef(
            source_id=args.source_id or Path(args.source).stem,
            file_path=str(Path(args.source).resolve()),
        ),
        provider={
            "provider": args.provider,
            "model": args.model,
        },
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the end-to-end media_graph_compiler pipeline.")
    parser.add_argument("--mode", choices=["video", "audio"], required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--prompt-profile", default="strict_json")
    parser.add_argument("--attach-frames", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bus = _build_bus(output_dir)
    asset_store = LocalOperatorAssetStore(output_dir / "assets")

    if args.mode == "video":
        request = _build_video_request(args, output_dir)
        result = compile_video(request, operator_bus=bus, asset_store=asset_store)
    else:
        request = _build_audio_request(args, output_dir)
        result = compile_audio(request, operator_bus=bus, asset_store=asset_store)

    report = {
        "status": result.status,
        "source_id": result.source_id,
        "window_digests": [item.model_dump() for item in result.window_digests],
        "visual_tracks": [item.model_dump() for item in result.visual_tracks],
        "speaker_tracks": [item.model_dump() for item in result.speaker_tracks],
        "face_voice_links": [item.model_dump() for item in result.face_voice_links],
        "utterances": [item.model_dump() for item in result.utterances],
        "evidence": [item.model_dump() for item in result.evidence],
        "asset_outputs": [item.model_dump() for item in result.asset_outputs],
        "graph_request": result.graph_request.model_dump() if result.graph_request else None,
        "trace": result.trace.model_dump(),
        "stats": result.stats,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"[done] report saved to {report_path}")


if __name__ == "__main__":
    main()
