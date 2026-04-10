from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

from modules.media_graph_compiler import (
    CompileVideoRequest,
    MediaRoutingContext,
    MediaSourceRef,
    compile_video,
)
from modules.media_graph_compiler.adapters import LocalOperatorAssetStore, OperatorBus
from modules.media_graph_compiler.adapters.default_operator_stages import (
    ensure_default_operator_stages,
)
from modules.media_graph_compiler.adapters.local_semantic_stage import (
    LocalSemanticStage,
)
from modules.media_graph_compiler.application import (
    FACE_VOICE_ASSOCIATION_STAGE,
    SEMANTIC_COMPILE_STAGE,
    SPEAKER_TRACK_STAGE,
    VISUAL_TRACK_STAGE,
)
from modules.memory import build_llm_from_byok


def _routing() -> MediaRoutingContext:
    return MediaRoutingContext(
        tenant_id="tenant_local",
        user_id=["u:demo"],
        memory_domain="media",
        run_id="run_logged_eval",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def _window_payload_summary(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for payload in payloads:
        summary.append(
            {
                "window_id": payload.get("window_id"),
                "modality": payload.get("modality"),
                "t_start_s": payload.get("t_start_s"),
                "t_end_s": payload.get("t_end_s"),
                "clip_frames": len(payload.get("clip_frames") or []),
                "representative_frames": len(payload.get("representative_frames") or []),
                "face_frames": len(payload.get("face_frames") or []),
                "visual_tracks": len(payload.get("visual_tracks") or []),
                "speaker_tracks": len(payload.get("speaker_tracks") or []),
                "face_voice_links": len(payload.get("face_voice_links") or []),
                "utterances": len(payload.get("utterances") or []),
                "evidence": len(payload.get("evidence") or []),
                "segment_vector_dim": int(
                    ((payload.get("segment_visual_profile") or {}).get("vector_summary") or {}).get("dim")
                    or 0
                ),
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
        "clip_start_s": ctx.get("clip_start_s"),
        "clip_end_s": ctx.get("clip_end_s"),
        "media_time_offset_s": ctx.get("media_time_offset_s"),
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
        print(
            json.dumps(
                {
                    "stage": name,
                    "input": _ctx_summary(ctx),
                    "output_keys": sorted((result or {}).keys()),
                },
                ensure_ascii=False,
                default=_json_default,
            )
        )
        return result

    return _wrapped


class LoggingSemanticAdapter:
    def __init__(self, *, inner, output_dir: Path) -> None:
        self._inner = inner
        self._output_dir = output_dir
        self._counter = 0
        self._counter_lock = threading.Lock()
        self.kind = getattr(inner, "kind", "unknown")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        with self._counter_lock:
            self._counter += 1
            request_id = f"semantic_call_{self._counter:03d}"
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        raw = ""
        error: Dict[str, Any] | None = None
        try:
            raw = self._inner.generate(messages, response_format)
            return raw
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            elapsed_ms = round((perf_counter() - started_perf) * 1000.0, 3)
            finished_at = datetime.now(timezone.utc)
            payload = {
                "request_id": request_id,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_ms": elapsed_ms,
                "adapter_kind": self.kind,
                "response_format": response_format,
                "messages": self._summarize_messages(messages),
                "raw_response": raw,
                "parsed_response": self._try_parse_json(raw),
                "error": error,
            }
            out_path = self._output_dir / f"{request_id}.json"
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

    def _summarize_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        summarized: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                summarized.append(
                    {
                        "role": message.get("role"),
                        "content_type": "text",
                        "text": content,
                    }
                )
                continue

            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, Mapping):
                        parts.append({"type": "unknown", "value": repr(item)})
                        continue
                    part_type = str(item.get("type") or "")
                    if part_type == "text":
                        text = str(item.get("text") or "")
                        maybe_json = self._try_parse_json(text)
                        if isinstance(maybe_json, dict) and "window_id" in maybe_json:
                            parts.append(
                                {
                                    "type": "structured_context",
                                    "window_id": maybe_json.get("window_id"),
                                    "modality": maybe_json.get("modality"),
                                    "t_start_s": maybe_json.get("t_start_s"),
                                    "t_end_s": maybe_json.get("t_end_s"),
                                    "images_map": maybe_json.get("images_map"),
                                    "visual_tracks": maybe_json.get("visual_tracks"),
                                    "speaker_tracks": maybe_json.get("speaker_tracks"),
                                    "face_voice_links": maybe_json.get("face_voice_links"),
                                    "utterances": maybe_json.get("utterances"),
                                    "evidence": maybe_json.get("evidence"),
                                    "window_stats": maybe_json.get("window_stats"),
                                    "segment_visual_profile": maybe_json.get("segment_visual_profile"),
                                }
                            )
                        else:
                            parts.append(
                                {
                                    "type": "text",
                                    "text": text,
                                }
                            )
                        continue
                    if part_type == "image_url":
                        url = str(((item.get("image_url") or {}) if isinstance(item.get("image_url"), Mapping) else {}).get("url") or "")
                        parts.append(
                            {
                                "type": "image_url",
                                "is_data_url": url.startswith("data:"),
                                "url_length": len(url),
                                "url_prefix": url[:64],
                            }
                        )
                        continue
                    parts.append({"type": part_type or "unknown", "value": dict(item)})

                summarized.append(
                    {
                        "role": message.get("role"),
                        "content_type": "parts",
                        "parts": parts,
                    }
                )
                continue

            summarized.append(
                {
                    "role": message.get("role"),
                    "content_type": type(content).__name__,
                    "value": repr(content),
                }
            )
        return summarized

    @staticmethod
    def _try_parse_json(raw: str) -> Any:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None


def _build_video_request(args: argparse.Namespace, output_dir: Path) -> CompileVideoRequest:
    metadata = {
        "artifacts_dir": str(output_dir / "artifacts"),
        "prompt_profile": args.prompt_profile,
        "attach_frames": args.attach_frames,
    }
    if args.semantic_max_concurrent is not None and int(args.semantic_max_concurrent) > 0:
        metadata["semantic_max_concurrent"] = int(args.semantic_max_concurrent)
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


def _build_bus(args: argparse.Namespace, output_dir: Path) -> OperatorBus:
    api_key = str(os.getenv(args.provider_api_key_env) or "").strip()
    if not api_key:
        raise RuntimeError(f"missing provider api key env: {args.provider_api_key_env}")

    adapter = build_llm_from_byok(
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=(args.base_url or None),
    )
    if adapter is None:
        raise RuntimeError("semantic adapter init failed")

    bus = OperatorBus()
    ensure_default_operator_stages(bus)
    stage_dump_dir = output_dir / "stages"
    semantic_dump_dir = output_dir / "semantic_requests"
    stage_dump_dir.mkdir(parents=True, exist_ok=True)
    semantic_dump_dir.mkdir(parents=True, exist_ok=True)

    for stage_name in (
        VISUAL_TRACK_STAGE,
        SPEAKER_TRACK_STAGE,
        FACE_VOICE_ASSOCIATION_STAGE,
    ):
        fn = bus.get(stage_name)
        if fn is not None:
            bus.register(stage_name, _wrap_stage(stage_name, fn, stage_dump_dir))

    semantic_stage = LocalSemanticStage(
        adapter=LoggingSemanticAdapter(inner=adapter, output_dir=semantic_dump_dir)
    )
    bus.register(
        SEMANTIC_COMPILE_STAGE,
        _wrap_stage(SEMANTIC_COMPILE_STAGE, lambda ctx: semantic_stage.run(ctx), stage_dump_dir),
    )
    return bus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run logged end-to-end video evaluation.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", default="")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--provider-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--prompt-profile", default="strict_json")
    parser.add_argument("--attach-frames", type=int, default=2)
    parser.add_argument("--semantic-max-concurrent", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = _build_video_request(args, output_dir)
    bus = _build_bus(args, output_dir)
    asset_store = LocalOperatorAssetStore(output_dir / "assets")

    started_at = datetime.now(timezone.utc)
    started_perf = perf_counter()
    result = compile_video(request, operator_bus=bus, asset_store=asset_store)
    elapsed_ms = round((perf_counter() - started_perf) * 1000.0, 3)
    finished_at = datetime.now(timezone.utc)

    report = {
        "run": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_ms": elapsed_ms,
            "source_path": str(Path(args.source).resolve()),
            "provider": args.provider,
            "model": args.model,
            "prompt_profile": args.prompt_profile,
            "attach_frames": args.attach_frames,
            "semantic_max_concurrent": args.semantic_max_concurrent,
            "duration_seconds": args.duration_seconds,
            "semantic_requests_dir": str(output_dir / "semantic_requests"),
        },
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
