from __future__ import annotations

from typing import Any, Dict, Mapping

from modules.media_graph_compiler.types import CompileAudioRequest, CompileVideoRequest


class OptimizationPlanBuilder:
    """Build an explicit stage-facing optimization plan from request policy.

    The current pipeline is still in the "runtime wrappers + orchestration"
    phase, so the highest-value thing we can do now is make these knobs
    explicit and stable instead of burying them in old operator code.
    """

    def build_video_plan(
        self,
        *,
        request: CompileVideoRequest,
        runtime_inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "system": self._build_system_plan(request=request),
            "media": {
                "normalized_video_path": runtime_inputs.get("normalized_video_path"),
                "extracted_audio_path": runtime_inputs.get("extracted_audio_path"),
                "duration_seconds": runtime_inputs.get("duration_seconds"),
                "frame_rate": runtime_inputs.get("frame_rate"),
                "extract_audio_once": bool(runtime_inputs.get("extracted_audio_path")),
                "dual_resolution_frames": True,
            },
            "visual": {
                "prefer_clip_frames": request.optimization.prefer_clip_frames,
                "enable_frame_dedup": request.optimization.enable_visual_dedup,
                "similarity_threshold": request.optimization.visual_similarity_threshold,
                "max_frames_per_window": request.optimization.max_visual_frames_per_window,
                "max_frames_per_source": request.optimization.max_visual_frames_per_source,
                "sample_fps": request.visual_policy.sample_fps,
                "min_track_length_s": request.visual_policy.min_track_length_s,
            },
            "audio": {
                "enable_vad": request.optimization.enable_audio_vad,
                "min_turn_length_s": max(
                    request.speaker_policy.min_turn_length_s,
                    request.optimization.min_audio_turn_length_s,
                ),
                "enable_speaker_embedding": request.speaker_policy.enable_voice_features,
                "max_embedding_chunks_per_track": 4,
                "enable_asr_rtf_adaptation": request.optimization.enable_asr_rtf_adaptation,
                "asr_rtf_threshold": request.optimization.asr_rtf_threshold,
                "fallback_asr_model": request.optimization.fallback_asr_model,
            },
            "semantic": {
                "mode_hint": "window_compile",
                "prefer_native_video": request.optimization.prefer_native_video_mode,
                "allow_frame_bundle": request.optimization.allow_frame_bundle_mode,
                "allow_realtime_stream": request.optimization.allow_realtime_stream_mode,
                "video_window_seconds": request.windowing.video_window_seconds,
            },
        }

    def build_audio_plan(
        self,
        *,
        request: CompileAudioRequest,
        runtime_inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "system": self._build_system_plan(request=request),
            "media": {
                "normalized_audio_path": runtime_inputs.get("normalized_audio_path"),
                "duration_seconds": runtime_inputs.get("duration_seconds"),
            },
            "audio": {
                "enable_vad": request.optimization.enable_audio_vad,
                "min_turn_length_s": max(
                    request.speaker_policy.min_turn_length_s,
                    request.optimization.min_audio_turn_length_s,
                ),
                "enable_speaker_embedding": request.speaker_policy.enable_voice_features,
                "max_embedding_chunks_per_track": 4,
                "enable_asr_rtf_adaptation": request.optimization.enable_asr_rtf_adaptation,
                "asr_rtf_threshold": request.optimization.asr_rtf_threshold,
                "fallback_asr_model": request.optimization.fallback_asr_model,
            },
            "semantic": {
                "mode_hint": "window_compile",
                "allow_realtime_stream": request.optimization.allow_realtime_stream_mode,
                "audio_window_seconds": request.windowing.audio_window_seconds,
                "audio_chunk_seconds": request.windowing.audio_chunk_seconds,
            },
        }

    @staticmethod
    def _build_system_plan(*, request: CompileVideoRequest | CompileAudioRequest) -> Dict[str, Any]:
        return {
            "prefer_asset_replay": request.optimization.prefer_asset_replay,
            "drop_full_video_payload": request.optimization.drop_full_video_payload,
            "max_detection_frames_per_source": request.optimization.max_detection_frames_per_source,
        }


__all__ = ["OptimizationPlanBuilder"]
