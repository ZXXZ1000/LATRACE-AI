from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from modules.media_graph_compiler.adapters.multimodal_input_builder import (
    MultimodalInputBuilder,
)


class SemanticRuntime:
    """Thin runtime wrapper around the copied multimodal input-building flow."""

    def __init__(
        self,
        *,
        processor: Any,
        process_media_fn: Callable[..., Any],
    ) -> None:
        self._processor = processor
        self._process_media_fn = process_media_fn
        self._builder = MultimodalInputBuilder()

    def build_transformers_inputs(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        use_audio_in_video: bool = True,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._builder.build_transformers_inputs(
            processor=self._processor,
            process_mm_info=self._process_media_fn,
            messages=messages,
            use_audio_in_video=use_audio_in_video,
            processor_kwargs=processor_kwargs,
        )

    def build_vllm_inputs(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        use_audio_in_video: bool = True,
    ) -> Dict[str, Any]:
        return self._builder.build_vllm_inputs(
            processor=self._processor,
            process_mm_info=self._process_media_fn,
            messages=messages,
            use_audio_in_video=use_audio_in_video,
        )


__all__ = ["SemanticRuntime"]
