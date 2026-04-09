from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


class MultimodalInputBuilder:
    """Borrow the stable processor/input split from Qwen3-Omni.

    This adapter does not depend on the real Qwen runtime. Callers provide the
    processor object and the `process_mm_info` function, which keeps this module
    light and testable while preserving the reference pipeline shape.
    """

    def build_prompt(
        self,
        *,
        processor: Any,
        messages: Sequence[Mapping[str, Any]],
        add_generation_prompt: bool = True,
    ) -> str:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def process_media(
        self,
        *,
        process_mm_info: Any,
        messages: Sequence[Mapping[str, Any]],
        use_audio_in_video: bool,
    ) -> tuple[Any, Any, Any]:
        return process_mm_info(messages, use_audio_in_video=use_audio_in_video)

    def build_transformers_inputs(
        self,
        *,
        processor: Any,
        process_mm_info: Any,
        messages: Sequence[Mapping[str, Any]],
        use_audio_in_video: bool,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        text = self.build_prompt(
            processor=processor,
            messages=messages,
            add_generation_prompt=True,
        )
        audios, images, videos = self.process_media(
            process_mm_info=process_mm_info,
            messages=messages,
            use_audio_in_video=use_audio_in_video,
        )
        kwargs = dict(processor_kwargs or {})
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", True)
        kwargs["use_audio_in_video"] = use_audio_in_video
        return processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            **kwargs,
        )

    def build_vllm_inputs(
        self,
        *,
        processor: Any,
        process_mm_info: Any,
        messages: Sequence[Mapping[str, Any]],
        use_audio_in_video: bool,
    ) -> Dict[str, Any]:
        text = self.build_prompt(
            processor=processor,
            messages=messages,
            add_generation_prompt=True,
        )
        audios, images, videos = self.process_media(
            process_mm_info=process_mm_info,
            messages=messages,
            use_audio_in_video=use_audio_in_video,
        )
        inputs: Dict[str, Any] = {
            "prompt": text,
            "multi_modal_data": {},
            "mm_processor_kwargs": {
                "use_audio_in_video": use_audio_in_video,
            },
        }
        if images is not None:
            inputs["multi_modal_data"]["image"] = images
        if videos is not None:
            inputs["multi_modal_data"]["video"] = videos
        if audios is not None:
            inputs["multi_modal_data"]["audio"] = audios
        return inputs


__all__ = ["MultimodalInputBuilder"]
