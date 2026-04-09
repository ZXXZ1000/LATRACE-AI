from __future__ import annotations

from typing import Any, Dict, List, Tuple


def generate_messages(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将迁移来的通用 multimodal inputs 转成对话消息。"""
    sys = {"role": "system", "content": "You are a helpful speech segmentation assistant."}
    user: Dict[str, Any] = {"role": "user", "content": ""}
    media: List[Dict[str, Any]] = []
    for item in inputs or []:
        item_type = str(item.get("type", "")).lower()
        content = item.get("content")
        if item_type.startswith("text") and content:
            user["content"] = str(content)
        elif item_type.startswith("video_base64/") and content:
            mime = item_type.split("/", 1)[1]
            media.append({"type": "video", "data_url": f"data:{mime};base64,{content}"})
    if media:
        user["media"] = media
    return [sys, user]


def get_response(model: str, messages: List[Dict[str, Any]], timeout: int = 30) -> Tuple[str, Dict[str, Any]]:
    """沿用旧实现，通过 memory.llm_adapter 做兜底多模态调用。"""
    try:
        from modules.memory.application.llm_adapter import build_llm_from_config  # type: ignore

        adapter = build_llm_from_config("multimodal")
        raw = adapter.generate(messages, response_format={"type": "json_object"})
        return str(raw), {}
    except Exception as exc:  # pragma: no cover - legacy fallback path
        raise RuntimeError(f"llm_adapter_generate_failed: {exc}")


def parallel_get_whisper(*args, **kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("parallel_get_whisper is not implemented in this integration.")


__all__ = ["generate_messages", "get_response", "parallel_get_whisper"]
