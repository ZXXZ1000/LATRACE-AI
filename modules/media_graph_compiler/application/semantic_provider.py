from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from modules.media_graph_compiler.application.prompt_loader import PromptLoader
from modules.media_graph_compiler.types import WindowDigest


class SemanticAdapter(Protocol):
    kind: str

    def generate(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str: ...


class RichBatchSemanticProvider:
    """Semantic provider adapted from the proven memorization-agent LLM shell.

    The provider does not own raw media preprocessing. It only consumes
    already-stabilized per-window payloads and compiles them into
    `WindowDigest` objects via a hosted/local multimodal model adapter.
    """

    def __init__(
        self,
        *,
        adapter: SemanticAdapter | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self._adapter = adapter
        self._prompt_loader = prompt_loader or PromptLoader()
        self._adapter_cache: Dict[Tuple[str, str, str], SemanticAdapter | None] = {}
        self._data_url_cache: Dict[str, str | None] = {}

    def generate_window_digests(self, ctx: Mapping[str, Any]) -> Dict[str, Any]:
        request = ctx.get("request")
        payloads = list(ctx.get("window_payloads") or [])
        optimization_plan = dict(ctx.get("optimization_plan") or {})
        adapter = self._adapter or self._resolve_adapter(request)
        digests: List[WindowDigest] = []

        for payload in payloads:
            if adapter is None:
                digests.append(
                    self._fallback_digest(
                        payload,
                        warning="semantic_provider_not_configured",
                    )
                )
                continue
            try:
                messages = self._build_messages(
                    request=request,
                    payload=payload,
                    optimization_plan=optimization_plan,
                    adapter_kind=str(getattr(adapter, "kind", "") or ""),
                )
                raw = adapter.generate(
                    messages,
                    response_format={"type": "json_object"},
                )
                parsed = self._parse_structured_response(raw)
                digests.append(
                    self._build_digest_from_response(
                        payload=payload,
                        parsed=parsed,
                        raw=raw,
                    )
                )
            except Exception as exc:
                digest = self._fallback_digest(
                    payload,
                    warning=f"semantic_provider_error:{type(exc).__name__}",
                )
                digest.semantic_payload["error"] = str(exc)
                digests.append(digest)

        return {"window_digests": [item.model_dump() for item in digests]}

    def _resolve_adapter(self, request: Any) -> SemanticAdapter | None:
        if request is None:
            return None
        provider = str(
            request.provider.provider or os.getenv("MGC_SEMANTIC_PROVIDER") or ""
        ).strip()
        model = str(
            request.provider.model or os.getenv("MGC_SEMANTIC_MODEL") or ""
        ).strip()
        if not provider or not model:
            return None

        base_url = str(
            request.metadata.get("provider_base_url")
            or os.getenv("MGC_SEMANTIC_BASE_URL")
            or ""
        ).strip()
        cache_key = (provider.lower(), model, base_url)
        if cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        api_key = self._resolve_api_key(provider, metadata=request.metadata)
        if not api_key:
            self._adapter_cache[cache_key] = None
            return None

        from modules.memory import build_llm_from_byok

        adapter = build_llm_from_byok(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url or None,
        )
        self._adapter_cache[cache_key] = adapter
        return adapter

    @staticmethod
    def _resolve_api_key(provider: str, metadata: Mapping[str, Any]) -> str:
        explicit = str(metadata.get("provider_api_key") or "").strip()
        if explicit:
            return explicit

        env_name = str(metadata.get("provider_api_key_env") or "").strip()
        if env_name:
            return str(os.getenv(env_name) or "").strip()

        provider_key_env = {
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "google": "GOOGLE_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openai_compat": "OPENAI_API_KEY",
            "openai-compatible": "OPENAI_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "aliyun": "DASHSCOPE_API_KEY",
            "glm": "ZHIPUAI_API_KEY",
            "zhipuai": "ZHIPUAI_API_KEY",
        }
        return str(os.getenv(provider_key_env.get(provider.lower(), "")) or "").strip()

    def _build_messages(
        self,
        *,
        request: Any,
        payload: Mapping[str, Any],
        optimization_plan: Mapping[str, Any],
        adapter_kind: str,
    ) -> List[Dict[str, Any]]:
        metadata = dict(getattr(request, "metadata", {}) or {})
        profile = str(
            metadata.get("prompt_profile") or os.getenv("MGC_PROMPT_PROFILE") or "strict_json"
        ).strip()
        prompt = self._prompt_loader.load_prompt(profile)

        frame_bundle = self._select_frame_bundle(
            payload,
            attach_frames=self._resolve_attach_frames(
                payload=payload,
                optimization_plan=optimization_plan,
                adapter_kind=adapter_kind,
                metadata=metadata,
            ),
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": self._build_user_content(payload, frame_bundle)},
        ]
        return messages

    def _build_user_content(
        self,
        payload: Mapping[str, Any],
        frame_bundle: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        frame_map = [
            {
                "image_id": item["image_id"],
                "t_media_s": item.get("t_media_s"),
                "frame_index": item.get("frame_index"),
            }
            for item in frame_bundle
        ]
        structured_context = {
            "window_id": payload.get("window_id"),
            "modality": payload.get("modality"),
            "t_start_s": payload.get("t_start_s"),
            "t_end_s": payload.get("t_end_s"),
            "images_map": frame_map,
            "visual_tracks": payload.get("visual_tracks") or [],
            "speaker_tracks": payload.get("speaker_tracks") or [],
            "face_voice_links": payload.get("face_voice_links") or [],
            "utterances": payload.get("utterances") or [],
            "evidence": payload.get("evidence") or [],
        }
        content.append(
            {
                "type": "text",
                "text": (
                    "请基于以下窗口级稳定上下文输出严格 JSON。"
                    "如果有图像，请通过 images_map 中的 image_id 引用，例如 img1/img2。"
                ),
            }
        )
        content.append(
            {
                "type": "text",
                "text": json.dumps(structured_context, ensure_ascii=False),
            }
        )
        for item in frame_bundle:
            url = self._to_data_url(item.get("file_path"))
            if not url:
                continue
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            )
        return content

    @staticmethod
    def _resolve_attach_frames(
        *,
        payload: Mapping[str, Any],
        optimization_plan: Mapping[str, Any],
        adapter_kind: str,
        metadata: Mapping[str, Any],
    ) -> int:
        if metadata.get("attach_frames") is not None:
            try:
                return max(0, int(metadata["attach_frames"]))
            except Exception:
                return 0

        clip_frames = list(payload.get("clip_frames") or [])
        if not clip_frames:
            return 0

        max_frames_per_window = int(
            ((optimization_plan.get("visual") or {}).get("max_frames_per_window") or 0)
        )
        default_limit = 2 if adapter_kind.lower() in {"openrouter_http", "openrouter"} else 4
        if max_frames_per_window > 0:
            default_limit = min(default_limit, max_frames_per_window)
        return max(1, min(default_limit, len(clip_frames)))

    @staticmethod
    def _select_frame_bundle(
        payload: Mapping[str, Any],
        *,
        attach_frames: int,
    ) -> List[Dict[str, Any]]:
        frames = list(payload.get("clip_frames") or [])
        if attach_frames <= 0 or not frames:
            return []
        if len(frames) <= attach_frames:
            indices = list(range(len(frames)))
        else:
            step = len(frames) / float(attach_frames)
            indices = [min(len(frames) - 1, int(i * step + step / 2.0)) for i in range(attach_frames)]
        out: List[Dict[str, Any]] = []
        for offset, index in enumerate(indices, start=1):
            frame = dict(frames[index] or {})
            frame["image_id"] = f"img{offset}"
            out.append(frame)
        return out

    def _to_data_url(self, val: Any) -> Optional[str]:
        """Copied-and-adapted image attachment handling from memorization-agent."""
        if not isinstance(val, str) or not val:
            return None
        if val.startswith("http://") or val.startswith("https://"):
            return val
        if val.startswith("data:image"):
            return val
        if val in self._data_url_cache:
            return self._data_url_cache[val]
        if os.path.exists(val):
            try:
                raw = open(val, "rb").read()
                encoded = base64.b64encode(raw).decode("ascii")
                data_url = f"data:image/jpeg;base64,{encoded}"
                self._data_url_cache[val] = data_url
                return data_url
            except Exception:
                self._data_url_cache[val] = None
                return None
        try:
            base64.b64decode(val, validate=True)
            if len(val) > 32:
                data_url = f"data:image/jpeg;base64,{val}"
                self._data_url_cache[val] = data_url
                return data_url
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_structured_response(text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {}
        raw = text.strip()
        fenced = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
        if fenced:
            raw = fenced.group(1).strip()
        boxed = re.search(r"<\|begin_of_box\|>([\s\S]*?)<\|end_of_box\|>", raw)
        if boxed:
            raw = boxed.group(1).strip()
        for candidate in (raw, RichBatchSemanticProvider._extract_json_block(raw)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                continue
        return {}

    @staticmethod
    def _extract_json_block(text: str) -> str:
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else ""

    def _build_digest_from_response(
        self,
        *,
        payload: Mapping[str, Any],
        parsed: Mapping[str, Any],
        raw: str,
    ) -> WindowDigest:
        summary = self._extract_summary(payload, parsed)
        participant_refs = self._extract_participant_refs(payload, parsed)
        evidence_refs = [
            str(item.get("evidence_id"))
            for item in (payload.get("evidence") or [])
            if item.get("evidence_id")
        ]
        warnings: List[str] = []
        if not parsed:
            warnings.append("semantic_parse_failed")
        semantic_payload = {
            "provider_response": dict(parsed or {}),
            "frame_batch": len(payload.get("clip_frames") or []),
            "face_batch": len(payload.get("face_frames") or []),
            "speaker_batch": len(payload.get("speaker_tracks") or []),
            "association_batch": len(payload.get("face_voice_links") or []),
            "utterance_batch": len(payload.get("utterances") or []),
        }
        if not parsed and raw:
            semantic_payload["provider_raw"] = raw
        return WindowDigest(
            window_id=str(payload.get("window_id")),
            modality=str(payload.get("modality")),
            t_start_s=float(payload.get("t_start_s") or 0.0),
            t_end_s=float(payload.get("t_end_s") or 0.0),
            summary=summary,
            participant_refs=participant_refs,
            semantic_payload=semantic_payload,
            evidence_refs=evidence_refs,
            warnings=warnings,
        )

    def _fallback_digest(
        self,
        payload: Mapping[str, Any],
        *,
        warning: str,
    ) -> WindowDigest:
        utterances = payload.get("utterances") or []
        summary = " ".join(
            self._denoise_text(str(item.get("text") or ""))
            for item in utterances[:2]
            if str(item.get("text") or "").strip()
        )
        if not summary:
            summary = (
                f"{payload.get('modality')} activity in "
                f"{float(payload.get('t_start_s') or 0.0):.1f}-"
                f"{float(payload.get('t_end_s') or 0.0):.1f}s"
            )
        participant_refs = self._extract_participant_refs(payload, {})
        evidence_refs = [
            str(item.get("evidence_id"))
            for item in (payload.get("evidence") or [])
            if item.get("evidence_id")
        ]
        return WindowDigest(
            window_id=str(payload.get("window_id")),
            modality=str(payload.get("modality")),
            t_start_s=float(payload.get("t_start_s") or 0.0),
            t_end_s=float(payload.get("t_end_s") or 0.0),
            summary=summary,
            participant_refs=participant_refs,
            semantic_payload={
                "frame_batch": len(payload.get("clip_frames") or []),
                "face_batch": len(payload.get("face_frames") or []),
                "speaker_batch": len(payload.get("speaker_tracks") or []),
                "association_batch": len(payload.get("face_voice_links") or []),
                "utterance_batch": len(payload.get("utterances") or []),
            },
            evidence_refs=evidence_refs,
            warnings=[warning],
        )

    def _extract_summary(
        self,
        payload: Mapping[str, Any],
        parsed: Mapping[str, Any],
    ) -> str:
        direct = parsed.get("summary")
        if isinstance(direct, str) and direct.strip():
            return self._denoise_text(direct, max_len=240)

        timeline = parsed.get("semantic_timeline") or parsed.get("events") or []
        if isinstance(timeline, list):
            texts = [
                self._denoise_text(str(item.get("text") or ""), max_len=180)
                for item in timeline[:2]
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            if texts:
                return " ".join(texts)

        for key in ("semantic", "episodic"):
            values = parsed.get(key) or []
            if isinstance(values, list):
                texts = [self._denoise_text(str(item), max_len=160) for item in values[:2] if str(item).strip()]
                if texts:
                    return " ".join(texts)

        utterances = payload.get("utterances") or []
        texts = [self._denoise_text(str(item.get("text") or ""), max_len=120) for item in utterances[:2]]
        texts = [item for item in texts if item]
        if texts:
            return " ".join(texts)
        return f"{payload.get('modality')} activity"

    @staticmethod
    def _extract_participant_refs(
        payload: Mapping[str, Any],
        parsed: Mapping[str, Any],
    ) -> List[str]:
        refs: List[str] = []
        refs.extend(
            str(item.get("track_id"))
            for item in (payload.get("visual_tracks") or [])
            if item.get("track_id")
        )
        refs.extend(
            str(item.get("track_id"))
            for item in (payload.get("speaker_tracks") or [])
            if item.get("track_id")
        )
        timeline = parsed.get("semantic_timeline") or parsed.get("events") or []
        if isinstance(timeline, list):
            for item in timeline:
                if not isinstance(item, dict):
                    continue
                actor_tag = item.get("actor_tag")
                if isinstance(actor_tag, str) and actor_tag.strip():
                    refs.append(actor_tag.strip())
        for pair in parsed.get("equivalence") or []:
            if isinstance(pair, list) and pair and isinstance(pair[0], str):
                refs.append(pair[0])
        deduped: List[str] = []
        for ref in refs:
            if ref and ref not in deduped:
                deduped.append(ref)
        return deduped

    @staticmethod
    def _denoise_text(text: str, max_len: int = 120) -> str:
        s = re.sub(r"\s+", " ", str(text)).strip()
        s = re.sub(r"([!?,。；，、\.])\1{3,}", r"\1\1\1", s)
        if len(s) > max_len:
            s = s[: max_len - 1].rstrip() + "…"
        return s


__all__ = ["RichBatchSemanticProvider", "SemanticAdapter"]
