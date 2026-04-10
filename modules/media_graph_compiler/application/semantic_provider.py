from __future__ import annotations

import base64
import concurrent.futures
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
        if not payloads:
            return {"window_digests": []}

        if adapter is None:
            return {
                "window_digests": [
                    self._fallback_digest(
                        payload,
                        warning="semantic_provider_not_configured",
                    ).model_dump()
                    for payload in payloads
                ]
            }

        adapter_kind = str(getattr(adapter, "kind", "") or "")
        max_concurrent = self._resolve_max_concurrent(
            request=request,
            adapter_kind=adapter_kind,
            payload_count=len(payloads),
        )
        if max_concurrent <= 1 or len(payloads) <= 1:
            digests = [
                self._compile_single_payload(
                    request=request,
                    payload=payload,
                    optimization_plan=optimization_plan,
                    adapter=adapter,
                    adapter_kind=adapter_kind,
                )
                for payload in payloads
            ]
            return {"window_digests": [item.model_dump() for item in digests]}

        digests_by_index: Dict[int, WindowDigest] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(
                    self._compile_single_payload,
                    request=request,
                    payload=payload,
                    optimization_plan=optimization_plan,
                    adapter=adapter,
                    adapter_kind=adapter_kind,
                ): index
                for index, payload in enumerate(payloads)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                digests_by_index[index] = future.result()

        digests = [digests_by_index[index] for index in range(len(payloads))]

        return {"window_digests": [item.model_dump() for item in digests]}

    def _compile_single_payload(
        self,
        *,
        request: Any,
        payload: Mapping[str, Any],
        optimization_plan: Mapping[str, Any],
        adapter: SemanticAdapter,
        adapter_kind: str,
    ) -> WindowDigest:
        try:
            messages = self._build_messages(
                request=request,
                payload=payload,
                optimization_plan=optimization_plan,
                adapter_kind=adapter_kind,
            )
            raw, parsed, attempts = self._generate_structured_response(
                adapter=adapter,
                messages=messages,
                request=request,
            )
            digest = self._build_digest_from_response(
                payload=payload,
                parsed=parsed,
                raw=raw,
            )
            digest.semantic_payload["request_attempts"] = attempts
            return digest
        except Exception as exc:
            digest = self._fallback_digest(
                payload,
                warning=f"semantic_provider_error:{type(exc).__name__}",
            )
            digest.semantic_payload["error"] = str(exc)
            return digest

    def _generate_structured_response(
        self,
        *,
        adapter: SemanticAdapter,
        messages: List[Dict[str, Any]],
        request: Any,
    ) -> Tuple[str, Dict[str, Any], int]:
        metadata = dict(getattr(request, "metadata", {}) or {})
        raw = ""
        parsed: Dict[str, Any] = {}
        attempts = max(
            1,
            int(
                metadata.get("semantic_parse_max_attempts")
                or os.getenv("MGC_SEMANTIC_PARSE_MAX_ATTEMPTS")
                or 2
            ),
        )
        response_formats: List[Optional[Dict[str, Any]]] = [{"type": "json_object"}]
        if attempts > 1:
            response_formats.extend([None] * (attempts - 1))
        for attempt, response_format in enumerate(response_formats, start=1):
            raw = adapter.generate(messages, response_format=response_format)
            parsed = self._parse_structured_response(raw)
            if parsed:
                return raw, parsed, attempt
        return raw, parsed, len(response_formats)

    @staticmethod
    def _resolve_max_concurrent(
        *,
        request: Any,
        adapter_kind: str,
        payload_count: int,
    ) -> int:
        metadata = dict(getattr(request, "metadata", {}) or {})
        explicit = metadata.get("semantic_max_concurrent")
        if explicit is None:
            explicit = os.getenv("MGC_SEMANTIC_MAX_CONCURRENT")
        try:
            explicit_value = int(explicit) if explicit is not None else 0
        except Exception:
            explicit_value = 0
        if explicit_value > 0:
            return max(1, min(explicit_value, payload_count))

        provider_name = str(
            getattr(getattr(request, "provider", None), "provider", "") or ""
        ).lower()
        adapter_key = f"{provider_name} {adapter_kind}".lower()
        if any(key in adapter_key for key in ("openrouter", "openai", "gemini", "google", "qwen", "dashscope", "aliyun")):
            return max(1, min(4, payload_count))
        return 1

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
        provider_name = str(
            getattr(getattr(request, "provider", None), "provider", "") or ""
        ).strip()
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
                provider_name=provider_name,
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
        structured_context = self._build_structured_context(
            payload=payload,
            frame_map=frame_map,
            with_images=bool(frame_bundle),
        )
        binding_hint = self._build_face_voice_binding_hint(payload)
        instruction = (
            "请基于以下窗口级稳定上下文输出严格 JSON。"
            "如果有图像，请通过 images_map 中的 image_id 引用，例如 img1/img2。"
        )
        if binding_hint:
            instruction = f"{instruction}\n{binding_hint}"
        content.append(
            {
                "type": "text",
                "text": instruction,
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
    def _build_face_voice_binding_hint(payload: Mapping[str, Any]) -> str:
        links = sorted(
            (
                dict(item)
                for item in (payload.get("face_voice_links") or [])
                if item.get("speaker_track_id") and item.get("visual_track_id")
            ),
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )[:4]
        if not links:
            return ""
        pairs: List[str] = []
        for item in links:
            speaker_id = str(item.get("speaker_track_id") or "").strip()
            visual_id = str(item.get("visual_track_id") or "").strip()
            confidence = round(float(item.get("confidence") or 0.0), 2)
            method = str(((item.get("metadata") or {}) if isinstance(item.get("metadata"), Mapping) else {}).get("method") or "").strip()
            suffix = f", conf={confidence}"
            if method:
                suffix += f", method={method}"
            pairs.append(f"{speaker_id}->{visual_id}({suffix})")
        joined = "; ".join(pairs)
        return (
            "声脸绑定提示："
            f"{joined}。若某条 utterance 的 speaker_track_id 已有对应 face_#，"
            "请优先把该发言归到对应 face_#，actor_tag 也优先用 face_#；"
            "不要把同一个已绑定角色同时写成一个 face_# 和另一个 voice_#。"
        )

    @staticmethod
    def _resolve_attach_frames(
        *,
        payload: Mapping[str, Any],
        optimization_plan: Mapping[str, Any],
        adapter_kind: str,
        provider_name: str,
        metadata: Mapping[str, Any],
    ) -> int:
        if metadata.get("attach_frames") is not None:
            try:
                return max(0, int(metadata["attach_frames"]))
            except Exception:
                return 0

        clip_frames = list(payload.get("representative_frames") or payload.get("clip_frames") or [])
        if not clip_frames:
            return 0

        max_frames_per_window = int(
            ((optimization_plan.get("visual") or {}).get("max_frames_per_window") or 0)
        )
        provider_key = f"{provider_name} {adapter_kind}".lower()
        explicit_multi_image = bool(metadata.get("allow_multi_image"))
        if "openrouter" in provider_key:
            default_limit = 1 if not explicit_multi_image else 2
        elif "openai" in provider_key:
            default_limit = 3
        elif any(key in provider_key for key in ("gemini", "google", "qwen", "dashscope", "aliyun")):
            default_limit = 4
        else:
            default_limit = 4
        provider_cap = metadata.get("max_images_per_request")
        if provider_cap is None:
            provider_cap = os.getenv("MGC_SEMANTIC_MAX_IMAGES_PER_REQUEST")
        try:
            provider_cap_value = int(provider_cap) if provider_cap is not None else 0
        except Exception:
            provider_cap_value = 0
        if provider_cap_value > 0:
            default_limit = min(default_limit, provider_cap_value)
        if max_frames_per_window > 0:
            default_limit = min(default_limit, max_frames_per_window)
        return max(1, min(default_limit, len(clip_frames)))

    @staticmethod
    def _select_frame_bundle(
        payload: Mapping[str, Any],
        *,
        attach_frames: int,
    ) -> List[Dict[str, Any]]:
        frames = list(payload.get("representative_frames") or payload.get("clip_frames") or [])
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

    def _build_structured_context(
        self,
        *,
        payload: Mapping[str, Any],
        frame_map: Sequence[Mapping[str, Any]],
        with_images: bool,
    ) -> Dict[str, Any]:
        limits = self._resolve_context_limits(with_images=with_images)
        return {
            "window_id": payload.get("window_id"),
            "modality": payload.get("modality"),
            "t_start_s": payload.get("t_start_s"),
            "t_end_s": payload.get("t_end_s"),
            "images_map": list(frame_map),
            "window_stats": dict(payload.get("window_stats") or self._derive_window_stats(payload)),
            "segment_visual_profile": self._compact_segment_visual_profile(
                payload.get("segment_visual_profile") or {}
            ),
            "visual_tracks": self._compact_visual_tracks(
                payload.get("visual_tracks") or [],
                limit=limits["visual_tracks"],
            ),
            "speaker_tracks": self._compact_speaker_tracks(
                payload.get("speaker_tracks") or [],
                limit=limits["speaker_tracks"],
            ),
            "face_voice_links": self._compact_face_voice_links(
                payload.get("face_voice_links") or [],
                limit=limits["face_voice_links"],
            ),
            "utterances": self._compact_utterances(
                payload.get("utterances") or [],
                limit=limits["utterances"],
                max_text_len=limits["text_len"],
            ),
            "evidence": self._compact_evidence(
                payload.get("evidence") or [],
                limit=limits["evidence"],
                max_text_len=limits["text_len"],
            ),
        }

    @staticmethod
    def _resolve_context_limits(*, with_images: bool) -> Dict[str, int]:
        def _read(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return max(1, int(raw))
            except Exception:
                return default

        if with_images:
            return {
                "visual_tracks": _read("MGC_SEMANTIC_MAX_VISUAL_TRACKS", 4),
                "speaker_tracks": _read("MGC_SEMANTIC_MAX_SPEAKER_TRACKS", 4),
                "face_voice_links": _read("MGC_SEMANTIC_MAX_FACE_VOICE_LINKS", 4),
                "utterances": _read("MGC_SEMANTIC_MAX_UTTERANCES_WITH_IMAGES", 6),
                "evidence": _read("MGC_SEMANTIC_MAX_EVIDENCE_WITH_IMAGES", 6),
                "text_len": _read("MGC_SEMANTIC_MAX_TEXT_LEN", 160),
            }
        return {
            "visual_tracks": _read("MGC_SEMANTIC_MAX_VISUAL_TRACKS_NO_IMAGES", 6),
            "speaker_tracks": _read("MGC_SEMANTIC_MAX_SPEAKER_TRACKS_NO_IMAGES", 6),
            "face_voice_links": _read("MGC_SEMANTIC_MAX_FACE_VOICE_LINKS_NO_IMAGES", 6),
            "utterances": _read("MGC_SEMANTIC_MAX_UTTERANCES_NO_IMAGES", 10),
            "evidence": _read("MGC_SEMANTIC_MAX_EVIDENCE_NO_IMAGES", 10),
            "text_len": _read("MGC_SEMANTIC_MAX_TEXT_LEN", 180),
        }

    @staticmethod
    def _derive_window_stats(payload: Mapping[str, Any]) -> Dict[str, int]:
        return {
            "clip_frames_total": len(payload.get("clip_frames") or []),
            "clip_frames_selected": len(payload.get("clip_frames") or []),
            "face_frames_total": len(payload.get("face_frames") or []),
            "face_frames_selected": len(payload.get("face_frames") or []),
            "representative_frames": len(payload.get("representative_frames") or []),
            "visual_tracks": len(payload.get("visual_tracks") or []),
            "speaker_tracks": len(payload.get("speaker_tracks") or []),
            "face_voice_links": len(payload.get("face_voice_links") or []),
            "utterances": len(payload.get("utterances") or []),
            "evidence": len(payload.get("evidence") or []),
        }

    def _compact_segment_visual_profile(
        self,
        profile: Mapping[str, Any],
    ) -> Dict[str, Any]:
        thumbnails = list(profile.get("representative_thumbnails") or [])[:3]
        vector_summary = dict(profile.get("vector_summary") or {})
        return {
            "representative_thumbnails": [
                {
                    "frame_index": item.get("frame_index"),
                    "t_media_s": item.get("t_media_s"),
                }
                for item in thumbnails
            ],
            "thumbnail_evidence_refs": list(profile.get("thumbnail_evidence_refs") or [])[:4],
            "vector_summary": {
                "provider": vector_summary.get("provider"),
                "strategy": vector_summary.get("strategy"),
                "sample_count": vector_summary.get("sample_count"),
                "dim": vector_summary.get("dim"),
                "preview": list(vector_summary.get("preview") or [])[:8],
            },
        }

    def _compact_visual_tracks(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            (dict(item) for item in items),
            key=lambda item: float(item.get("t_end_s") or 0.0)
            - float(item.get("t_start_s") or 0.0),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for item in ordered[:limit]:
            out.append(
                {
                    "track_id": item.get("track_id"),
                    "category": item.get("category"),
                    "t_start_s": item.get("t_start_s"),
                    "t_end_s": item.get("t_end_s"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "evidence_refs": list(item.get("evidence_refs") or [])[:2],
                    "metadata": self._compact_metadata(
                        item.get("metadata") or {},
                        keys=("segment_id", "track_quality", "source", "identity_hint"),
                    ),
                }
            )
        return out

    def _compact_speaker_tracks(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            (dict(item) for item in items),
            key=lambda item: float(item.get("t_end_s") or 0.0)
            - float(item.get("t_start_s") or 0.0),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for item in ordered[:limit]:
            out.append(
                {
                    "track_id": item.get("track_id"),
                    "t_start_s": item.get("t_start_s"),
                    "t_end_s": item.get("t_end_s"),
                    "utterance_count": len(item.get("utterance_ids") or []),
                    "evidence_refs": list(item.get("evidence_refs") or [])[:2],
                    "metadata": self._compact_metadata(
                        item.get("metadata") or {},
                        keys=("language", "segment_count", "embedding_dim", "method"),
                    ),
                }
            )
        return out

    def _compact_face_voice_links(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            (dict(item) for item in items),
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for item in ordered[:limit]:
            out.append(
                {
                    "link_id": item.get("link_id"),
                    "speaker_track_id": item.get("speaker_track_id"),
                    "visual_track_id": item.get("visual_track_id"),
                    "t_start_s": item.get("t_start_s"),
                    "t_end_s": item.get("t_end_s"),
                    "confidence": item.get("confidence"),
                    "overlap_s": item.get("overlap_s"),
                    "support_evidence_refs": list(item.get("support_evidence_refs") or [])[:3],
                    "support_utterance_ids": list(item.get("support_utterance_ids") or [])[:3],
                    "metadata": self._compact_metadata(
                        item.get("metadata") or {},
                        keys=("method", "scoring", "asd_used"),
                    ),
                }
            )
        return out

    def _compact_utterances(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
        max_text_len: int,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            (dict(item) for item in items if str(item.get("text") or "").strip()),
            key=lambda item: float(item.get("t_start_s") or 0.0),
        )
        out: List[Dict[str, Any]] = []
        for item in ordered[:limit]:
            out.append(
                {
                    "utterance_id": item.get("utterance_id"),
                    "speaker_track_id": item.get("speaker_track_id"),
                    "t_start_s": item.get("t_start_s"),
                    "t_end_s": item.get("t_end_s"),
                    "text": self._denoise_text(
                        str(item.get("text") or ""),
                        max_len=max_text_len,
                    ),
                    "language": item.get("language"),
                    "confidence": item.get("confidence"),
                }
            )
        return out

    def _compact_evidence(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        limit: int,
        max_text_len: int,
    ) -> List[Dict[str, Any]]:
        priority = {"thumbnail": 0, "frame_crop": 1, "audio_chunk": 2, "transcript": 3, "mask": 4}
        ordered = sorted(
            (dict(item) for item in items),
            key=lambda item: (
                priority.get(str(item.get("kind") or ""), 9),
                float(item.get("t_start_s") or 0.0),
            ),
        )
        out: List[Dict[str, Any]] = []
        for item in ordered[:limit]:
            out.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "kind": item.get("kind"),
                    "t_start_s": item.get("t_start_s"),
                    "t_end_s": item.get("t_end_s"),
                    "metadata": self._compact_metadata(
                        item.get("metadata") or {},
                        keys=(
                            "track_id",
                            "speaker_track_id",
                            "frame_index",
                            "transcript",
                            "score",
                            "algorithm",
                        ),
                        max_text_len=max_text_len,
                    ),
                }
            )
        return out

    def _compact_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        keys: Sequence[str],
        max_text_len: int = 120,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in keys:
            if key not in metadata:
                continue
            value = metadata.get(key)
            if isinstance(value, str):
                out[key] = self._denoise_text(value, max_len=max_text_len)
            elif isinstance(value, (int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, list):
                out[key] = value[:4]
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
            "representative_batch": len(payload.get("representative_frames") or []),
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
                "representative_batch": len(payload.get("representative_frames") or []),
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
