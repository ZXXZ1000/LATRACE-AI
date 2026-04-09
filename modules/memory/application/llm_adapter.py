from __future__ import annotations

"""LLM Adapter using LiteLLM to unify providers with minimal Python-side code.

Usage:
- Provide a callable: LLMAdapter(callable).generate(messages, response_format)
- Or build from env via build_llm_from_env() (returns None if not configured)

Supported via LiteLLM by model string + env keys (examples):
- OpenAI:            model="gpt-4o-mini"             + OPENAI_API_KEY
- OpenRouter:        model="openrouter/<model>"      + OPENROUTER_API_KEY (base handled by LiteLLM)
- DeepSeek:          model="deepseek-chat"           + DEEPSEEK_API_KEY
- Moonshot (Kimi):   model="moonshot-v1-8k"          + MOONSHOT_API_KEY
- Qwen (DashScope):  model="qwen2.5"                 + DASHSCOPE_API_KEY
- GLM (ZhipuAI):     model="glm-4"                   + ZHIPUAI_API_KEY
- Gemini:            model="gemini/gemini-1.5-flash" + GOOGLE_API_KEY
- OpenAI-compatible custom endpoint: set LLM_BASE_URL + LLM_API_KEY + LLM_MODEL

If not configured, returns None and callers should fallback to heuristics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import contextvars
import os
import json
from modules.memory.contracts.usage_models import LLMUsage

# Be robust about env loading: load both repo root .env and module-specific .env
try:
    from dotenv import load_dotenv  # type: ignore
    # load root .env (CWD)
    load_dotenv()
    # load memory module config .env with override to ensure keys present
    _MEM_ENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
    if os.path.exists(_MEM_ENV):
        load_dotenv(_MEM_ENV, override=True)
except Exception:
    pass

# Normalize Gemini keys: accept GOOGLE_API_KEY as GEMINI_API_KEY equivalent
try:
    if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
except Exception:
    pass


import threading

# Global rate limiter for LLM calls to prevent overwhelming API
# Default: 2 concurrent calls. Override via MEMORY_LLM_MAX_CONCURRENT env var.
def _parse_llm_max_concurrent() -> int:
    """Parse and validate MEMORY_LLM_MAX_CONCURRENT env var.
    
    Returns:
        Validated concurrency limit (minimum 1, default 2).
        Falls back to default on any parsing error to prevent service failure.
    """
    default = 8
    raw = os.environ.get("MEMORY_LLM_MAX_CONCURRENT", str(default))
    if not raw or not raw.strip():
        return default
    try:
        value = int(raw.strip())
        # Enforce minimum of 1 to prevent Semaphore(0) deadlock
        if value < 1:
            return default
        return value
    except (ValueError, TypeError):
        # Invalid value (non-integer, None, etc.) - fallback to default
        return default

_LLM_MAX_CONCURRENT = _parse_llm_max_concurrent()
_LLM_SEMAPHORE = threading.Semaphore(_LLM_MAX_CONCURRENT)


class LLMAdapter:
    def __init__(self, fn: Callable[[List[Dict[str, Any]], Optional[Dict[str, Any]]], str], *, kind: str = "unknown") -> None:
        self._fn = fn
        # debug marker for upstream to introspect which path is used
        self.kind = kind

    def generate(self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        # Rate limit LLM calls to prevent API overload
        with _LLM_SEMAPHORE:
            try:
                from modules.memory.application.metrics import gauge_inc, gauge_dec  # local import to avoid cycles
            except Exception:
                gauge_inc = gauge_dec = None  # type: ignore[assignment]
            if callable(gauge_inc):
                try:
                    gauge_inc("llm_inflight", 1)
                except Exception:
                    pass
            try:
                return self._fn(messages, response_format)
            finally:
                if callable(gauge_dec):
                    try:
                        gauge_dec("llm_inflight", 1)
                    except Exception:
                        pass

    def generate_with_usage(
        self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[LLMUsage]]:
        """Generate content and capture usage via hook mechanism."""
        captured: List[Dict[str, Any]] = []

        def _capture(payload: Dict[str, Any]) -> None:
            captured.append(payload)

        # Rate limit LLM calls to prevent API overload
        with _LLM_SEMAPHORE:
            try:
                from modules.memory.application.metrics import gauge_inc, gauge_dec  # local import to avoid cycles
            except Exception:
                gauge_inc = gauge_dec = None  # type: ignore[assignment]
            if callable(gauge_inc):
                try:
                    gauge_inc("llm_inflight", 1)
                except Exception:
                    pass
            # Hook into the contextvar
            token = _LLM_USAGE_HOOK.set(_capture)
            try:
                content = self._fn(messages, response_format)
                usage_obj = None
                if captured:
                    # Assuming the last event is the one (or the only one)
                    last = captured[-1]
                    usage_obj = LLMUsage(
                        provider=str(last.get("provider") or "unknown"),
                        model=str(last.get("model") or "unknown"),
                        prompt_tokens=int(last.get("prompt_tokens") or 0),
                        completion_tokens=int(last.get("completion_tokens") or 0),
                        total_tokens=int(last.get("total_tokens") or 0),
                        cost_usd=None,
                    )
                return content, usage_obj
            finally:
                _LLM_USAGE_HOOK.reset(token)
                if callable(gauge_dec):
                    try:
                        gauge_dec("llm_inflight", 1)
                    except Exception:
                        pass


class LLMUsageContext:
    def __init__(
        self,
        *,
        tenant_id: Optional[str],
        api_key_id: Optional[str],
        request_id: Optional[str],
        stage: Optional[str],
        job_id: Optional[str],
        session_id: Optional[str],
        call_index: Optional[int],
        source: Optional[str],
        byok_route: Optional[str] = None,
        credential_fingerprint: Optional[str] = None,
        resolver_status: Optional[str] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.api_key_id = api_key_id
        self.request_id = request_id
        self.stage = stage
        self.job_id = job_id
        self.session_id = session_id
        self.call_index = call_index
        self.source = source
        self.byok_route = byok_route
        self.credential_fingerprint = credential_fingerprint
        self.resolver_status = resolver_status


_LLM_USAGE_CTX: contextvars.ContextVar[Optional[LLMUsageContext]] = contextvars.ContextVar(
    "llm_usage_ctx", default=None
)
_LLM_USAGE_HOOK: contextvars.ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = contextvars.ContextVar(
    "llm_usage_hook", default=None
)


def set_llm_usage_context(ctx: Optional[LLMUsageContext]) -> contextvars.Token[Optional[LLMUsageContext]]:
    return _LLM_USAGE_CTX.set(ctx)


def reset_llm_usage_context(token: contextvars.Token[Optional[LLMUsageContext]]) -> None:
    _LLM_USAGE_CTX.reset(token)


def set_llm_usage_hook(
    hook: Optional[Callable[[Dict[str, Any]], None]]
) -> contextvars.Token[Optional[Callable[[Dict[str, Any]], None]]]:
    return _LLM_USAGE_HOOK.set(hook)


def reset_llm_usage_hook(token: contextvars.Token[Optional[Callable[[Dict[str, Any]], None]]]) -> None:
    _LLM_USAGE_HOOK.reset(token)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
        if n < 0:
            return None
        return n
    except Exception:
        return None


def _extract_usage_from_response(resp: Any) -> Dict[str, Optional[int]]:
    usage: Dict[str, Optional[int]] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if resp is None:
        return usage
    try:
        meta = getattr(resp, "usage_metadata", None)
    except Exception:
        meta = None
    if meta is None:
        try:
            meta = resp.get("usage_metadata") if isinstance(resp, dict) else None
        except Exception:
            meta = None
    if meta:
        if isinstance(meta, dict):
            usage["prompt_tokens"] = _safe_int(
                meta.get("prompt_token_count") or meta.get("input_token_count") or meta.get("prompt_tokens")
            )
            usage["completion_tokens"] = _safe_int(
                meta.get("candidates_token_count") or meta.get("output_token_count") or meta.get("completion_tokens")
            )
            usage["total_tokens"] = _safe_int(meta.get("total_token_count") or meta.get("total_tokens"))
            return usage
        usage["prompt_tokens"] = _safe_int(
            getattr(meta, "prompt_token_count", None)
            or getattr(meta, "input_token_count", None)
            or getattr(meta, "prompt_tokens", None)
        )
        usage["completion_tokens"] = _safe_int(
            getattr(meta, "candidates_token_count", None)
            or getattr(meta, "output_token_count", None)
            or getattr(meta, "completion_tokens", None)
        )
        usage["total_tokens"] = _safe_int(
            getattr(meta, "total_token_count", None) or getattr(meta, "total_tokens", None)
        )
        return usage
    data = getattr(resp, "usage", None)
    if data is None:
        try:
            data = resp.get("usage") if isinstance(resp, dict) else None
        except Exception:
            data = None
    if not data:
        return usage
    if isinstance(data, dict):
        usage["prompt_tokens"] = _safe_int(data.get("prompt_tokens") or data.get("input_tokens"))
        usage["completion_tokens"] = _safe_int(data.get("completion_tokens") or data.get("output_tokens"))
        usage["total_tokens"] = _safe_int(data.get("total_tokens"))
        return usage
    for key in ("prompt_tokens", "input_tokens"):
        val = getattr(data, key, None)
        if val is not None:
            usage["prompt_tokens"] = _safe_int(val)
            break
    for key in ("completion_tokens", "output_tokens"):
        val = getattr(data, key, None)
        if val is not None:
            usage["completion_tokens"] = _safe_int(val)
            break
    usage["total_tokens"] = _safe_int(getattr(data, "total_tokens", None))
    return usage


def _fetch_openrouter_generation_stats(
    *,
    generation_id: str,
    api_key: str,
    api_base: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch OpenRouter generation stats to obtain native token counts and cost."""
    try:
        import requests  # type: ignore
    except Exception:
        return None
    gen_id = str(generation_id or "").strip()
    if not gen_id:
        return None
    base = (api_base or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base}/generation"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, params={"id": gen_id}, timeout=8)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json() or {}
    except Exception:
        return None
    # OpenRouter returns a normalized payload; keep parsing defensive.
    payload = data.get("data") if isinstance(data, dict) else None
    if payload is None and isinstance(data, dict):
        payload = data
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage") or {}
    cost = (
        payload.get("cost")
        or payload.get("cost_usd")
        or payload.get("total_cost")
        or payload.get("total_cost_usd")
    )
    try:
        cost_usd = float(cost) if cost is not None else None
    except Exception:
        cost_usd = None
    usage_dict = _extract_usage_from_response({"usage": usage})
    return {"usage": usage_dict, "cost_usd": cost_usd}


def _emit_llm_usage(
    *,
    provider: str,
    model: str,
    usage: Optional[Dict[str, Optional[int]]] = None,
    status: str = "ok",
    error_code: Optional[str] = None,
    error_detail: Optional[str] = None,
    generation_id: Optional[str] = None,
    cost_usd: Optional[float] = None,
) -> None:
    hook = _LLM_USAGE_HOOK.get()
    if hook is None:
        return
    ctx = _LLM_USAGE_CTX.get()
    call_index: Optional[int] = None
    if ctx is not None:
        try:
            if ctx.call_index is None:
                ctx.call_index = 0
            else:
                ctx.call_index = int(ctx.call_index) + 1
            call_index = int(ctx.call_index)
        except Exception:
            call_index = ctx.call_index
    usage = usage or {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    payload: Dict[str, Any] = {
        "event_type": "llm",
        "tenant_id": getattr(ctx, "tenant_id", None) if ctx else None,
        "api_key_id": getattr(ctx, "api_key_id", None) if ctx else None,
        "request_id": getattr(ctx, "request_id", None) if ctx else None,
        "job_id": getattr(ctx, "job_id", None) if ctx else None,
        "session_id": getattr(ctx, "session_id", None) if ctx else None,
        "stage": getattr(ctx, "stage", None) if ctx else None,
        "call_index": call_index if call_index is not None else (getattr(ctx, "call_index", None) if ctx else None),
        "source": getattr(ctx, "source", None) if ctx else None,
        "byok_route": getattr(ctx, "byok_route", None) if ctx else None,
        "credential_fingerprint": getattr(ctx, "credential_fingerprint", None) if ctx else None,
        "resolver_status": getattr(ctx, "resolver_status", None) if ctx else None,
        "provider": provider,
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": cost_usd,
        "generation_id": generation_id,
        "status": status,
        "error_code": error_code,
        "error_detail": error_detail,
        "tokens_missing": bool(usage.get("prompt_tokens") is None and usage.get("completion_tokens") is None),
    }
    try:
        hook(payload)
    except Exception:
        return


def _map_messages_for_multimodal(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map our generic messages (may contain 'media') into a format that LiteLLM/OpenAI-compatible endpoints understand.

    Our convention:
    - Each message may have a 'media': [{"type": "image", "data_url": "data:image/jpeg;base64,..."}, ...]
    - We transform such a message into OpenAI vision-style content parts:
        {"role": "user", "content": [
            {"type": "text", "text": "...original content string..."},
            {"type": "image_url", "image_url": {"url": "data:image/..."}},
            ...
        ]}
    - If content is not a string, we keep it as-is and append image_url parts if possible.

    This generic mapping tends to work with many providers via LiteLLM. Providers that don't
    support images should ignore the non-text parts or return an error upstream.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        media = m.get("media")
        if not media:
            out.append(m)
            continue
        parts: List[Dict[str, Any]] = []
        content = m.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            # assume already in parts form; keep
            parts.extend(content)
        elif content is not None:
            parts.append({"type": "text", "text": str(content)})
        # add images
        for x in media:
            if isinstance(x, dict) and x.get("type") == "image" and x.get("data_url"):
                parts.append({"type": "image_url", "image_url": {"url": str(x.get("data_url"))}})
        nm = {k: v for k, v in m.items() if k != "content" and k != "media"}
        nm["content"] = parts
        out.append(nm)
    return out


def _litellm_adapter(
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
) -> Optional[LLMAdapter]:
    """Build an adapter using LiteLLM to unify providers.

    If api_base/api_key is provided, route via custom OpenAI-compatible endpoint.
    Otherwise, LiteLLM will infer provider from model string + env keys.
    """
    try:
        import litellm  # type: ignore

        def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
            params: Dict[str, Any] = {"model": model, "messages": messages}
            # Respect YAML-configured mapping strategy
            try:
                from modules.memory.application.config import load_memory_config, get_llm_multimodal_mapping  # type: ignore
                cfg = load_memory_config()
                strategy = get_llm_multimodal_mapping(cfg)
            except Exception:
                strategy = "generic_image_url"
            if strategy == "generic_image_url":
                try:
                    params["messages"] = _map_messages_for_multimodal(messages)
                except Exception:
                    pass
            if api_base:
                params["api_base"] = api_base
            if api_key:
                params["api_key"] = api_key
            if custom_llm_provider:
                params["custom_llm_provider"] = custom_llm_provider
            if response_format is not None:
                params["response_format"] = response_format
            try:
                resp = litellm.completion(**params)
                usage = _extract_usage_from_response(resp)
                generation_id = None
                try:
                    generation_id = getattr(resp, "id", None) or (resp.get("id") if isinstance(resp, dict) else None)
                except Exception:
                    generation_id = None
                cost_usd = None
                if str(model).startswith("openrouter/"):
                    key = api_key or os.getenv("OPENROUTER_API_KEY")
                    base = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
                    if key and generation_id:
                        stats = _fetch_openrouter_generation_stats(
                            generation_id=str(generation_id),
                            api_key=str(key),
                            api_base=base,
                        )
                        if stats:
                            usage = stats.get("usage") or usage
                            cost_usd = stats.get("cost_usd")
                _emit_llm_usage(
                    provider="litellm",
                    model=str(model),
                    usage=usage,
                    generation_id=(str(generation_id) if generation_id else None),
                    cost_usd=cost_usd,
                )
                try:
                    # chat style response
                    content = resp.choices[0].message["content"]  # type: ignore
                except Exception:
                    # text style response
                    content = getattr(resp, "choices", [{}])[0].get("text", "{}")  # type: ignore
                return content or "{}"
            except Exception as exc:
                _emit_llm_usage(
                    provider="litellm",
                    model=str(model),
                    status="fail",
                    error_code="LITELLM_ERROR",
                    error_detail=str(exc)[:500],
                )
                raise

        return LLMAdapter(_fn, kind="litellm")
    except Exception:
        return None


def _glm_http_adapter(model: str, api_key: str, api_base: Optional[str] = None) -> Optional[LLMAdapter]:
    """Direct HTTP adapter for GLM chat completions (avoids provider autodetection issues).

    - Accepts OpenAI-style messages with content parts (text / image_url)
    - POSTs to https://open.bigmodel.cn/api/coding/paas/v4/chat/completions by default
    - Ignores response_format (service may not support OpenAI response_format)
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    base = (api_base or "https://open.bigmodel.cn/api/coding/paas/v4").rstrip("/")
    url = f"{base}/chat/completions"

    # normalize model – GLM 接口对大小写宽松，但与直连探针一致更稳妥
    mdl = model
    try:
        if model.strip().lower() == "glm-4.5v":
            mdl = "GLM-4.5V"
    except Exception:
        pass

    def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Map chat_api style `media` → OpenAI content parts to ensure images are delivered
        try:
            mapped_msgs = _map_messages_for_multimodal(messages)
        except Exception:
            mapped_msgs = messages

        payload: Dict[str, Any] = {
            "model": mdl,
            "messages": mapped_msgs,
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.0,
        }
        # GLM 官方未宣称支持 OpenAI response_format；若被要求严格 JSON，请在调用者侧做提示词约束与解析兜底
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            usage = _extract_usage_from_response(data)
            _emit_llm_usage(provider="glm_http", model=str(mdl), usage=usage)
            # 兼容多种承载位置：message.content / message.reasoning_content / choices[].content / 顶层 output_text
            content = ""
            try:
                msg = (data.get("choices") or [{}])[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning_content") or ""
            except Exception:
                content = ""
            if not content:
                try:
                    content = (data.get("choices") or [{}])[0].get("content") or ""
                except Exception:
                    pass
            if not content:
                content = (
                    data.get("output_text")
                    or (data.get("output") or {}).get("text")
                    or ""
                )
            # fallbacks: 部分返回将 JSON 包在文本中（如 <|begin_of_box|>...）—调用者可自行剥壳
            return content or json.dumps(data, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"glm_http_error: {e}")

    return LLMAdapter(_fn, kind="glm_http")



def _openrouter_http_adapter(model: str, api_key: str, api_base: Optional[str] = None) -> Optional[LLMAdapter]:
    """Direct HTTP adapter for OpenRouter chat completions (Google Gemini via OpenRouter).

    - Accepts OpenAI-style messages with optional image_url parts.
    - Adds optional Referer/X-Title headers from env (`OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME`).
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    base = (api_base or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base}/chat/completions"

    referer = os.getenv("OPENROUTER_SITE_URL")
    site_title = os.getenv("OPENROUTER_SITE_NAME")

    def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if referer:
            headers["HTTP-Referer"] = referer
        if site_title:
            headers["X-Title"] = site_title

        try:
            mapped_msgs = _map_messages_for_multimodal(messages)
        except Exception:
            mapped_msgs = messages

        try:
            max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "30000") or "30000")
        except Exception:
            max_tokens = 30000
        if max_tokens <= 0:
            max_tokens = 30000
        max_tokens = min(max_tokens, 65536)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": mapped_msgs,
            "stream": False,
            # Many OpenRouter upstream providers expect these to be present.
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code >= 400:
                # Include response body for debuggability (OpenRouter often returns JSON error details).
                body = ""
                try:
                    body = resp.text or ""
                except Exception:
                    body = ""
                _emit_llm_usage(
                    provider="openrouter_http",
                    model=str(model),
                    status="fail",
                    error_code=f"HTTP_{resp.status_code}",
                    error_detail=body[:800],
                )
                raise RuntimeError(f"openrouter_http_{resp.status_code}: {body[:800]}")
            data = resp.json()
            usage = _extract_usage_from_response(data)
            generation_id = None
            try:
                generation_id = data.get("id")
            except Exception:
                generation_id = None
            cost_usd = None
            if generation_id:
                stats = _fetch_openrouter_generation_stats(
                    generation_id=str(generation_id),
                    api_key=api_key,
                    api_base=base,
                )
                if stats:
                    usage = stats.get("usage") or usage
                    cost_usd = stats.get("cost_usd")
            _emit_llm_usage(
                provider="openrouter_http",
                model=str(model),
                usage=usage,
                generation_id=(str(generation_id) if generation_id else None),
                cost_usd=cost_usd,
            )
            # Follow OpenAI-compatible structure
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # Concatenate text segments
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                content = "\n".join([t for t in texts if t])
            if not content:
                content = choice.get("text") or data.get("output_text") or ""
            return content or json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            _emit_llm_usage(
                provider="openrouter_http",
                model=str(model),
                status="fail",
                error_code="OPENROUTER_HTTP_ERROR",
                error_detail=str(exc)[:800],
            )
            raise RuntimeError(f"openrouter_http_error: {exc}")

    return LLMAdapter(_fn, kind="openrouter_http")


def _dashscope_http_adapter(model: str, api_key: str, api_base: Optional[str] = None) -> Optional[LLMAdapter]:
    """Direct HTTP adapter for DashScope (Qwen) chat completions.

    - Accepts OpenAI-style messages
    - Uses DashScope's OpenAI-compatible API endpoint
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    base = (api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    url = f"{base}/chat/completions"

    def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            mapped_msgs = _map_messages_for_multimodal(messages)
        except Exception:
            mapped_msgs = messages

        payload: Dict[str, Any] = {
            "model": model,
            "messages": mapped_msgs,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            usage = _extract_usage_from_response(data)
            _emit_llm_usage(provider="dashscope_http", model=str(model), usage=usage)
            # Follow OpenAI-compatible structure
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # Concatenate text segments
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                content = "\n".join([t for t in texts if t])
            if not content:
                content = choice.get("text") or data.get("output_text") or ""
            return content or json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError(f"dashscope_http_error: {exc}")

    return LLMAdapter(_fn, kind="dashscope_http")




def _sglang_http_adapter(
    model: str,
    base_url: str = "http://localhost:30000",
    api_key: Optional[str] = None,
) -> Optional[LLMAdapter]:
    """Direct HTTP adapter for SGLang OpenAI-compatible API.

    SGLang provides OpenAI-compatible endpoints at /v1/chat/completions.
    This adapter passes messages through as-is; multimodal mapping (if needed)
    is handled by the caller based on mapping_strategy configuration.

    Args:
        model: Model name (as served by SGLang, e.g., "Qwen/Qwen3-VL-2B-Instruct")
        base_url: SGLang server URL (default: http://localhost:30000)
        api_key: Optional API key (SGLang typically doesn't require one)
    """
    try:
        import requests  # type: ignore
    except Exception:
        return None

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Pass messages through as-is; mapping is applied upstream if configured
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            usage = _extract_usage_from_response(data)
            _emit_llm_usage(provider="sglang_http", model=str(model), usage=usage)
            # OpenAI-compatible response structure
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # Concatenate text segments
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                content = "\n".join([t for t in texts if t])
            if not content:
                content = choice.get("text") or ""
            return content or json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise RuntimeError(f"sglang_http_error: {exc}")

    return LLMAdapter(_fn, kind="sglang_http")


def _gemini_genai_adapter(model: str, api_key: str) -> Optional[LLMAdapter]:
    """Adapter for Google official genai SDK with multi-image support.

    Maps our generic messages (OpenAI-style content parts) into Google GenAI contents:
    - Text parts → strings
    - image_url parts:
        * Local file path → upload via client.files.upload
        * Data URL (base64) → types.Part.from_bytes
        * HTTP(S) URL → try types.Part.from_uri; fallback to include URL as text
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        return None

    client = genai.Client(api_key=api_key)

    def _is_data_url(url: str) -> bool:
        return isinstance(url, str) and url.startswith("data:image")

    def _from_data_url(url: str):
        try:
            import base64
            head, b64 = url.split(",", 1)
            mime = "image/jpeg"
            if ";" in head and head.startswith("data:"):
                mime = head[5:].split(";")[0] or mime
            raw = base64.b64decode(b64)
            return types.Part.from_bytes(data=raw, mime_type=mime)
        except Exception:
            return None

    def _messages_to_contents(messages: List[Dict[str, Any]]):
        contents: List[Any] = []
        # flatten all messages into a single prompt
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                if content.strip():
                    contents.append(content)
            elif isinstance(content, list):
                for part in content:
                    try:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t = str(part.get("text") or "").strip()
                            if t:
                                contents.append(t)
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            url = (part.get("image_url") or {}).get("url")
                            if not isinstance(url, str):
                                continue
                            if _is_data_url(url):
                                p = _from_data_url(url)
                                if p is not None:
                                    contents.append(p)
                                    continue
                            # local file path
                            if os.path.exists(url):
                                try:
                                    uploaded = client.files.upload(file=url)
                                    contents.append(uploaded)
                                    continue
                                except Exception:
                                    pass
                            # http(s) url
                            try:
                                p2 = types.Part.from_uri(url=url)
                                contents.append(p2)
                            except Exception:
                                # fallback: keep url as text (last resort)
                                contents.append(url)
                    except Exception:
                        continue
            elif content is not None:
                contents.append(str(content))
        return contents

    def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        try:
            mdl = os.getenv("GEMINI_MODEL", model) or model
            contents = _messages_to_contents(messages)
            resp = client.models.generate_content(model=mdl, contents=contents)
            usage = _extract_usage_from_response(resp)
            _emit_llm_usage(provider="google_genai", model=str(mdl), usage=usage)
            # google genai SDK: resp.text contains concatenated text
            out = getattr(resp, "text", None)
            if isinstance(out, str) and out.strip():
                return out
            # fallback to raw JSON
            try:
                return json.dumps(resp, ensure_ascii=False)  # type: ignore[arg-type]
            except Exception:
                return "{}"
        except Exception as exc:
            raise RuntimeError(f"gemini_genai_error: {exc}")

    return LLMAdapter(_fn, kind="google_genai")


def build_llm_from_env() -> Optional[LLMAdapter]:
    """Construct an LLMAdapter from environment variables.

    Priority:
    1. SGLang local server (SGLANG_BASE_URL + SGLANG_MODEL)
    2. Custom OpenAI-compatible base (LLM_BASE_URL + LLM_API_KEY + LLM_MODEL)
    3. LLM_MODEL only -> let LiteLLM route by model name + env keys
    4. Specific provider env keys (OPENAI_API_KEY, etc.)
    5. None
    """
    # SGLang local server (highest priority for local development)
    sglang_base = os.getenv("SGLANG_BASE_URL")
    sglang_model = os.getenv("SGLANG_MODEL")
    if sglang_base and sglang_model:
        sglang_key = os.getenv("SGLANG_API_KEY")  # Optional
        adapter = _sglang_http_adapter(model=sglang_model, base_url=sglang_base, api_key=sglang_key)
        if adapter:
            return adapter

    # Custom OpenAI-compatible base
    base = os.getenv("LLM_BASE_URL")
    key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if base and key and model:
        return _litellm_adapter(
            model=model,
            api_base=base,
            api_key=key,
            custom_llm_provider="openai",
        )

    # Direct model routing via LiteLLM
    model2 = os.getenv("LLM_MODEL")
    if model2:
        return _litellm_adapter(model=model2)

    # Specific providers via env keys (heuristic)
    # OpenRouter: prefer direct HTTP adapter (LiteLLM routing has had compatibility issues for some deployments)
    if os.getenv("OPENROUTER_API_KEY"):
        m = str(os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "").strip()
        if not m:
            m = "openai/gpt-4o-mini"
        # If user gave a bare model name (e.g. "gpt-4o-mini"), assume OpenAI namespace.
        if "/" not in m:
            m = f"openai/{m}"
        key2 = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
        adapter = _openrouter_http_adapter(model=m, api_key=key2, api_base=os.getenv("OPENROUTER_BASE_URL"))
        if adapter is not None:
            return adapter
        # Last-resort fallback: LiteLLM routing
        m2 = m if m.startswith("openrouter/") else f"openrouter/{m}"
        return _litellm_adapter(model=m2, api_key=key2)
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        m = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return _litellm_adapter(model=m)
    # DeepSeek
    if os.getenv("DEEPSEEK_API_KEY"):
        m = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return _litellm_adapter(model=m, api_base="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
    # Moonshot
    if os.getenv("MOONSHOT_API_KEY"):
        m = os.getenv("MOONSHOT_MODEL", "moonshot-v1-8k")
        return _litellm_adapter(model=m, api_base="https://api.moonshot.cn/v1", api_key=os.getenv("MOONSHOT_API_KEY"))
    # Qwen
    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"):
        m = os.getenv("QWEN_MODEL", "qwen2.5")
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        return _litellm_adapter(model=m, api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key)
    # GLM
    if os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY"):
        m = os.getenv("GLM_MODEL", "glm-4.6")
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
        # GLM Coding 接口使用全量路径 https://open.bigmodel.cn/api/coding/paas/v4/chat/completions
        # LiteLLM 会在 api_base 后附加 /chat/completions，因此这里使用到 /paas/v4 级别
        return _litellm_adapter(
            model=m,
            api_base="https://open.bigmodel.cn/api/coding/paas/v4",
            api_key=api_key,
        )
    # Gemini
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        m = os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-flash")
        return _litellm_adapter(model=m)

    return None


def build_llm_from_config(kind: str = "text") -> Optional[LLMAdapter]:
    """Construct an LLMAdapter from memory.config.yaml selections for the given kind.

    - kind: 'text' or 'multimodal'
    - Uses env-provided API keys; picks model/provider from YAML
    - Applies multimodal mapping_strategy (generic_image_url/none) when kind='multimodal'
    """
    try:
        from modules.memory.application.config import load_memory_config, get_llm_selection, get_llm_multimodal_mapping  # type: ignore
    except Exception:
        return None
    cfg = load_memory_config()
    sel = get_llm_selection(cfg, kind)
    provider = (sel.get("provider") or "").lower()
    model = sel.get("model") or ""
    if not model:
        return build_llm_from_env()

    # Map provider to LiteLLM model string
    m_final = model
    adapter: Optional[LLMAdapter] = None

    # SGLang local server (highest priority for local/on-premise deployment)
    if provider == "sglang":
        base_url = os.getenv("SGLANG_BASE_URL", "http://localhost:30000")
        sglang_key = os.getenv("SGLANG_API_KEY")  # Optional
        adapter = _sglang_http_adapter(model=model, base_url=base_url, api_key=sglang_key)
        if adapter is None:
            # Fallback to litellm with custom base if sglang adapter fails
            adapter = _litellm_adapter(model=model, api_base=f"{base_url}/v1", api_key=sglang_key)
        # Don't return early; continue to mapping_strategy logic below

    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            adapter = _openrouter_http_adapter(model=model, api_key=key, api_base=os.getenv("OPENROUTER_BASE_URL"))
        if adapter is None and not model.startswith("openrouter/"):
            m_final = f"openrouter/{model}"
    
    elif provider == "qwen" or provider == "dashscope":
        # Use DashScope HTTP adapter for Qwen models
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if key:
            adapter = _dashscope_http_adapter(model=model, api_key=key, api_base=os.getenv("DASHSCOPE_BASE_URL"))
    
    # For openai/ gemini/ glm, LiteLLM recognizes by model string and env; keep as-is
    # For openai_compat/custom, prefer env LLM_BASE_URL/LLM_API_KEY; if set, handled by build_llm_from_env fallback

    # Build adapter via LiteLLM if not already set by provider-specific logic
    if adapter is None:
        try:
            pass  # type: ignore
        except Exception:
            return None

        # reuse _litellm_adapter and apply mapping based on strategy
        base = None
        key = None
        if provider == "openai_compat" and os.getenv("LLM_BASE_URL") and os.getenv("LLM_API_KEY"):
            base = os.getenv("LLM_BASE_URL")
            key = os.getenv("LLM_API_KEY")
        elif provider == "glm":
            # Prefer direct HTTP adapter for GLM to avoid provider autodetection issues in litellm
            key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
            if key:
                adapter = _glm_http_adapter(model=m_final, api_key=key, api_base=os.getenv("GLM_API_BASE"))
            # Fallback: try litellm with explicit namespacing if key missing or requests not available
            if adapter is None:
                if not m_final.lower().startswith("zhipuai/"):
                    m_final = f"zhipuai/{m_final}"
                base = None

        if provider == "gemini":
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            mdl = os.getenv("GEMINI_MODEL", model) or model
            if key:
                adapter = _gemini_genai_adapter(model=mdl, api_key=key)
            # fallback to litellm if SDK not available
            if adapter is None:
                m_final = mdl
                if not m_final.startswith("gemini/"):
                    m_final = f"gemini/{m_final}"

        # Final fallback to litellm
        if adapter is None:
            custom_provider = "openai" if provider == "openai_compat" and base and key else None
            adapter = _litellm_adapter(
                model=m_final,
                api_base=base,
                api_key=key,
                custom_llm_provider=custom_provider,
            )
            if adapter is None:
                return build_llm_from_env()

    # Final check: if no adapter was created, fallback to env-based
    if adapter is None:
        return build_llm_from_env()

    # Wrap to enforce mapping strategy for multimodal
    if kind == "multimodal":
        strategy = get_llm_multimodal_mapping(cfg)
        if strategy == "generic_image_url":
            inner = adapter

            def _fn(messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
                try:
                    msgs = _map_messages_for_multimodal(messages)
                except Exception:
                    msgs = messages
                return inner.generate(msgs, response_format)

            return LLMAdapter(_fn, kind=getattr(inner, "kind", "unknown"))
    return adapter


def build_llm_from_byok(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
) -> Optional[LLMAdapter]:
    prov = str(provider or "").strip().lower()
    mdl = str(model or "").strip()
    key = str(api_key or "").strip()
    if not prov or not mdl or not key:
        return None

    if prov in {"openrouter"}:
        adapter = _openrouter_http_adapter(model=mdl, api_key=key, api_base=base_url)
        if adapter is not None:
            return adapter
        m_final = mdl if mdl.startswith("openrouter/") else f"openrouter/{mdl}"
        return _litellm_adapter(model=m_final, api_key=key, api_base=base_url)

    if prov in {"qwen", "dashscope", "aliyun"}:
        return _dashscope_http_adapter(model=mdl, api_key=key, api_base=base_url)

    if prov in {"glm", "zhipuai"}:
        adapter = _glm_http_adapter(model=mdl, api_key=key, api_base=base_url)
        if adapter is not None:
            return adapter
        return _litellm_adapter(model=mdl, api_key=key, api_base=base_url)

    if prov in {"gemini", "google", "google_genai"}:
        adapter = _gemini_genai_adapter(model=mdl, api_key=key)
        if adapter is not None:
            return adapter
        m_final = mdl if mdl.startswith("gemini/") else f"gemini/{mdl}"
        return _litellm_adapter(model=m_final, api_key=key, api_base=base_url)

    if prov in {"deepseek"}:
        return _litellm_adapter(
            model=mdl,
            api_base=base_url or "https://api.deepseek.com",
            api_key=key,
            custom_llm_provider="deepseek",
        )

    if prov in {"moonshot", "kimi"}:
        return _litellm_adapter(
            model=mdl,
            api_base=base_url or "https://api.moonshot.cn/v1",
            api_key=key,
        )

    if prov in {"openai", "openai_compat", "openai-compatible", "openai_compatible"}:
        return _litellm_adapter(
            model=mdl,
            api_base=base_url,
            api_key=key,
            custom_llm_provider="openai",
        )

    # Default: let LiteLLM infer provider from model.
    return _litellm_adapter(model=mdl, api_base=base_url, api_key=key)


def _norm_opt_str(value: Optional[str]) -> Optional[str]:
    raw = str(value or "").strip()
    return raw or None


def _normalize_sglang_base_url(base_url: Optional[str]) -> Optional[str]:
    base = _norm_opt_str(base_url)
    if not base:
        return None
    trimmed = base.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return f"{trimmed}/v1"


def resolve_openai_compatible_chat_target(
    *,
    kind: str = "agentic_router",
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """Resolve provider/model/api_key/base_url for OpenAI-compatible chat calls.

    - Primary source: memory.config.yaml `memory.llm.<kind>`
    - Fallback source: `memory.llm.text`
    - Optional env/runtime overrides:
      - provider_override
      - model_override
      - base_url_override
    """

    try:
        from modules.memory.application.config import get_llm_selection, load_memory_config
    except Exception:
        return None

    cfg = load_memory_config()
    selected = get_llm_selection(cfg, kind)
    fallback = get_llm_selection(cfg, "text")

    provider = (
        _norm_opt_str(provider_override)
        or _norm_opt_str(selected.get("provider"))
        or _norm_opt_str(fallback.get("provider"))
        or _norm_opt_str(os.getenv("LLM_PROVIDER"))
        or "openai"
    ).lower()
    model = (
        _norm_opt_str(model_override)
        or _norm_opt_str(selected.get("model"))
        or _norm_opt_str(fallback.get("model"))
        or _norm_opt_str(os.getenv("LLM_MODEL"))
    )
    if not model:
        if provider == "openai":
            model = _norm_opt_str(os.getenv("OPENAI_MODEL"))
        elif provider == "openrouter":
            model = _norm_opt_str(os.getenv("OPENROUTER_MODEL"))
        elif provider in {"qwen", "dashscope", "aliyun"}:
            model = _norm_opt_str(os.getenv("QWEN_MODEL"))
        elif provider in {"glm", "zhipuai"}:
            model = _norm_opt_str(os.getenv("GLM_MODEL"))
        elif provider == "deepseek":
            model = _norm_opt_str(os.getenv("DEEPSEEK_MODEL"))
        elif provider in {"moonshot", "kimi"}:
            model = _norm_opt_str(os.getenv("MOONSHOT_MODEL"))
        elif provider in {"gemini", "google"}:
            model = _norm_opt_str(os.getenv("GEMINI_MODEL"))
    if not model:
        return None

    base_override = _norm_opt_str(base_url_override)
    api_key = ""
    base_url: Optional[str] = None

    if provider == "openrouter":
        api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
        base_url = base_override or str(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        if model.startswith("openrouter/"):
            model = model[len("openrouter/") :]
    elif provider in {"qwen", "dashscope", "aliyun"}:
        api_key = str(os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY") or "").strip()
        base_url = (
            base_override
            or str(os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        )
    elif provider in {"glm", "zhipuai"}:
        api_key = str(os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY") or "").strip()
        base_url = base_override or str(os.getenv("GLM_API_BASE") or "https://open.bigmodel.cn/api/coding/paas/v4").strip()
    elif provider == "deepseek":
        api_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
        base_url = base_override or str(os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1").strip()
    elif provider in {"moonshot", "kimi"}:
        api_key = str(os.getenv("MOONSHOT_API_KEY") or "").strip()
        base_url = base_override or str(os.getenv("MOONSHOT_API_BASE") or "https://api.moonshot.cn/v1").strip()
    elif provider in {"sglang"}:
        api_key = str(os.getenv("SGLANG_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
        base_url = _normalize_sglang_base_url(base_override or os.getenv("SGLANG_BASE_URL") or "http://localhost:30000")
    elif provider in {"openai_compat", "openai-compatible", "openai_compatible"}:
        api_key = str(os.getenv("LLM_API_KEY") or "").strip()
        base_url = base_override or _norm_opt_str(os.getenv("LLM_BASE_URL"))
    elif provider in {"gemini", "google"}:
        # Gemini native SDK path is not OpenAI-compatible by default.
        api_key = str(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        base_url = base_override or _norm_opt_str(os.getenv("GEMINI_BASE_URL"))
    elif provider in {"openai", ""}:
        api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
        base_url = base_override
    else:
        # Unknown provider: treat as OpenAI-compatible custom endpoint.
        api_key = str(os.getenv("LLM_API_KEY") or "").strip()
        base_url = base_override or _norm_opt_str(os.getenv("LLM_BASE_URL"))

    base_url = _norm_opt_str(base_url)

    key_required = provider in {
        "openrouter",
        "qwen",
        "dashscope",
        "aliyun",
        "glm",
        "zhipuai",
        "deepseek",
        "moonshot",
        "kimi",
        "gemini",
        "google",
    }
    if key_required and not api_key:
        return None
    if provider in {"openai_compat", "openai-compatible", "openai_compatible", "gemini", "google"} and not base_url:
        return None
    if provider in {"openai", ""} and not api_key and not base_url:
        return None
    if not api_key and base_url:
        # Allow keyless local/self-hosted OpenAI-compatible endpoints
        # (e.g. sglang/vllm/ollama proxy) while still rejecting missing-key
        # provider SaaS paths above.
        api_key = "EMPTY"

    return {
        "provider": provider or "openai",
        "model": model,
        "api_key": api_key or None,
        "base_url": base_url,
    }
