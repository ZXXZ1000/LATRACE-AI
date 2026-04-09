from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
import hashlib
from pathlib import Path

try:
    # 自动加载根目录与模块级 .env，让嵌入配置无需手动 export
    from dotenv import load_dotenv  # type: ignore

    _ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"
    _MOD_ENV = Path(__file__).resolve().parents[2] / "config" / ".env"
    if _ROOT_ENV.exists():
        load_dotenv(_ROOT_ENV, override=False)
    if _MOD_ENV.exists():
        load_dotenv(_MOD_ENV, override=False)
except Exception:
    pass

from modules.memory.contracts.usage_models import EmbeddingUsage


def _sanitize_placeholder(value: Optional[str]) -> str:
    """去除 ${VAR} 或 ${VAR:default} 形式的占位符，避免传递非法 base/key。"""
    if value is None:
        return ""
    s = str(value).strip()
    if s.startswith("${") and s.endswith("}"):
        return ""
    return s


def _iter_tokens(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    if " " in text:
        return text.split()
    # simple char bigrams for languages without spaces (e.g., Chinese)
    toks = []
    s = text
    if len(s) == 1:
        return [s]
    for i in range(len(s) - 1):
        toks.append(s[i : i + 2])
    return toks


def _hash_embed(text: str, dim: int = 1536) -> List[float]:
    toks = _iter_tokens(text)
    if not toks:
        return [0.0] * dim
    buckets = [0.0] * dim
    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = -1.0 if (h >> 1) & 1 else 1.0
        buckets[idx] += sign
    norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
    return [v / norm for v in buckets]



# Usage Hook Mechanism (Analogue to LLMAdapter)
import contextvars
_EMBEDDING_USAGE_HOOK: contextvars.ContextVar[Optional[Callable[[EmbeddingUsage], None]]] = contextvars.ContextVar(
    "embedding_usage_hook", default=None
)

def set_embedding_usage_hook(
    hook: Optional[Callable[[EmbeddingUsage], None]]
) -> contextvars.Token[Optional[Callable[[EmbeddingUsage], None]]]:
    return _EMBEDDING_USAGE_HOOK.set(hook)

def reset_embedding_usage_hook(token: contextvars.Token[Optional[Callable[[EmbeddingUsage], None]]]) -> None:
    _EMBEDDING_USAGE_HOOK.reset(token)

def _emit_embedding_usage(usage: Optional[EmbeddingUsage]) -> None:
    if usage is None:
        return
    hook = _EMBEDDING_USAGE_HOOK.get()
    if hook:
        try:
            hook(usage)
        except Exception:
            pass

def _l2_normalize(vec: List[float]) -> List[float]:
    try:
        import math as _m
        n = _m.sqrt(sum((float(x) * float(x)) for x in vec)) or 1.0
        return [float(x) / n for x in vec]
    except Exception:
        return vec


def _build_openai_sdk_embedder(
    model: str,
    *,
    api_base: Optional[str],
    api_key: Optional[str],
    dim: int,
    embed_concurrency: Optional[int] = None,
) -> Optional[Callable[[str], List[float]]]:
    """通过 OpenAI 官方 SDK 访问兼容端点（DashScope/Qwen 等）。"""
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        OpenAI = None  # type: ignore

    import threading
    import requests

    # 缺少必要密钥时直接放弃，回退哈希
    if not api_key:
        return None

    model_name = model or "text-embedding-v2"
    base = (api_base or "").rstrip("/") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    is_openrouter = "openrouter.ai" in base or "openrouter" in base
    url = f"{base}/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    session_local = threading.local()

    def _get_session() -> requests.Session:
        sess = getattr(session_local, "session", None)
        if sess is None:
            sess = requests.Session()
            session_local.session = sess
        return sess

    # For OpenRouter, prefer direct HTTP to:
    # - capture generation id headers (for cost lookup)
    # - avoid SDK/provider compatibility issues
    client = None
    if not is_openrouter and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key, base_url=base)
        except Exception:
            client = None
            
    pricing_cache: Dict[str, Any] = {"ts": 0.0, "models": {}}

    def _openrouter_estimate_cost_usd(tokens: int) -> Optional[float]:
        """Best-effort cost estimation via OpenRouter /models pricing."""
        try:
            import time
            import requests  # type: ignore
        except Exception:
            return None
        if tokens <= 0:
            return 0.0
        try:
            now = float(time.time())
        except Exception:
            now = 0.0
        # refresh at most once per hour
        if not pricing_cache["models"] or (now - float(pricing_cache.get("ts") or 0.0)) > 3600:
            try:
                resp = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=8)
                if resp.status_code == 200:
                    data = resp.json() or {}
                    items = data.get("data") if isinstance(data, dict) else None
                    if isinstance(items, list):
                        m: Dict[str, Any] = {}
                        for it in items:
                            if not isinstance(it, dict):
                                continue
                            mid = str(it.get("id") or "").strip()
                            if not mid:
                                continue
                            m[mid] = it.get("pricing") or {}
                        pricing_cache["models"] = m
                        pricing_cache["ts"] = now
            except Exception:
                pass
        pricing = (pricing_cache.get("models") or {}).get(str(model_name), {})
        if not isinstance(pricing, dict):
            return None
        # OpenRouter pricing commonly uses USD-per-token strings.
        token_price = (
            pricing.get("prompt")
            or pricing.get("input")
            or pricing.get("prompt_token")
            or pricing.get("input_token")
        )
        request_fee = pricing.get("request")
        try:
            per_tok = float(token_price) if token_price is not None else None
        except Exception:
            per_tok = None
        try:
            per_req = float(request_fee) if request_fee is not None else 0.0
        except Exception:
            per_req = 0.0
        if per_tok is None:
            return None
        try:
            return float(tokens) * per_tok + per_req
        except Exception:
            return None

    def _extract_usage_from_data(
        data: Dict[str, Any],
        provider_k: str = "openai_compat",
        *,
        generation_id: Optional[str] = None,
    ) -> Optional[EmbeddingUsage]:
        try:
            u = data.get("usage")
            if not u:
                return None
            # usage can be dict or object
            if isinstance(u, dict):
                pt = int(u.get("prompt_tokens") or 0)
                tt = int(u.get("total_tokens") or 0)
            else:
                pt = int(getattr(u, "prompt_tokens", 0) or 0)
                tt = int(getattr(u, "total_tokens", 0) or 0)
            tokens = int(pt if pt > 0 else tt)
            cost_usd: Optional[float] = None
            src = "response"
            if is_openrouter:
                # 1) cost fields (if OpenRouter provided them directly)
                raw_cost = (
                    data.get("cost")
                    or data.get("cost_usd")
                    or data.get("total_cost")
                    or data.get("total_cost_usd")
                )
                try:
                    if raw_cost is not None:
                        cost_usd = float(raw_cost)
                except Exception:
                    cost_usd = None
                # 2) generation stats (preferred)
                if cost_usd is None and generation_id:
                    try:
                        from modules.memory.application.llm_adapter import _fetch_openrouter_generation_stats

                        stats = _fetch_openrouter_generation_stats(
                            generation_id=str(generation_id),
                            api_key=str(api_key),
                            api_base=base,
                        )
                        if stats:
                            cost_usd = stats.get("cost_usd")
                            src = "openrouter_generation" if cost_usd is not None else src
                            # prefer OpenRouter native token counts if present
                            u2 = stats.get("usage") or {}
                            try:
                                total2 = int(u2.get("total_tokens") or 0)
                                if total2 > 0:
                                    tokens = total2
                            except Exception:
                                pass
                    except Exception:
                        pass
                # 3) pricing estimation (fallback; still better than null)
                if cost_usd is None:
                    est = _openrouter_estimate_cost_usd(tokens)
                    if est is not None:
                        cost_usd = float(est)
                        src = "estimation"

            return EmbeddingUsage(
                provider=("openrouter" if is_openrouter else provider_k),
                model=model_name,
                tokens=tokens,
                cost_usd=cost_usd,
                source=src,  # type: ignore[arg-type]
            )
        except Exception:
            return None

    def _post_full(payload: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[EmbeddingUsage]]:
        try:
            session = _get_session()
            resp = session.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                return None, None
            data = resp.json()
            embedding = (data.get("data") or [{}])[0].get("embedding")
            gen_id = None
            try:
                gen_id = data.get("id") if isinstance(data, dict) else None
            except Exception:
                gen_id = None
            if not gen_id:
                # OpenRouter may return generation id in headers for OpenAI-compatible endpoints
                for hk in (
                    "x-openrouter-generation-id",
                    "x-openrouter-generation",
                    "x-openrouter-id",
                ):
                    try:
                        v = resp.headers.get(hk)
                        if v:
                            gen_id = str(v).strip()
                            break
                    except Exception:
                        continue
            usage = _extract_usage_from_data(data, "openai_rest", generation_id=(str(gen_id) if gen_id else None))
            if embedding:
                vec = [float(v) for v in embedding]
                return vec, usage
        except Exception:
            return None, None
        return None, None

    def _post_batch(payload: Dict[str, Any]) -> Optional[List[List[float]]]:
        try:
            session = _get_session()
            resp = session.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list):
                return None
            vecs: List[List[float]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                emb = item.get("embedding")
                if emb:
                    vecs.append([float(v) for v in emb])
            return vecs or None
        except Exception:
            return None

    def _embed_with_usage(text: str) -> Tuple[List[float], Optional[EmbeddingUsage]]:
        payload: Dict[str, Any] = {"model": model_name, "input": text}
        if dim and dim > 0:
            payload["dimensions"] = dim
        vec: List[float] = []
        usage: Optional[EmbeddingUsage] = None
        try:
            from modules.memory.application.metrics import gauge_inc, gauge_dec  # local import to avoid cycles
        except Exception:
            gauge_inc = gauge_dec = None  # type: ignore[assignment]
        if callable(gauge_inc):
            try:
                gauge_inc("embedding_inflight", 1)
            except Exception:
                pass

        try:
            if client is not None:
                try:
                    resp = client.embeddings.create(**payload)
                    data = getattr(resp, "data", None) or []
                    embedding = data[0].embedding if data else None  # type: ignore[index]
                    if embedding:
                        vec = [float(v) for v in embedding]
                        # extract usage from SDK response
                        raw_usage = getattr(resp, "usage", None)
                        if raw_usage:
                            usage = _extract_usage_from_data({"usage": raw_usage}, "openai_sdk")
                except Exception:
                    vec = []
                    usage = None

            if not vec:
                vec, usage = _post_full(payload)
        finally:
            if callable(gauge_dec):
                try:
                    gauge_dec("embedding_inflight", 1)
                except Exception:
                    pass
            
        if not vec:
            # fallback hash -> source="missing"
            fallback_vec = _hash_embed(text, dim)
            return fallback_vec, EmbeddingUsage(
                provider="hash_fallback",
                model=model_name,
                tokens=len(_iter_tokens(text)),
                cost_usd=None,
                source="missing"
            )
            
        # normalize
        final_vec = vec
        if len(vec) > dim:
            final_vec = vec[:dim]
        elif len(vec) < dim:
            final_vec = vec + [0.0] * (dim - len(vec))
            
        if usage is None:
            usage = EmbeddingUsage(
                provider="openai_unknown",
                model=model_name,
                tokens=0,
                cost_usd=None,
                source="missing"
            )
        # Emit usage via hook if listeners are present (e.g. QdrantStore context)
        _emit_embedding_usage(usage)
        return final_vec, usage

    def _embed(text: str) -> List[float]:
        # Legacy stub that uses cache
        return _embed_cached(text)

    def _embed_batch(texts: List[str], *, bsz: Optional[int] = None) -> List[List[float]]:
        # ... (Same as before, keep logic for batch without usage support for now)
        # For Phase 1 we don't strictly require batch usage support unless Ingest uses it heavily.
        # But wait, QdrantStore uses encode_batch. If we don't update this, batch calls won't have usage.
        # Currently _embed_batch implementation in original file DOES NOT support usage return.
        # Minimal implementation: just delegate to original logic.
        if not texts:
            return []
        out: List[List[float]] = []
        try:
            b = int(bsz or 32)
        except Exception:
            b = 32
        b = max(1, b)
        if "dashscope.aliyuncs.com" in base:
            b = min(b, 10)
        try:
            embed_conc = int(embed_concurrency or 0)
        except Exception:
            embed_conc = 0
        embed_conc = max(1, embed_conc or 1)

        if client is None:
            def _run_chunk(chunk: List[str]) -> List[List[float]]:
                try:
                    from modules.memory.application.metrics import gauge_inc, gauge_dec  # local import to avoid cycles
                except Exception:
                    gauge_inc = gauge_dec = None  # type: ignore[assignment]
                if callable(gauge_inc):
                    try:
                        gauge_inc("embedding_inflight", 1)
                    except Exception:
                        pass
                payload: Dict[str, Any] = {"model": model_name, "input": chunk}
                if dim and dim > 0:
                    payload["dimensions"] = dim
                try:
                    vecs = _post_batch(payload)
                finally:
                    if callable(gauge_dec):
                        try:
                            gauge_dec("embedding_inflight", 1)
                        except Exception:
                            pass
                if not vecs or len(vecs) < len(chunk):
                    return [_embed(s) for s in chunk]
                if len(vecs) > len(chunk):
                    vecs = vecs[: len(chunk)]
                out_local: List[List[float]] = []
                for vec in vecs:
                    if len(vec) > dim:
                        vec = vec[:dim]
                    elif len(vec) < dim:
                        vec = vec + [0.0] * (dim - len(vec))
                    out_local.append(vec)
                if len(out_local) < len(chunk):
                    out_local.extend([_embed(s) for s in chunk[len(out_local):]])
                return out_local

            chunks = [texts[i : i + b] for i in range(0, len(texts), b)]
            if embed_conc <= 1 or len(chunks) <= 1:
                for chunk in chunks:
                    out.extend(_run_chunk(chunk))
            else:
                results: Dict[int, List[List[float]]] = {}
                with ThreadPoolExecutor(max_workers=embed_conc) as ex:
                    futures = {ex.submit(_run_chunk, chunk): idx for idx, chunk in enumerate(chunks)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        results[idx] = fut.result()
                for idx in sorted(results.keys()):
                    out.extend(results[idx])
            if len(out) < len(texts):
                out.extend([[0.0] * dim for _ in range(len(texts) - len(out))])
            elif len(out) > len(texts):
                out = out[: len(texts)]
            return out

        def _run_chunk_client(chunk: List[str]) -> List[List[float]]:
            payload: Dict[str, Any] = {"model": model_name, "input": chunk}
            if dim and dim > 0:
                payload["dimensions"] = dim
            try:
                try:
                    from modules.memory.application.metrics import gauge_inc, gauge_dec  # local import to avoid cycles
                except Exception:
                    gauge_inc = gauge_dec = None  # type: ignore[assignment]
                if callable(gauge_inc):
                    try:
                        gauge_inc("embedding_inflight", 1)
                    except Exception:
                        pass
                resp = client.embeddings.create(**payload)
                if callable(gauge_dec):
                    try:
                        gauge_dec("embedding_inflight", 1)
                    except Exception:
                        pass
                data = getattr(resp, "data", None) or []
                out_local: List[List[float]] = []
                # WE LOSE USAGE HERE FOR BATCH, ACCEPTABLE FOR PHASE 1.
                for item in data:
                    emb = getattr(item, "embedding", None) or []
                    try:
                        vec = [float(v) for v in emb]
                    except Exception:
                        vec = [0.0] * dim
                    if len(vec) > dim:
                        vec = vec[:dim]
                    elif len(vec) < dim:
                        vec = vec + [0.0] * (dim - len(vec))
                    out_local.append(vec)
                if len(out_local) < len(chunk):
                    out_local.extend([_embed(s) for s in chunk[len(out_local):]])
                return out_local
            except Exception:
                try:
                    from modules.memory.application.metrics import gauge_dec  # local import to avoid cycles
                    if callable(gauge_dec):
                        gauge_dec("embedding_inflight", 1)
                except Exception:
                    pass
                return [_embed(s) for s in chunk]

        chunks = [texts[i : i + b] for i in range(0, len(texts), b)]
        if embed_conc <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                out.extend(_run_chunk_client(chunk))
        else:
            results: Dict[int, List[List[float]]] = {}
            with ThreadPoolExecutor(max_workers=embed_conc) as ex:
                futures = {ex.submit(_run_chunk_client, chunk): idx for idx, chunk in enumerate(chunks)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
            for idx in sorted(results.keys()):
                out.extend(results[idx])

        if len(out) < len(texts):
            out.extend([[0.0] * dim for _ in range(len(texts) - len(out))])
        elif len(out) > len(texts):
            out = out[: len(texts)]
        return out

    # LRU cache to reduce repeated HTTP calls for identical inputs (legacy path)
    try:
        from functools import lru_cache

        @lru_cache(maxsize=4096)
        def _cached(s: str) -> tuple:
            # We call _embed_with_usage internally but discard usage to support caching
            v, _ = _embed_with_usage(s)
            return tuple(v)

        def _embed_cached(s: str) -> List[float]:
            return list(_cached(s))

        try:
            setattr(_embed_cached, "encode_batch", _embed_batch)
        except Exception:
            pass
        
        # Attach new capability
        setattr(_embed_cached, "embed_with_usage", _embed_with_usage)

        return _embed_cached
    except Exception:
        try:
            setattr(_embed, "encode_batch", _embed_batch)
            setattr(_embed, "embed_with_usage", _embed_with_usage)
        except Exception:
            pass
        return _embed


def build_embedding_from_settings(settings: Optional[Dict[str, Any]] = None) -> Callable[[str], List[float]]:
    """Return an embed(text) -> vector function.

    settings example:
      {"provider": "gemini", "model": "gemini-embedding-001", "dim": 1536}

    Fallback: feature-hash embedding.
    """
    cfg = settings or {}
    provider = (cfg.get("provider") or os.getenv("EMBEDDING_PROVIDER") or "").lower()
    model = cfg.get("model") or os.getenv("EMBEDDING_MODEL") or ""
    dim = int(cfg.get("dim") or os.getenv("EMBEDDING_DIM") or 1536)

    if provider == "gemini" and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model_name = model or "models/embedding-001"
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"

            def _embed(text: str) -> List[float]:
                try:
                    resp = genai.embed_content(model=model_name, content=text)
                    vec = resp.get("embedding") or resp.get("data", [{}])[0].get("embedding")
                    if not vec:
                        return _hash_embed(text, dim)
                    # adjust dim
                    if len(vec) == dim:
                        return vec
                    if len(vec) > dim:
                        return vec[:dim]
                    # pad
                    return vec + [0.0] * (dim - len(vec))
                except Exception:
                    return _hash_embed(text, dim)

            # LRU cache for identical inputs
            try:
                from functools import lru_cache

                @lru_cache(maxsize=4096)
                def _cached(s: str) -> tuple:
                    v = _embed(s)
                    return tuple(v)

                def _embed_cached(s: str) -> List[float]:
                    return list(_cached(s))

                return _embed_cached
            except Exception:
                return _embed
        except Exception:
            pass

    # OpenAI-compatible embedding endpoints via LiteLLM (text-only)
    if provider in {"openai_compat", "openai", "openai-compatible"}:
        api_base = _sanitize_placeholder(
            cfg.get("api_base")
            or os.getenv("OPENAI_COMPAT_API_BASE")
            or os.getenv("OPENAI_API_BASE")
        )
        api_key = _sanitize_placeholder(
            cfg.get("api_key")
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        mdl = model or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=cfg.get("embed_concurrency"),
        )
        if fn is not None:
            return fn

    # DashScope/Qwen：走 OpenAI SDK 兼容模式
    if provider in {"qwen", "dashscope", "aliyun"}:
        api_base = _sanitize_placeholder(
            cfg.get("api_base")
            or os.getenv("EMBEDDING_API_BASE")
        )
        if not api_base:
            api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key_candidates = [
            cfg.get("api_key"),
            os.getenv("EMBEDDING_API_KEY"),
            os.getenv("DASHSCOPE_API_KEY"),
            os.getenv("QWEN_API_KEY"),
        ]
        api_key = _sanitize_placeholder(next((v for v in api_key_candidates if v), None))
        mdl = model or "text-embedding-v2"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=cfg.get("embed_concurrency"),
        )
        if fn is not None:
            return fn

    # OpenRouter：走 OpenAI SDK 兼容模式（路由到 OpenAI embedding 等）
    if provider in {"openrouter", "open_router"}:
        api_base = _sanitize_placeholder(
            cfg.get("api_base")
            or os.getenv("OPENROUTER_EMBEDDING_API_BASE")
            or os.getenv("OPENROUTER_BASE_URL")
        )
        if not api_base:
            api_base = "https://openrouter.ai/api/v1"
        api_key_candidates = [
            cfg.get("api_key"),
            os.getenv("OPENROUTER_EMBEDDING_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
        ]
        api_key = _sanitize_placeholder(next((v for v in api_key_candidates if v), None))
        # OpenRouter 支持的 embedding 模型，默认使用 OpenAI text-embedding-3-large
        mdl = model or os.getenv("OPENROUTER_EMBEDDING_MODEL") or "openai/text-embedding-3-large"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=cfg.get("embed_concurrency"),
        )
        if fn is not None:
            return fn

    # Local provider: sentence-transformers or transformers from local path
    if provider in {"local", "offline", "native"}:
        raw_local = (cfg.get("local_path") if cfg else None) or (cfg.get("model") if cfg else None)
        local_path = _sanitize_placeholder(raw_local)
        batch_size = int((cfg.get("batch_size") if cfg else 0) or os.getenv("EMBEDDING_LOCAL_BATCH", 32))
        normalize = bool((cfg.get("normalize") if cfg else True))

        st_model = None
        hf_tokenizer = None
        hf_model = None
        device = "cpu"
        # Avoid accidental remote pulls: only try SentenceTransformer when a valid local_path exists。
        # 优先按配置路径解析；若为相对路径，则尝试以仓库根目录为基准进行解析，确保
        # memory.config.yaml 中的路径（如 modules/memorization_agent/ops/...）在任意 CWD 下都可用。
        from pathlib import Path as _P
        resolved_local: Optional[str] = None
        if local_path:
            p = _P(local_path)
            if not p.is_absolute():
                try:
                    repo_root = Path(__file__).resolve().parents[3]
                    p_root = repo_root / local_path
                    if p_root.exists():
                        p = p_root
                except Exception:
                    # 若无法解析仓库根目录，仍然退回使用相对路径 p
                    pass
            if p.exists():
                resolved_local = str(p)

        if resolved_local:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                st_model = SentenceTransformer(resolved_local, device=None, trust_remote_code=True)
                try:
                    device = str(getattr(st_model, "device", "cpu"))
                except Exception:
                    device = "cpu"
            except Exception:
                st_model = None

        if st_model is None and resolved_local:
            # Fallback: raw transformers with mean pooling
            try:
                from transformers import AutoTokenizer, AutoModel  # type: ignore
                import torch  # type: ignore

                hf_tokenizer = AutoTokenizer.from_pretrained(resolved_local, local_files_only=True)
                hf_model = AutoModel.from_pretrained(resolved_local, local_files_only=True)
                hf_model.eval()
                if torch.cuda.is_available():
                    device = "cuda"
                    hf_model.to(device)
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                    hf_model.to(device)
                else:
                    device = "cpu"
            except Exception:
                hf_tokenizer = None
                hf_model = None

        # Single encode
        def _encode_one(text: str) -> List[float]:
            try:
                if st_model is not None:
                    v = st_model.encode([text], batch_size=1, normalize_embeddings=False, show_progress_bar=False)
                    vec = v[0].tolist()
                elif hf_model is not None and hf_tokenizer is not None:
                    import torch  # type: ignore
                    with torch.no_grad():
                        toks = hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                        if device != "cpu":
                            toks = {k: v.to(device) for k, v in toks.items()}
                        out = hf_model(**toks)
                        last_hidden = out.last_hidden_state  # [1, seq, hidden]
                        attn = toks["attention_mask"].unsqueeze(-1)  # [1, seq, 1]
                        sum_emb = (last_hidden * attn).sum(dim=1)
                        denom = attn.sum(dim=1).clamp(min=1e-6)
                        pooled = (sum_emb / denom).squeeze(0)
                        vec = pooled.detach().cpu().float().numpy().tolist()
                else:
                    vec = _hash_embed(text, dim)
                if normalize:
                    vec = _l2_normalize(vec)
                # defensive pad/trim
                if len(vec) > dim:
                    return vec[:dim]
                if len(vec) < dim:
                    return vec + [0.0] * (dim - len(vec))
                return vec
            except Exception:
                return _hash_embed(text, dim)

        # Batch encode
        def _encode_batch(texts: List[str], *, bsz: Optional[int] = None) -> List[List[float]]:
            if not texts:
                return []
            b = int(bsz or batch_size or 32)
            out: List[List[float]] = []
            try:
                if st_model is not None:
                    for i in range(0, len(texts), b):
                        chunk = texts[i:i + b]
                        arr = st_model.encode(chunk, batch_size=b, normalize_embeddings=False, show_progress_bar=False)
                        for row in arr:
                            vec = row.tolist()
                            if normalize:
                                vec = _l2_normalize(vec)
                            if len(vec) > dim:
                                vec = vec[:dim]
                            elif len(vec) < dim:
                                vec = vec + [0.0] * (dim - len(vec))
                            out.append(vec)
                    return out
                elif hf_model is not None and hf_tokenizer is not None:
                    import torch  # type: ignore
                    for i in range(0, len(texts), b):
                        chunk = texts[i:i + b]
                        toks = hf_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                        if device != "cpu":
                            toks = {k: v.to(device) for k, v in toks.items()}
                        with torch.no_grad():
                            out_h = hf_model(**toks).last_hidden_state  # [B, seq, hidden]
                            attn = toks["attention_mask"].unsqueeze(-1)  # [B, seq, 1]
                            sum_emb = (out_h * attn).sum(dim=1)
                            denom = attn.sum(dim=1).clamp(min=1e-6)
                            pooled = (sum_emb / denom).detach().cpu().float().numpy()
                        for row in pooled:
                            vec = row.tolist()
                            if normalize:
                                vec = _l2_normalize(vec)
                            if len(vec) > dim:
                                vec = vec[:dim]
                            elif len(vec) < dim:
                                vec = vec + [0.0] * (dim - len(vec))
                            out.append(vec)
                    return out
            except Exception:
                pass
            # fallback: loop single
            return [_encode_one(t) for t in texts]

        # LRU cache for single encodes
        try:
            from functools import lru_cache

            @lru_cache(maxsize=4096)
            def _cached_one(s: str) -> tuple:
                return tuple(_encode_one(s))

            def _embed_cached(s: str) -> List[float]:
                return list(_cached_one(s))

            # Attach batch encoder for callers that can utilize it
            setattr(_embed_cached, "encode_batch", _encode_batch)
            return _embed_cached
        except Exception:
            def _embed_plain(s: str) -> List[float]:
                return _encode_one(s)
            setattr(_embed_plain, "encode_batch", _encode_batch)
            return _embed_plain

    # fallback
    # fallback
    def fn_fallback(text):
        return _hash_embed(text, dim)

    # Wrap the chosen embedder with a small LRU cache to avoid repeated HTTP calls for identical texts
    from functools import lru_cache

    def _wrap_cache(fn: Callable[[str], List[float]]) -> Callable[[str], List[float]]:
        encode_batch = getattr(fn, "encode_batch", None)
        embed_with_usage = getattr(fn, "embed_with_usage", None)
        try:
            @lru_cache(maxsize=4096)
            def _cached(s: str) -> tuple:
                vec = fn(s)
                # normalize length to dim defensively
                if len(vec) == dim:
                    return tuple(vec)
                if len(vec) > dim:
                    return tuple(vec[:dim])
                return tuple(vec + [0.0] * (dim - len(vec)))

            def _embed(s: str) -> List[float]:
                return list(_cached(s))

            if callable(encode_batch):
                try:
                    setattr(_embed, "encode_batch", encode_batch)
                except Exception:
                    pass
            if callable(embed_with_usage):
                try:
                    setattr(_embed, "embed_with_usage", embed_with_usage)
                except Exception:
                    pass
            return _embed
        except Exception:
            return fn

    # assemble priority order chosen above, then wrap with cache
    chosen: Optional[Callable[[str], List[float]]] = None
    # Re-evaluate the same provider selection to avoid refactor churn; returns early above where possible
    if provider in {"openai_compat", "openai", "openai-compatible"}:
        api_base = _sanitize_placeholder(
            settings.get("api_base") if settings else None
        ) or _sanitize_placeholder(os.getenv("OPENAI_COMPAT_API_BASE") or os.getenv("OPENAI_API_BASE"))
        api_key = _sanitize_placeholder(
            (settings.get("api_key") if settings else None)
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        mdl = (settings.get("model") if settings else None) or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-large"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=(settings or {}).get("embed_concurrency"),
        )
        if fn is not None:
            chosen = fn
    elif provider in {"qwen", "dashscope", "aliyun"}:
        api_base = _sanitize_placeholder((settings or {}).get("api_base") or os.getenv("EMBEDDING_API_BASE")) or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key_candidates = [
            (settings or {}).get("api_key"),
            os.getenv("EMBEDDING_API_KEY"),
            os.getenv("DASHSCOPE_API_KEY"),
            os.getenv("QWEN_API_KEY"),
        ]
        api_key = _sanitize_placeholder(next((v for v in api_key_candidates if v), None))
        mdl = (settings or {}).get("model") or "text-embedding-v2"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=(settings or {}).get("embed_concurrency"),
        )
        if fn is not None:
            chosen = fn
    elif provider in {"openrouter", "open_router"}:
        api_base = _sanitize_placeholder((settings or {}).get("api_base") or os.getenv("OPENROUTER_EMBEDDING_API_BASE") or os.getenv("OPENROUTER_BASE_URL")) or "https://openrouter.ai/api/v1"
        api_key_candidates = [
            (settings or {}).get("api_key"),
            os.getenv("OPENROUTER_EMBEDDING_API_KEY"),
            os.getenv("OPENROUTER_API_KEY"),
        ]
        api_key = _sanitize_placeholder(next((v for v in api_key_candidates if v), None))
        mdl = (settings or {}).get("model") or os.getenv("OPENROUTER_EMBEDDING_MODEL") or "openai/text-embedding-3-large"
        fn = _build_openai_sdk_embedder(
            mdl,
            api_base=api_base,
            api_key=api_key,
            dim=dim,
            embed_concurrency=(settings or {}).get("embed_concurrency"),
        )
        if fn is not None:
            chosen = fn
    elif provider == "gemini" and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        # already returned above in success path; if it falls through, treat as fallback
        pass

    if chosen is None:
        chosen = fn_fallback

    return _wrap_cache(chosen)


def build_image_embedding_from_settings(settings: Optional[Dict[str, Any]] = None) -> Callable[[str], List[float]]:
    """Return an embed(content) -> vector function for image modality.

    行为：
    - provider = clip|open_clip：使用 OpenCLIP（open_clip_torch）加载模型；
      - 若 content 形似 base64 图像（或 data:image/...;base64,...），则用图像编码器；
      - 否则将 content 视为文本，使用文本编码器（便于查询路径）。
      - 运行时按需 lazy-load，若依赖缺失或模型不可用→回退哈希嵌入。
    - 其他 provider：回退哈希嵌入。

    settings 示例：
      {"provider": "clip", "model": "ViT-B-32", "pretrained": "openai", "dim": 512}
    环境变量可覆盖：IMAGE_EMBEDDING_PROVIDER / IMAGE_EMBEDDING_MODEL / IMAGE_EMBEDDING_DIM
    """
    cfg = settings or {}
    provider = (cfg.get("provider") or os.getenv("IMAGE_EMBEDDING_PROVIDER") or "").strip().lower()
    model = (cfg.get("model") or os.getenv("IMAGE_EMBEDDING_MODEL") or "ViT-B-32").strip()
    pretrained = (cfg.get("pretrained") or os.getenv("IMAGE_EMBEDDING_PRETRAINED") or "openai").strip()
    dim = int(cfg.get("dim") or os.getenv("IMAGE_EMBEDDING_DIM") or 512)

    # lazy state captured in closure
    _state: Dict[str, Any] = {"ready": False, "model": None, "tokenizer": None, "preprocess": None, "device": "cpu"}

    def _lazy_init() -> bool:
        if _state["ready"]:
            return True
        try:
            import torch  # type: ignore
            import open_clip  # type: ignore
            from PIL import Image  # type: ignore

            # Prefer Apple Metal (MPS) on macOS, then CUDA, else CPU
            device = "cpu"
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"
            mdl, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)  # type: ignore
            tokenizer = open_clip.get_tokenizer(model)
            _state["model"] = mdl
            _state["preprocess"] = preprocess
            _state["tokenizer"] = tokenizer
            _state["device"] = device
            _state["Image"] = Image
            _state["torch"] = torch
            _state["ready"] = True
            return True
        except Exception:
            return False

    def _maybe_decode_image(s: str):
        # try data URL
        try:
            if s.startswith("data:image/") and ";base64," in s:
                b64 = s.split(",", 1)[1]
                import base64
                import io
                raw = base64.b64decode(b64)
                return _state["Image"].open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
        # try plain base64
        try:
            import base64
            import io
            if len(s) > 256 and all(c.isalnum() or c in "+/=\n\r" for c in s.strip()):
                raw = base64.b64decode(s)
                return _state["Image"].open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
        return None

    def _norm(vec) -> List[float]:
        try:
            import numpy as _np  # type: ignore
            arr = _np.array(vec, dtype=_np.float32).ravel()
            n = float(_np.linalg.norm(arr) or 1.0)
            arr = arr / n
            out = arr.tolist()
        except Exception:
            # best-effort python fallback
            arr = [float(v) for v in vec]
            n2 = sum(v * v for v in arr) or 1.0
            n = n2 ** 0.5
            out = [v / n for v in arr]
        # pad/truncate to dim
        if len(out) > dim:
            return out[:dim]
        if len(out) < dim:
            return out + [0.0] * (dim - len(out))
        return out

    if provider in {"clip", "open_clip"}:
        def _embed(content: str) -> List[float]:
            if not _lazy_init():
                return _hash_embed(content, dim)
            try:
                mdl = _state["model"]
                preprocess = _state["preprocess"]
                tokenizer = _state["tokenizer"]
                torch = _state["torch"]
                device = _state["device"]

                img = _maybe_decode_image(content)
                if img is not None:
                    img_t = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_feat = mdl.encode_image(img_t)  # type: ignore
                        vec = img_feat[0].detach().cpu().numpy()
                    return _norm(vec)
                # fallback: treat as text query
                toks = tokenizer([content])
                with torch.no_grad():
                    txt = toks.to(device)
                    txt_feat = mdl.encode_text(txt)  # type: ignore
                    vec = txt_feat[0].detach().cpu().numpy()
                return _norm(vec)
            except Exception:
                return _hash_embed(content, dim)

        def _encode_many(contents: List[str]) -> List[List[float]]:
            if not _lazy_init():
                return [_hash_embed(c, dim) for c in contents]
            mdl = _state["model"]
            preprocess = _state["preprocess"]
            tokenizer = _state["tokenizer"]
            torch = _state["torch"]
            device = _state["device"]
            # split inputs as images vs texts
            imgs = []  # tuples (pos, PIL)
            txts = []  # tuples (pos, str)
            from PIL import Image as _Image  # type: ignore
            import base64 as _b64
            import io as _io
            for i, s in enumerate(contents or []):
                im = None
                # try data URL
                try:
                    if isinstance(s, str) and s.startswith("data:image/") and ";base64," in s:
                        b64 = s.split(",", 1)[1]
                        raw = _b64.b64decode(b64)
                        im = _Image.open(_io.BytesIO(raw)).convert("RGB")
                except Exception:
                    im = None
                if im is None:
                    # try plain base64
                    try:
                        if isinstance(s, str) and len(s) > 256 and all(c.isalnum() or c in "+/=\n\r" for c in s.strip()):
                            raw = _b64.b64decode(s)
                            im = _Image.open(_io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        im = None
                if im is not None:
                    imgs.append((i, im))
                else:
                    txts.append((i, str(s)))
            out: List[List[float]] = [[0.0] * dim for _ in range(len(contents or []))]
            # batch images
            if imgs:
                try:
                    batch = torch.stack([preprocess(im) for _, im in imgs], dim=0).to(device)
                    with torch.no_grad():
                        feats = mdl.encode_image(batch)  # type: ignore
                        arr = feats.detach().cpu().numpy()
                    for (pos, _), vec in zip(imgs, arr):
                        out[pos] = _norm(vec)
                except Exception:
                    for pos, im in imgs:
                        try:
                            t = preprocess(im).unsqueeze(0).to(device)
                            with torch.no_grad():
                                f = mdl.encode_image(t)  # type: ignore
                                vec = f[0].detach().cpu().numpy()
                            out[pos] = _norm(vec)
                        except Exception:
                            out[pos] = _hash_embed("", dim)
            # batch texts
            if txts:
                try:
                    toks = tokenizer([s for _, s in txts])
                    with torch.no_grad():
                        t = toks.to(device)
                        feats = mdl.encode_text(t)  # type: ignore
                        arr = feats.detach().cpu().numpy()
                    for (pos, _), vec in zip(txts, arr):
                        out[pos] = _norm(vec)
                except Exception:
                    for pos, s in txts:
                        try:
                            toks = tokenizer([s])
                            with torch.no_grad():
                                t = toks.to(device)
                                f = mdl.encode_text(t)  # type: ignore
                                vec = f[0].detach().cpu().numpy()
                            out[pos] = _norm(vec)
                        except Exception:
                            out[pos] = _hash_embed(s, dim)
            return out

        # attach batch method for callers that can leverage it
        try:
            setattr(_embed, "encode_many", _encode_many)
        except Exception:
            pass

        return _embed

    # fallback provider
    return lambda text: _hash_embed(text, dim)


def build_audio_embedding_from_settings(settings: Optional[Dict[str, Any]] = None) -> Callable[[str], List[float]]:
    """Return an embed(audio_desc) -> vector function for audio modality.

    当前占位：若未配置或提供商不可用，回退到基于描述文本的特征哈希（dim 默认 256）。
    可通过 settings = {"provider": "eres2net", "model": "...", "dim": 256} 扩展真实嵌入。
    """
    cfg = settings or {}
    dim = int(cfg.get("dim") or os.getenv("AUDIO_EMBEDDING_DIM") or 256)
    # TODO: 接入真实语音嵌入（ERes2NetV2 等）。此处先回退哈希嵌入。
    def _audio_fallback(text: str) -> List[float]:
        return _hash_embed(text, dim)

    return _audio_fallback
