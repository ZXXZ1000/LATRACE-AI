from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import os


@dataclass
class RerankDocument:
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    document_id: str
    relevance_score: float
    rank: int
    index: Optional[int] = None


@dataclass
class RerankResponse:
    results: List[RerankResult]
    usage_total_tokens: Optional[int] = None
    request_id: Optional[str] = None
    model: Optional[str] = None


class NativeRerankerClient:
    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[RerankDocument],
        top_n: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> RerankResponse:
        raise NotImplementedError


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _resolve_dashscope_rerank_base_url(api_base: Optional[str]) -> str:
    raw = (
        str(api_base or "").strip()
        or str(os.getenv("DASHSCOPE_RERANK_BASE_URL") or "").strip()
        or str(os.getenv("DASHSCOPE_BASE_URL") or "").strip()
    )
    if not raw:
        return "https://dashscope.aliyuncs.com/compatible-api/v1"
    base = raw.rstrip("/")
    if base.endswith("/compatible-mode/v1"):
        return f"{base[:-len('/compatible-mode/v1')]}/compatible-api/v1"
    if base.endswith("/compatible-api/v1"):
        return base
    if base.endswith("/v1"):
        return base
    return f"{base}/compatible-api/v1"


class DashScopeNativeRerankerClient(NativeRerankerClient):
    def __init__(self, *, model: str, api_key: str, api_base: Optional[str] = None, timeout_s: float = 60.0) -> None:
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        self.api_base = _resolve_dashscope_rerank_base_url(api_base)
        self.timeout_s = float(timeout_s)
        if not self.model:
            raise ValueError("reranker model is required")
        if not self.api_key:
            raise ValueError("reranker api_key is required")

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[RerankDocument],
        top_n: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> RerankResponse:
        try:
            import requests  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"requests_unavailable: {exc}") from exc

        docs = list(documents or [])
        if not docs:
            return RerankResponse(results=[], usage_total_tokens=0, model=self.model)

        url = f"{self.api_base.rstrip('/')}/reranks"
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": str(query or ""),
            "documents": [str(doc.text or "") for doc in docs],
            "top_n": int(top_n or len(docs)),
        }
        if instruct:
            payload["instruct"] = str(instruct)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        except Exception as exc:
            raise RuntimeError(f"dashscope_rerank_request_failed: {exc}") from exc

        if resp.status_code >= 400:
            body = ""
            try:
                body = resp.text[:500]
            except Exception:
                body = ""
            raise RuntimeError(f"dashscope_rerank_http_{resp.status_code}: {body}")

        try:
            data = resp.json() or {}
        except Exception as exc:
            raise RuntimeError(f"dashscope_rerank_invalid_json: {exc}") from exc

        raw_results = list(data.get("results") or [])
        out: List[RerankResult] = []
        for rank, item in enumerate(raw_results, start=1):
            idx = _coerce_int(item.get("index"))
            doc_id: Optional[str] = None
            if idx is not None and 0 <= idx < len(docs):
                doc_id = docs[idx].document_id
            if not doc_id:
                doc_id = str(item.get("document_id") or "").strip() or None
            if not doc_id:
                continue
            try:
                relevance = float(item.get("relevance_score") or 0.0)
            except Exception:
                relevance = 0.0
            out.append(
                RerankResult(
                    document_id=doc_id,
                    relevance_score=relevance,
                    rank=rank,
                    index=idx,
                )
            )

        usage = data.get("usage") or {}
        usage_total_tokens = _coerce_int(usage.get("total_tokens") or usage.get("input_tokens"))
        request_id = str(data.get("id") or data.get("request_id") or "").strip() or None
        model = str(data.get("model") or self.model).strip() or self.model
        return RerankResponse(
            results=out,
            usage_total_tokens=usage_total_tokens,
            request_id=request_id,
            model=model,
        )


def build_reranker_from_selection(
    *,
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[NativeRerankerClient]:
    prov = str(provider or "").strip().lower()
    mdl = str(model or "").strip()
    key = str(api_key or "").strip()
    if not prov or not mdl or not key:
        return None
    if prov in {"dashscope", "qwen", "aliyun"}:
        return DashScopeNativeRerankerClient(model=mdl, api_key=key, api_base=api_base)
    return None


def build_reranker_from_config(kind: str = "reranker") -> Optional[NativeRerankerClient]:
    try:
        from modules.memory.application.config import get_llm_selection, load_memory_config
    except Exception:
        return None

    cfg = load_memory_config()
    sel = get_llm_selection(cfg, kind)
    provider = str(sel.get("provider") or "").strip().lower()
    model = str(sel.get("model") or "").strip()
    if not provider or not model:
        return None

    if provider in {"dashscope", "qwen", "aliyun"}:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        api_base = os.getenv("DASHSCOPE_RERANK_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
        return build_reranker_from_selection(
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    return None
