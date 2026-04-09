from __future__ import annotations

import os
import math
import pytest

from modules.memory.application.config import load_memory_config
from modules.memory.application.embedding_adapter import build_embedding_from_settings


def _iter_tokens(text: str) -> list[str]:
    s = (text or "").strip().lower()
    if not s:
        return []
    if " " in s:
        return s.split()
    # char bigrams fallback for CJK
    if len(s) == 1:
        return [s]
    return [s[i : i + 2] for i in range(len(s) - 1)]


def _hash_embed(text: str, dim: int) -> list[float]:
    import hashlib

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


def _has_provider_credentials(provider: str) -> bool:
    p = (provider or "").strip().lower()
    if p in {"qwen", "dashscope", "aliyun"}:
        return any(os.getenv(k) for k in ("DASHSCOPE_API_KEY", "EMBEDDING_API_KEY", "QWEN_API_KEY"))
    if p in {"openai", "openai_compat", "openai-compatible"}:
        return any(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY"))
    if p in {"openrouter", "open_router"}:
        return any(os.getenv(k) for k in ("OPENROUTER_EMBEDDING_API_KEY", "OPENROUTER_API_KEY"))
    if p in {"gemini"}:
        return any(os.getenv(k) for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    # other providers are not supported explicitly in embedding_adapter
    return False


def test_has_provider_credentials_accepts_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _has_provider_credentials("openrouter") is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    assert _has_provider_credentials("openrouter") is True


def test_embedding_provider_connectivity_or_fallback():
    cfg = load_memory_config()
    emb_cfg = (
        cfg.get("memory", {})
        .get("vector_store", {})
        .get("embedding", {})
        or {}
    )
    dim = int(emb_cfg.get("dim", 1536))
    provider = str(emb_cfg.get("provider", "")).strip().lower()

    embed = build_embedding_from_settings(emb_cfg)
    text = "embedding connectivity smoke test 连接性测试"
    vec = embed(text)

    assert isinstance(vec, list) and len(vec) == dim

    require_connectivity = str(os.getenv("REQUIRE_EMBEDDING_CONNECTIVITY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Local provider: insist on real model when可用，否则跳过并提示缺依赖
    if provider == "local":
        from pathlib import Path
        local_path = Path(str(emb_cfg.get("local_path") or emb_cfg.get("model") or ""))
        deps_ok = False
        try:
            import sentence_transformers  # type: ignore  # noqa: F401
            deps_ok = True
        except Exception:
            try:
                import transformers  # type: ignore  # noqa: F401
                deps_ok = True
            except Exception:
                deps_ok = False
        if local_path.exists() and not deps_ok:
            pytest.skip("Local embedding provider configured but sentence-transformers/transformers not installed; install to run real embedding test.")
        # if deps exist, require non-hash embedding
        baseline = _hash_embed(text, dim)
        diff = sum(abs(a - b) for a, b in zip(vec, baseline))
        assert diff > 1e-3, "Local embedding provider should return non-hash vectors; check local model load."
        return

    # If credentials are present (or CI requires connectivity), expect a non-hash vector
    if _has_provider_credentials(provider):
        baseline = _hash_embed(text, dim)
        diff = sum(abs(a - b) for a, b in zip(vec, baseline))
        # if diff is ~0, very likely fell back to hash (connectivity failure or server unreachable)
        assert diff > 1e-3, (
            f"Embedding provider '{provider}' appears unreachable; got fallback hash vector.\n"
            f"Check modules/memory/config/.env credentials and network access."
        )
    else:
        if require_connectivity:
            pytest.fail(
                f"Embedding connectivity was explicitly required but no credentials detected for provider '{provider}'.\n"
                f"Please set provider credentials in environment variables or CI secrets."
            )
        else:
            pytest.skip(
                f"No credentials detected for provider '{provider}', skipping real connectivity check in default CI/open-source runs."
            )
