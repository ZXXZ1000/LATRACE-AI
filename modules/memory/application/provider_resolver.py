"""Shared provider name normalization and environment variable resolution.

This module is the single source of truth for:
1. Canonicalizing provider name aliases (e.g., ``open_router`` → ``openrouter``).
2. Resolving the first non-empty value from a list of environment variable names.
3. Mapping providers to their expected credential environment variables.

Both ``llm_adapter`` and ``service`` (health check) MUST import from here
instead of maintaining local copies.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

def norm_opt_str(value: Optional[str]) -> Optional[str]:
    """Strip whitespace; return ``None`` for empty / missing values."""
    raw = str(value or "").strip()
    return raw or None


def normalize_provider_name(value: Optional[str]) -> str:
    """Canonicalize a provider name string.

    Rules:
    - Lowercase, replace hyphens with underscores.
    - ``open_router`` → ``openrouter``
    - ``openai_compatible`` / ``openai-compatible`` → ``openai_compat``

    Returns empty string for ``None`` / blank input.
    """
    raw = norm_opt_str(value)
    if not raw:
        return ""
    provider = raw.lower().replace("-", "_")
    if provider == "open_router":
        return "openrouter"
    if provider in {"openai_compatible", "openai_compat"}:
        return "openai_compat"
    return provider


def first_env_value(*names: str) -> Optional[str]:
    """Return the first non-empty env value from *names*, or ``None``."""
    for name in names:
        value = norm_opt_str(os.getenv(name))
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Provider → credential mapping
# ---------------------------------------------------------------------------

# Each entry: (api_key_env_names, base_url_env_names, default_base_url, key_required)
_PROVIDER_CREDENTIALS: Dict[str, Tuple[List[str], List[str], Optional[str], bool]] = {
    "openrouter": (
        ["OPENROUTER_API_KEY"],
        ["OPENROUTER_BASE_URL"],
        "https://openrouter.ai/api/v1",
        True,
    ),
    "openai_compat": (
        ["LLM_API_KEY", "OPENAI_COMPAT_API_KEY"],
        ["LLM_BASE_URL", "OPENAI_COMPAT_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"],
        None,
        True,
    ),
    "openai": (
        ["OPENAI_API_KEY"],
        ["OPENAI_BASE_URL", "OPENAI_API_BASE"],
        None,
        True,
    ),
    "qwen": (["DASHSCOPE_API_KEY", "QWEN_API_KEY"], ["DASHSCOPE_BASE_URL"], "https://dashscope.aliyuncs.com/compatible-mode/v1", True),
    "dashscope": (["DASHSCOPE_API_KEY", "QWEN_API_KEY"], ["DASHSCOPE_BASE_URL"], "https://dashscope.aliyuncs.com/compatible-mode/v1", True),
    "aliyun": (["DASHSCOPE_API_KEY", "QWEN_API_KEY"], ["DASHSCOPE_BASE_URL"], "https://dashscope.aliyuncs.com/compatible-mode/v1", True),
    "glm": (["ZHIPUAI_API_KEY", "GLM_API_KEY"], ["GLM_API_BASE"], "https://open.bigmodel.cn/api/coding/paas/v4", True),
    "zhipuai": (["ZHIPUAI_API_KEY", "GLM_API_KEY"], ["GLM_API_BASE"], "https://open.bigmodel.cn/api/coding/paas/v4", True),
    "deepseek": (["DEEPSEEK_API_KEY"], ["DEEPSEEK_API_BASE"], "https://api.deepseek.com/v1", True),
    "moonshot": (["MOONSHOT_API_KEY"], ["MOONSHOT_API_BASE"], "https://api.moonshot.cn/v1", True),
    "kimi": (["MOONSHOT_API_KEY"], ["MOONSHOT_API_BASE"], "https://api.moonshot.cn/v1", True),
    "gemini": (["GEMINI_API_KEY", "GOOGLE_API_KEY"], ["GEMINI_BASE_URL"], None, True),
    "google": (["GEMINI_API_KEY", "GOOGLE_API_KEY"], ["GEMINI_BASE_URL"], None, True),
}

# Fallback for unknown providers (same as openai_compat)
_UNKNOWN_PROVIDER_CREDENTIALS: Tuple[List[str], List[str], Optional[str], bool] = (
    ["LLM_API_KEY", "OPENAI_COMPAT_API_KEY"],
    ["LLM_BASE_URL", "OPENAI_COMPAT_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"],
    None,
    False,  # key not strictly required for unknown (could be local endpoint)
)


def resolve_provider_credentials(provider: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (api_key, base_url) for a normalized provider name.

    Returns the first non-empty value found for each credential.
    Uses ``_PROVIDER_CREDENTIALS`` lookup table to ensure consistent
    resolution order across all call sites.
    """
    entry = _PROVIDER_CREDENTIALS.get(provider, _UNKNOWN_PROVIDER_CREDENTIALS)
    key_names, base_names, default_base, _ = entry

    api_key = first_env_value(*key_names)
    raw_base = first_env_value(*base_names)
    base_url = raw_base or default_base

    return api_key, base_url


def is_key_required(provider: str) -> bool:
    """Whether the given provider mandates an API key."""
    entry = _PROVIDER_CREDENTIALS.get(provider, _UNKNOWN_PROVIDER_CREDENTIALS)
    return entry[3]


# Provider model env var mapping (for resolve_openai_compatible_chat_target fallback)
_PROVIDER_MODEL_ENV: Dict[str, str] = {
    "openrouter": "OPENROUTER_MODEL",
    "qwen": "QWEN_MODEL",
    "dashscope": "QWEN_MODEL",
    "aliyun": "QWEN_MODEL",
    "glm": "GLM_MODEL",
    "zhipuai": "GLM_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "moonshot": "MOONSHOT_MODEL",
    "kimi": "MOONSHOT_MODEL",
    "gemini": "GEMINI_MODEL",
    "google": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
}


def resolve_model_from_env(provider: str) -> Optional[str]:
    """Try to resolve a model name from provider-specific env vars."""
    env_name = _PROVIDER_MODEL_ENV.get(provider)
    if env_name:
        return norm_opt_str(os.getenv(env_name))
    return None
