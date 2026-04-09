from __future__ import annotations

from modules.memory.application import config as cfgmod
from modules.memory.application.llm_adapter import resolve_openai_compatible_chat_target


def test_resolve_chat_target_openrouter_from_agentic_router_config(monkeypatch):
    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {
            "memory": {
                "llm": {
                    "agentic_router": {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
                    "text": {"provider": "openai", "model": "gpt-4o-mini"},
                }
            }
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is not None
    assert target["provider"] == "openrouter"
    assert target["model"] == "google/gemini-2.5-flash-lite"
    assert target["api_key"] == "sk-openrouter"
    assert target["base_url"] == "https://openrouter.ai/api/v1"


def test_resolve_chat_target_falls_back_to_text_when_agentic_router_missing(monkeypatch):
    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {
            "memory": {
                "llm": {
                    "text": {"provider": "openai", "model": "gpt-4o-mini"},
                }
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is not None
    assert target["provider"] == "openai"
    assert target["model"] == "gpt-4o-mini"
    assert target["api_key"] == "sk-openai"
    assert target["base_url"] is None


def test_resolve_chat_target_sglang_allows_empty_api_key(monkeypatch):
    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {
            "memory": {
                "llm": {
                    "agentic_router": {"provider": "sglang", "model": "Qwen/Qwen3-VL-2B-Instruct"},
                }
            }
        },
    )
    monkeypatch.delenv("SGLANG_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("SGLANG_BASE_URL", "http://localhost:30000")

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is not None
    assert target["provider"] == "sglang"
    assert target["model"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert target["api_key"] == "EMPTY"
    assert target["base_url"] == "http://localhost:30000/v1"


def test_resolve_chat_target_returns_none_when_provider_key_missing(monkeypatch):
    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {
            "memory": {
                "llm": {
                    "agentic_router": {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
                }
            }
        },
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is None


def test_resolve_chat_target_falls_back_to_env_model_when_config_missing(monkeypatch):
    monkeypatch.setattr(cfgmod, "load_memory_config", lambda: {})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is not None
    assert target["provider"] == "openai"
    assert target["model"] == "gpt-4o-mini"


def test_resolve_chat_target_does_not_cross_fallback_other_provider_models(monkeypatch):
    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {
            "memory": {
                "llm": {
                    "agentic_router": {"provider": "openrouter"},
                }
            }
        },
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("GLM_MODEL", "glm-4.6")

    target = resolve_openai_compatible_chat_target(kind="agentic_router")

    assert target is None
