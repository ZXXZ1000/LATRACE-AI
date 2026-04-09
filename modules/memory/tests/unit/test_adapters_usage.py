import pytest
from unittest.mock import MagicMock, patch
from modules.memory.application.embedding_adapter import _build_openai_sdk_embedder
from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_config, build_llm_from_env
from modules.memory.contracts.usage_models import EmbeddingUsage, LLMUsage

class MockOpenAIResponse:
    def __init__(self, embedding_vector, prompt_tokens=5, total_tokens=5):
        self.data = [MagicMock(embedding=embedding_vector)]
        self.usage = MagicMock(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

@pytest.mark.anyio
async def test_embedding_adapter_usage_extraction():
    pytest.importorskip("openai")
    # Mock OpenAI client
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MockOpenAIResponse([0.1, 0.2], 10, 10)
        
        embedder = _build_openai_sdk_embedder(
            model="test-model",
            api_base="http://fake",
            api_key="sk-fake",
            dim=2
        )
        assert hasattr(embedder, "embed_with_usage")
        
        # Test usage extraction
        vec, usage = embedder.embed_with_usage("hello")
        assert vec == [0.1, 0.2]
        assert isinstance(usage, EmbeddingUsage)
        assert usage.tokens == 10
        assert usage.provider == "openai_sdk"
        
        # Test backward compatibility
        vec_legacy = embedder("hello")
        assert vec_legacy == [0.1, 0.2]

def test_llm_adapter_usage_capture():
    # Create a mock LLM adapter that emits usage via internal mechanism
    def mock_fn(msgs, fmt=None):
        from modules.memory.application.llm_adapter import _emit_llm_usage
        _emit_llm_usage(
            provider="test_prov",
            model="test_model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        return "response text"

    adapter = LLMAdapter(mock_fn, kind="test")
    
    content, usage = adapter.generate_with_usage([{"role":"user", "content":"hi"}])
    assert content == "response text"
    assert isinstance(usage, LLMUsage)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.provider == "test_prov"


def test_openrouter_embedding_cost_is_not_null(monkeypatch):
    # Ensure OpenRouter embedding usage emits non-null cost via stats/pricing fallback.
    from modules.memory.application import embedding_adapter as ea
    import requests

    class _Resp:
        status_code = 200

        def __init__(self):
            self.headers = {"x-openrouter-generation-id": "gen_test_1"}

        def json(self):
            return {
                "data": [{"embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }

    class _Sess:
        def post(self, *_args, **_kwargs):
            return _Resp()

    monkeypatch.setattr(requests, "Session", lambda: _Sess())

    # Patch OpenRouter generation stats helper to return cost.
    monkeypatch.setattr(
        "modules.memory.application.llm_adapter._fetch_openrouter_generation_stats",
        lambda **_kwargs: {"usage": {"total_tokens": 10}, "cost_usd": 0.0002},
    )

    embedder = ea._build_openai_sdk_embedder(
        model="openai/text-embedding-3-small",
        api_base="https://openrouter.ai/api/v1",
        api_key="sk-openrouter",
        dim=2,
    )
    assert embedder is not None
    vec, usage = embedder.embed_with_usage("hello")  # type: ignore[attr-defined]
    assert vec == [0.1, 0.2]
    assert usage is not None
    assert usage.provider == "openrouter"
    assert usage.cost_usd is not None


def test_openai_embedder_honors_embed_concurrency(monkeypatch):
    pytest.importorskip("openai")
    from modules.memory.application import embedding_adapter as ea

    observed = {"max_workers": None}

    class _FakeResponse:
        def __init__(self, texts):
            self.data = [MagicMock(embedding=[float(idx), float(idx) + 0.5]) for idx, _ in enumerate(texts)]

    class _FakeClient:
        class embeddings:
            @staticmethod
            def create(**payload):
                inputs = payload["input"]
                if not isinstance(inputs, list):
                    inputs = [inputs]
                return _FakeResponse(inputs)

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _ImmediateExecutor:
        def __init__(self, *, max_workers):
            observed["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _ImmediateFuture(fn(*args, **kwargs))

    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: _FakeClient())
    monkeypatch.setattr(ea, "ThreadPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(ea, "as_completed", lambda futures: list(futures))

    embedder = _build_openai_sdk_embedder(
        model="test-model",
        api_base="http://fake",
        api_key="sk-fake",
        dim=2,
        embed_concurrency=3,
    )
    assert embedder is not None

    vectors = embedder.encode_batch(["a", "b", "c"], bsz=1)  # type: ignore[attr-defined]
    assert len(vectors) == 3
    assert observed["max_workers"] == 3


def test_build_llm_from_config_openai_compat_sets_custom_provider(monkeypatch):
    from modules.memory.application import config as cfgmod
    from modules.memory.application import llm_adapter as ladapter

    captured = {}

    monkeypatch.setattr(
        cfgmod,
        "load_memory_config",
        lambda: {"memory": {"llm": {"extract": {"provider": "openai_compat", "model": "Kimi-K2.5"}}}},
    )
    monkeypatch.setattr(cfgmod, "get_llm_selection", lambda _cfg, kind: {"provider": "openai_compat", "model": "Kimi-K2.5"})
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    def _fake_litellm_adapter(*, model, api_base=None, api_key=None, custom_llm_provider=None):
        captured.update(
            {
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "custom_llm_provider": custom_llm_provider,
            }
        )
        return LLMAdapter(lambda *_args, **_kwargs: "{}")

    monkeypatch.setattr(ladapter, "_litellm_adapter", _fake_litellm_adapter)

    adapter = build_llm_from_config("extract")
    assert adapter is not None
    assert captured["model"] == "Kimi-K2.5"
    assert captured["api_base"] == "https://example.com/v1"
    assert captured["api_key"] == "sk-test"
    assert captured["custom_llm_provider"] == "openai"


def test_build_llm_from_env_openai_compat_sets_custom_provider(monkeypatch):
    from modules.memory.application import llm_adapter as ladapter

    captured = {}
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "Kimi-K2.5")
    monkeypatch.delenv("SGLANG_BASE_URL", raising=False)

    def _fake_litellm_adapter(*, model, api_base=None, api_key=None, custom_llm_provider=None):
        captured.update(
            {
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "custom_llm_provider": custom_llm_provider,
            }
        )
        return LLMAdapter(lambda *_args, **_kwargs: "{}")

    monkeypatch.setattr(ladapter, "_litellm_adapter", _fake_litellm_adapter)

    adapter = build_llm_from_env()
    assert adapter is not None
    assert captured["model"] == "Kimi-K2.5"
    assert captured["api_base"] == "https://example.com/v1"
    assert captured["api_key"] == "sk-test"
    assert captured["custom_llm_provider"] == "openai"
