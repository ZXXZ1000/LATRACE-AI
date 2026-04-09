from __future__ import annotations


def test_resolve_llm_adapter_from_client_meta(monkeypatch) -> None:
    from modules.memory.api import server as srv

    sentinel = object()

    def _fake_build_llm_from_byok(*, provider: str, model: str, api_key: str, base_url=None):
        assert provider == "openai"
        assert model == "gpt-4o-mini"
        assert api_key == "sk-test"
        assert base_url == "https://api.openai.com/v1"
        return sentinel

    monkeypatch.setattr(srv, "build_llm_from_byok", _fake_build_llm_from_byok)
    adapter, route, status = srv._resolve_llm_adapter_from_client_meta(
        {
            "llm_mode": "byok",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": "sk-test",
            "llm_base_url": "https://api.openai.com/v1",
        }
    )
    assert adapter is sentinel
    assert route == "byok"
    assert status == "hit"


def test_resolve_llm_adapter_from_client_meta_missing() -> None:
    from modules.memory.api import server as srv

    adapter, route, status = srv._resolve_llm_adapter_from_client_meta(None)
    assert adapter is None
    assert route == "platform"
    assert status == "missing"


def test_resolve_llm_adapter_from_client_meta_incomplete() -> None:
    from modules.memory.api import server as srv

    adapter, route, status = srv._resolve_llm_adapter_from_client_meta(
        {"llm_mode": "byok", "llm_provider": "openai"}
    )
    assert adapter is None
    assert route == "platform"
    assert status == "config_incomplete"


def test_resolve_llm_adapter_from_client_meta_adapter_missing(monkeypatch) -> None:
    from modules.memory.api import server as srv

    def _fake_build_llm_from_byok(*, provider: str, model: str, api_key: str, base_url=None):
        return None

    monkeypatch.setattr(srv, "build_llm_from_byok", _fake_build_llm_from_byok)
    adapter, route, status = srv._resolve_llm_adapter_from_client_meta(
        {
            "llm_mode": "byok",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": "sk-test",
        }
    )
    assert adapter is None
    assert route == "platform"
    assert status == "adapter_missing"
