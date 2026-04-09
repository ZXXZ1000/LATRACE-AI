from __future__ import annotations

from cryptography.fernet import Fernet

from modules.memory.byok_control import ByokRegistry, FernetCredentialCipher, SqliteByokStore
from modules.memory.application.byok_resolver import ByokResolver


def _resolver_and_registry() -> tuple[ByokResolver, ByokRegistry]:
    key = Fernet.generate_key().decode("utf-8")
    store = SqliteByokStore(sqlite_path=":memory:")
    cipher = FernetCredentialCipher(current_key=key)
    registry = ByokRegistry(store=store, cipher=cipher)
    resolver = ByokResolver(store=store, cipher=cipher, cache_ttl_s=0)
    return resolver, registry


def test_byok_resolver_missing_binding_returns_platform() -> None:
    resolver, _ = _resolver_and_registry()
    res = resolver.resolve(tenant_id="tenant_x", api_key_id="key_x")
    assert res.route == "platform"
    assert res.status == "binding_missing"


def test_byok_resolver_materialize_config_success() -> None:
    resolver, registry = _resolver_and_registry()
    profile = registry.create_profile(provider="openai", default_model="gpt-4o-mini")
    cred = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-123")
    registry.bind_api_key(
        tenant_id="tenant_a",
        api_key_id="api_key_1",
        profile_id=profile.profile_id,
        credential_id=cred.credential_id,
    )

    res = resolver.resolve(tenant_id="tenant_a", api_key_id="api_key_1")
    assert res.route == "byok"
    assert res.status == "hit"
    cfg = resolver.materialize_config(res)
    assert cfg is not None
    assert cfg.api_key == "sk-test-123"
    assert cfg.model == "gpt-4o-mini"


def test_byok_resolver_revoked_credential_falls_back() -> None:
    resolver, registry = _resolver_and_registry()
    profile = registry.create_profile(provider="openai", default_model="gpt-4o-mini")
    cred = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-456")
    registry.bind_api_key(
        tenant_id="tenant_b",
        api_key_id="api_key_2",
        profile_id=profile.profile_id,
        credential_id=cred.credential_id,
    )
    registry.revoke_credential(cred.credential_id)

    res = resolver.resolve(tenant_id="tenant_b", api_key_id="api_key_2")
    assert res.route == "platform"
    assert res.status == "credential_inactive"
