from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from modules.memory.byok_control import AuditSink, ByokRegistry, FernetCredentialCipher, SqliteByokStore


class ListAuditSink:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event: str, obj_id: str, payload: dict, *, reason: str | None = None) -> None:
        self.events.append({"event": event, "obj_id": obj_id, "payload": payload, "reason": reason})


def _registry(audit: AuditSink | None = None) -> ByokRegistry:
    key = Fernet.generate_key().decode("utf-8")
    store = SqliteByokStore(sqlite_path=":memory:")
    cipher = FernetCredentialCipher(current_key=key)
    return ByokRegistry(store=store, cipher=cipher, audit_sink=audit)


def test_registry_profile_credential_binding_roundtrip() -> None:
    audit = ListAuditSink()
    registry = _registry(audit)
    profile = registry.create_profile(provider="openai", default_model="gpt-4o-mini")
    credential = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-123")

    assert credential.secret_cipher != "sk-test-123"
    assert credential.key_fingerprint
    assert len(credential.key_fingerprint) == 16

    binding = registry.bind_api_key(
        tenant_id="tenant_a",
        api_key_id="api_key_1",
        profile_id=profile.profile_id,
        credential_id=credential.credential_id,
    )
    assert binding.active_credential_id == credential.credential_id
    fetched = registry.get_binding("tenant_a", "api_key_1")
    assert fetched is not None
    assert fetched.active_profile_id == profile.profile_id
    assert any(evt["event"] == "byok.profile.created" for evt in audit.events)
    assert any(evt["event"] == "byok.credential.created" for evt in audit.events)
    assert any(evt["event"] == "byok.binding.upserted" for evt in audit.events)


def test_registry_rejects_missing_profile() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="profile_not_found"):
        registry.create_credential(profile_id="missing", plaintext_secret="sk-test-1")


def test_registry_revoke_credential() -> None:
    registry = _registry()
    profile = registry.create_profile(provider="openai")
    credential = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-2")
    revoked = registry.revoke_credential(credential.credential_id)
    assert revoked.status == "revoked"
    assert revoked.rotated_at


def test_registry_rotate_credential_creates_new_one() -> None:
    registry = _registry()
    profile = registry.create_profile(provider="openai")
    credential = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-3")
    new_cred = registry.rotate_credential(credential_id=credential.credential_id, new_plaintext_secret="sk-test-4")
    assert new_cred.credential_id != credential.credential_id
    rotated = registry.get_credential(credential.credential_id)
    assert rotated is not None
    assert rotated.status == "rotated"


def test_registry_update_profile_and_binding_status() -> None:
    registry = _registry()
    profile = registry.create_profile(provider="openai", base_url="https://example.com")
    updated = registry.update_profile(profile_id=profile.profile_id, default_model="gpt-4o-mini")
    assert updated.default_model == "gpt-4o-mini"

    credential = registry.create_credential(profile_id=profile.profile_id, plaintext_secret="sk-test-5")
    binding = registry.bind_api_key(
        tenant_id="tenant_b",
        api_key_id="api_key_2",
        profile_id=profile.profile_id,
        credential_id=credential.credential_id,
    )
    assert binding.status == "active"
    disabled = registry.set_binding_status("tenant_b", "api_key_2", status="disabled")
    assert disabled.status == "disabled"
