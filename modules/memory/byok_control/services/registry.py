from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict, Optional, Protocol

from modules.memory.byok_control.adapters.credential_cipher import CredentialCipher
from modules.memory.byok_control.adapters.sqlite_store import SqliteByokStore
from modules.memory.byok_control.domain.models import (
    CredentialStatus,
    LlmBinding,
    ProviderCredential,
    ProviderProfile,
    ProfileStatus,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return digest[:16]


class AuditSink(Protocol):
    def emit(self, event: str, obj_id: str, payload: Dict[str, Any], *, reason: Optional[str] = None) -> None:
        ...


class NullAuditSink:
    def emit(self, event: str, obj_id: str, payload: Dict[str, Any], *, reason: Optional[str] = None) -> None:
        return None


class ByokRegistry:
    def __init__(
        self,
        *,
        store: SqliteByokStore,
        cipher: CredentialCipher,
        audit_sink: Optional[AuditSink] = None,
    ) -> None:
        self._store = store
        self._cipher = cipher
        self._audit = audit_sink or NullAuditSink()

    def _emit(self, event: str, obj_id: str, payload: Dict[str, Any], *, reason: Optional[str] = None) -> None:
        self._audit.emit(event, obj_id, payload, reason=reason)

    def create_profile(
        self,
        *,
        provider: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        status: ProfileStatus = "active",
    ) -> ProviderProfile:
        now = _now_iso()
        profile = ProviderProfile(
            profile_id=f"profile_{uuid.uuid4().hex[:12]}",
            provider=str(provider or "").strip().lower(),
            base_url=(str(base_url).strip() if base_url else None),
            default_model=(str(default_model).strip() if default_model else None),
            status=status,
            created_at=now,
            updated_at=now,
        )
        if not profile.provider:
            raise ValueError("provider is required")
        self._store.upsert_profile(profile)
        self._emit(
            "byok.profile.created",
            profile.profile_id,
            {"provider": profile.provider, "status": profile.status},
        )
        return profile

    def set_profile_status(self, profile_id: str, *, status: ProfileStatus) -> ProviderProfile:
        current = self._store.get_profile(profile_id)
        if current is None:
            raise ValueError("profile_not_found")
        updated = replace(current, status=status, updated_at=_now_iso())
        self._store.upsert_profile(updated)
        self._emit(
            "byok.profile.status_changed",
            updated.profile_id,
            {"status": updated.status},
        )
        return updated

    def update_profile(
        self,
        *,
        profile_id: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> ProviderProfile:
        current = self._store.get_profile(profile_id)
        if current is None:
            raise ValueError("profile_not_found")
        next_base = current.base_url
        if base_url is not None:
            raw = str(base_url).strip()
            next_base = raw or None
        next_model = current.default_model
        if default_model is not None:
            raw = str(default_model).strip()
            next_model = raw or None
        updated = replace(current, base_url=next_base, default_model=next_model, updated_at=_now_iso())
        self._store.upsert_profile(updated)
        self._emit(
            "byok.profile.updated",
            updated.profile_id,
            {"base_url": updated.base_url, "default_model": updated.default_model},
        )
        return updated

    def create_credential(
        self,
        *,
        profile_id: str,
        plaintext_secret: str,
        status: CredentialStatus = "active",
    ) -> ProviderCredential:
        profile = self._store.get_profile(profile_id)
        if profile is None:
            raise ValueError("profile_not_found")
        secret = str(plaintext_secret or "")
        if not secret:
            raise ValueError("plaintext_secret is required")
        now = _now_iso()
        encrypted = self._cipher.encrypt(secret)
        cred = ProviderCredential(
            credential_id=f"cred_{uuid.uuid4().hex[:12]}",
            profile_id=profile.profile_id,
            secret_cipher=encrypted,
            key_fingerprint=_fingerprint(secret),
            status=status,
            created_at=now,
            rotated_at=None,
        )
        self._store.upsert_credential(cred)
        self._emit(
            "byok.credential.created",
            cred.credential_id,
            {"profile_id": cred.profile_id, "status": cred.status, "key_fingerprint": cred.key_fingerprint},
        )
        return cred

    def revoke_credential(self, credential_id: str) -> ProviderCredential:
        current = self._store.get_credential(credential_id)
        if current is None:
            raise ValueError("credential_not_found")
        updated = replace(current, status="revoked", rotated_at=_now_iso())
        self._store.upsert_credential(updated)
        self._emit(
            "byok.credential.revoked",
            updated.credential_id,
            {"profile_id": updated.profile_id, "status": updated.status},
        )
        return updated

    def rotate_credential(self, *, credential_id: str, new_plaintext_secret: str) -> ProviderCredential:
        current = self._store.get_credential(credential_id)
        if current is None:
            raise ValueError("credential_not_found")
        rotated = replace(current, status="rotated", rotated_at=_now_iso())
        self._store.upsert_credential(rotated)
        self._emit(
            "byok.credential.rotated",
            rotated.credential_id,
            {"profile_id": rotated.profile_id, "status": rotated.status},
        )
        return self.create_credential(profile_id=current.profile_id, plaintext_secret=new_plaintext_secret)

    def bind_api_key(
        self,
        *,
        tenant_id: str,
        api_key_id: str,
        profile_id: str,
        credential_id: str,
        status: str = "active",
    ) -> LlmBinding:
        if self._store.get_profile(profile_id) is None:
            raise ValueError("profile_not_found")
        if self._store.get_credential(credential_id) is None:
            raise ValueError("credential_not_found")
        now = _now_iso()
        binding = LlmBinding(
            tenant_id=str(tenant_id or "").strip(),
            api_key_id=str(api_key_id or "").strip(),
            active_profile_id=str(profile_id),
            active_credential_id=str(credential_id),
            status=str(status or "active"),
            created_at=now,
            updated_at=now,
        )
        if not binding.tenant_id or not binding.api_key_id:
            raise ValueError("tenant_id and api_key_id are required")
        self._store.upsert_binding(binding)
        self._emit(
            "byok.binding.upserted",
            f"{binding.tenant_id}:{binding.api_key_id}",
            {
                "tenant_id": binding.tenant_id,
                "api_key_id": binding.api_key_id,
                "profile_id": binding.active_profile_id,
                "credential_id": binding.active_credential_id,
                "status": binding.status,
            },
        )
        return binding

    def set_binding_status(self, tenant_id: str, api_key_id: str, *, status: str) -> LlmBinding:
        current = self._store.get_binding(tenant_id, api_key_id)
        if current is None:
            raise ValueError("binding_not_found")
        updated = replace(current, status=str(status or "active"), updated_at=_now_iso())
        self._store.upsert_binding(updated)
        self._emit(
            "byok.binding.status_changed",
            f"{updated.tenant_id}:{updated.api_key_id}",
            {"status": updated.status},
        )
        return updated

    def get_binding(self, tenant_id: str, api_key_id: str) -> Optional[LlmBinding]:
        return self._store.get_binding(tenant_id, api_key_id)

    def get_profile(self, profile_id: str) -> Optional[ProviderProfile]:
        return self._store.get_profile(profile_id)

    def get_credential(self, credential_id: str) -> Optional[ProviderCredential]:
        return self._store.get_credential(credential_id)
