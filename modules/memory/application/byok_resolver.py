from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import os

from modules.memory.byok_control import CredentialCipher, FernetCredentialCipher, SqliteByokStore


@dataclass(frozen=True)
class ByokResolution:
    route: str  # byok | platform
    status: str  # hit | binding_missing | binding_disabled | profile_missing | profile_disabled | credential_missing | credential_inactive | invalid_model | decrypt_failed
    provider: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    credential_fingerprint: Optional[str]
    secret_cipher: Optional[str]
    profile_id: Optional[str]
    credential_id: Optional[str]


@dataclass(frozen=True)
class ByokProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str]
    credential_fingerprint: Optional[str]


class ByokResolver:
    def __init__(
        self,
        *,
        store: SqliteByokStore,
        cipher: CredentialCipher,
        cache_ttl_s: int = 300,
    ) -> None:
        self._store = store
        self._cipher = cipher
        self._cache_ttl_s = max(0, int(cache_ttl_s))
        self._cache: dict[str, tuple[float, ByokResolution]] = {}
        self._lock = threading.Lock()

    def resolve(self, *, tenant_id: str, api_key_id: str) -> ByokResolution:
        key = f"{tenant_id}:{api_key_id}"
        now = time.time()
        if self._cache_ttl_s > 0:
            with self._lock:
                cached = self._cache.get(key)
                if cached and cached[0] > now:
                    return cached[1]

        res = self._resolve_uncached(tenant_id=str(tenant_id), api_key_id=str(api_key_id))
        if self._cache_ttl_s > 0:
            with self._lock:
                self._cache[key] = (now + self._cache_ttl_s, res)
        return res

    def materialize_config(self, res: ByokResolution) -> Optional[ByokProviderConfig]:
        if res.route != "byok" or not res.secret_cipher or not res.provider or not res.model:
            return None
        try:
            api_key = str(self._cipher.decrypt(res.secret_cipher) or "")
        except Exception:
            return None
        if not api_key:
            return None
        return ByokProviderConfig(
            provider=str(res.provider),
            model=str(res.model),
            api_key=api_key,
            base_url=(str(res.base_url) if res.base_url else None),
            credential_fingerprint=(str(res.credential_fingerprint) if res.credential_fingerprint else None),
        )

    def _resolve_uncached(self, *, tenant_id: str, api_key_id: str) -> ByokResolution:
        tenant = str(tenant_id or "").strip()
        key_id = str(api_key_id or "").strip()
        if not tenant or not key_id:
            return ByokResolution(
                route="platform",
                status="binding_missing",
                provider=None,
                model=None,
                base_url=None,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=None,
                credential_id=None,
            )
        binding = self._store.get_binding(tenant, key_id)
        if binding is None:
            return ByokResolution(
                route="platform",
                status="binding_missing",
                provider=None,
                model=None,
                base_url=None,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=None,
                credential_id=None,
            )
        if str(binding.status).lower() != "active":
            return ByokResolution(
                route="platform",
                status="binding_disabled",
                provider=None,
                model=None,
                base_url=None,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=binding.active_profile_id,
                credential_id=binding.active_credential_id,
            )
        profile = self._store.get_profile(binding.active_profile_id)
        if profile is None:
            return ByokResolution(
                route="platform",
                status="profile_missing",
                provider=None,
                model=None,
                base_url=None,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=binding.active_profile_id,
                credential_id=binding.active_credential_id,
            )
        if str(profile.status).lower() != "active":
            return ByokResolution(
                route="platform",
                status="profile_disabled",
                provider=profile.provider,
                model=profile.default_model,
                base_url=profile.base_url,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=profile.profile_id,
                credential_id=binding.active_credential_id,
            )
        credential = self._store.get_credential(binding.active_credential_id)
        if credential is None:
            return ByokResolution(
                route="platform",
                status="credential_missing",
                provider=profile.provider,
                model=profile.default_model,
                base_url=profile.base_url,
                credential_fingerprint=None,
                secret_cipher=None,
                profile_id=profile.profile_id,
                credential_id=binding.active_credential_id,
            )
        if str(credential.status).lower() != "active":
            return ByokResolution(
                route="platform",
                status="credential_inactive",
                provider=profile.provider,
                model=profile.default_model,
                base_url=profile.base_url,
                credential_fingerprint=credential.key_fingerprint,
                secret_cipher=None,
                profile_id=profile.profile_id,
                credential_id=credential.credential_id,
            )
        model = str(profile.default_model or "").strip()
        if not model:
            return ByokResolution(
                route="platform",
                status="invalid_model",
                provider=profile.provider,
                model=None,
                base_url=profile.base_url,
                credential_fingerprint=credential.key_fingerprint,
                secret_cipher=None,
                profile_id=profile.profile_id,
                credential_id=credential.credential_id,
            )
        return ByokResolution(
            route="byok",
            status="hit",
            provider=profile.provider,
            model=model,
            base_url=profile.base_url,
            credential_fingerprint=credential.key_fingerprint,
            secret_cipher=credential.secret_cipher,
            profile_id=profile.profile_id,
            credential_id=credential.credential_id,
        )


def build_byok_resolver_from_env() -> Optional[ByokResolver]:
    raw_keys = str(os.getenv("BYOK_FERNET_KEYS") or "").strip()
    if not raw_keys:
        raw_keys = str(os.getenv("BYOK_MASTER_KEY") or "").strip()
    if not raw_keys:
        return None
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        return None
    current = keys[0]
    previous = keys[1:]
    db_path = str(os.getenv("MEMORY_BYOK_REGISTRY_DB_PATH") or "").strip() or "modules/memory/outputs/byok_registry.db"
    ttl_raw = os.getenv("BYOK_RESOLVER_CACHE_TTL_S") or os.getenv("BYOK_RESOLVER_CACHE_TTL") or "300"
    try:
        ttl = int(ttl_raw)
    except Exception:
        ttl = 300
    cipher = FernetCredentialCipher(current_key=current, previous_keys=previous, version="v1")
    store = SqliteByokStore(sqlite_path=db_path)
    return ByokResolver(store=store, cipher=cipher, cache_ttl_s=ttl)


__all__ = [
    "ByokProviderConfig",
    "ByokResolution",
    "ByokResolver",
    "build_byok_resolver_from_env",
]
