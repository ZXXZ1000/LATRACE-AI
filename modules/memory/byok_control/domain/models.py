from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ProfileStatus = Literal["active", "disabled"]
CredentialStatus = Literal["active", "rotated", "revoked"]
BindingStatus = Literal["active", "disabled"]


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    base_url: Optional[str]
    default_model: Optional[str]
    status: ProfileStatus
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ProviderCredential:
    credential_id: str
    profile_id: str
    secret_cipher: str
    key_fingerprint: str
    status: CredentialStatus
    created_at: str
    rotated_at: Optional[str]


@dataclass(frozen=True)
class LlmBinding:
    tenant_id: str
    api_key_id: str
    active_profile_id: str
    active_credential_id: str
    status: BindingStatus
    created_at: str
    updated_at: str
