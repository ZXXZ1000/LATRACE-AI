from __future__ import annotations

from typing import Optional, Protocol, Sequence

from cryptography.fernet import Fernet, MultiFernet


class CredentialCipher(Protocol):
    def encrypt(self, plaintext: str) -> str:
        ...

    def decrypt(self, cipher_text: str) -> str:
        ...


class FernetCredentialCipher:
    def __init__(
        self,
        *,
        current_key: str,
        previous_keys: Optional[Sequence[str]] = None,
        version: str = "v1",
    ) -> None:
        if not str(current_key or "").strip():
            raise ValueError("current_key is required")
        keys = [current_key] + list(previous_keys or [])
        fernet_keys = [Fernet(k.encode() if isinstance(k, str) else k) for k in keys]
        self._fernet = MultiFernet(fernet_keys)
        self._version = str(version or "v1")

    def encrypt(self, plaintext: str) -> str:
        raw = str(plaintext or "")
        if not raw:
            raise ValueError("plaintext is required")
        token = self._fernet.encrypt(raw.encode("utf-8")).decode("utf-8")
        return f"{self._version}:{token}"

    def decrypt(self, cipher_text: str) -> str:
        raw = str(cipher_text or "")
        if not raw:
            raise ValueError("cipher_text is required")
        token = raw
        if ":" in raw:
            prefix, rest = raw.split(":", 1)
            if prefix == self._version and rest:
                token = rest
        return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
