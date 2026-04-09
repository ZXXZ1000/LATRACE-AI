from __future__ import annotations

from typing import Optional
import sqlite3
import threading

from modules.memory.byok_control.domain.models import LlmBinding, ProviderCredential, ProviderProfile


class SqliteByokStore:
    def __init__(self, *, sqlite_path: str) -> None:
        path = str(sqlite_path or "").strip()
        if not path:
            raise ValueError("sqlite_path is required")
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS byok_provider_profiles (
                profile_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                base_url TEXT,
                default_model TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS byok_provider_credentials (
                credential_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                secret_cipher TEXT NOT NULL,
                key_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                rotated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS byok_bindings (
                tenant_id TEXT NOT NULL,
                api_key_id TEXT NOT NULL,
                active_profile_id TEXT NOT NULL,
                active_credential_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, api_key_id)
            )
            """
        )
        self._conn.commit()

    def upsert_profile(self, profile: ProviderProfile) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO byok_provider_profiles(
                    profile_id, provider, base_url, default_model, status, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    profile.profile_id,
                    profile.provider,
                    profile.base_url,
                    profile.default_model,
                    profile.status,
                    profile.created_at,
                    profile.updated_at,
                ),
            )
            self._conn.commit()

    def get_profile(self, profile_id: str) -> Optional[ProviderProfile]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT profile_id, provider, base_url, default_model, status, created_at, updated_at
                FROM byok_provider_profiles
                WHERE profile_id=?
                """,
                (str(profile_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ProviderProfile(
                profile_id=str(row[0]),
                provider=str(row[1]),
                base_url=(str(row[2]) if row[2] else None),
                default_model=(str(row[3]) if row[3] else None),
                status=str(row[4]),
                created_at=str(row[5]),
                updated_at=str(row[6]),
            )

    def upsert_credential(self, credential: ProviderCredential) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO byok_provider_credentials(
                    credential_id, profile_id, secret_cipher, key_fingerprint, status, created_at, rotated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    credential.credential_id,
                    credential.profile_id,
                    credential.secret_cipher,
                    credential.key_fingerprint,
                    credential.status,
                    credential.created_at,
                    credential.rotated_at,
                ),
            )
            self._conn.commit()

    def get_credential(self, credential_id: str) -> Optional[ProviderCredential]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT credential_id, profile_id, secret_cipher, key_fingerprint, status, created_at, rotated_at
                FROM byok_provider_credentials
                WHERE credential_id=?
                """,
                (str(credential_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return ProviderCredential(
                credential_id=str(row[0]),
                profile_id=str(row[1]),
                secret_cipher=str(row[2]),
                key_fingerprint=str(row[3]),
                status=str(row[4]),
                created_at=str(row[5]),
                rotated_at=(str(row[6]) if row[6] else None),
            )

    def upsert_binding(self, binding: LlmBinding) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO byok_bindings(
                    tenant_id, api_key_id, active_profile_id, active_credential_id, status, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    binding.tenant_id,
                    binding.api_key_id,
                    binding.active_profile_id,
                    binding.active_credential_id,
                    binding.status,
                    binding.created_at,
                    binding.updated_at,
                ),
            )
            self._conn.commit()

    def get_binding(self, tenant_id: str, api_key_id: str) -> Optional[LlmBinding]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT tenant_id, api_key_id, active_profile_id, active_credential_id, status, created_at, updated_at
                FROM byok_bindings
                WHERE tenant_id=? AND api_key_id=?
                """,
                (str(tenant_id), str(api_key_id)),
            )
            row = cur.fetchone()
            if not row:
                return None
            return LlmBinding(
                tenant_id=str(row[0]),
                api_key_id=str(row[1]),
                active_profile_id=str(row[2]),
                active_credential_id=str(row[3]),
                status=str(row[4]),
                created_at=str(row[5]),
                updated_at=str(row[6]),
            )
