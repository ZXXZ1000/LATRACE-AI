# BYOK Control Plane (Registry)

## Responsibilities

- Manage provider profiles (provider/base_url/default_model).
- Manage encrypted credentials (secret cipher + fingerprint).
- Bind `tenant_id + api_key_id` to active provider/credential.

## Public API

- `ByokRegistry`: CRUD for profiles/credentials/bindings + audit hooks.
- `SqliteByokStore`: SQLite persistence adapter.
- `FernetCredentialCipher`: encrypt/decrypt secrets (Fernet; versioned prefix).

## Input / Output Contract

- Input: `provider/base_url/default_model`, `plaintext_secret`, `tenant_id/api_key_id`.
- Output: `ProviderProfile/ProviderCredential/LlmBinding` with encrypted secret.

## Dependencies

- `cryptography` for Fernet encryption.
- SQLite (standard library).

## Data Flow

1. `ByokRegistry` validates input.
2. `CredentialCipher` encrypts plaintext secret.
3. `SqliteByokStore` persists profile/credential/binding.
4. Optional `AuditSink` receives structured events (no plaintext).

## Boundaries

- No plaintext secret is stored in DB or logs.
- Decryption is internal-only; SDK never sees keys.
