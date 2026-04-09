from __future__ import annotations

from modules.memory.application.config import load_memory_config


def test_memory_config_audio_dim_is_192():
    cfg = load_memory_config()
    aud = ((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("embedding", {}).get("audio", {})
    assert int(aud.get("dim", 0)) == 192

