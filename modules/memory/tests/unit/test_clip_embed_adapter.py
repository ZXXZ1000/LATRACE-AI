from __future__ import annotations

from modules.memory.application.embedding_adapter import build_image_embedding_from_settings


def test_clip_embedder_fallback_shape():
    # In CI/vanilla env open_clip may be unavailable → should fallback to hash with correct dim
    emb = build_image_embedding_from_settings({
        "provider": "clip",
        "model": "ViT-B-32",
        "pretrained": "openai",
        "dim": 512,
    })
    v = emb("a small cat sitting on a sofa")
    assert isinstance(v, list)
    assert len(v) == 512

