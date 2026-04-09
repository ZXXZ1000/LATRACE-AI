from __future__ import annotations

from typing import Any, Dict

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.contracts.memory_models import MemoryEntry


def test_upsert_image_entry_sends_face_and_clip_image_vectors(monkeypatch):
    # Prepare store with extended collections
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {
            "image": "memory_image",
            "clip_image": "memory_clip_image",
            "face": "memory_face",
        },
        "embedding": {
            "dim": 768,
            "image": {"dim": 512},
            "clip_image": {"dim": 512},
            "face": {"dim": 512},
        },
    }
    store = QdrantStore(settings)

    captured = {"puts": []}

    class _Resp:
        def raise_for_status(self):
            return None

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None

    def fake_put(url: str, json: Dict[str, Any], timeout: int = 60):
        captured["puts"].append({"url": url, "json": json})
        return _Resp()

    store.session.put = fake_put  # type: ignore

    # Create an image entry with both face and clip_image vectors
    face_vec = list(range(512))
    ci_vec = [0.01] * 512
    e = MemoryEntry(
        kind="semantic",
        modality="image",
        contents=["img://dummy"],
        vectors={"face": face_vec, "clip_image": ci_vec},
        metadata={"source": "test"},
    )

    import asyncio

    asyncio.run(store.upsert_vectors([e]))

    # Should have produced at least two PUTs: one to face collection and one to clip_image
    urls = [p["url"] for p in captured["puts"]]
    assert any("/collections/memory_face/points" in u for u in urls)
    assert any("/collections/memory_clip_image/points" in u for u in urls)
    # And each payload should contain a vector field with expected length
    face_payloads = [p for p in captured["puts"] if "/memory_face/" in p["url"]]
    ci_payloads = [p for p in captured["puts"] if "/memory_clip_image/" in p["url"]]
    assert face_payloads and ci_payloads
    fvec = face_payloads[0]["json"]["points"][0]["vector"]
    cvec = ci_payloads[0]["json"]["points"][0]["vector"]
    assert isinstance(fvec, list) and len(fvec) == 512
    assert isinstance(cvec, list) and len(cvec) == 512
