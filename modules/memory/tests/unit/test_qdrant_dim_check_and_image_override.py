from __future__ import annotations

import asyncio

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.contracts.memory_models import MemoryEntry


class _FakeResp:
    def __init__(self):
        self.status_code = 200
    def json(self):
        return {"result": {}}
    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, capture):
        self.headers = {}
        self._cap = capture
    def put(self, url, json=None, timeout=10):
        # capture payload for assertions
        self._cap["payload"] = json
        return _FakeResp()


def test_dim_mismatch_raises_before_network():
    # expect image dim=2, but provide vector of len=3
    s = QdrantStore({
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"image": "cimg"},
        "embedding": {"image": {"dim": 2}},
    })
    e = MemoryEntry(kind="semantic", modality="image", contents=["data:image/jpeg;base64,/9j/"], vectors={"image": [1.0, 2.0, 3.0]}, metadata={})
    # override embed_image to avoid heavy deps; we won't reach it due to pre-check
    s.embed_image = lambda _: [1.0, 2.0, 3.0]  # type: ignore
    try:
        asyncio.run(s.upsert_vectors([e]))
        assert False, "should have raised due to dim mismatch"
    except RuntimeError as ex:
        assert "vector_dim_mismatch" in str(ex)


def test_image_precomputed_vectors_are_preferred_when_present():
    cap = {}
    s = QdrantStore({
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"image": "cimg"},
        "embedding": {"image": {"dim": 2}, "clip_image": {"dim": 2}},
    })
    # monkeypatch session to avoid real HTTP
    s.session = _FakeSession(cap)
    # force embedder to return [9,9] while entry carries different vectors; we expect precomputed to be used
    s.embed_image = lambda _: [9.0, 9.0]  # type: ignore
    s.embed_clip_image = lambda _: [9.0, 9.0]  # type: ignore
    e = MemoryEntry(kind="semantic", modality="image", contents=["data:image/jpeg;base64,/9j/"], vectors={"image": [1.0, 2.0]}, metadata={})
    asyncio.run(s.upsert_vectors([e]))
    pts = cap.get("payload", {}).get("points", [])
    assert len(pts) == 1
    assert pts[0]["vector"] == [1.0, 2.0]
