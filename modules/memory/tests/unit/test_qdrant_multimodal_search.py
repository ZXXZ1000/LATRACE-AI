from __future__ import annotations

import types
from typing import Any, Dict

from modules.memory.infra.qdrant_store import QdrantStore


class _FakeResp:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_qdrant_multimodal_union_and_filters(monkeypatch):
    store = QdrantStore({
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}
    })

    calls = []

    def _fake_post(url: str, json: Dict[str, Any], timeout: int = 10):
        calls.append((url, json))
        if url.endswith("/collections/memory_text/points/search"):
            return _FakeResp(200, {
                "result": [
                    {"id": "t1", "score": 0.8, "payload": {"id": "t1", "kind": "semantic", "modality": "text", "contents": ["我 喜欢 科幻 电影"], "metadata": {}}},
                ]
            })
        if url.endswith("/collections/memory_image/points/search"):
            return _FakeResp(200, {
                "result": [
                    {"id": "i1", "score": 0.75, "payload": {"id": "i1", "kind": "semantic", "modality": "image", "contents": ["电影 海报"], "metadata": {}}},
                ]
            })
        if url.endswith("/collections/memory_audio/points/search"):
            return _FakeResp(200, {
                "result": [
                    {"id": "a1", "score": 0.70, "payload": {"id": "a1", "kind": "semantic", "modality": "audio", "contents": ["说话人 偏好"], "metadata": {}}},
                ]
            })
        return _FakeResp(404, {})

    # monkeypatch the session.post
    store.session.post = _fake_post  # type: ignore[assignment]

    # request all modalities
    types.SimpleNamespace()
    import asyncio
    results = asyncio.run(store.search_vectors("科幻", {"modality": ["text", "image", "audio"]}, topk=3, threshold=None))

    ids = [r["id"] for r in results]
    assert set(ids) == {"t1", "i1", "a1"}
    # ensure three calls made
    urls = [u for (u, _j) in calls]
    assert any("/memory_text/points/search" in u for u in urls)
    assert any("/memory_image/points/search" in u for u in urls)
    assert any("/memory_audio/points/search" in u for u in urls)


def test_qdrant_modality_filter_text_only(monkeypatch):
    store = QdrantStore({
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}
    })

    calls = []

    def _fake_post(url: str, json: Dict[str, Any], timeout: int = 10):
        calls.append(url)
        if "/collections/memory_text/points/search" in url:
            return _FakeResp(200, {"result": []})
        return _FakeResp(404, {})

    store.session.post = _fake_post  # type: ignore[assignment]

    import asyncio
    _ = asyncio.run(store.search_vectors("科幻", {"modality": ["text"]}, topk=5, threshold=None))

    # only text collection queried
    assert any("/collections/memory_text/points/search" in u for u in calls)
    assert not any("/collections/memory_image/points/search" in u for u in calls)
    assert not any("/collections/memory_audio/points/search" in u for u in calls)
