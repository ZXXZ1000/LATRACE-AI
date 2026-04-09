from __future__ import annotations

from typing import Any, Dict

import asyncio

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.contracts.memory_models import MemoryEntry


def test_search_dedup_prefers_higher_score(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text", "clip_image": "memory_clip_image"},
        "embedding": {"dim": 4, "clip_image": {"dim": 4}},
    }
    store = QdrantStore(settings)

    # embed functions return trivial vectors
    store.embed_text = lambda q: [0.1, 0.2, 0.3, 0.4]
    store.embed_clip_image = lambda q: [0.4, 0.3, 0.2, 0.1]

    class _Resp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
        def json(self):
            return self._data
        def raise_for_status(self):
            return None

    # Simulate Qdrant search with same id in both modalities, different scores
    def fake_post(url: str, json: Dict[str, Any], timeout: int = 10):
        if "/memory_text/" in url:
            data = {"result": [{"id": "X", "score": 0.5, "payload": MemoryEntry(kind="semantic", modality="text", contents=["a"], metadata={}).model_dump()}]}
            return _Resp(data)
        if "/memory_clip_image/" in url:
            data = {"result": [{"id": "X", "score": 0.8, "payload": MemoryEntry(kind="semantic", modality="image", contents=["url://thumb"], metadata={}).model_dump()}]}
            return _Resp(data)
        return _Resp({"result": []})

    store.session.post = fake_post  # type: ignore

    res = asyncio.run(store.search_vectors("q", {"modality": ["text", "clip_image"]}, topk=5))
    assert res and len(res) == 1  # deduped by id
    assert abs(float(res[0]["score"]) - 0.8) < 1e-6


def test_upsert_raises_on_dim_mismatch(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"image": "memory_image"},
        "embedding": {"dim": 768, "image": {"dim": 512}},
    }
    store = QdrantStore(settings)

    # prevent real network
    class _Resp:
        def raise_for_status(self):
            return None
    store.session.put = lambda url, json, timeout=60: _Resp()  # type: ignore

    e = MemoryEntry(kind="semantic", modality="image", contents=["img://"], vectors={"image": [0.1] * 128}, metadata={})
    try:
        asyncio.run(store.upsert_vectors([e]))
        assert False, "expected vector_dim_mismatch"
    except RuntimeError as ex:
        assert "vector_dim_mismatch" in str(ex)


def test_ensure_collections_success(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text"},
        "embedding": {"dim": 3},
    }
    store = QdrantStore(settings)

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"result": {"payload_schema": {}}}

    store.session.get = lambda url, timeout=10: _Resp()  # type: ignore
    store.session.put = lambda url, json, timeout=10: _Resp()  # type: ignore

    asyncio.run(store.ensure_collections())


def test_ensure_collections_creates_payload_indexes(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text"},
        "embedding": {"dim": 3},
    }
    store = QdrantStore(settings)

    calls = []

    class _Resp:
        status_code = 200
        def json(self):
            return {"result": {"payload_schema": {}}}

    def _fake_put(url, json, timeout=10):  # type: ignore[no-untyped-def]
        calls.append((url, json, timeout))
        return _Resp()

    store.session.get = lambda url, timeout=10: _Resp()  # type: ignore
    store.session.put = _fake_put  # type: ignore

    asyncio.run(store.ensure_collections())

    index_calls = [
        body.get("field_name")
        for (url, body, _timeout) in calls
        if isinstance(url, str) and url.endswith("/collections/memory_text/index")
    ]
    assert "metadata.tenant_id" in index_calls
    assert "metadata.user_id" in index_calls
    assert "metadata.timestamp" in index_calls
    assert "published" in index_calls


def test_ensure_collections_skips_existing_payload_indexes(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text"},
        "embedding": {"dim": 3},
    }
    store = QdrantStore(settings)

    calls = []

    class _GetResp:
        status_code = 200
        def json(self):
            return {
                "result": {
                    "payload_schema": {
                        "metadata.tenant_id": {"data_type": "keyword"},
                        "metadata.user_id": {"data_type": "keyword"},
                        "metadata.timestamp": {"data_type": "float"},
                        "published": {"data_type": "bool"},
                    }
                }
            }

    class _PutResp:
        status_code = 200

    def _fake_put(url, json, timeout=10):  # type: ignore[no-untyped-def]
        calls.append((url, json, timeout))
        return _PutResp()

    store.session.get = lambda url, timeout=10: _GetResp()  # type: ignore
    store.session.put = _fake_put  # type: ignore

    asyncio.run(store.ensure_collections())

    index_calls = {
        body.get("field_name")
        for (url, body, _timeout) in calls
        if isinstance(url, str) and url.endswith("/collections/memory_text/index")
    }
    assert "metadata.tenant_id" not in index_calls
    assert "metadata.user_id" not in index_calls
    assert "metadata.timestamp" not in index_calls
    assert "published" not in index_calls
    assert "metadata.memory_domain" in index_calls


def test_only_text_compat(monkeypatch):
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"text": "memory_text"},
        "embedding": {"dim": 4},
    }
    store = QdrantStore(settings)
    store.embed_text = lambda q: [0.1, 0.2, 0.3, 0.4]

    class _Resp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
        def json(self):
            return self._data
        def raise_for_status(self):
            return None

    def fake_post(url: str, json: Dict[str, Any], timeout: int = 10):
        data = {"result": [{"id": "T1", "score": 0.6, "payload": MemoryEntry(kind="semantic", modality="text", contents=["x"], metadata={}).model_dump()}]}
        return _Resp(data)

    store.session.post = fake_post  # type: ignore
    res = asyncio.run(store.search_vectors("hello", {}, topk=3))
    assert res and res[0]["id"] == "T1"
