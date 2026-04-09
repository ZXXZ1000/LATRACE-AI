from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.qdrant_store import QdrantStore


class _Resp:
    def __init__(self, status_code: int = 200, data: Dict[str, Any] | None = None) -> None:
        self.status_code = int(status_code)
        self._data = data or {"status": "ok", "result": []}
        self.text = str(self._data)

    def json(self) -> Dict[str, Any]:
        return dict(self._data)


def _make_store() -> QdrantStore:
    store = QdrantStore(
        {
            "host": "127.0.0.1",
            "port": 6333,
            "collections": {"text": "memory_text"},
            "embedding": {"dim": 3},
            "sharding": {
                "enabled": True,
                "method": "custom",
                "key_field": "tenant_id",
                "namespace_ids_by_tenant": True,
            },
        }
    )
    store.embed_text = lambda _q: [0.0, 0.0, 0.0]  # type: ignore[attr-defined]
    return store


def test_qdrant_upsert_creates_custom_shard_and_passes_shard_key() -> None:
    store = _make_store()
    calls: List[Dict[str, Any]] = []

    class _Sess:
        def put(self, url: str, json: Dict[str, Any], timeout: int = 60):  # type: ignore[no-untyped-def]
            calls.append({"url": url, "json": json, "timeout": timeout})
            return _Resp()

    store.session = _Sess()  # type: ignore[assignment]
    entry = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={"tenant_id": "tenant-a"})

    asyncio.run(store.upsert_vectors([entry]))

    assert len(calls) >= 2
    assert calls[0]["url"].endswith("/collections/memory_text/shards")
    assert calls[0]["json"] == {"shard_key": "tenant-a"}
    assert calls[1]["url"].endswith("/collections/memory_text/points?wait=true")
    assert calls[1]["json"]["shard_key"] == "tenant-a"
    assert calls[1]["json"]["points"]


def test_qdrant_search_passes_shard_key_for_tenant_scoped_queries() -> None:
    store = _make_store()
    seen: List[Dict[str, Any]] = []

    class _Sess:
        def post(self, url: str, json: Dict[str, Any], timeout: int = 10):  # type: ignore[no-untyped-def]
            seen.append({"url": url, "json": json, "timeout": timeout})
            payload = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={"tenant_id": "tenant-a"})
            return _Resp(data={"result": [{"id": "p1", "score": 0.9, "payload": payload.model_dump()}]})

    store.session = _Sess()  # type: ignore[assignment]
    res = asyncio.run(store.search_vectors("hello", {"tenant_id": "tenant-a", "modality": ["text"]}, topk=3))

    assert res and res[0]["id"] == "p1"
    assert seen
    assert seen[0]["json"]["shard_key"] == "tenant-a"


def test_qdrant_ensure_collections_uses_custom_sharding_spec() -> None:
    store = _make_store()
    calls: List[Dict[str, Any]] = []

    class _Sess:
        def get(self, url: str, timeout: int = 10):  # type: ignore[no-untyped-def]
            return _Resp(
                data={
                    "result": {
                        "config": {"params": {"sharding_method": "custom"}},
                        "payload_schema": {},
                    }
                }
            )

        def put(self, url: str, json: Dict[str, Any], timeout: int = 10):  # type: ignore[no-untyped-def]
            calls.append({"url": url, "json": json, "timeout": timeout})
            return _Resp()

    store.session = _Sess()  # type: ignore[assignment]
    asyncio.run(store.ensure_collections())

    create_call = next(call for call in calls if call["url"].endswith("/collections/memory_text"))
    assert create_call["json"]["sharding_method"] == "custom"
    assert create_call["json"]["shard_number"] == 1


def test_qdrant_upsert_requires_tenant_when_sharding_enabled() -> None:
    store = _make_store()
    entry = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={})

    try:
        asyncio.run(store.upsert_vectors([entry]))
        assert False, "expected missing shard key failure"
    except RuntimeError as exc:
        assert "qdrant_shard_key_missing" in str(exc)
