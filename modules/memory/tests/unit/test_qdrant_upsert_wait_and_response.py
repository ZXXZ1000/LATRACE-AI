from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.qdrant_store import QdrantStore


def test_qdrant_upsert_uses_wait_true_and_validates_response(monkeypatch) -> None:
    store = QdrantStore(
        {
            "host": "127.0.0.1",
            "port": 6333,
            "collections": {"text": "memory_text"},
            "embedding": {"dim": 3},
        }
    )

    # deterministic embedder, avoid real network
    store.embed_text = lambda q: [0.0, 0.0, 0.0]  # type: ignore[attr-defined]

    calls: List[Dict[str, Any]] = []

    class _Resp:
        status_code = 200
        text = '{"status":"ok"}'

        def json(self):
            return {"status": "ok"}

    class _Sess:
        def put(self, url: str, json: Dict[str, Any], timeout: int = 60):
            calls.append({"url": url, "json": json, "timeout": timeout})
            return _Resp()

    store.session = _Sess()  # type: ignore[assignment]
    store._upsert_wait = True  # ensure deterministic test behavior

    e = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={"tenant_id": "t1"})
    asyncio.run(store.upsert_vectors([e]))

    assert calls, "expected a PUT call"
    assert "?wait=true" in calls[0]["url"]

