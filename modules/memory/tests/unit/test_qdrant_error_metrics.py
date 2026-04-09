from __future__ import annotations

import asyncio

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.application.metrics import get_metrics


def test_qdrant_search_errors_increment_metrics():
    async def _run():
        store = QdrantStore({"host": "127.0.0.1", "port": 6333, "collections": {"text": "memory_text"}})

        # minimal embedder
        store.embed_text = lambda q: [0.0, 0.0, 0.0]  # type: ignore[attr-defined]

        class _Sess:
            def post(self, *args, **kwargs):
                raise RuntimeError("network error")

        store.session = _Sess()  # type: ignore
        before = int(get_metrics().get("errors_total", 0))
        res = await store.search_vectors("hello", {"modality": ["text"]}, topk=1, threshold=None)
        after = int(get_metrics().get("errors_total", 0))
        assert res == []
        assert after >= before + 1

    asyncio.run(_run())

