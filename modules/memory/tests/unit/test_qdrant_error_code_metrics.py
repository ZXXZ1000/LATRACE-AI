from __future__ import annotations

import asyncio

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.application.metrics import get_metrics


def test_qdrant_error_code_metrics_4xx_5xx():
    async def _run():
        store = QdrantStore({"host": "127.0.0.1", "port": 6333, "collections": {"text": "memory_text"}})
        store.embed_text = lambda q: [0.0, 0.0, 0.0]  # type: ignore

        class _Resp:
            def __init__(self, code):
                self.status_code = code

            def json(self):
                return {"result": []}

        class _Sess:
            def __init__(self, code):
                self.code = code

            def post(self, *args, **kwargs):
                return _Resp(self.code)

        # test 4xx
        store.session = _Sess(400)  # type: ignore
        before4 = int(get_metrics().get("errors_4xx_total", 0))
        await store.search_vectors("q", {"modality": ["text"]}, topk=1, threshold=None)
        after4 = int(get_metrics().get("errors_4xx_total", 0))
        assert after4 == before4 + 1

        # test 5xx
        store.session = _Sess(500)  # type: ignore
        before5 = int(get_metrics().get("errors_5xx_total", 0))
        await store.search_vectors("q", {"modality": ["text"]}, topk=1, threshold=None)
        after5 = int(get_metrics().get("errors_5xx_total", 0))
        assert after5 == before5 + 1

    asyncio.run(_run())

