from __future__ import annotations

import asyncio


from modules.memory.infra.qdrant_store import QdrantStore


def test_qdrant_search_uses_embed_text_and_handles_no_network():
    async def _run():
        called = {
            "embed": 0,
            "post": 0,
        }

        store = QdrantStore({"host": "127.0.0.1", "port": 6333, "collections": {"text": "memory_text"}})

        # stub embedder to verify it is called
        def _embed_text(q: str):
            called["embed"] += 1
            return [0.0, 0.0, 0.0]

        store.embed_text = _embed_text  # type: ignore[attr-defined]

        class _Resp:
            status_code = 200

            def json(self):
                return {"result": []}

        class _Sess:
            def post(self, *args, **kwargs):
                called["post"] += 1
                return _Resp()

        store.session = _Sess()  # type: ignore
        res = await store.search_vectors("hello", {"modality": ["text"]}, topk=1, threshold=None)
        assert isinstance(res, list)
        assert called["embed"] == 1
        assert called["post"] >= 1

    asyncio.run(_run())

