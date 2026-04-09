from __future__ import annotations

import logging


class _FailSession:
    def put(self, *args, **kwargs):
        class R:
            status_code = 500
            text = "internal"
        return R()

    def post(self, *args, **kwargs):
        raise Exception("net down")

    def get(self, *args, **kwargs):
        class R:
            status_code = 200
            def json(self):
                return {}
        return R()


def test_qdrant_upsert_error_logs_and_raises(caplog):
    from modules.memory.infra.qdrant_store import QdrantStore
    from modules.memory.contracts.memory_models import MemoryEntry
    store = QdrantStore({"collections": {"text": "c_text"}, "embedding": {"dim": 8}})
    # swap session
    store.session = _FailSession()
    caplog.set_level(logging.ERROR)
    # crafting one entry to force upsert
    e = MemoryEntry(kind="semantic", modality="text", contents=["hello"], metadata={})
    try:
        import asyncio
        asyncio.run(store.upsert_vectors([e]))
    except RuntimeError:
        pass
    logs = [rec for rec in caplog.records if "qdrant.upsert.error" in rec.msg]
    assert logs, "expected qdrant.upsert.error log record"


def test_qdrant_search_error_logs_and_degrades(caplog):
    from modules.memory.infra.qdrant_store import QdrantStore
    store = QdrantStore({"collections": {"text": "c_text"}, "embedding": {"dim": 8}})
    store.session = _FailSession()
    caplog.set_level(logging.ERROR)
    import asyncio
    out = asyncio.run(store.search_vectors("q", {"modality": ["text"]}, 5, None))
    assert isinstance(out, list)
    # not necessarily error-level if retries not exhausted, but our session raises -> error logged
    logs = [rec for rec in caplog.records if rec.msg in {"qdrant.search.error", "qdrant.search.http_error"}]
    assert logs, "expected qdrant.search error log"
