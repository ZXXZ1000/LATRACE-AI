from __future__ import annotations

import logging


class _Http500:
    status_code = 500
    def json(self):
        return {}


class _FailSession:
    def __init__(self, mode: str):
        self.mode = mode
    def post(self, *args, **kwargs):
        if self.mode == "http":
            return _Http500()
        raise RuntimeError("net down")


def test_scroll_http_error_logs(caplog):
    from modules.memory.infra.qdrant_store import QdrantStore
    store = QdrantStore({"collections": {"text": "c_text"}, "embedding": {"dim": 8}})
    store.session = _FailSession("http")
    caplog.set_level(logging.WARNING)
    import asyncio
    out = asyncio.run(store.fetch_text_corpus({}, limit=10))
    assert out == []
    msgs = [r.msg for r in caplog.records]
    assert "qdrant.scroll.http_error" in msgs


def test_scroll_exception_logs(caplog):
    from modules.memory.infra.qdrant_store import QdrantStore
    store = QdrantStore({"collections": {"text": "c_text"}, "embedding": {"dim": 8}})
    store.session = _FailSession("raise")
    caplog.set_level(logging.ERROR)
    import asyncio
    out = asyncio.run(store.fetch_text_corpus({}, limit=10))
    assert out == []
    msgs = [r.msg for r in caplog.records]
    assert "qdrant.scroll.error" in msgs
