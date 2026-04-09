from __future__ import annotations

import logging
import asyncio


class _BadSession:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def run(self, *args, **kwargs):
        raise RuntimeError("neo down")


class _BadDriver:
    def session(self, *args, **kwargs):
        return _BadSession()


def test_neo4j_merge_nodes_edges_logs_error(caplog):
    from modules.memory.infra.neo4j_store import Neo4jStore
    from modules.memory.contracts.memory_models import MemoryEntry
    store = Neo4jStore({})
    store._driver = _BadDriver()  # type: ignore[attr-defined]
    caplog.set_level(logging.ERROR)
    e = MemoryEntry(kind="semantic", modality="text", contents=["x"], metadata={})
    asyncio.run(store.merge_nodes_edges([e], None))
    logs = [rec for rec in caplog.records if rec.msg == "neo4j.merge_nodes_edges.error"]
    assert logs, "expected neo4j.merge_nodes_edges.error log"
