import asyncio
from typing import Any, Dict, List

from modules.memory.application.graph_service import GraphService
from modules.memory.infra.neo4j_store import Neo4jStore


class _Sess:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        class _Row:
            def single(self_inner):
                return {"updated": 2}
        return _Row()


class _Drv:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def session(self, *args, **kwargs):
        return _Sess(self.calls)


def test_touch_calls_store():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Drv(calls)  # type: ignore[attr-defined]
    svc = GraphService(store)

    res = asyncio.run(svc.touch(tenant_id="t", node_ids=["a", "b"], extend_seconds=10))
    assert res["updated"] == 2
    assert calls, "expected cypher execution"
    assert "last_accessed_at" in calls[0]["cypher"]
