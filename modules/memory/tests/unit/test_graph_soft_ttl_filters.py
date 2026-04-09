from typing import Any, Dict, List

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
        return []


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def session(self, *args, **kwargs):
        return _Sess(self.calls)


def test_query_segments_includes_soft_ttl_filter():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    # invoke query to record cypher
    import asyncio

    asyncio.run(store.query_segments_by_time(tenant_id="t", limit=10))
    assert calls, "expected session run calls"
    cypher = calls[0]["cypher"]
    assert "expires_at" in cypher
    assert "s.expires_at IS NULL OR s.expires_at > datetime()" in cypher
