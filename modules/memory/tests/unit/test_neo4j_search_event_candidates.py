from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class _Sess:
    def __init__(self, calls: List[Dict[str, Any]], behavior: str) -> None:
        self.calls = calls
        self.behavior = behavior

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        if self.behavior == "fulltext_ok":
            if "tkg_event_summary_desc_v1" in cypher or "tkg_event_summary_v1" in cypher:
                return [{"event": {"id": "e1", "summary": "hello"}, "score": 2.5, "source_id": "s1"}]
            return []
        if self.behavior == "fulltext_missing":
            if "CALL db.index.fulltext.queryNodes('tkg_event_summary_desc_v1'" in cypher:
                raise RuntimeError("index not found")
            if "CALL db.index.fulltext.queryNodes('tkg_event_summary_v1'" in cypher:
                raise RuntimeError("index not found")
            if "toLower(ev.summary) CONTAINS" in cypher:
                return [{"event": {"id": "e2", "summary": "fallback"}, "score": 1.0, "source_id": None}]
            return []
        return []


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]], behavior: str) -> None:
        self.calls = calls
        self.behavior = behavior

    def session(self, *args, **kwargs):
        return _Sess(self.calls, self.behavior)


def test_search_event_candidates_uses_fulltext_when_available():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls, "fulltext_ok")  # type: ignore[attr-defined]
    res = asyncio.run(store.search_event_candidates(tenant_id="t1", query="hello", limit=5, source_id="s1"))
    assert res and res[0]["event_id"] == "e1"
    assert res[0]["matched"] == "summary"
    assert any("CALL db.index.fulltext.queryNodes('tkg_event_summary_desc_v1'" in c["cypher"] for c in calls)


def test_search_event_candidates_falls_back_to_contains_when_fulltext_missing():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls, "fulltext_missing")  # type: ignore[attr-defined]
    res = asyncio.run(store.search_event_candidates(tenant_id="t1", query="fallback", limit=5))
    assert res and res[0]["event_id"] == "e2"
    assert any("toLower(ev.summary) CONTAINS" in c["cypher"] for c in calls), "expected CONTAINS fallback query"
