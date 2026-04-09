from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List


class _Rows(list):
    def single(self):
        return self[0] if self else None


class _Sess:
    def __init__(self, calls: List[Dict[str, Any]], responder: Callable[[str, Dict[str, Any]], _Rows]) -> None:
        self.calls = calls
        self.responder = responder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": dict(params)})
        return self.responder(cypher, params)


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]], responder: Callable[[str, Dict[str, Any]], _Rows]) -> None:
        self.calls = calls
        self.responder = responder

    def session(self, *args, **kwargs):
        return _Sess(self.calls, self.responder)


def test_query_entities_by_ids_preserves_input_order_and_applies_name_fallback():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []

    def _responder(cypher: str, params: Dict[str, Any]) -> _Rows:
        if "MATCH (ent:Entity {tenant_id: $tenant})" in cypher and "WHERE ent.id IN $ids" in cypher:
            return _Rows(
                [
                    {"entity": {"id": "ent-2", "name": "", "manual_name": "Bob", "type": "PERSON"}},
                    {"entity": {"id": "ent-1", "name": "Alice", "type": "PERSON"}},
                ]
            )
        return _Rows([])

    store = Neo4jStore({})
    store._driver = _Driver(calls, _responder)  # type: ignore[attr-defined]
    out = asyncio.run(
        store.query_entities_by_ids(
            tenant_id="t1",
            entity_ids=["ent-1", "ent-2", "ent-x"],
            user_ids=["u:alice"],
            memory_domain="work",
            limit=50,
        )
    )

    assert [x["id"] for x in out] == ["ent-1", "ent-2"]
    assert out[0]["name"] == "Alice"
    assert out[1]["name"] == "Bob"
    assert calls and calls[0]["params"]["user_ids"] == ["u:alice"]
    assert calls[0]["params"]["memory_domain"] == "work"


def test_query_event_id_by_logical_id_returns_latest_match():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []

    def _responder(cypher: str, params: Dict[str, Any]) -> _Rows:
        if "logical_event_id" in cypher:
            return _Rows([{"event_id": "evt-9"}])
        return _Rows([])

    store = Neo4jStore({})
    store._driver = _Driver(calls, _responder)  # type: ignore[attr-defined]
    out = asyncio.run(
        store.query_event_id_by_logical_id(
            tenant_id="tenant-a",
            logical_event_id="logic-9",
        )
    )

    assert out == "evt-9"
    assert calls and calls[0]["params"] == {
        "tenant": "tenant-a",
        "logical_event_id": "logic-9",
    }


def test_query_event_id_by_logical_id_blank_input_short_circuit():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []

    def _responder(cypher: str, params: Dict[str, Any]) -> _Rows:
        return _Rows([])

    store = Neo4jStore({})
    store._driver = _Driver(calls, _responder)  # type: ignore[attr-defined]
    out = asyncio.run(
        store.query_event_id_by_logical_id(
            tenant_id="tenant-a",
            logical_event_id="  ",
        )
    )

    assert out is None
    assert calls == []


def test_query_entities_by_name_forwards_user_and_domain_filters():
    from modules.memory.infra.neo4j_store import Neo4jStore

    calls: List[Dict[str, Any]] = []

    def _responder(cypher: str, params: Dict[str, Any]) -> _Rows:
        if "CALL db.index.fulltext.queryNodes('tkg_entity_name_v1'" in cypher:
            return _Rows(
                [
                    {
                        "entity": {"id": "ent-1", "name": "Alice", "type": "PERSON"},
                        "score": 2.0,
                    }
                ]
            )
        return _Rows([])

    store = Neo4jStore({})
    store._driver = _Driver(calls, _responder)  # type: ignore[attr-defined]
    out = asyncio.run(
        store.query_entities_by_name(
            tenant_id="tenant-b",
            name="Alice",
            entity_type="PERSON",
            user_ids=["u:bob"],
            memory_domain="home",
            limit=3,
        )
    )

    assert out and out[0]["entity_id"] == "ent-1"
    assert calls and calls[0]["params"]["uids"] == ["u:bob"]
    assert calls[0]["params"]["domain"] == "home"
