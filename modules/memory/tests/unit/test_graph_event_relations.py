from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.infra.neo4j_store import Neo4jStore


class _Session:
    def __init__(self, calls: List[Dict[str, Any]], rows: List[Dict[str, Any]]):
        self.calls = calls
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        # First query returns rows (events), subsequent queries record cypher/params
        if "ORDER BY ev.t_abs_start" in cypher:
            return self.rows
        self.calls.append({"cypher": cypher, "params": params})
        return []


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]], rows: List[Dict[str, Any]]):
        self.calls = calls
        self.rows = rows

    def session(self, *args, **kwargs):
        return _Session(self.calls, self.rows)


def test_build_event_relations_creates_edges():
    calls: List[Dict[str, Any]] = []
    events = [
        {"id": "ev1", "t_abs_start": 1, "t_abs_end": None, "source_id": "src", "place_id": "p1"},
        {"id": "ev2", "t_abs_start": 2, "t_abs_end": None, "source_id": "src", "place_id": "p1"},
        {"id": "ev3", "t_abs_start": 3, "t_abs_end": None, "source_id": "src", "place_id": "p2"},
    ]
    store = Neo4jStore({})
    store._driver = _Driver(calls, events)  # type: ignore[attr-defined]

    res = asyncio.run(store.build_event_relations(tenant_id="t1"))
    assert res["next_event"] == 2
    assert res["causes"] == 1  # only ev1->ev2 same place

    # Two MERGE batches: NEXT_EVENT and CAUSES
    cyphers = "\n".join(call["cypher"] for call in calls)
    assert "NEXT_EVENT" in cyphers
    assert "CAUSES" in cyphers
