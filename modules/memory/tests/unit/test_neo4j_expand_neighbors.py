from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.infra.neo4j_store import Neo4jStore


class _FakeResult:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, hop1_rows: Dict[str, List[Dict[str, Any]]], hop2_rows: Dict[str, List[Dict[str, Any]]]):
        self._h1 = hop1_rows
        self._h2 = hop2_rows
        self._last_sid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, **params):
        sid = params.get("sid")
        if "-[r]->(n)" in query:
            return _FakeResult(self._h1.get(sid, []))
        if "-[r1]->(m)-[r2]->(n)" in query:
            return _FakeResult(self._h2.get(sid, []))
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, hop1_rows: Dict[str, List[Dict[str, Any]]], hop2_rows: Dict[str, List[Dict[str, Any]]]):
        self._h1 = hop1_rows
        self._h2 = hop2_rows

    def session(self, *args, **kwargs):
        return _FakeSession(self._h1, self._h2)


def test_expand_neighbors_two_hops_and_filters():
    # Prepare fake graph rows: for seed 'a'
    hop1 = {
        "a": [
            {"nid": "b", "rel": "prefer", "w": 0.9},
            {"nid": "c", "rel": "appears_in", "w": 0.5},
        ]
    }
    hop2 = {
        "a": [
            {"nid": "d", "rel": "located_in", "w": 0.7},
        ]
    }
    store = Neo4jStore({})
    # inject fake driver
    store._driver = _FakeDriver(hop1, hop2)  # type: ignore[attr-defined]

    res = asyncio.run(
        store.expand_neighbors(
            ["a"],
            rel_whitelist=["prefer", "appears_in", "located_in"],
            max_hops=2,
            neighbor_cap_per_seed=5,
            user_ids=["u1"],
            memory_domain="home",
            restrict_to_user=True,
            restrict_to_domain=True,
        )
    )

    assert "neighbors" in res and "a" in res["neighbors"]
    nbrs = res["neighbors"]["a"]
    # expect b (hop1), c (hop1), d (hop2) with descending weight
    ids = [n.get("to") for n in nbrs]
    assert "b" in ids and "c" in ids and "d" in ids
