from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.infra.neo4j_store import Neo4jStore


class _SessTS:
    def __init__(self, calls: List[Dict[str, Any]], segments: List[Dict[str, Any]]):
        self.calls = calls
        self.segments = segments
        self._first = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        # First query (MATCH MediaSegment) returns rows, subsequent MERGE captures cypher
        if "MATCH (s:MediaSegment" in cypher:
            return self.segments
        self.calls.append({"cypher": cypher, "params": params})
        return []


class _DriverTS:
    def __init__(self, calls: List[Dict[str, Any]], segments: List[Dict[str, Any]]):
        self.calls = calls
        self.segments = segments

    def session(self, *args, **kwargs):
        return _SessTS(self.calls, self.segments)


def test_build_time_slices_from_segments_creates_windows_and_edges():
    calls: List[Dict[str, Any]] = []
    seg_rows = [
        {"id": "segA", "source_id": "src", "t_start": 0.0, "t_end": 10.0},
        {"id": "segB", "source_id": "src", "t_start": 3700.0, "t_end": 3710.0},
    ]
    store = Neo4jStore({})
    store._driver = _DriverTS(calls, seg_rows)  # type: ignore[attr-defined]

    res = asyncio.run(store.build_time_slices_from_segments(tenant_id="t1", window_seconds=3600.0, modes=["media_window"]))
    # Expect two slices, two covers for provided segments
    assert res["timeslices"] == 2
    assert res["edges"] == 2
    cyphers = "\n".join(call["cypher"] for call in calls)
    assert "COVERS_SEGMENT" in cyphers
    assert "TimeSlice" in cyphers or "timeslices" in cyphers
