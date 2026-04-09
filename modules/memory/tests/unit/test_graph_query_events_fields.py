from __future__ import annotations

import asyncio

from modules.memory.infra.neo4j_store import Neo4jStore


class FakeRow:
    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default=None):
        return self._data.get(key, default)


class FakeSession:
    def __init__(self, rows: list[FakeRow]):
        self._rows = rows

    def run(self, *args, **kwargs):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeDriver:
    def __init__(self, rows: list[FakeRow]):
        self._rows = rows

    def session(self, database=None):
        return FakeSession(self._rows)


def test_query_events_includes_event_fields_and_relations():
    store = Neo4jStore(settings={})
    row = FakeRow(
        {
            "event": {
                "id": "event-1",
                "summary": "Person arrives home",
                "event_type": "arrival",
                "action": "arrive_home",
                "actor_id": "person::t::p1",
                "t_abs_start": "2025-01-01T00:00:00Z",
                "t_abs_end": "2025-01-01T00:00:05Z",
            },
            "segment_id": "seg-a",
            "source_id": "demo.mp4",
            "entity_ids": ["person::t::p1"],
            "place_ids": ["place-1"],
            "relations": [
                {
                    "type": "NEXT_EVENT",
                    "target_event_id": "event-2",
                    "layer": "fact",
                    "status": None,
                    "kind": "observed",
                }
            ],
        }
    )
    store._driver = FakeDriver([row])

    res = asyncio.run(store.query_events(tenant_id="t1"))

    assert res[0]["event_type"] == "arrival"
    assert res[0]["action"] == "arrive_home"
    assert res[0]["actor_id"] == "person::t::p1"
    assert res[0]["relations"] == [
        {
            "type": "NEXT_EVENT",
            "target_event_id": "event-2",
            "layer": "fact",
            "status": None,
            "kind": "observed",
        }
    ]
