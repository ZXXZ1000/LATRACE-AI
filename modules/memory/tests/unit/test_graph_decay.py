from datetime import datetime, timezone, timedelta

from modules.memory.application.graph_service import GraphService
from modules.memory.infra.neo4j_store import Neo4jStore


def test_decay_half_life_config(monkeypatch):
    store = Neo4jStore({})
    # set long half-life to reduce decay
    monkeypatch.setenv("GRAPH_DECAY_HALF_LIFE_DAYS", "100")
    svc = GraphService(store)
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    old = now - timedelta(days=10)
    items = [
        {"id": "old", "importance": 1.0, "memory_strength": 1.0, "last_accessed_at": old.isoformat()},
        {"id": "new", "importance": 1.0, "memory_strength": 1.0, "last_accessed_at": now.isoformat()},
    ]
    scored = svc._apply_decay(items, key="id", time_fields=None)
    assert scored[0]["id"] == "new"
    # decay of old item should still be significant with long half-life
    old_score = svc._decay_score(items[0], now=now, time_fields=None)
    assert old_score > 0.2


def test_decay_uses_half_life(monkeypatch):
    store = Neo4jStore({})
    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    old = now - timedelta(days=10)
    items = [{"importance": 1.0, "memory_strength": 1.0, "last_accessed_at": old.isoformat()}]

    monkeypatch.setenv("GRAPH_DECAY_HALF_LIFE_DAYS", "1")
    svc_short = GraphService(store)
    short_score = svc_short._decay_score(items[0], now=now, time_fields=None)

    monkeypatch.setenv("GRAPH_DECAY_HALF_LIFE_DAYS", "100")
    svc_long = GraphService(store)
    long_score = svc_long._decay_score(items[0], now=now, time_fields=None)

    assert long_score > short_score
