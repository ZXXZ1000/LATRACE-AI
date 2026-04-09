from __future__ import annotations

import asyncio
from typing import Any, Dict


class _StoreCapture:
    def __init__(self) -> None:
        self.calls: Dict[str, Dict[str, Any]] = {}

    async def query_events_by_ids(self, **kwargs):
        self.calls["query_events_by_ids"] = dict(kwargs)
        return [{"id": "evt-1"}]

    async def query_entities_by_ids(self, **kwargs):
        self.calls["query_entities_by_ids"] = dict(kwargs)
        return [{"id": "ent-1"}]

    async def expand_neighbors(self, **kwargs):
        self.calls["expand_neighbors"] = dict(kwargs)
        return {"neighbors": {"evt-1": [{"to": "ent-1"}]}, "edges": []}

    async def query_entities_by_name(self, **kwargs):
        self.calls["query_entities_by_name"] = dict(kwargs)
        return [{"entity_id": "ent-1", "name": "Alice"}]

    async def query_event_id_by_logical_id(self, **kwargs):
        self.calls["query_event_id_by_logical_id"] = dict(kwargs)
        return "evt-1"


class _StoreWithoutPhase0Methods:
    async def query_events_by_ids(self, **kwargs):
        return []

    async def query_entities_by_name(self, **kwargs):
        return []

    async def expand_neighbors(self, **kwargs):
        return {"neighbors": {}, "edges": []}


def test_graph_service_phase0_forwards_events_and_entities_filters():
    from modules.memory.application.graph_service import GraphService

    store = _StoreCapture()
    svc = GraphService(store)  # type: ignore[arg-type]

    events = asyncio.run(
        svc.list_events_by_ids(
            tenant_id="t1",
            event_ids=["evt-1"],
            user_ids=["u:alice"],
            memory_domain="work",
            limit=7,
        )
    )
    entities = asyncio.run(
        svc.list_entities_by_ids(
            tenant_id="t1",
            entity_ids=["ent-1"],
            user_ids=["u:alice"],
            memory_domain="work",
            limit=9,
        )
    )

    assert events == [{"id": "evt-1"}]
    assert entities == [{"id": "ent-1"}]
    assert store.calls["query_events_by_ids"] == {
        "tenant_id": "t1",
        "event_ids": ["evt-1"],
        "user_ids": ["u:alice"],
        "memory_domain": "work",
        "limit": 7,
    }
    assert store.calls["query_entities_by_ids"] == {
        "tenant_id": "t1",
        "entity_ids": ["ent-1"],
        "user_ids": ["u:alice"],
        "memory_domain": "work",
        "limit": 9,
    }


def test_graph_service_phase0_expand_neighbors_enforces_restrict_flags():
    from modules.memory.application.graph_service import GraphService

    store = _StoreCapture()
    svc = GraphService(store)  # type: ignore[arg-type]

    res = asyncio.run(
        svc.expand_neighbors(
            seed_ids=["evt-1"],
            rel_whitelist=["INVOLVES"],
            max_hops=2,
            neighbor_cap_per_seed=3,
            user_ids=["u:alice"],
            memory_domain="work",
            memory_scope="scope::x",
        )
    )

    assert res["neighbors"]["evt-1"][0]["to"] == "ent-1"
    assert store.calls["expand_neighbors"] == {
        "seed_ids": ["evt-1"],
        "rel_whitelist": ["INVOLVES"],
        "max_hops": 2,
        "neighbor_cap_per_seed": 3,
        "user_ids": ["u:alice"],
        "memory_domain": "work",
        "memory_scope": "scope::x",
        "restrict_to_user": True,
        "restrict_to_domain": True,
        "restrict_to_scope": True,
    }


def test_graph_service_phase0_resolve_entities_forwards_scope_filters():
    from modules.memory.application.graph_service import GraphService

    store = _StoreCapture()
    svc = GraphService(store)  # type: ignore[arg-type]

    out = asyncio.run(
        svc.resolve_entities(
            tenant_id="t1",
            name="Alice",
            entity_type="PERSON",
            user_ids=["u:alice"],
            memory_domain="work",
            limit=6,
        )
    )

    assert out == [{"entity_id": "ent-1", "name": "Alice"}]
    assert store.calls["query_entities_by_name"] == {
        "tenant_id": "t1",
        "name": "Alice",
        "entity_type": "PERSON",
        "user_ids": ["u:alice"],
        "memory_domain": "work",
        "limit": 6,
    }


def test_graph_service_phase0_optional_methods_fallback_safe():
    from modules.memory.application.graph_service import GraphService

    store = _StoreWithoutPhase0Methods()
    svc = GraphService(store)  # type: ignore[arg-type]

    entities = asyncio.run(
        svc.list_entities_by_ids(
            tenant_id="t1",
            entity_ids=["ent-1"],
        )
    )
    event_id = asyncio.run(
        svc.event_id_by_logical_id(
            tenant_id="t1",
            logical_event_id="logic-1",
        )
    )

    assert entities == []
    assert event_id is None


def test_graph_service_phase0_event_id_by_logical_id_forwarding():
    from modules.memory.application.graph_service import GraphService

    store = _StoreCapture()
    svc = GraphService(store)  # type: ignore[arg-type]

    event_id = asyncio.run(
        svc.event_id_by_logical_id(
            tenant_id="tenant-x",
            logical_event_id="logic-123",
        )
    )

    assert event_id == "evt-1"
    assert store.calls["query_event_id_by_logical_id"] == {
        "tenant_id": "tenant-x",
        "logical_event_id": "logic-123",
    }
