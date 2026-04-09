import asyncio
from typing import Any, Dict, List

from modules.memory.application.graph_service import GraphService
from modules.memory.contracts.graph_models import (
    Event,
    GraphEdge,
    GraphUpsertRequest,
    Entity,
)
from modules.memory.infra.neo4j_store import Neo4jStore


class _StubStore(Neo4jStore):
    def __init__(self, calls: List[Dict[str, Any]]):
        super().__init__({})
        self.calls = calls
        self._driver = None  # disable real driver

    async def upsert_graph_v0(self, **kwargs):
        self.calls.append(kwargs)


def test_gating_filters_by_confidence_and_topk():
    calls: List[Dict[str, Any]] = []
    store = _StubStore(calls)
    svc = GraphService(store, gating={"confidence_threshold": 0.5, "importance_threshold": 1.0, "rel_topk": 1})

    req = GraphUpsertRequest(
        entities=[Entity(id="ent1", tenant_id="t", type="PERSON")],
        events=[
            Event(id="ev_keep", tenant_id="t", summary="keep", importance=2),
            Event(id="ev_drop", tenant_id="t", summary="drop", importance=0),  # below threshold
        ],
        edges=[
            GraphEdge(src_id="s1", dst_id="d1", rel_type="REL", tenant_id="t", confidence=0.9, weight=0.9),
            GraphEdge(src_id="s1", dst_id="d2", rel_type="REL", tenant_id="t", confidence=0.4, weight=0.4),  # below conf
            GraphEdge(src_id="s1", dst_id="d3", rel_type="REL", tenant_id="t", confidence=0.8, weight=0.8),
        ],
    )

    asyncio.run(svc.upsert(req))

    assert len(calls) == 1
    payload = calls[0]
    # events filtered by importance
    assert len(payload["events"]) == 1
    assert payload["events"][0].id == "ev_keep"
    # edges filtered by confidence + topK=1 (keep highest weight)
    kept_edges = payload["edges"]
    assert len(kept_edges) == 1
    assert kept_edges[0].dst_id == "d1"
