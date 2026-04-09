import asyncio
from datetime import datetime

from modules.memory.application.graph_service import GraphService
from modules.memory.contracts.graph_models import (
    Event,
    GraphEdge,
    GraphUpsertRequest,
)
from modules.memory.infra.neo4j_store import Neo4jStore


class _StubStore(Neo4jStore):
    def __init__(self):
        super().__init__({})
        self.calls = []
        self._driver = None  # disable driver

    async def upsert_graph_v0(self, **kwargs):
        self.calls.append(kwargs)


def test_ttl_defaults_applied():
    store = _StubStore()
    svc = GraphService(store, gating={"ttl_default_days": 1})
    evt = Event(id="ev1", tenant_id="t", summary="hello")
    req = GraphUpsertRequest(events=[evt], edges=[GraphEdge(src_id="ev1", dst_id="ev1", rel_type="SELF", tenant_id="t")])

    asyncio.run(svc.upsert(req))

    assert len(store.calls) == 1
    payload = store.calls[0]
    ev = payload["events"][0]
    assert ev.ttl is not None and ev.ttl > 0
    assert ev.created_at is not None
    assert ev.expires_at is not None
    assert isinstance(ev.expires_at, datetime)
    assert ev.expires_at.tzinfo is not None
