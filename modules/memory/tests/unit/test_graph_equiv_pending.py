from modules.memory.contracts.graph_models import PendingEquiv, GraphUpsertRequest
from modules.memory.infra.equiv_store import EquivStore
from modules.memory.infra.neo4j_store import Neo4jStore


def test_pending_equiv_defaults_and_registry_status():
    pe = PendingEquiv(
        id="peq-1",
        tenant_id="t",
        entity_id="ent-a",
        candidate_id="ent-b",
        confidence=0.42,
    )
    assert pe.status == "pending"
    assert pe.entity_id == "ent-a"
    assert pe.candidate_id == "ent-b"
    assert pe.confidence == 0.42


def test_graph_upsert_accepts_pending_equivs():
    req = GraphUpsertRequest(
        pending_equivs=[
            PendingEquiv(id="peq-1", tenant_id="t", entity_id="ent-a", candidate_id="ent-b")
        ]
    )
    assert len(req.pending_equivs) == 1
    assert req.pending_equivs[0].status == "pending"


def test_equiv_store_approve_and_reject(monkeypatch):
    calls = []

    class _Sess:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, cypher: str, **params):
            calls.append({"cypher": cypher, "params": params})
            # return a single-like object for .single() when needed
            class _Row:
                def __init__(self, ok=True):
                    self._ok = ok

                def single(self):
                    return {"p": {"id": "peq-1"}} if self._ok else None

                def __iter__(self):
                    return iter([])

                def __next__(self):
                    raise StopIteration

                def __getitem__(self, item):
                    return {"p": {"id": "peq-1"}}

            return _Row(ok=True)

    class _Drv:
        def session(self, *args, **kwargs):
            return _Sess()

    neo = Neo4jStore({})
    neo._driver = _Drv()  # type: ignore[attr-defined]
    store = EquivStore(neo)

    res_list = store.list_pending(tenant_id="t")
    assert isinstance(res_list, list)

    store.upsert_pending(tenant_id="t", records=[PendingEquiv(id="peq-1", tenant_id="t", entity_id="a", candidate_id="b")])

    res_approve = store.approve(tenant_id="t", pending_id="peq-1", reviewer="r1")
    assert res_approve.get("merged", 0) == 1

    res_reject = store.reject(tenant_id="t", pending_id="peq-2", reviewer="r1")
    assert isinstance(res_reject, dict)
    assert "updated" in res_reject
