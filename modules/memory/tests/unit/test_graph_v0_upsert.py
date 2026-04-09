from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.application.graph_service import GraphService, GraphValidationError
from modules.memory.contracts.graph_models import (
    MediaSegment,
    Evidence,
    UtteranceEvidence,
    Entity,
    Event,
    SpatioTemporalRegion,
    State as GraphState,
    Knowledge as GraphKnowledge,
    PendingEquiv,
    GraphEdge,
    GraphUpsertRequest,
    TimeSlice,
)


class _Tx:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        return []


class _Sess:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        return []


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def session(self, *args, **kwargs):
        return _Sess(self.calls)


def test_graph_service_rejects_mixed_tenant():
    svc = GraphService(Neo4jStore({}))
    req = GraphUpsertRequest(
        segments=[MediaSegment(id="s1", tenant_id="a", source_id="src", t_media_start=0.0, t_media_end=1.0)],
        evidences=[Evidence(id="e1", tenant_id="b", source_id="src", algorithm="a", algorithm_version="1", confidence=0.9)],
    )
    try:
        asyncio.run(svc.upsert(req))
    except GraphValidationError:
        return
    assert False, "expected GraphValidationError for mixed tenant_id"


def test_graph_service_rejects_mixed_user_scope():
    svc = GraphService(Neo4jStore({}))
    req = GraphUpsertRequest(
        segments=[
            MediaSegment(
                id="s1",
                tenant_id="t",
                source_id="src",
                t_media_start=0.0,
                t_media_end=1.0,
                user_id=["u1"],
                memory_domain="d",
            )
        ],
        entities=[Entity(id="en1", tenant_id="t", type="PERSON", user_id=["u2"], memory_domain="d")],
    )
    try:
        asyncio.run(svc.upsert(req))
    except GraphValidationError:
        return
    assert False, "expected GraphValidationError for mixed user_id scope"


def test_graph_service_rejects_mixed_domain_scope():
    svc = GraphService(Neo4jStore({}))
    req = GraphUpsertRequest(
        segments=[
            MediaSegment(
                id="s1",
                tenant_id="t",
                source_id="src",
                t_media_start=0.0,
                t_media_end=1.0,
                user_id=["u1"],
                memory_domain="d1",
            )
        ],
        entities=[Entity(id="en1", tenant_id="t", type="PERSON", user_id=["u1"], memory_domain="d2")],
    )
    try:
        asyncio.run(svc.upsert(req))
    except GraphValidationError:
        return
    assert False, "expected GraphValidationError for mixed memory_domain scope"


def test_upsert_graph_v0_merges_nodes_and_edges():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    seg = MediaSegment(
        id="s1",
        tenant_id="t",
        source_id="src",
        t_media_start=0.0,
        t_media_end=1.0,
        has_physical_time=False,
        time_origin="media",
    )
    ent = Entity(id="en1", tenant_id="t", type="PERSON")
    ev = Evidence(id="ev1", tenant_id="t", source_id="src", algorithm="algo", algorithm_version="1", confidence=0.8)
    edge = GraphEdge(src_id="s1", dst_id="ev1", rel_type="CONTAINS_EVIDENCE", tenant_id="t")

    asyncio.run(
        store.upsert_graph_v0(
            segments=[seg],
            evidences=[ev],
            utterances=[],
            entities=[ent],
            events=[],
            places=[],
            time_slices=[],
            regions=[],
            states=[],
            knowledge=[],
            edges=[edge],
        )
    )

    # Expect MERGE statements for nodes and edges
    cyphers = "\n".join(c["cypher"] for c in calls)
    assert "MediaSegment" in cyphers
    assert "Evidence" in cyphers
    assert "CONTAINS_EVIDENCE" in cyphers


def test_upsert_graph_v0_with_timeslice_and_v02_edges():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    ts = GraphUpsertRequest(
        segments=[],
        evidences=[],
        utterances=[],
        entities=[],
        events=[],
        places=[],
        time_slices=[
            TimeSlice(
                id="ts1",
                tenant_id="t",
                kind="physical",
                t_abs_start=None,
                t_abs_end=None,
                t_media_start=0.0,
                t_media_end=10.0,
            )
        ],
        edges=[
            GraphEdge(
                src_id="ts1",
                dst_id="e1",
                rel_type="COVERS_EVENT",
                tenant_id="t",
                src_type="TimeSlice",
                dst_type="Event",
            ),
            GraphEdge(
                src_id="ev_a",
                dst_id="ev_b",
                rel_type="NEXT_EVENT",
                tenant_id="t",
            ),
            GraphEdge(
                src_id="ent_a",
                dst_id="ent_b",
                rel_type="CO_OCCURS_WITH",
                tenant_id="t",
            ),
            GraphEdge(
                src_id="ev_a",
                dst_id="ev_b",
                rel_type="CAUSES",
                tenant_id="t",
                status="candidate",
                layer="hypothesis",
            ),
        ],
        regions=[],
        states=[],
        knowledge=[],
    )

    asyncio.run(
        store.upsert_graph_v0(
            segments=[],
            evidences=[],
            utterances=[],
            entities=[],
            events=[],
            places=[],
            time_slices=ts.time_slices,
            regions=[],
            states=[],
            knowledge=[],
            edges=ts.edges,
        )
    )

    cyphers = "\n".join(c["cypher"] for c in calls)
    assert "TimeSlice" in cyphers
    assert "COVERS_EVENT" in cyphers
    assert "NEXT_EVENT" in cyphers
    assert "CO_OCCURS_WITH" in cyphers
    assert "CAUSES" in cyphers


def test_upsert_graph_v03_nodes_and_new_rels():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    req = GraphUpsertRequest(
        segments=[],
        evidences=[],
        utterances=[
            UtteranceEvidence(
                id="utt1",
                tenant_id="t",
                raw_text="hi",
                t_media_start=0.0,
                t_media_end=1.0,
            )
        ],
        entities=[Entity(id="ent1", tenant_id="t", type="PERSON")],
        events=[Event(id="ev1", tenant_id="t", summary="hello")],
        places=[],
        time_slices=[
            TimeSlice(
                id="ts1",
                tenant_id="t",
                kind="media_window",
                t_media_start=0.0,
                t_media_end=10.0,
            )
        ],
        regions=[SpatioTemporalRegion(id="reg1", tenant_id="t", region_type="room")],
        states=[GraphState(id="state1", tenant_id="t", subject_id="ent1", property="mood", value="happy")],
        knowledge=[GraphKnowledge(id="kn1", tenant_id="t", summary="fact")],
        edges=[
            GraphEdge(
                src_id="utt1",
                dst_id="ent1",
                rel_type="SPOKEN_BY",
                tenant_id="t",
                src_type="UtteranceEvidence",
                dst_type="Entity",
            ),
            GraphEdge(
                src_id="ts1",
                dst_id="ev1",
                rel_type="TEMPORALLY_CONTAINS",
                tenant_id="t",
            ),
            GraphEdge(
                src_id="ent1",
                dst_id="state1",
                rel_type="HAS_STATE",
                tenant_id="t",
                dst_type="State",
            ),
            GraphEdge(
                src_id="kn1",
                dst_id="ev1",
                rel_type="DERIVED_FROM",
                tenant_id="t",
                src_type="Knowledge",
            ),
        GraphEdge(
            src_id="ent1",
            dst_id="ent2",
            rel_type="EQUIV",
            tenant_id="t",
        ),
    ],
    pending_equivs=[
        PendingEquiv(
            id="peq1",
            tenant_id="t",
            entity_id="ent1",
            candidate_id="ent2",
            confidence=0.8,
        )
    ],
)

    asyncio.run(
        store.upsert_graph_v0(
            segments=req.segments,
            evidences=req.evidences,
            utterances=req.utterances,
            entities=req.entities,
            events=req.events,
            places=req.places,
            time_slices=req.time_slices,
            regions=req.regions,
            states=req.states,
            knowledge=req.knowledge,
            pending_equivs=req.pending_equivs,
            edges=req.edges,
        )
    )

    cyphers = "\n".join(c["cypher"] for c in calls)
    assert "UtteranceEvidence" in cyphers
    assert "SpatioTemporalRegion" in cyphers or "regions" in cyphers  # MERGE regions block present
    assert "State" in cyphers or "states" in cyphers
    assert "Knowledge" in cyphers or "knowledge" in cyphers
    assert "SPOKEN_BY" in cyphers
    assert "TEMPORALLY_CONTAINS" in cyphers
    assert "HAS_STATE" in cyphers
    assert "DERIVED_FROM" in cyphers
    assert "EQUIV" in cyphers
