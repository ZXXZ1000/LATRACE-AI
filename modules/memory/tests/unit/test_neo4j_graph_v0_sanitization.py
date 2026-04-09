from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class _Sess:
    def __init__(self, calls: List[Dict[str, Any]]) -> None:
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        return []


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]]) -> None:
        self.calls = calls

    def session(self, *args, **kwargs):
        return _Sess(self.calls)


def _has_nested_maps(obj: Any) -> bool:
    if isinstance(obj, dict):
        return True
    if isinstance(obj, list):
        return any(_has_nested_maps(x) for x in obj)
    if isinstance(obj, tuple):
        return any(_has_nested_maps(x) for x in obj)
    return False


def test_upsert_graph_v0_sanitizes_nested_properties():
    from modules.memory.infra.neo4j_store import Neo4jStore
    from modules.memory.contracts.graph_models import Evidence, GraphEdge, MediaSegment, Provenance

    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    seg = MediaSegment(
        id="seg-1",
        tenant_id="t1",
        source_id="demo.mp4",
        t_media_start=0.0,
        t_media_end=1.0,
    )
    ev = Evidence(
        id="ev-1",
        tenant_id="t1",
        source_id="demo.mp4",
        algorithm="face_cluster",
        algorithm_version="v1",
        confidence=0.9,
        extras={"cluster_id": "face_0", "frame_id": 1, "image_ref": ".artifacts/x.jpg"},
        provenance=Provenance(source="mema", model_version="x"),
    )
    edge = GraphEdge(
        src_id="seg-1",
        dst_id="ev-1",
        rel_type="CONTAINS_EVIDENCE",
        tenant_id="t1",
        provenance=Provenance(source="mema", model_version="x"),
    )

    asyncio.run(
        store.upsert_graph_v0(
            segments=[seg],
            evidences=[ev],
            utterances=[],
            entities=[],
            events=[],
            places=[],
            time_slices=[],
            regions=[],
            states=[],
            knowledge=[],
            pending_equivs=[],
            edges=[edge],
        )
    )

    # Ensure evidence payloads are JSON-encoded (no direct dict/map properties).
    ev_calls = [c for c in calls if "UNWIND $evidences AS e" in c["cypher"]]
    assert ev_calls, "expected Evidence upsert cypher call"
    payload = ev_calls[0]["params"]["evidences"][0]
    assert "extras" not in payload
    assert "extras_json" in payload and isinstance(payload["extras_json"], str)
    assert "provenance" not in payload
    assert "provenance_json" in payload and isinstance(payload["provenance_json"], str)

    # Ensure relationship payloads are JSON-encoded (no direct dict/map properties).
    edge_calls = [c for c in calls if "UNWIND $edges AS e" in c["cypher"]]
    assert edge_calls, "expected Edge upsert cypher call"
    edge_payload = edge_calls[0]["params"]["edges"][0]
    assert "provenance" not in edge_payload
    assert "provenance_json" in edge_payload and isinstance(edge_payload["provenance_json"], str)

    # Hard guard: no map/list-of-map passed as Neo4j property values.
    for c in calls:
        for key, val in (c.get("params") or {}).items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                for item in val:
                    for v in item.values():
                        assert not _has_nested_maps(v), f"nested map leaked into params for key={key}"

