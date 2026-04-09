from __future__ import annotations

from modules.memory.scripts.backup_qdrant import build_scroll_payload
from modules.memory.scripts.export_neo4j_graph import build_queries


def test_backup_qdrant_build_scroll_payload():
    p = build_scroll_payload(batch=123)
    assert p["with_payload"] is True and p["limit"] == 123
    off = {"point_id": 42}
    p2 = build_scroll_payload(offset=off, batch=10)
    assert p2["offset"] == off and p2["limit"] == 10


def test_export_neo4j_build_queries():
    qn, qr = build_queries()
    assert "MATCH (n:Entity)" in qn and "labels(n)" in qn
    assert "MATCH (a:Entity)-[r]->(b:Entity)" in qr and "type(r)" in qr

