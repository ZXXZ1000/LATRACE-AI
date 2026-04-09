from __future__ import annotations

from modules.memory.infra.qdrant_store import QdrantStore


def test_qdrant_filter_user_any_domain_run():
    qs = QdrantStore({"host": "127.0.0.1", "port": 6333})
    flt = qs._build_filter({  # type: ignore[attr-defined]
        "user_id": ["alice", "bob"],
        "user_match": "any",
        "memory_domain": "work",
        "run_id": "r1",
    })
    assert flt is not None
    must = flt.get("must", [])
    should = flt.get("should", [])
    assert {"key": "metadata.memory_domain", "match": {"value": "work"}} in must
    assert {"key": "metadata.run_id", "match": {"value": "r1"}} in must
    # any 语义：两个 user_id 都应该在 should（OR）
    assert {"key": "metadata.user_id", "match": {"value": "alice"}} in should
    assert {"key": "metadata.user_id", "match": {"value": "bob"}} in should
    assert "minimum_should_match" not in flt


def test_qdrant_filter_user_all():
    qs = QdrantStore({"host": "127.0.0.1", "port": 6333})
    flt = qs._build_filter({  # type: ignore[attr-defined]
        "user_id": ["alice", "bob"],
        "user_match": "all",
    })
    assert flt is not None
    must = flt.get("must", [])
    assert {"key": "metadata.user_id", "match": {"value": "alice"}} in must
    assert {"key": "metadata.user_id", "match": {"value": "bob"}} in must
    assert "should" not in flt or not flt.get("should")
