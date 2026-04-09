from __future__ import annotations

from modules.memory.infra.qdrant_store import QdrantStore


def test_build_filter_includes_memory_scope():
    store = QdrantStore(settings={"host": "127.0.0.1", "port": 6333})
    f = store._build_filter({"memory_scope": "vh::abcd1234"})  # type: ignore[attr-defined]
    assert isinstance(f, dict)
    must = f.get("must") or []
    # flatten must entries and look for metadata.memory_scope match
    found = any(d.get("key") == "metadata.memory_scope" for d in must if isinstance(d, dict))
    assert found, f"memory_scope filter not present: {f}"

