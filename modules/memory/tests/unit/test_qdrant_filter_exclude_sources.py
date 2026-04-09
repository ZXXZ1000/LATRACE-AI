from __future__ import annotations

from modules.memory.infra.qdrant_store import QdrantStore


def test_qdrant_build_filter_supports_must_not_exclude_sources() -> None:
    qs = QdrantStore({"host": "127.0.0.1", "port": 6333})
    flt = qs._build_filter({  # type: ignore[attr-defined]
        "__exclude_sources": ["tkg_dialog_utterance_index_v1", "tkg_dialog_event_index_v1"],
        "tenant_id": "t",
        "memory_domain": "dialog",
    })
    assert flt is not None
    must = flt.get("must", [])
    must_not = flt.get("must_not", [])
    assert {"key": "metadata.tenant_id", "match": {"value": "t"}} in must
    assert {"key": "metadata.memory_domain", "match": {"value": "dialog"}} in must
    assert {"key": "metadata.source", "match": {"value": "tkg_dialog_utterance_index_v1"}} in must_not
    assert {"key": "metadata.source", "match": {"value": "tkg_dialog_event_index_v1"}} in must_not
