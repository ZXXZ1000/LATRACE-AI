from __future__ import annotations

from modules.memory.infra.qdrant_store import QdrantStore


def test_qdrant_filter_topic_tag_keyword_time_bucket() -> None:
    qs = QdrantStore({"host": "127.0.0.1", "port": 6333})
    flt = qs._build_filter({  # type: ignore[attr-defined]
        "topic_path": ["travel/japan", "travel/korea"],
        "tags": ["travel", "planning"],
        "keywords": ["日本", "机票"],
        "time_bucket": ["2026", "2026-01", "2026-01-29"],
        "tags_vocab_version": "v1",
    })
    assert flt is not None
    must = flt.get("must", [])
    should = flt.get("should", [])

    # tags_vocab_version should be a hard filter
    assert {"key": "metadata.tags_vocab_version", "match": {"value": "v1"}} in must

    # list filters should be required (match any)
    assert {"key": "metadata.topic_path", "match": {"any": ["travel/japan", "travel/korea"]}} in must
    assert {"key": "metadata.tags", "match": {"any": ["travel", "planning"]}} in must
    assert {"key": "metadata.keywords", "match": {"any": ["日本", "机票"]}} in must
    assert {"key": "metadata.time_bucket", "match": {"any": ["2026", "2026-01", "2026-01-29"]}} in must
    assert not should
