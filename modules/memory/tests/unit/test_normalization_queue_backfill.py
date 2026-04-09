from __future__ import annotations

from modules.memory.scripts import normalization_queue_backfill as backfill


def test_build_update_fields_skips_uncategorized() -> None:
    ev = {"topic_path": "_uncategorized/general", "tags": ["travel"], "keywords": ["日本"]}
    out = backfill._build_update_fields(ev)
    assert out == {}


def test_build_update_fields_includes_valid_fields() -> None:
    ev = {
        "topic_path": "travel/japan",
        "tags": ["travel", "japan"],
        "keywords": ["日本"],
        "time_bucket": ["2026", "2026-01"],
        "tags_vocab_version": "v1",
    }
    out = backfill._build_update_fields(ev)
    assert out["topic_path"] == "travel/japan"
    assert out["tags"] == ["travel", "japan"]
    assert out["keywords"] == ["日本"]
    assert out["time_bucket"] == ["2026", "2026-01"]
    assert out["tags_vocab_version"] == "v1"
