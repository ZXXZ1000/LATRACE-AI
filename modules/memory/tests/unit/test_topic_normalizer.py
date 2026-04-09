from __future__ import annotations

from modules.memory.application.topic_normalizer import TopicNormalizer
from modules.memory.application.topic_normalizer import (
    normalize_events,
    normalize_topic_text,
    TopicNormalization,
    get_topic_registry,
)


def test_normalizer_rule_match() -> None:
    norm = TopicNormalizer()
    ev = {
        "summary": "计划日本旅行，准备机票和酒店",
        "keywords": ["日本", "旅行", "机票"],
    }
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "travel/japan"
    assert "travel" in (out.get("tags") or [])


def test_normalizer_synonym_match() -> None:
    norm = TopicNormalizer()
    ev = {
        "summary": "最近在学编程",
        "topic_id": "学编程",
    }
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "learning/programming"


def test_normalizer_fallback_uncategorized() -> None:
    norm = TopicNormalizer()
    ev = {
        "summary": "随便聊聊今天的天气",
        "keywords": ["随便聊"],
    }
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "_uncategorized/high_entropy"


def test_normalizer_priority_conflict() -> None:
    norm = TopicNormalizer()
    ev = {"summary": "最近很焦虑", "keywords": ["焦虑"]}
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "health/mental/anxiety"


def test_normalizer_existing_topic_path_passthrough() -> None:
    norm = TopicNormalizer()
    ev = {"summary": "日本旅行计划", "topic_path": "travel/japan", "keywords": ["日本", "旅行"]}
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "travel/japan"


def test_normalizer_uncategorized_rewrite() -> None:
    norm = TopicNormalizer()
    ev = {"summary": "参加舞蹈比赛", "topic_id": "dance competition", "topic_path": "_uncategorized/general"}
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "entertainment/dance"


def test_normalizer_derives_keywords() -> None:
    norm = TopicNormalizer()
    ev = {"summary": "日本旅行计划，准备机票和酒店"}
    out = norm.normalize_event(ev)
    kws = out.get("keywords") or []
    assert isinstance(kws, list)
    assert len(kws) > 0


def test_normalizer_empty_event_defaults() -> None:
    norm = TopicNormalizer()
    ev = {}
    out = norm.normalize_event(ev)
    assert out.get("topic_path") == "_uncategorized/general"


def test_normalize_events_async_queue(tmp_path, monkeypatch) -> None:
    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.setenv("MEMORY_TOPIC_NORMALIZATION_MODE", "async")
    monkeypatch.setenv("MEMORY_TOPIC_NORMALIZATION_QUEUE_PATH", str(queue_path))
    events = [{"id": "evt-1", "summary": "随便聊聊今天的天气", "keywords": ["随便聊"]}]
    out = normalize_events(events)
    assert out and out[0].get("topic_path") == "_uncategorized/high_entropy"
    assert queue_path.exists()
    lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_normalize_topic_text_returns_structured_payload() -> None:
    normalize_topic_text.cache_clear()  # type: ignore[attr-defined]
    out = normalize_topic_text("日本旅行计划，准备机票和酒店")
    assert isinstance(out, TopicNormalization)
    assert out.topic_id
    assert out.topic_path
    assert isinstance(out.tags, tuple)
    assert isinstance(out.keywords, tuple)


def test_normalize_topic_text_uses_lru_cache() -> None:
    normalize_topic_text.cache_clear()  # type: ignore[attr-defined]
    _ = normalize_topic_text("最近在学编程")
    info1 = normalize_topic_text.cache_info()  # type: ignore[attr-defined]
    _ = normalize_topic_text("最近在学编程")
    info2 = normalize_topic_text.cache_info()  # type: ignore[attr-defined]
    assert info2.hits == info1.hits + 1


def test_topic_registry_deterministic_id_merges_case_variants(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_TOPIC_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_TOPIC_REGISTRY_OVERRIDE_TOPIC_ID", "true")
    normalize_topic_text.cache_clear()  # type: ignore[attr-defined]
    get_topic_registry().clear()

    out = normalize_events(
        [
            {"summary": "讨论范围", "topic_id": "Project Alpha", "keywords": ["alpha"]},
            {"summary": "讨论范围", "topic_id": "project   alpha", "keywords": ["Alpha"]},
        ]
    )
    assert len(out) == 2
    assert out[0].get("topic_id") == out[1].get("topic_id")
    assert str(out[0].get("topic_id") or "").startswith("tpk_")
    assert out[0].get("topic_id_raw") == "Project Alpha"
    assert out[1].get("topic_id_raw") == "project   alpha"
    assert out[0].get("topic_registry_key") == out[1].get("topic_registry_key")


def test_topic_registry_query_write_alignment(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_TOPIC_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_TOPIC_REGISTRY_OVERRIDE_TOPIC_ID", "true")
    normalize_topic_text.cache_clear()  # type: ignore[attr-defined]
    get_topic_registry().clear()

    write_ev = normalize_events([{"summary": "日本旅行计划", "topic_id": "日本旅行"}])[0]
    query_norm = normalize_topic_text("日本旅行")
    assert write_ev.get("topic_id") == query_norm.topic_id
