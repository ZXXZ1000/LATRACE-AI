from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

# Skip the entire module when the benchmark directory is not present.
# The benchmark scripts are not part of the public repository; these alignment
# tests are only meaningful for contributors who have the full monorepo layout.
if not (ROOT / "benchmark" / "scripts" / "step3_build_graph.py").exists():
    pytest.skip("benchmark directory not present, skipping alignment tests", allow_module_level=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench_step3 = _load_module(
    "benchmark_step3_build_graph",
    ROOT / "benchmark" / "scripts" / "step3_build_graph.py",
)

from modules.memory.domain.dialog_text_pipeline_v1 import (
    build_entries_and_links,
    event_record_to_entry,
    fact_item_to_entry,
    generate_uuid,
    parse_datetime,
    timeslice_record_to_entry,
)


def test_dialog_v1_uuid_generation_matches_benchmark() -> None:
    assert generate_uuid("locomo.events", "conv-26_D1_3") == bench_step3._generate_uuid("locomo.events", "conv-26_D1_3")
    assert generate_uuid("locomo.timeslices", "conv-26_session_1_ts") == bench_step3._generate_uuid("locomo.timeslices", "conv-26_session_1_ts")
    assert generate_uuid("locomo.facts", "fact:conv-26:0") == bench_step3._generate_uuid("locomo.facts", "fact:conv-26:0")


def test_dialog_v1_parse_datetime_parses_locomo_datetime() -> None:
    _ts, iso = parse_datetime("1:14 pm on 25 May, 2023")
    assert iso.startswith("2023-05-25T13:14")


def test_dialog_v1_event_conversion_matches_benchmark_step3() -> None:
    event = {
        "kind": "event",
        "id": "conv-26_D1_1",
        "tenant_id": "locomo_bench",
        "user_id": "locomo_user_conv-26",
        "text": "D1:1 Alice: hello",
        "timestamp": 123.0,
        "timestamp_iso": "2023-05-08T14:00:00",
        "timeslice": "conv-26_session_1_ts",
        "prev_event": None,
        "participants": ["Alice"],
        "metadata": {"dia_id": "D1:1", "session": 1, "turn": 1, "speaker": "Alice", "sample_id": "conv-26"},
    }

    entry, event_uuid = event_record_to_entry(event, tenant_id="locomo_bench")
    expected, expected_uuid = bench_step3._convert_event_to_entry(event, "locomo_bench")
    expected.pop("tenant_id", None)
    expected["metadata"]["tenant_id"] = "locomo_bench"

    assert event_uuid == expected_uuid == entry.id
    assert entry.model_dump(exclude_none=True) == expected


def test_dialog_v1_timeslice_conversion_matches_benchmark_step3() -> None:
    ts = {
        "kind": "timeslice",
        "id": "conv-26_session_1_ts",
        "tenant_id": "locomo_bench",
        "user_id": "locomo_user_conv-26",
        "label": "conv-26 Session 1",
        "start_at": 123.0,
        "end_at": 456.0,
        "start_iso": "2023-05-08T14:00:00",
        "end_iso": "2023-05-08T15:00:00",
    }

    entry, ts_uuid = timeslice_record_to_entry(ts, tenant_id="locomo_bench")
    expected, expected_uuid = bench_step3._convert_timeslice_to_entry(ts, "locomo_bench")
    expected.pop("tenant_id", None)
    expected["metadata"]["tenant_id"] = "locomo_bench"

    assert ts_uuid == expected_uuid == entry.id
    assert entry.model_dump(exclude_none=True) == expected


def test_dialog_v1_fact_conversion_matches_benchmark_step3() -> None:
    fact = {
        "op": "ADD",
        "type": "fact",
        "statement": "Alice attended an LGBTQ support group on May 7, 2023.",
        "speaker": "Alice",
        "importance": "high",
        "status": "done",
        "scope": "permanent",
        "source_sample_id": "conv-26",
        "source_turn_ids": ["D1:1"],
        "mentions": ["LGBTQ support group"],
        "temporal_grounding": [{"text": "yesterday", "reference_time": "2023-05-08", "estimated": "2023-05-07", "confidence": "high"}],
    }

    entry, fact_uuid = fact_item_to_entry(fact, fact_idx=0, tenant_id="locomo_bench", user_prefix="locomo_user_")
    assert entry is not None and fact_uuid is not None

    expected, expected_uuid = bench_step3._convert_fact_to_entry(fact, 0, "locomo_bench", "locomo_user_")
    expected.pop("tenant_id", None)
    expected["metadata"]["tenant_id"] = "locomo_bench"

    assert fact_uuid == expected_uuid == entry.id
    assert entry.model_dump(exclude_none=True) == expected


def test_dialog_v1_build_entries_and_links_counts_and_edge_types() -> None:
    async def _run() -> None:
        events_raw = [
            {
                "kind": "timeslice",
                "id": "conv-26_session_1_ts",
                "tenant_id": "locomo_bench",
                "user_id": "locomo_user_conv-26",
                "label": "conv-26 Session 1",
                "start_iso": "2023-05-08T14:00:00",
                "end_iso": "2023-05-08T15:00:00",
            },
            {
                "kind": "event",
                "id": "conv-26_D1_1",
                "tenant_id": "locomo_bench",
                "user_id": "locomo_user_conv-26",
                "text": "D1:1 Alice: hello",
                "timestamp_iso": "2023-05-08T14:00:00",
                "timeslice": "conv-26_session_1_ts",
                "metadata": {"dia_id": "D1:1", "session": 1, "turn": 1, "speaker": "Alice", "sample_id": "conv-26"},
            },
            {
                "kind": "event",
                "id": "conv-26_D1_2",
                "tenant_id": "locomo_bench",
                "user_id": "locomo_user_conv-26",
                "text": "D1:2 Bob: hi",
                "timestamp_iso": "2023-05-08T14:01:00",
                "timeslice": "conv-26_session_1_ts",
                "metadata": {"dia_id": "D1:2", "session": 1, "turn": 2, "speaker": "Bob", "sample_id": "conv-26"},
            },
        ]

        facts_raw = [
            {
                "op": "ADD",
                "type": "fact",
                "statement": "Alice said hello on May 8, 2023.",
                "importance": "low",
                "status": "done",
                "scope": "permanent",
                "source_sample_id": "conv-26",
                "source_turn_ids": ["D1:1", "D1:2"],
            }
        ]

        entries, links = build_entries_and_links(events_raw=events_raw, facts_raw=facts_raw, tenant_id="locomo_bench")

        assert len(entries) == 4  # 1 timeslice + 2 events + 1 fact
        rels = sorted([e.rel_type for e in links])
        assert rels.count("OCCURS_AT") == 2
        assert rels.count("REFERENCES") == 2
        assert rels.count("PART_OF") == 1

    asyncio.run(_run())
