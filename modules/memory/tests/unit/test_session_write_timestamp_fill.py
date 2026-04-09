from __future__ import annotations

from datetime import datetime

from modules.memory.session_write import _fill_turn_timestamps


def test_fill_turn_timestamps_from_first_turn() -> None:
    turns = [
        {"text": "a", "timestamp_iso": "2026-01-01T00:00:00+00:00"},
        {"text": "b", "timestamp_iso": None},
        {"text": "c"},
    ]
    out, _ = _fill_turn_timestamps(turns, reference_time_iso=None, turn_interval_seconds=60)
    assert out[0]["timestamp_iso"] == "2026-01-01T00:00:00+00:00"
    assert out[1]["timestamp_iso"] == "2026-01-01T00:01:00+00:00"
    assert out[2]["timestamp_iso"] == "2026-01-01T00:02:00+00:00"


def test_fill_turn_timestamps_when_missing_all() -> None:
    turns = [{"text": "a"}, {"text": "b"}]
    out, ref = _fill_turn_timestamps(turns, reference_time_iso=None, turn_interval_seconds=30)
    assert ref is not None
    # timestamps should be parseable
    t0 = datetime.fromisoformat(out[0]["timestamp_iso"].replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(out[1]["timestamp_iso"].replace("Z", "+00:00"))
    assert t0.tzinfo is not None
    assert t1.tzinfo is not None
    assert (t1 - t0).total_seconds() == 30
