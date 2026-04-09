from __future__ import annotations

from modules.memory.scripts.phase5.metrics import (
    absolute_time_error_days,
    order_consistency,
    precision_at_k,
    speaker_precision,
)


def test_precision_at_k() -> None:
    pred = ["a", "b", "c"]
    gt = ["b", "d"]
    assert precision_at_k(pred, gt) == 1 / 3


def test_order_consistency() -> None:
    pred = ["e1", "e2", "e3"]
    exp = ["e1", "e3"]
    assert order_consistency(pred, exp) == 1.0
    assert order_consistency(["e2", "e1"], ["e1", "e2"]) == 0.0


def test_speaker_precision() -> None:
    pred = [{"utterance_id": "u1", "speaker_id": "s1"}, {"utterance_id": "u2", "speaker_id": "s2"}]
    gt = [{"utterance_id": "u1", "speaker_id": "s1"}, {"utterance_id": "u2", "speaker_id": "s3"}]
    assert speaker_precision(pred, gt) == 0.5


def test_absolute_time_error_days() -> None:
    err = absolute_time_error_days("2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z")
    assert round(err, 2) == 1.0
