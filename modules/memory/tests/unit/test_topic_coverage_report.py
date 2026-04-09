from __future__ import annotations

import json
import sys

from modules.memory.scripts import topic_coverage_report as report


def test_topic_coverage_report_output(tmp_path, capsys, monkeypatch) -> None:
    data = [
        {"topic_path": "travel/japan", "tags": ["travel"], "keywords": ["日本"], "time_bucket": ["2026"]},
        {"topic_path": "_uncategorized/general", "tags": [], "keywords": [], "time_bucket": []},
    ]
    fp = tmp_path / "events.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(fp)])
    rc = report.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["total_events"] == 2
    assert parsed["topic_path_coverage"] == 1.0
    assert parsed["uncategorized_ratio"] == 0.5
    assert parsed["tags_coverage"] == 0.5
    assert parsed["keywords_coverage"] == 0.5
    assert parsed["time_bucket_coverage"] == 0.5


def test_topic_coverage_report_event_only(tmp_path, capsys, monkeypatch) -> None:
    data = [
        {
            "topic_path": "travel/japan",
            "tags": ["travel"],
            "keywords": [],
            "time_bucket": ["2026"],
            "event_id": "evt_1",
        },
        {
            "topic_path": "lifestyle/food",
            "tags": ["food"],
            "keywords": ["美食"],
            "time_bucket": ["2026"],
            "node_type": "Chunk",
        },
    ]
    fp = tmp_path / "events.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(fp), "--event-only"])
    rc = report.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["total_events"] == 1
    assert parsed["topic_path_coverage"] == 1.0
    assert parsed["uncategorized_ratio"] == 0.0


def test_topic_coverage_report_tenant_filter(tmp_path, capsys, monkeypatch) -> None:
    data = [
        {"topic_path": "travel/japan", "tenant_id": "t1", "event_id": "evt_1"},
        {"topic_path": "work/project", "tenant_id": "t2", "event_id": "evt_2"},
    ]
    fp = tmp_path / "events.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(fp), "--event-only", "--tenant", "t1"])
    rc = report.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["total_events"] == 1
    assert parsed["topic_path_coverage"] == 1.0


def test_topic_coverage_report_event_index_only(tmp_path, capsys, monkeypatch) -> None:
    data = [
        {
            "topic_path": "travel/japan",
            "source": "tkg_dialog_event_index_v1",
            "event_id": "evt_1",
        },
        {
            "topic_path": "work/project",
            "source": "dialog_session_write_v1",
            "event_id": "evt_2",
        },
    ]
    fp = tmp_path / "events.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(fp), "--event-index-only"])
    rc = report.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["total_events"] == 1
    assert parsed["topic_path_coverage"] == 1.0
