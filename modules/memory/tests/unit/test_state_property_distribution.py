from __future__ import annotations

import json
import sys

from modules.memory.scripts import state_property_distribution as dist


def test_state_property_distribution_basic(tmp_path, capsys, monkeypatch) -> None:
    data = [
        {
            "payload": {
                "content": "I got a new job offer and feel excited.",
                "metadata": {"source": "tkg_dialog_event_index_v1"},
            }
        },
        {
            "payload": {
                "content": "最近压力很大，感觉有点崩溃。",
                "metadata": {"source": "tkg_dialog_event_index_v1"},
            }
        },
        {
            "payload": {
                "content": "我们去北京玩了几天。",
                "metadata": {"source": "other"},
            }
        },
    ]
    fp = tmp_path / "events.jsonl"
    fp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in data), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--input", str(fp), "--source", "tkg_dialog_event_index_v1"],
    )
    rc = dist.main()
    assert rc == 0
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    counts = parsed["counts"]
    assert counts["job_status"] >= 1
    assert counts["mood"] >= 1
