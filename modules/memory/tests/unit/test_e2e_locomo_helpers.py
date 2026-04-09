from __future__ import annotations

from pathlib import Path


def test_e2e_locomo_resolve_queries_path_file(tmp_path: Path) -> None:
    from modules.memory.scripts.e2e_dialog_conv26_session_write_and_retrieval import _resolve_queries_path

    q = tmp_path / "queries.jsonl"
    q.write_text("{}", encoding="utf-8")
    assert _resolve_queries_path(str(q), "conv-26") == q


def test_e2e_locomo_resolve_queries_path_dir(tmp_path: Path) -> None:
    from modules.memory.scripts.e2e_dialog_conv26_session_write_and_retrieval import _resolve_queries_path

    d = tmp_path / "step1_events"
    (d / "conv-26").mkdir(parents=True)
    q = d / "conv-26" / "queries.jsonl"
    q.write_text("{}", encoding="utf-8")
    assert _resolve_queries_path(str(d), "conv-26") == q


def test_e2e_locomo_discover_sample_ids(tmp_path: Path) -> None:
    from modules.memory.scripts.e2e_dialog_conv26_session_write_and_retrieval import _discover_sample_ids

    root = tmp_path / "step1_events"
    (root / "conv-26").mkdir(parents=True)
    (root / "conv-30").mkdir(parents=True)
    (root / "conv-26" / "queries.jsonl").write_text("{}", encoding="utf-8")
    (root / "conv-30" / "queries.jsonl").write_text("{}", encoding="utf-8")

    assert _discover_sample_ids(str(root)) == ["conv-26", "conv-30"]

