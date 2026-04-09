from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.memory.application.qa_dialog_v1 import QA_SYSTEM_PROMPT_GENERAL, build_qa_user_prompt

_BENCHMARK_ROOT = Path(__file__).resolve().parents[4] / "benchmark"
pytestmark = pytest.mark.skipif(
    not _BENCHMARK_ROOT.exists(),
    reason="benchmark directory not present, skipping alignment tests",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_benchmark_file(*relative_paths: str) -> str:
    root = _repo_root()
    for rel in relative_paths:
        path = root / rel
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"benchmark prompt source not found in any of: {relative_paths}")


def _extract_benchmark_prompt() -> str:
    text = _read_benchmark_file(
        "benchmark/shared/adapters/moyan_memory_qa_adapter.py",
        "benchmark/adapters/moyan_memory_qa_adapter.py",
    )
    m = re.search(r'QA_SYSTEM_PROMPT_GENERAL\s*=\s*\"\"\"(.*?)\"\"\"', text, re.S)
    assert m, "QA_SYSTEM_PROMPT_GENERAL not found in benchmark/shared/adapters/moyan_memory_qa_adapter.py"
    return m.group(1)


def _norm(s: str) -> str:
    return "\n".join([ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]).strip()


def test_dialog_qa_system_prompt_matches_benchmark() -> None:
    bench = _extract_benchmark_prompt()
    assert _norm(QA_SYSTEM_PROMPT_GENERAL) == _norm(bench)


def test_build_qa_user_prompt_matches_benchmark_format() -> None:
    prompt = build_qa_user_prompt(
        query="Q1",
        task="GENERAL",
        evidence=[
            {"event_id": "e1", "text": "t1", "source": "fact_search", "timestamp": None},
            {"event_id": "e2", "text": "t2", "source": "reference_trace"},
            {"event_id": "e3", "text": "t3", "source": "event_search"},
        ],
    )
    expected = (
        "Question: Q1\n"
        "Task type: GENERAL\n\n"
        "Evidence from memory (Fact=summarized memory, Event=original dialogue):\n"
        "[1] (Fact) id=e1, ts=None\n"
        "    t1\n"
        "[2] (Reference) id=e2, ts=None\n"
        "    t2\n"
        "[3] (Event) id=e3, ts=None\n"
        "    t3\n\n"
        "Based on the evidence above, provide the best answer. Focus on facts and details mentioned."
    )
    assert prompt == expected


def test_build_qa_user_prompt_caps_evidence_to_30() -> None:
    evidence = [{"event_id": f"e{i}", "text": f"t{i}", "source": "event_search", "timestamp": None} for i in range(1, 40)]
    prompt = build_qa_user_prompt(query="Q", task="GENERAL", evidence=evidence)
    assert "[30]" in prompt
    assert "id=e30" in prompt
    assert "[31]" not in prompt
