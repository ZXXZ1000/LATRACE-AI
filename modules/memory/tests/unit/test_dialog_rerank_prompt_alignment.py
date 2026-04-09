from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.memory.application.rerank_dialog_v1 import RERANK_PROMPT

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
        "benchmark/shared/adapters/rerank_service.py",
        "benchmark/adapters/rerank_service.py",
    )
    m = re.search(r'RERANK_PROMPT\s*=\s*\"\"\"(.*?)\"\"\"', text, re.S)
    assert m, "RERANK_PROMPT not found in benchmark/shared/adapters/rerank_service.py"
    return m.group(1)


def _norm(s: str) -> str:
    return "\n".join([ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]).strip()


def test_dialog_rerank_prompt_matches_benchmark() -> None:
    bench = _extract_benchmark_prompt()
    assert _norm(RERANK_PROMPT) == _norm(bench)
