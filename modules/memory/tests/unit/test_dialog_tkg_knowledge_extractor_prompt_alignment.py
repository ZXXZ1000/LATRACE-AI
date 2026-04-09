from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.memory.application.knowledge_extractor_dialog_tkg_v1 import SYSTEM_PROMPT

_BENCHMARK_ROOT = Path(__file__).resolve().parents[4] / "benchmark"
pytestmark = pytest.mark.skipif(
    not _BENCHMARK_ROOT.exists(),
    reason="benchmark directory not present, skipping alignment tests",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_benchmark_prompt() -> str:
    candidates = (
        _repo_root() / "benchmark" / "archive" / "v2" / "prompts" / "tkg_knowledge_extractor_system_prompt_v1.txt",
        _repo_root() / "benchmark" / "v2" / "prompts" / "tkg_knowledge_extractor_system_prompt_v1.txt",
    )
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"benchmark prompt source not found in any of: {candidates}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def test_dialog_tkg_knowledge_extractor_prompt_matches_benchmark_v2() -> None:
    bench = _read_benchmark_prompt()
    assert _norm(SYSTEM_PROMPT) == _norm(bench)
