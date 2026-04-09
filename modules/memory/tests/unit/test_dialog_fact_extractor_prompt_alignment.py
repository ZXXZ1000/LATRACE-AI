from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.memory.application.fact_extractor_dialog_v1 import SYSTEM_PROMPT, parse_facts_json

_BENCHMARK_ROOT = Path(__file__).resolve().parents[4] / "benchmark"
pytestmark = pytest.mark.skipif(
    not _BENCHMARK_ROOT.exists(),
    reason="benchmark directory not present, skipping alignment tests",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _extract_benchmark_system_prompt() -> str:
    text = (_repo_root() / "benchmark" / "scripts" / "step2_extract_facts.py").read_text(encoding="utf-8")
    m = re.search(r'SYSTEM_PROMPT\s*=\s*\"\"\"(.*?)\"\"\"', text, re.S)
    assert m, "SYSTEM_PROMPT not found in benchmark/scripts/step2_extract_facts.py"
    return m.group(1)


def _norm(s: str) -> str:
    # ignore trailing spaces and normalize line endings for robust equality
    return "\n".join([ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]).strip()


def test_dialog_v1_system_prompt_matches_benchmark_step2() -> None:
    bench = _extract_benchmark_system_prompt()
    assert _norm(SYSTEM_PROMPT) == _norm(bench)


def test_parse_facts_json_accepts_code_fence_and_returns_list() -> None:
    raw = """```json
{
  "facts": [
    {"op":"ADD","type":"fact","statement":"A","source_turn_ids":["D1:1"]}
  ]
}
```"""
    facts = parse_facts_json(raw)
    assert isinstance(facts, list)
    assert facts and facts[0]["statement"] == "A"
