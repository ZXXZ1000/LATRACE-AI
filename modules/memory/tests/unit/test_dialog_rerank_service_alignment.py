from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

# Skip the entire module when the benchmark directory is not present.
# The benchmark scripts are not part of the public repository; these alignment
# tests are only meaningful for contributors who have the full monorepo layout.
if not (ROOT / "benchmark" / "shared" / "adapters" / "rerank_types.py").exists():
    pytest.skip("benchmark directory not present, skipping alignment tests", allow_module_level=True)


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("benchmark", ROOT / "benchmark")
_ensure_package("benchmark.shared", ROOT / "benchmark" / "shared")
_ensure_package("benchmark.shared.adapters", ROOT / "benchmark" / "shared" / "adapters")
bench_types = _load_module(
    "benchmark.shared.adapters.rerank_types",
    ROOT / "benchmark" / "shared" / "adapters" / "rerank_types.py",
)
bench_service = _load_module(
    "benchmark.shared.adapters.rerank_service",
    ROOT / "benchmark" / "shared" / "adapters" / "rerank_service.py",
)
BenchEvidenceType = bench_types.EvidenceType
BenchCandidate = bench_types.RetrievalCandidate
BenchConfig = bench_types.RerankConfig

from modules.memory.application.rerank_dialog_v1 import (
    EvidenceType,
    RetrievalCandidate,
    RerankConfig,
    build_llm_client_from_fn,
    create_rerank_service,
)


def test_dialog_rerank_service_matches_benchmark_scores_and_ranks() -> None:
    def _llm(system_prompt: str, user_prompt: str) -> str:
        # passage 2 should win
        return '{"1": 0.1, "2": 0.9, "3": 0.2}'

    # benchmark
    bench_llm = type("LLM", (), {"generate": staticmethod(_llm)})
    bench_cfg = BenchConfig(enabled=True, model="llm", top_n=2)
    bench_candidates = [
        BenchCandidate(query_text="q", evidence_text="a", evidence_type=BenchEvidenceType.FACT, event_id="e1", base_score=2.0),
        BenchCandidate(query_text="q", evidence_text="b", evidence_type=BenchEvidenceType.EVENT, event_id="e2", base_score=1.0),
        BenchCandidate(query_text="q", evidence_text="c", evidence_type=BenchEvidenceType.REFERENCE, event_id="e3", base_score=0.5),
    ]
    bench = bench_service.create_rerank_service(bench_cfg, llm_client=bench_llm)
    bench_results = bench.rerank("q", bench_candidates)

    # ours
    ours_cfg = RerankConfig(enabled=True, model="llm", top_n=2)
    ours_candidates = [
        RetrievalCandidate(query_text="q", evidence_text="a", evidence_type=EvidenceType.FACT, event_id="e1", base_score=2.0),
        RetrievalCandidate(query_text="q", evidence_text="b", evidence_type=EvidenceType.EVENT, event_id="e2", base_score=1.0),
        RetrievalCandidate(query_text="q", evidence_text="c", evidence_type=EvidenceType.REFERENCE, event_id="e3", base_score=0.5),
    ]
    ours = create_rerank_service(ours_cfg, llm_client=build_llm_client_from_fn(_llm))
    ours_results = ours.rerank("q", ours_candidates)

    assert [r.candidate.event_id for r in ours_results] == [r.candidate.event_id for r in bench_results]
    for a, b in zip(ours_results, bench_results, strict=True):
        assert a.rank == b.rank
        assert abs(a.rerank_score - b.rerank_score) < 1e-9
        assert abs(a.final_score - b.final_score) < 1e-9
