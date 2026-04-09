from __future__ import annotations

"""Dialog v1 rerank (benchmark-aligned).

This module intentionally mirrors:
- benchmark/shared/adapters/rerank_types.py
- benchmark/shared/adapters/rerank_service.py

We keep the prompt here (not importing benchmark code) so that production usage does not depend on `benchmark/`.
Tests should assert prompt equality to prevent drift.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Callable, List, Optional, Protocol


class EvidenceType(Enum):
    """Evidence type mapping (fact/event/reference)."""

    FACT = "fact"
    EVENT = "event"
    REFERENCE = "reference"
    UNKNOWN = "unknown"

    @classmethod
    def from_source(cls, source: str) -> "EvidenceType":
        mapping = {
            "fact_search": cls.FACT,
            "event_search": cls.EVENT,
            "reference_trace": cls.REFERENCE,
        }
        return mapping.get(str(source or ""), cls.UNKNOWN)


@dataclass
class RetrievalCandidate:
    """Unified rerank candidate payload (mirrors benchmark)."""

    query_text: str
    evidence_text: str
    evidence_type: EvidenceType
    event_id: str
    base_score: float
    fact_id: Optional[str] = None
    turn_id: Optional[str] = None
    timestamp: Optional[str] = None
    importance: str = "medium"
    metadata: dict = field(default_factory=dict)

    def to_display_text(self) -> str:
        type_label = {
            EvidenceType.FACT: "(Fact)",
            EvidenceType.EVENT: "(Event)",
            EvidenceType.REFERENCE: "(Reference)",
            EvidenceType.UNKNOWN: "(Unknown)",
        }
        return f"{type_label[self.evidence_type]} {self.evidence_text}"


@dataclass
class RerankResult:
    candidate: RetrievalCandidate
    rerank_score: float
    final_score: float
    rank: int

    @property
    def event_id(self) -> str:
        return self.candidate.event_id

    @property
    def evidence_text(self) -> str:
        return self.candidate.evidence_text


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "noop"  # "llm" | "cross_encoder" | "noop"
    top_n: int = 20
    weight_base: float = 0.4
    weight_rerank: float = 0.5
    weight_type: float = 0.1
    type_bias: dict = field(
        default_factory=lambda: {
            "fact": 0.15,
            "reference": 0.10,
            "event": 0.0,
            "unknown": 0.0,
        }
    )

    @classmethod
    def default(cls) -> "RerankConfig":
        return cls(enabled=False, model="noop")

    @classmethod
    def llm_rerank(cls, top_n: int = 20) -> "RerankConfig":
        return cls(enabled=True, model="llm", top_n=int(top_n))


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def _normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    if max_val <= min_val:
        return 0.5
    return max(0.0, min(1.0, (float(score) - float(min_val)) / (float(max_val) - float(min_val))))


def _compute_final_score(
    base_score: float,
    rerank_score: float,
    evidence_type: EvidenceType,
    config: RerankConfig,
) -> float:
    type_bias = float(config.type_bias.get(evidence_type.value, 0.0))
    return (
        float(config.weight_base) * float(base_score)
        + float(config.weight_rerank) * float(rerank_score)
        + float(config.weight_type) * type_bias
    )


class NoopRerankService:
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig.default()

    def rerank(self, query: str, candidates: List[RetrievalCandidate]) -> List[RerankResult]:
        if not candidates:
            return []
        base_scores = [float(c.base_score) for c in candidates]
        min_base = min(base_scores)
        max_base = max(base_scores)
        results: List[RerankResult] = []
        for candidate in candidates:
            norm_base = _normalize_score(float(candidate.base_score), min_base, max_base)
            final_score = _compute_final_score(
                base_score=norm_base,
                rerank_score=norm_base,
                evidence_type=candidate.evidence_type,
                config=self.config,
            )
            results.append(RerankResult(candidate=candidate, rerank_score=norm_base, final_score=final_score, rank=0))

        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_rerank_prompt_v1.txt"
RERANK_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class LLMRerankService:
    def __init__(self, llm_client: LLMClient, config: Optional[RerankConfig] = None):
        self.llm = llm_client
        self.config = config or RerankConfig.llm_rerank()

    def rerank(self, query: str, candidates: List[RetrievalCandidate]) -> List[RerankResult]:
        if not candidates:
            return []

        base_scores_all = [float(c.base_score) for c in candidates]
        min_base = min(base_scores_all)
        max_base = max(base_scores_all)

        to_rerank = candidates[: int(self.config.top_n)]
        rest = candidates[int(self.config.top_n) :]

        passages_text = "\n".join([f"[{i+1}] {c.to_display_text()[:300]}" for i, c in enumerate(to_rerank)])
        prompt = RERANK_PROMPT.format(query=str(query or ""), passages=passages_text)

        llm_failed = False
        try:
            raw = self.llm.generate("", prompt)
            scores = self._parse_scores(raw, len(to_rerank))
            if not scores:
                llm_failed = True
                scores = {i: float(c.base_score) for i, c in enumerate(to_rerank)}
        except Exception:
            llm_failed = True
            scores = {i: float(c.base_score) for i, c in enumerate(to_rerank)}

        results: List[RerankResult] = []
        for i, candidate in enumerate(to_rerank):
            rerank_score = float(scores.get(i, float(candidate.base_score)))
            norm_base = _normalize_score(float(candidate.base_score), min_base, max_base)
            if llm_failed:
                norm_rerank = norm_base
            else:
                norm_rerank = _normalize_score(rerank_score, 0.0, 1.0)
            final_score = _compute_final_score(
                base_score=norm_base,
                rerank_score=norm_rerank,
                evidence_type=candidate.evidence_type,
                config=self.config,
            )
            results.append(RerankResult(candidate=candidate, rerank_score=norm_rerank, final_score=final_score, rank=0))

        for candidate in rest:
            norm_base = _normalize_score(float(candidate.base_score), min_base, max_base)
            final_score = _compute_final_score(
                base_score=norm_base,
                rerank_score=0.0,
                evidence_type=candidate.evidence_type,
                config=self.config,
            )
            results.append(RerankResult(candidate=candidate, rerank_score=0.0, final_score=final_score, rank=0))

        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    def _parse_scores(self, raw: str, n: int) -> dict:
        try:
            start = str(raw).find("{")
            end = str(raw).rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(str(raw)[start:end])
                return {int(k) - 1: float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
        return {}


def create_rerank_service(config: RerankConfig, llm_client: Optional[LLMClient] = None):
    if not config.enabled or str(config.model or "noop") == "noop":
        return NoopRerankService(config)
    if str(config.model) == "llm":
        if llm_client is None:
            return NoopRerankService(config)
        return LLMRerankService(llm_client, config)
    return NoopRerankService(config)


def build_llm_client_from_fn(fn: Callable[[str, str], str]) -> LLMClient:
    class _FnClient:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return str(fn(system_prompt, user_prompt))

    return _FnClient()
