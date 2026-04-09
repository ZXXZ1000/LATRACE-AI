from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_CFG = REPO_ROOT / "modules" / "memory" / "config" / "memory.config.yaml"
HYDRA_CFG = REPO_ROOT / "modules" / "memory" / "config" / "hydra" / "memory.yaml"


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _get_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def test_formal_default_keys_match_hydra_mirror():
    base = _load_yaml(BASE_CFG)
    hydra = _load_yaml(HYDRA_CFG)

    mirrored_paths = [
        "memory.llm.qa.provider",
        "memory.llm.qa.model",
        "memory.llm.judge.provider",
        "memory.llm.judge.model",
        "memory.search.lexical_hybrid.enabled",
        "memory.search.lexical_hybrid.corpus_limit",
        "memory.search.lexical_hybrid.lexical_topn",
        "memory.search.lexical_hybrid.normalize_scores",
        "memory.search.scoping.default_scope",
        "memory.search.scoping.user_match_mode",
        "memory.search.scoping.require_user",
        "memory.search.scoping.fallback_order",
        "memory.search.rerank.alpha_vector",
        "memory.search.rerank.beta_bm25",
        "memory.search.rerank.gamma_graph",
        "memory.search.rerank.delta_recency",
        "memory.search.rerank.user_boost",
        "memory.search.rerank.domain_boost",
        "memory.search.rerank.session_boost",
        "memory.search.dialog_v2.ranking.rrf_k",
        "memory.search.dialog_v2.ranking.weights.event_vec",
        "memory.search.dialog_v2.ranking.weights.vec",
        "memory.search.dialog_v2.ranking.weights.knowledge",
        "memory.search.dialog_v2.ranking.weights.entity",
        "memory.search.dialog_v2.ranking.weights.time",
        "memory.search.dialog_v2.ranking.weights.match",
        "memory.search.dialog_v2.ranking.weights.recency",
        "memory.search.dialog_v2.ranking.weights.signal",
        "memory.search.dialog_v2.ranking.weights.score_blend_alpha",
    ]

    mismatches = []
    for path in mirrored_paths:
        left = _get_path(base, path)
        right = _get_path(hydra, path)
        if left != right:
            mismatches.append((path, left, right))

    assert not mismatches, mismatches
