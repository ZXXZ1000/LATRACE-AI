from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.suites.locomo.dataset import event_ids_to_dia_ids  # noqa: E402
from benchmark.suites.locomo.official_metrics import compute_context_recall  # noqa: E402

ROUTE_ORDER: Tuple[str, ...] = ("event_vec", "vec", "knowledge", "entity", "time")
WEIGHT_KEYS: Tuple[str, ...] = (
    "w_event_vec",
    "w_vec",
    "w_knowledge",
    "w_entity",
    "w_time",
    "w_match",
    "w_recency",
    "w_signal",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "modules" / "memory" / "config" / "parameter_optimization" / "outputs"
DEFAULT_SEARCH_SPACE = REPO_ROOT / "modules" / "memory" / "config" / "parameter_optimization" / "data" / "default_search_space.json"


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    evaluation_event_ids: Tuple[str, ...]
    route_ranks: Tuple[int | None, ...]
    route_norm_scores: Tuple[float, ...]
    match_fidelity_score: float
    recency_score: float
    graph_signal: float
    base_rank: int
    recorded_final_rank: int
    recorded_in_final_topk: bool
    text_preview: str


@dataclass(frozen=True, slots=True)
class QueryRecord:
    sample_id: str
    query_id: str
    question: str
    gold_event_ids: Tuple[str, ...]
    gold_context_ids: Tuple[str, ...]
    candidates: Tuple[CandidateRecord, ...]


@dataclass(frozen=True, slots=True)
class QueryEval:
    query_id: str
    gold_hit: bool
    support_recall: float
    official_recall: float
    weighted_support_recall_at_topk: float
    ndcg_at_topk: float
    selected_event_ids: Tuple[str, ...]
    selected_context_ids: Tuple[str, ...]
    top_candidates: Tuple[Tuple[str, float], ...]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _stable_unique(items: Iterable[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _normalize_scores(values: Sequence[float | None]) -> List[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return [0.0 for _ in values]
    lo = min(present)
    hi = max(present)
    if abs(hi - lo) <= 1e-12:
        return [1.0 if v is not None else 0.0 for v in values]
    return [((float(v) - lo) / (hi - lo)) if v is not None else 0.0 for v in values]


def _route_rank(route_ranks: Dict[str, Any], route: str) -> int | None:
    key = f"{route}_rank"
    value = route_ranks.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_candidate_records(raw_candidates: Sequence[Dict[str, Any]]) -> Tuple[CandidateRecord, ...]:
    score_columns: Dict[str, List[float | None]] = {route: [] for route in ROUTE_ORDER}
    for raw in raw_candidates:
        route_scores = raw.get("route_scores") if isinstance(raw.get("route_scores"), dict) else {}
        route_ranks = raw.get("route_ranks") if isinstance(raw.get("route_ranks"), dict) else {}
        for route in ROUTE_ORDER:
            rank = _route_rank(route_ranks, route)
            score = route_scores.get(route)
            score_columns[route].append(_safe_float(score) if rank is not None and score is not None else None)

    normalized_by_route = {route: _normalize_scores(values) for route, values in score_columns.items()}

    candidates: List[CandidateRecord] = []
    for index, raw in enumerate(raw_candidates):
        route_ranks = raw.get("route_ranks") if isinstance(raw.get("route_ranks"), dict) else {}
        evaluation_event_ids = tuple(str(v) for v in (raw.get("evaluation_event_ids") or []) if str(v or "").strip())
        if not evaluation_event_ids:
            fallback_id = str(raw.get("logical_event_id") or "").strip()
            evaluation_event_ids = ((fallback_id,) if fallback_id else tuple())
        candidates.append(
            CandidateRecord(
                candidate_id=str(raw.get("candidate_id") or f"candidate_{index}"),
                evaluation_event_ids=evaluation_event_ids,
                route_ranks=tuple(_route_rank(route_ranks, route) for route in ROUTE_ORDER),
                route_norm_scores=tuple(float(normalized_by_route[route][index]) for route in ROUTE_ORDER),
                match_fidelity_score=_safe_float(raw.get("match_fidelity_score")),
                recency_score=_safe_float(raw.get("recency_score")),
                graph_signal=_safe_float(raw.get("graph_signal")),
                base_rank=_safe_int(raw.get("base_rank"), default=10**9),
                recorded_final_rank=_safe_int(raw.get("final_rank"), default=10**9),
                recorded_in_final_topk=bool(raw.get("in_final_topk")),
                text_preview=str(raw.get("text_preview") or ""),
            )
        )
    return tuple(candidates)


def load_query_records(results_jsonl: Path, *, query_limit: int | None = None) -> Tuple[QueryRecord, ...]:
    queries: List[QueryRecord] = []
    lines = results_jsonl.read_text(encoding="utf-8").splitlines()
    if query_limit is not None and query_limit > 0:
        lines = lines[:query_limit]
    for line in lines:
        raw = json.loads(line)
        debug = raw.get("debug") if isinstance(raw.get("debug"), dict) else {}
        candidate_details = debug.get("candidate_details") if isinstance(debug.get("candidate_details"), list) else []
        sample_id = str(raw.get("sample_id") or "").strip()
        gold_event_ids = _stable_unique(str(v) for v in (raw.get("gold_evidence_ids") or []))
        gold_context_ids = _stable_unique(event_ids_to_dia_ids(sample_id, list(gold_event_ids)))
        queries.append(
            QueryRecord(
                sample_id=sample_id,
                query_id=str(raw.get("query_id") or ""),
                question=str(raw.get("question") or ""),
                gold_event_ids=gold_event_ids,
                gold_context_ids=gold_context_ids,
                candidates=_build_candidate_records(candidate_details),
            )
        )
    return tuple(queries)


def _retrieved_ids_from_candidates(candidates: Sequence[CandidateRecord]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    event_ids = _stable_unique(
        event_id
        for candidate in candidates
        for event_id in candidate.evaluation_event_ids
    )
    if not candidates:
        return event_ids, tuple()
    sample_id = ""
    for candidate in candidates:
        for event_id in candidate.evaluation_event_ids:
            if "_" in event_id:
                sample_id = event_id.split("_", 1)[0] + "_" + event_id.split("_", 2)[1].split("_", 1)[0]
                break
        if sample_id:
            break
    return event_ids, tuple()


def _build_scored_candidates(query: QueryRecord, config: Dict[str, float | int]) -> List[Tuple[float, float, float, int, CandidateRecord]]:
    rrf_k = int(config["rrf_k"])
    alpha = float(config["score_blend_alpha"])
    route_weights = (
        float(config["w_event_vec"]),
        float(config["w_vec"]),
        float(config["w_knowledge"]),
        float(config["w_entity"]),
        float(config["w_time"]),
    )
    w_match = float(config["w_match"])
    w_recency = float(config["w_recency"])
    w_signal = float(config["w_signal"])
    scored: List[Tuple[float, float, float, int, CandidateRecord]] = []
    for candidate in query.candidates:
        route_support = 0.0
        for idx, route_weight in enumerate(route_weights):
            rank = candidate.route_ranks[idx]
            if rank is None or route_weight == 0.0:
                continue
            rank_term = 1.0 / float(rrf_k + rank)
            blended = alpha * rank_term + (1.0 - alpha) * candidate.route_norm_scores[idx]
            route_support += route_weight * blended
        preselect = route_support + w_match * candidate.match_fidelity_score + w_recency * candidate.recency_score
        final_score = preselect + w_signal * candidate.graph_signal
        scored.append((final_score, preselect, route_support, -candidate.base_rank, candidate))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
            item[3],
            item[4].candidate_id,
        ),
        reverse=True,
    )
    return scored


def _rank_discount(rank: int) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def _weighted_support_recall_at_topk(
    *,
    top_candidates: Sequence[CandidateRecord],
    gold_event_ids: Sequence[str],
) -> float:
    gold_ids = tuple(str(event_id) for event_id in gold_event_ids if str(event_id or "").strip())
    if not gold_ids:
        return 1.0
    first_hit_rank: Dict[str, int] = {}
    gold_set = set(gold_ids)
    for rank, candidate in enumerate(top_candidates, start=1):
        for event_id in candidate.evaluation_event_ids:
            if event_id in gold_set and event_id not in first_hit_rank:
                first_hit_rank[event_id] = rank
    total = 0.0
    for event_id in gold_ids:
        rank = first_hit_rank.get(event_id)
        if rank is not None:
            total += _rank_discount(rank)
    return total / float(len(gold_ids))


def _ndcg_at_topk(
    *,
    ranked_candidates: Sequence[CandidateRecord],
    all_candidates: Sequence[CandidateRecord],
    gold_event_ids: Sequence[str],
    topk: int,
) -> float:
    gold_set = {str(event_id) for event_id in gold_event_ids if str(event_id or "").strip()}
    if not gold_set:
        return 1.0
    dcg = 0.0
    for rank, candidate in enumerate(ranked_candidates[:topk], start=1):
        relevance = 1.0 if gold_set.intersection(candidate.evaluation_event_ids) else 0.0
        if relevance > 0.0:
            dcg += relevance * _rank_discount(rank)
    relevant_in_pool = sum(1 for candidate in all_candidates if gold_set.intersection(candidate.evaluation_event_ids))
    ideal_hits = min(int(topk), int(relevant_in_pool))
    if ideal_hits <= 0:
        return 0.0
    ideal_dcg = sum(_rank_discount(rank) for rank in range(1, ideal_hits + 1))
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_config(query_records: Sequence[QueryRecord], config: Dict[str, float | int]) -> Tuple[Dict[str, Any], List[QueryEval]]:
    topk = max(1, int(config["topk"]))
    query_results: List[QueryEval] = []
    gold_hit_values: List[float] = []
    support_values: List[float] = []
    official_values: List[float] = []
    weighted_support_values: List[float] = []
    ndcg_values: List[float] = []

    for query in query_records:
        scored = _build_scored_candidates(query, config)
        top_candidates = [item[4] for item in scored[:topk]]
        selected_event_ids = _stable_unique(
            event_id
            for candidate in top_candidates
            for event_id in candidate.evaluation_event_ids
        )
        selected_context_ids = _stable_unique(event_ids_to_dia_ids(query.sample_id, list(selected_event_ids)))
        set(query.gold_event_ids)
        gold_context_ids = set(query.gold_context_ids)
        support_hits = sum(1 for event_id in query.gold_event_ids if event_id in set(selected_event_ids))
        support_recall = (float(support_hits) / float(len(query.gold_event_ids))) if query.gold_event_ids else 1.0
        official_recall = float(compute_context_recall(list(query.gold_context_ids), list(selected_context_ids)) or 0.0)
        gold_hit = bool(gold_context_ids & set(selected_context_ids)) if query.gold_context_ids else False
        weighted_support_recall = _weighted_support_recall_at_topk(
            top_candidates=top_candidates,
            gold_event_ids=query.gold_event_ids,
        )
        ndcg_at_topk = _ndcg_at_topk(
            ranked_candidates=top_candidates,
            all_candidates=query.candidates,
            gold_event_ids=query.gold_event_ids,
            topk=topk,
        )
        top_debug = tuple((item[4].candidate_id, float(item[0])) for item in scored[: min(topk, 5)])
        query_results.append(
            QueryEval(
                query_id=query.query_id,
                gold_hit=gold_hit,
                support_recall=support_recall,
                official_recall=official_recall,
                weighted_support_recall_at_topk=weighted_support_recall,
                ndcg_at_topk=ndcg_at_topk,
                selected_event_ids=selected_event_ids,
                selected_context_ids=selected_context_ids,
                top_candidates=top_debug,
            )
        )
        if query.gold_context_ids:
            gold_hit_values.append(1.0 if gold_hit else 0.0)
        if query.gold_event_ids:
            support_values.append(support_recall)
        if query.gold_context_ids:
            official_values.append(official_recall)
        if query.gold_event_ids:
            weighted_support_values.append(weighted_support_recall)
            ndcg_values.append(ndcg_at_topk)

    metrics = {
        "queries": len(query_records),
        "gold_hit_at_topk": (sum(gold_hit_values) / len(gold_hit_values) if gold_hit_values else None),
        "support_recall": (sum(support_values) / len(support_values) if support_values else None),
        "official_overall_recall": (sum(official_values) / len(official_values) if official_values else None),
        "weighted_support_recall_at_topk": (
            sum(weighted_support_values) / len(weighted_support_values) if weighted_support_values else None
        ),
        "ndcg_at_topk": (sum(ndcg_values) / len(ndcg_values) if ndcg_values else None),
    }
    return metrics, query_results


def evaluate_recorded_ranking(query_records: Sequence[QueryRecord], *, topk: int) -> Tuple[Dict[str, Any], List[QueryEval]]:
    query_results: List[QueryEval] = []
    gold_hit_values: List[float] = []
    support_values: List[float] = []
    official_values: List[float] = []
    weighted_support_values: List[float] = []
    ndcg_values: List[float] = []
    for query in query_records:
        recorded = sorted(
            [candidate for candidate in query.candidates if candidate.recorded_in_final_topk],
            key=lambda candidate: (candidate.recorded_final_rank, candidate.base_rank, candidate.candidate_id),
        )[:topk]
        selected_event_ids = _stable_unique(event_id for candidate in recorded for event_id in candidate.evaluation_event_ids)
        selected_context_ids = _stable_unique(event_ids_to_dia_ids(query.sample_id, list(selected_event_ids)))
        support_hits = sum(1 for event_id in query.gold_event_ids if event_id in set(selected_event_ids))
        support_recall = (float(support_hits) / float(len(query.gold_event_ids))) if query.gold_event_ids else 1.0
        official_recall = float(compute_context_recall(list(query.gold_context_ids), list(selected_context_ids)) or 0.0)
        gold_hit = bool(set(query.gold_context_ids) & set(selected_context_ids)) if query.gold_context_ids else False
        weighted_support_recall = _weighted_support_recall_at_topk(
            top_candidates=recorded,
            gold_event_ids=query.gold_event_ids,
        )
        ndcg_at_topk = _ndcg_at_topk(
            ranked_candidates=recorded,
            all_candidates=query.candidates,
            gold_event_ids=query.gold_event_ids,
            topk=topk,
        )
        query_results.append(
            QueryEval(
                query_id=query.query_id,
                gold_hit=gold_hit,
                support_recall=support_recall,
                official_recall=official_recall,
                weighted_support_recall_at_topk=weighted_support_recall,
                ndcg_at_topk=ndcg_at_topk,
                selected_event_ids=selected_event_ids,
                selected_context_ids=selected_context_ids,
                top_candidates=tuple((candidate.candidate_id, float(candidate.recorded_final_rank)) for candidate in recorded[: min(topk, 5)]),
            )
        )
        if query.gold_context_ids:
            gold_hit_values.append(1.0 if gold_hit else 0.0)
        if query.gold_event_ids:
            support_values.append(support_recall)
        if query.gold_context_ids:
            official_values.append(official_recall)
        if query.gold_event_ids:
            weighted_support_values.append(weighted_support_recall)
            ndcg_values.append(ndcg_at_topk)

    return {
        "queries": len(query_records),
        "gold_hit_at_topk": (sum(gold_hit_values) / len(gold_hit_values) if gold_hit_values else None),
        "support_recall": (sum(support_values) / len(support_values) if support_values else None),
        "official_overall_recall": (sum(official_values) / len(official_values) if official_values else None),
        "weighted_support_recall_at_topk": (
            sum(weighted_support_values) / len(weighted_support_values) if weighted_support_values else None
        ),
        "ndcg_at_topk": (sum(ndcg_values) / len(ndcg_values) if ndcg_values else None),
    }, query_results


def _objective_value(metrics: Dict[str, Any], primary_metric: str) -> float:
    value = metrics.get(primary_metric)
    return float(value) if value is not None else -1.0


def load_search_space(search_space_path: Path) -> Dict[str, Any]:
    return json.loads(search_space_path.read_text(encoding="utf-8"))


def expand_configs(search_space: Dict[str, Any], *, preset_name: str, max_combinations: int | None = None) -> List[Dict[str, Any]]:
    baseline = dict(search_space.get("baseline") or {})
    presets = search_space.get("presets") if isinstance(search_space.get("presets"), dict) else {}
    if preset_name not in presets:
        raise ValueError(f"unknown preset: {preset_name}")
    preset = presets[preset_name] if isinstance(presets[preset_name], dict) else {}
    grid = preset.get("grid") if isinstance(preset.get("grid"), dict) else {}
    param_names = list(grid.keys())
    values = []
    for name in param_names:
        raw = grid.get(name)
        if isinstance(raw, list):
            values.append(list(raw))
        else:
            values.append([raw])
    configs: List[Dict[str, Any]] = []
    for index, combo in enumerate(itertools.product(*values), start=1):
        config = dict(baseline)
        for name, value in zip(param_names, combo):
            config[name] = value
        config["config_id"] = f"{preset_name}_{index:04d}"
        configs.append(config)
        if max_combinations is not None and len(configs) >= max_combinations:
            break
    return configs


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir(preset_name: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{_timestamp()}_{preset_name}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_best_query_results(path: Path, query_results: Sequence[QueryEval]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in query_results:
            handle.write(
                json.dumps(
                    {
                        "query_id": row.query_id,
                        "gold_hit": row.gold_hit,
                        "support_recall": row.support_recall,
                        "official_recall": row.official_recall,
                        "weighted_support_recall_at_topk": row.weighted_support_recall_at_topk,
                        "ndcg_at_topk": row.ndcg_at_topk,
                        "selected_event_ids": list(row.selected_event_ids),
                        "selected_context_ids": list(row.selected_context_ids),
                        "top_candidates": [
                            {"candidate_id": candidate_id, "score": score}
                            for candidate_id, score in row.top_candidates
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _render_summary(
    *,
    results_jsonl: Path,
    preset_name: str,
    primary_metric: str,
    recorded_metrics: Dict[str, Any],
    best_metrics: Dict[str, Any],
    best_config: Dict[str, Any],
    leaderboard_rows: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Offline Rerank Sweep Summary")
    lines.append("")
    lines.append(f"- results_jsonl: `{results_jsonl}`")
    lines.append(f"- preset: `{preset_name}`")
    lines.append(f"- primary_metric: `{primary_metric}`")
    lines.append(f"- total_queries: `{recorded_metrics.get('queries')}`")
    lines.append("")
    lines.append("## Current Recorded Ranking")
    lines.append(f"- gold_hit_at_topk: `{_format_pct(recorded_metrics.get('gold_hit_at_topk'))}`")
    lines.append(f"- support_recall: `{_format_pct(recorded_metrics.get('support_recall'))}`")
    lines.append(f"- official_overall_recall: `{_format_pct(recorded_metrics.get('official_overall_recall'))}`")
    lines.append(f"- weighted_support_recall_at_topk: `{_format_pct(recorded_metrics.get('weighted_support_recall_at_topk'))}`")
    lines.append(f"- ndcg_at_topk: `{_format_pct(recorded_metrics.get('ndcg_at_topk'))}`")
    lines.append("")
    lines.append("## Best Offline Config")
    lines.append(f"- config_id: `{best_config.get('config_id')}`")
    lines.append(f"- gold_hit_at_topk: `{_format_pct(best_metrics.get('gold_hit_at_topk'))}`")
    lines.append(f"- support_recall: `{_format_pct(best_metrics.get('support_recall'))}`")
    lines.append(f"- official_overall_recall: `{_format_pct(best_metrics.get('official_overall_recall'))}`")
    lines.append(f"- weighted_support_recall_at_topk: `{_format_pct(best_metrics.get('weighted_support_recall_at_topk'))}`")
    lines.append(f"- ndcg_at_topk: `{_format_pct(best_metrics.get('ndcg_at_topk'))}`")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best_config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Top 10")
    for row in leaderboard_rows[:10]:
        lines.append(
            "- {config_id}: objective={objective:.6f} weighted_support={weighted_support} ndcg={ndcg} "
            "gold_hit={gold_hit} support_recall={support} official_recall={official} rrf_k={rrf_k} alpha={alpha}".format(
                config_id=row["config_id"],
                objective=float(row["objective"]),
                weighted_support=_format_pct(row.get("weighted_support_recall_at_topk")),
                ndcg=_format_pct(row.get("ndcg_at_topk")),
                gold_hit=_format_pct(row.get("gold_hit_at_topk")),
                support=_format_pct(row.get("support_recall")),
                official=_format_pct(row.get("official_overall_recall")),
                rrf_k=row.get("rrf_k"),
                alpha=row.get("score_blend_alpha"),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ranking-only parameter sweep for dialog_v2 candidate_details.")
    parser.add_argument("--results-jsonl", required=True, help="Path to benchmark results jsonl with debug.candidate_details.")
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help="Path to search-space json.")
    parser.add_argument("--preset", default="priority_scan", help="Preset name inside search-space json.")
    parser.add_argument(
        "--primary-metric",
        default="weighted_support_recall_at_topk",
        choices=[
            "gold_hit_at_topk",
            "support_recall",
            "official_overall_recall",
            "weighted_support_recall_at_topk",
            "ndcg_at_topk",
        ],
        help="Metric used to rank configs.",
    )
    parser.add_argument("--query-limit", type=int, default=0, help="Optional query limit for smoke runs.")
    parser.add_argument("--max-combinations", type=int, default=0, help="Optional hard cap on scanned combinations.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to timestamped folder under outputs/.")
    parser.add_argument("--top-n", type=int, default=20, help="How many leaderboard rows to keep in output.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N combinations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_jsonl = Path(args.results_jsonl).resolve()
    search_space_path = Path(args.search_space).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else _default_output_dir(str(args.preset))
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = load_search_space(search_space_path)
    max_combinations = int(args.max_combinations) if int(args.max_combinations or 0) > 0 else None
    query_limit = int(args.query_limit) if int(args.query_limit or 0) > 0 else None
    query_records = load_query_records(results_jsonl, query_limit=query_limit)
    configs = expand_configs(search_space, preset_name=str(args.preset), max_combinations=max_combinations)
    if not query_records:
        raise SystemExit("No queries loaded from results jsonl.")
    if not configs:
        raise SystemExit("No configs resolved from search space.")

    recorded_metrics, _ = evaluate_recorded_ranking(query_records, topk=int((search_space.get("baseline") or {}).get("topk") or 20))

    leaderboard: List[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] | None = None
    best_query_results: List[QueryEval] = []
    best_config: Dict[str, Any] | None = None
    primary_metric = str(args.primary_metric)
    progress_every = max(1, int(args.progress_every or 100))

    print(f"[offline_rerank_sweep] queries={len(query_records)} configs={len(configs)} preset={args.preset}", flush=True)
    for index, config in enumerate(configs, start=1):
        metrics, query_results = evaluate_config(query_records, config)
        objective = _objective_value(metrics, primary_metric)
        row = {
            "config_id": config["config_id"],
            "objective": objective,
            "gold_hit_at_topk": metrics.get("gold_hit_at_topk"),
            "support_recall": metrics.get("support_recall"),
            "official_overall_recall": metrics.get("official_overall_recall"),
            "weighted_support_recall_at_topk": metrics.get("weighted_support_recall_at_topk"),
            "ndcg_at_topk": metrics.get("ndcg_at_topk"),
            "rrf_k": config["rrf_k"],
            "topk": config["topk"],
            "score_blend_alpha": config["score_blend_alpha"],
            "w_event_vec": config["w_event_vec"],
            "w_vec": config["w_vec"],
            "w_knowledge": config["w_knowledge"],
            "w_entity": config["w_entity"],
            "w_time": config["w_time"],
            "w_match": config["w_match"],
            "w_recency": config["w_recency"],
            "w_signal": config["w_signal"],
        }
        leaderboard.append(row)
        if best_metrics is None or objective > _objective_value(best_metrics, primary_metric):
            best_metrics = dict(metrics)
            best_query_results = list(query_results)
            best_config = dict(config)
        if index % progress_every == 0 or index == len(configs):
            print(f"[offline_rerank_sweep] evaluated {index}/{len(configs)}", flush=True)

    leaderboard.sort(
        key=lambda row: (
            float(row["objective"]),
            float(row.get("weighted_support_recall_at_topk") or -1.0),
            float(row.get("ndcg_at_topk") or -1.0),
            float(row.get("support_recall") or -1.0),
            float(row.get("gold_hit_at_topk") or -1.0),
            float(row.get("official_overall_recall") or -1.0),
        ),
        reverse=True,
    )
    keep_n = max(1, int(args.top_n or 20))
    leaderboard = leaderboard[:keep_n]

    assert best_metrics is not None
    assert best_config is not None

    resolved_search_space = {
        "results_jsonl": str(results_jsonl),
        "preset": str(args.preset),
        "primary_metric": primary_metric,
        "query_limit": query_limit,
        "max_combinations": max_combinations,
        "baseline": search_space.get("baseline") or {},
        "selected_preset": ((search_space.get("presets") or {}).get(str(args.preset)) or {}),
        "evaluated_combinations": len(configs),
    }
    _write_json(output_dir / "resolved_search_space.json", resolved_search_space)
    _write_json(output_dir / "leaderboard.json", leaderboard)
    _write_json(output_dir / "best_config.json", {"config": best_config, "metrics": best_metrics})
    _write_csv(output_dir / "leaderboard.csv", leaderboard)
    _write_best_query_results(output_dir / "best_query_results.jsonl", best_query_results)
    (output_dir / "summary.md").write_text(
        _render_summary(
            results_jsonl=results_jsonl,
            preset_name=str(args.preset),
            primary_metric=primary_metric,
            recorded_metrics=recorded_metrics,
            best_metrics=best_metrics,
            best_config=best_config,
            leaderboard_rows=leaderboard,
        ),
        encoding="utf-8",
    )

    print(f"[offline_rerank_sweep] output_dir={output_dir}", flush=True)
    print(
        "[offline_rerank_sweep] best {metric}: {value:.6f}".format(
            metric=primary_metric,
            value=_objective_value(best_metrics, primary_metric),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
