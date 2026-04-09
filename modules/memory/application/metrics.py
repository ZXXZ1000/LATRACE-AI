from __future__ import annotations

from typing import Dict, Any, Tuple

_metrics: Dict[str, Any] = {
    "writes_total": 0,
    "searches_total": 0,
    "search_latency_ms_sum": 0,
    "graph_rel_merges_total": 0,
    "rollbacks_total": 0,
    "errors_total": 0,
    "search_cache_hits_total": 0,
    "search_cache_misses_total": 0,
    "search_cache_evictions_total": 0,
    "write_batch_flush_total": 0,
    "backend_retries_total": 0,
    "circuit_breaker_open_total": 0,
    "circuit_breaker_short_total": 0,
    "auth_failures_total": 0,
    "signature_failures_total": 0,
    "throttled_requests_total": 0,
    "request_too_large_total": 0,
    "errors_4xx_total": 0,
    "errors_5xx_total": 0,
    # usage (global counters, best-effort)
    "usage_llm_calls_total": 0,
    "usage_llm_prompt_tokens_total": 0,
    "usage_llm_completion_tokens_total": 0,
    "usage_llm_total_tokens_total": 0,
    "usage_llm_cost_usd_total": 0.0,
    "usage_embedding_calls_total": 0,
    "usage_embedding_tokens_total": 0,
    "usage_embedding_cost_usd_total": 0.0,
    # inflight gauges (best-effort)
    "llm_inflight": 0,
    "llm_inflight_max": 0,
    "embedding_inflight": 0,
    "embedding_inflight_max": 0,
    # ingest latency sums/counts (ms)
    "ingest_stage2_ms_sum": 0,
    "ingest_stage2_ms_count": 0,
    "ingest_stage3_ms_sum": 0,
    "ingest_stage3_ms_count": 0,
    "ingest_stage3_extract_ms_sum": 0,
    "ingest_stage3_extract_ms_count": 0,
    "ingest_stage3_build_ms_sum": 0,
    "ingest_stage3_build_ms_count": 0,
    "ingest_stage3_graph_ms_sum": 0,
    "ingest_stage3_graph_ms_count": 0,
    "ingest_stage3_vector_ms_sum": 0,
    "ingest_stage3_vector_ms_count": 0,
    "ingest_stage3_publish_ms_sum": 0,
    "ingest_stage3_publish_ms_count": 0,
    "ingest_stage3_overwrite_delete_ms_sum": 0,
    "ingest_stage3_overwrite_delete_ms_count": 0,
}

# simple histogram buckets for search latency (ms)
_latency_buckets = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
_latency_hist: Dict[int, int] = {b: 0 for b in _latency_buckets}

# ANN latency per modality (sum/count)
_ann_latency_sum_mod: Dict[str, int] = {}
_ann_latency_count_mod: Dict[str, int] = {}

# payload items per entry (e.g., number of contents items) histogram per modality
_payload_count_buckets = [1, 2, 5, 10, 20, 50, 100]
_payload_hist: Dict[str, Dict[int, int]] = {}

# vector size (dimension) histogram per modality
_vector_dim_buckets = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
_vector_dim_hist: Dict[str, Dict[int, int]] = {}

_graph_request_counters: Dict[Tuple[str, str], int] = {}
_graph_latency_stats: Dict[str, Dict[str, int]] = {}


def record_ttl_cleanup(status: str, deleted_nodes: int, deleted_edges: int) -> None:
    """Record TTL cleanup outcome."""
    key = f"ttl_cleanup_total_{status}"
    inc(key, 1)
    add("ttl_cleanup_nodes_total", deleted_nodes)
    add("ttl_cleanup_edges_total", deleted_edges)


def inc(name: str, value: int = 1) -> None:
    cur = _metrics.get(name, 0)
    try:
        _metrics[name] = cur + value
    except Exception:
        try:
            _metrics[name] = float(cur) + float(value)
        except Exception:
            _metrics[name] = value


def add(name: str, value: float = 0.0) -> None:
    cur = _metrics.get(name, 0.0)
    try:
        _metrics[name] = float(cur) + float(value)
    except Exception:
        _metrics[name] = float(value)


def gauge_set(name: str, value: float) -> None:
    try:
        _metrics[name] = float(value)
    except Exception:
        _metrics[name] = value


def gauge_inc(name: str, value: int = 1) -> None:
    try:
        cur = int(_metrics.get(name, 0))
    except Exception:
        cur = 0
    new = cur + int(value)
    _metrics[name] = new
    # track max
    max_key = f"{name}_max"
    try:
        cur_max = int(_metrics.get(max_key, 0))
    except Exception:
        cur_max = 0
    if new > cur_max:
        _metrics[max_key] = new


def gauge_dec(name: str, value: int = 1) -> None:
    try:
        cur = int(_metrics.get(name, 0))
    except Exception:
        cur = 0
    new = cur - int(value)
    if new < 0:
        new = 0
    _metrics[name] = new


def observe_ingest_latency(stage: str, ms: float) -> None:
    """Record ingest latency for a specific stage (ms)."""
    try:
        key_sum = f"ingest_{str(stage)}_ms_sum"
        key_count = f"ingest_{str(stage)}_ms_count"
        add(key_sum, float(ms))
        inc(key_count, 1)
    except Exception:
        return


def add_latency_ms(ms: int) -> None:
    _metrics["search_latency_ms_sum"] = _metrics.get("search_latency_ms_sum", 0) + int(ms)
    # histogram bucket
    for b in _latency_buckets:
        if ms <= b:
            _latency_hist[b] = int(_latency_hist.get(b, 0)) + 1
            break


def add_ann_latency_ms(modality: str, ms: int) -> None:
    key = str(modality)
    _ann_latency_sum_mod[key] = int(_ann_latency_sum_mod.get(key, 0)) + int(ms)
    _ann_latency_count_mod[key] = int(_ann_latency_count_mod.get(key, 0)) + 1


def add_tx_latency_ms(ms: int) -> None:
    _metrics["neo4j_tx_ms_sum"] = int(_metrics.get("neo4j_tx_ms_sum", 0)) + int(ms)
    _metrics["neo4j_tx_ms_count"] = int(_metrics.get("neo4j_tx_ms_count", 0)) + 1


def observe_payload_items(modality: str, count: int) -> None:
    """Observe how many content items an entry carries, labeled by modality."""
    mod = str(modality)
    if mod not in _payload_hist:
        _payload_hist[mod] = {b: 0 for b in _payload_count_buckets}
    # place into first bucket >= count
    placed = False
    for b in _payload_count_buckets:
        if int(count) <= b:
            _payload_hist[mod][b] = int(_payload_hist[mod].get(b, 0)) + 1
            placed = True
            break
    if not placed:
        # implicit +Inf bucket handled in exposition by using total entries per modality if needed
        # Track a separate counter for overflow
        key = f"payload_items_overflow_total_{mod}"
        _metrics[key] = int(_metrics.get(key, 0)) + 1


def observe_vector_size(modality: str, dim: int) -> None:
    """Observe vector dimension per entry by modality."""
    mod = str(modality)
    if mod not in _vector_dim_hist:
        _vector_dim_hist[mod] = {b: 0 for b in _vector_dim_buckets}
    placed = False
    for b in _vector_dim_buckets:
        if int(dim) <= b:
            _vector_dim_hist[mod][b] = int(_vector_dim_hist[mod].get(b, 0)) + 1
            placed = True
            break
    if not placed:
        key = f"vector_dim_overflow_total_{mod}"
        _metrics[key] = int(_metrics.get(key, 0)) + 1


def get_metrics() -> Dict[str, Any]:
    return dict(_metrics)


def record_graph_request(endpoint: str, status: str) -> None:
    """Record a Graph API request outcome."""
    inc("graph_requests_total", 1)
    key = (endpoint, status)
    _graph_request_counters[key] = int(_graph_request_counters.get(key, 0)) + 1


def add_graph_latency(endpoint: str, ms: int) -> None:
    stats = _graph_latency_stats.setdefault(endpoint, {"sum": 0, "count": 0})
    stats["sum"] += int(ms)
    stats["count"] += 1


def as_prometheus_text() -> str:
    """Return metrics in Prometheus text exposition format (basic)."""
    m = get_metrics()
    lines = []
    # writes_total
    lines.append("# TYPE memory_writes_total counter")
    lines.append(f"memory_writes_total {int(m.get('writes_total', 0))}")
    # searches_total
    lines.append("# TYPE memory_searches_total counter")
    lines.append(f"memory_searches_total {int(m.get('searches_total', 0))}")
    # search_latency_ms_sum
    lines.append("# TYPE memory_search_latency_ms_sum counter")
    lines.append(f"memory_search_latency_ms_sum {int(m.get('search_latency_ms_sum', 0))}")
    # graph_rel_merges_total
    lines.append("# TYPE memory_graph_rel_merges_total counter")
    lines.append(f"memory_graph_rel_merges_total {int(m.get('graph_rel_merges_total', 0))}")
    # rollbacks_total
    lines.append("# TYPE memory_rollbacks_total counter")
    lines.append(f"memory_rollbacks_total {int(m.get('rollbacks_total', 0))}")
    # errors_total
    lines.append("# TYPE memory_errors_total counter")
    lines.append(f"memory_errors_total {int(m.get('errors_total', 0))}")
    # cache metrics
    lines.append("# TYPE memory_search_cache_hits_total counter")
    lines.append(f"memory_search_cache_hits_total {int(m.get('search_cache_hits_total', 0))}")
    lines.append("# TYPE memory_search_cache_misses_total counter")
    lines.append(f"memory_search_cache_misses_total {int(m.get('search_cache_misses_total', 0))}")
    lines.append("# TYPE memory_search_cache_evictions_total counter")
    lines.append(f"memory_search_cache_evictions_total {int(m.get('search_cache_evictions_total', 0))}")
    # batch flush
    lines.append("# TYPE memory_write_batch_flush_total counter")
    lines.append(f"memory_write_batch_flush_total {int(m.get('write_batch_flush_total', 0))}")
    # retries & circuit breaker
    lines.append("# TYPE memory_backend_retries_total counter")
    lines.append(f"memory_backend_retries_total {int(m.get('backend_retries_total', 0))}")
    lines.append("# TYPE memory_circuit_breaker_open_total counter")
    lines.append(f"memory_circuit_breaker_open_total {int(m.get('circuit_breaker_open_total', 0))}")
    lines.append("# TYPE memory_circuit_breaker_short_total counter")
    lines.append(f"memory_circuit_breaker_short_total {int(m.get('circuit_breaker_short_total', 0))}")
    # error classes
    lines.append("# TYPE memory_errors_4xx_total counter")
    lines.append(f"memory_errors_4xx_total {int(m.get('errors_4xx_total', 0))}")
    lines.append("# TYPE memory_errors_5xx_total counter")
    lines.append(f"memory_errors_5xx_total {int(m.get('errors_5xx_total', 0))}")
    lines.append("# TYPE memory_auth_failures_total counter")
    lines.append(f"memory_auth_failures_total {int(m.get('auth_failures_total', 0))}")
    # inflight gauges
    lines.append("# TYPE memory_llm_inflight gauge")
    lines.append(f"memory_llm_inflight {int(m.get('llm_inflight', 0))}")
    lines.append("# TYPE memory_llm_inflight_max gauge")
    lines.append(f"memory_llm_inflight_max {int(m.get('llm_inflight_max', 0))}")
    lines.append("# TYPE memory_embedding_inflight gauge")
    lines.append(f"memory_embedding_inflight {int(m.get('embedding_inflight', 0))}")
    lines.append("# TYPE memory_embedding_inflight_max gauge")
    lines.append(f"memory_embedding_inflight_max {int(m.get('embedding_inflight_max', 0))}")
    # ingest stage latencies (sum/count)
    for name in (
        "ingest_stage2_ms",
        "ingest_stage3_ms",
        "ingest_stage3_extract_ms",
        "ingest_stage3_build_ms",
        "ingest_stage3_graph_ms",
        "ingest_stage3_vector_ms",
        "ingest_stage3_publish_ms",
        "ingest_stage3_overwrite_delete_ms",
    ):
        lines.append(f"# TYPE memory_{name}_sum counter")
        lines.append(f"memory_{name}_sum {int(m.get(f'{name}_sum', 0))}")
        lines.append(f"# TYPE memory_{name}_count counter")
        lines.append(f"memory_{name}_count {int(m.get(f'{name}_count', 0))}")
    lines.append("# TYPE memory_signature_failures_total counter")
    lines.append(f"memory_signature_failures_total {int(m.get('signature_failures_total', 0))}")
    lines.append("# TYPE memory_throttled_requests_total counter")
    lines.append(f"memory_throttled_requests_total {int(m.get('throttled_requests_total', 0))}")
    lines.append("# TYPE memory_request_too_large_total counter")
    lines.append(f"memory_request_too_large_total {int(m.get('request_too_large_total', 0))}")
    # usage counters
    lines.append("# TYPE memory_usage_llm_calls_total counter")
    lines.append(f"memory_usage_llm_calls_total {int(m.get('usage_llm_calls_total', 0))}")
    lines.append("# TYPE memory_usage_llm_prompt_tokens_total counter")
    lines.append(f"memory_usage_llm_prompt_tokens_total {int(m.get('usage_llm_prompt_tokens_total', 0))}")
    lines.append("# TYPE memory_usage_llm_completion_tokens_total counter")
    lines.append(f"memory_usage_llm_completion_tokens_total {int(m.get('usage_llm_completion_tokens_total', 0))}")
    lines.append("# TYPE memory_usage_llm_total_tokens_total counter")
    lines.append(f"memory_usage_llm_total_tokens_total {int(m.get('usage_llm_total_tokens_total', 0))}")
    lines.append("# TYPE memory_usage_llm_cost_usd_total counter")
    lines.append(f"memory_usage_llm_cost_usd_total {float(m.get('usage_llm_cost_usd_total', 0.0))}")
    lines.append("# TYPE memory_usage_embedding_calls_total counter")
    lines.append(f"memory_usage_embedding_calls_total {int(m.get('usage_embedding_calls_total', 0))}")
    lines.append("# TYPE memory_usage_embedding_tokens_total counter")
    lines.append(f"memory_usage_embedding_tokens_total {int(m.get('usage_embedding_tokens_total', 0))}")
    lines.append("# TYPE memory_usage_embedding_cost_usd_total counter")
    lines.append(f"memory_usage_embedding_cost_usd_total {float(m.get('usage_embedding_cost_usd_total', 0.0))}")
    lines.append("# TYPE memory_graph_requests_total counter")
    lines.append(f"memory_graph_requests_total {int(m.get('graph_requests_total', 0))}")
    if _graph_request_counters:
        lines.append("# TYPE memory_graph_requests_endpoint_total counter")
        for (endpoint, status), val in _graph_request_counters.items():
            lines.append(
                f'memory_graph_requests_endpoint_total{{endpoint="{endpoint}",status="{status}"}} {int(val)}'
            )
    if _graph_latency_stats:
        lines.append("# TYPE memory_graph_latency_ms_sum counter")
        lines.append("# TYPE memory_graph_latency_ms_count counter")
        for endpoint, stats in _graph_latency_stats.items():
            lines.append(
                f'memory_graph_latency_ms_sum{{endpoint="{endpoint}"}} {int(stats.get("sum", 0))}'
            )
            lines.append(
                f'memory_graph_latency_ms_count{{endpoint="{endpoint}"}} {int(stats.get("count", 0))}'
            )
    # histogram (cumulative buckets)
    lines.append("# TYPE memory_search_latency_ms histogram")
    cum = 0
    for b in _latency_buckets:
        cnt = int(_latency_hist.get(b, 0))
        cum += cnt
        lines.append(f"memory_search_latency_ms_bucket{{le=\"{b}\"}} {cum}")
    # +Inf bucket
    total_searches = int(m.get("searches_total", 0))
    lines.append(f"memory_search_latency_ms_bucket{{le=\"+Inf\"}} {total_searches}")
    lines.append(f"memory_search_latency_ms_sum {int(m.get('search_latency_ms_sum', 0))}")
    lines.append(f"memory_search_latency_ms_count {total_searches}")
    # dynamic: scope usage
    for key, val in m.items():
        if key.startswith("search_scope_used_") and key.endswith("_total"):
            scope = key.removeprefix("search_scope_used_").removesuffix("_total")
            lines.append("# TYPE memory_search_scope_total counter")
            lines.append(f"memory_search_scope_total{{scope=\"{scope}\"}} {val}")
    # dynamic: cache hits by scope
    for key, val in m.items():
        if key.startswith("search_cache_hits_scope_") and key.endswith("_total"):
            scope = key.removeprefix("search_cache_hits_scope_").removesuffix("_total")
            lines.append("# TYPE memory_search_cache_hits_scope_total counter")
            lines.append(f"memory_search_cache_hits_scope_total{{scope=\"{scope}\"}} {val}")
    # filter applies
    for tag in ("user", "domain", "session"):
        name = f"search_filter_applied_{tag}_total"
        if name in m:
            lines.append("# TYPE memory_search_filter_applied_total counter")
            lines.append(f"memory_search_filter_applied_total{{key=\"{tag}\"}} {int(m.get(name, 0))}")
    # rerank boost sums (float)
    if "rerank_boost_user_sum" in m:
        lines.append("# TYPE memory_rerank_boost_sum counter")
        lines.append(f"memory_rerank_boost_sum{{factor=\"user\"}} {float(m.get('rerank_boost_user_sum', 0.0))}")
    if "rerank_boost_domain_sum" in m:
        lines.append("# TYPE memory_rerank_boost_sum counter")
        lines.append(f"memory_rerank_boost_sum{{factor=\"domain\"}} {float(m.get('rerank_boost_domain_sum', 0.0))}")
    if "rerank_boost_session_sum" in m:
        lines.append("# TYPE memory_rerank_boost_sum counter")
        lines.append(f"memory_rerank_boost_sum{{factor=\"session\"}} {float(m.get('rerank_boost_session_sum', 0.0))}")
    # domain distribution counters
    for key, val in m.items():
        if key.startswith("domain_distribution_") and key.endswith("_total"):
            dom = key.removeprefix("domain_distribution_").removesuffix("_total")
            lines.append("# TYPE memory_domain_distribution_total counter")
            lines.append(f"memory_domain_distribution_total{{domain=\"{dom}\"}} {int(val)}")
    # ANN modality call counters
    for key, val in m.items():
        if key.startswith("ann_calls_total_"):
            mod = key.removeprefix("ann_calls_total_")
            lines.append("# TYPE memory_ann_calls_total counter")
            lines.append(f"memory_ann_calls_total{{modality=\"{mod}\"}} {int(val)}")
    # ANN latency per modality (sum/count)
    for mod, sum_ms in _ann_latency_sum_mod.items():
        lines.append("# TYPE memory_ann_latency_ms_sum counter")
        lines.append(f"memory_ann_latency_ms_sum{{modality=\"{mod}\"}} {int(sum_ms)}")
    for mod, cnt in _ann_latency_count_mod.items():
        lines.append("# TYPE memory_ann_latency_ms_count counter")
        lines.append(f"memory_ann_latency_ms_count{{modality=\"{mod}\"}} {int(cnt)}")
    # Payload items histogram per modality
    lines.append("# TYPE memory_payload_items_per_entry histogram")
    # Emit cumulative buckets per modality
    for mod, hist in _payload_hist.items():
        cum = 0
        for b in _payload_count_buckets:
            cnt = int(hist.get(b, 0))
            cum += cnt
            lines.append(f"memory_payload_items_per_entry_bucket{{modality=\"{mod}\",le=\"{b}\"}} {cum}")
        # +Inf bucket approximated by cum (no explicit total count metric per modality here)
        lines.append(f"memory_payload_items_per_entry_bucket{{modality=\"{mod}\",le=\"+Inf\"}} {cum}")
    # Vector dimension histogram per modality
    lines.append("# TYPE memory_vector_size_per_entry histogram")
    for mod, hist in _vector_dim_hist.items():
        cum = 0
        for b in _vector_dim_buckets:
            cnt = int(hist.get(b, 0))
            cum += cnt
            lines.append(f"memory_vector_size_per_entry_bucket{{modality=\"{mod}\",le=\"{b}\"}} {cum}")
        lines.append(f"memory_vector_size_per_entry_bucket{{modality=\"{mod}\",le=\"+Inf\"}} {cum}")
    # Neo4j tx metrics
    if "neo4j_tx_ms_sum" in m:
        lines.append("# TYPE memory_neo4j_tx_ms_sum counter")
        lines.append(f"memory_neo4j_tx_ms_sum {int(m.get('neo4j_tx_ms_sum', 0))}")
    if "neo4j_tx_ms_count" in m:
        lines.append("# TYPE memory_neo4j_tx_ms_count counter")
        lines.append(f"memory_neo4j_tx_ms_count {int(m.get('neo4j_tx_ms_count', 0))}")
    if "neo4j_tx_retries_total" in m:
        lines.append("# TYPE memory_neo4j_tx_retries_total counter")
        lines.append(f"memory_neo4j_tx_retries_total {int(m.get('neo4j_tx_retries_total', 0))}")
    if "neo4j_tx_failures_total" in m:
        lines.append("# TYPE memory_neo4j_tx_failures_total counter")
        lines.append(f"memory_neo4j_tx_failures_total {int(m.get('neo4j_tx_failures_total', 0))}")
    if "neo4j_batch_nodes_total" in m:
        lines.append("# TYPE memory_neo4j_batch_nodes_total counter")
        lines.append(f"memory_neo4j_batch_nodes_total {int(m.get('neo4j_batch_nodes_total', 0))}")
    if "neo4j_batch_rels_total" in m:
        lines.append("# TYPE memory_neo4j_batch_rels_total counter")
        lines.append(f"memory_neo4j_batch_rels_total {int(m.get('neo4j_batch_rels_total', 0))}")
    # TTL cleanup metrics
    for key, val in m.items():
        if key.startswith("ttl_cleanup_total_"):
            status = key.removeprefix("ttl_cleanup_total_")
            lines.append("# TYPE memory_ttl_cleanup_total counter")
            lines.append(f"memory_ttl_cleanup_total{{status=\"{status}\"}} {int(val)}")
    if "ttl_cleanup_nodes_total" in m:
        lines.append("# TYPE memory_ttl_cleanup_nodes_total counter")
        lines.append(f"memory_ttl_cleanup_nodes_total {int(m.get('ttl_cleanup_nodes_total', 0))}")
    if "ttl_cleanup_edges_total" in m:
        lines.append("# TYPE memory_ttl_cleanup_edges_total counter")
        lines.append(f"memory_ttl_cleanup_edges_total {int(m.get('ttl_cleanup_edges_total', 0))}")
    return "\n".join(lines) + "\n"
