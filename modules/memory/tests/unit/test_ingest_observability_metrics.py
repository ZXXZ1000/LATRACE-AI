from modules.memory.application.metrics import (
    as_prometheus_text,
    gauge_inc,
    gauge_dec,
    observe_ingest_latency,
)


def test_ingest_observability_metrics_exposed() -> None:
    # Record a sample ingest latency and toggle inflight gauges
    observe_ingest_latency("stage2", 123)
    gauge_inc("llm_inflight", 1)
    gauge_dec("llm_inflight", 1)

    text = as_prometheus_text()
    # Basic presence checks (do not assert exact values to avoid cross-test interference)
    assert "memory_ingest_stage2_ms_sum" in text
    assert "memory_ingest_stage2_ms_count" in text
    assert "memory_llm_inflight" in text
