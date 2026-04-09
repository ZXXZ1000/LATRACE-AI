from __future__ import annotations

from modules.memory.application import metrics as m
from modules.memory.application.metrics import add_ann_latency_ms, as_prometheus_text, add_tx_latency_ms, inc


def test_ann_latency_and_tx_metrics_export():
    # Metrics are process-global; other tests may already have emitted ANN latency.
    base_sum = int(getattr(m, "_ann_latency_sum_mod", {}).get("text", 0))
    base_cnt = int(getattr(m, "_ann_latency_count_mod", {}).get("text", 0))
    add_ann_latency_ms("text", 120)
    add_ann_latency_ms("text", 80)
    add_tx_latency_ms(50)
    inc("neo4j_tx_retries_total", 2)
    inc("neo4j_tx_failures_total", 1)
    s = as_prometheus_text()
    assert f"memory_ann_latency_ms_sum{{modality=\"text\"}} {base_sum + 200}" in s
    assert f"memory_ann_latency_ms_count{{modality=\"text\"}} {base_cnt + 2}" in s
    assert "memory_neo4j_tx_ms_sum" in s and "memory_neo4j_tx_ms_count" in s
    assert "memory_neo4j_tx_retries_total 2" in s
    assert "memory_neo4j_tx_failures_total 1" in s
