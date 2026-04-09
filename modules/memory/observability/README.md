# Memory 模块可观测性与监控建议（v1.0）

> 目标：给出一份“最低可行”的监控基线，用于线上观察 Memory 服务在 Qdrant + Neo4j 后端下的健康度与性能。

---

## 1. 指标暴露与采集

- 暴露端点：`GET /metrics_prom`  
  - 暴露所有 Memory 相关 Prometheus 指标（包括搜索/图/TTL/后端错误等）。  
- 采集建议：  
  - 把 Memory API 实例加入 Prometheus 抓取目标，scrape 间隔建议 15s–30s。  
  - 使用仓库中提供的：  
    - Grafana Dashboard：`modules/memory/observability/grafana_memory.json`  
    - Prometheus AlertRules：`modules/memory/observability/alerts/memory_rules.yml`

---

## 2. 关键指标分组

### 2.1 搜索与延迟

- `memory_searches_total{}`  
  - 含义：search 请求总次数（包括成功/失败）。  
  - 用途：监控 QPS、突发流量、低流量异常。

- `memory_search_latency_ms_bucket{}` / `memory_search_latency_ms_sum` / `memory_search_latency_ms_count`  
  - 用途：计算 P50/P95/P99 延迟：  
    - 示例：`histogram_quantile(0.95, sum(rate(memory_search_latency_ms_bucket[5m])) by (le))`

- `memory_errors_total{}`  
  - 含义：Memory 层面错误总数（包括后端错误、配置错误等）。  
  - 用途：配合 `rate(...)` 报警错误突刺。

### 2.2 缓存与 ANN 行为

- `memory_search_cache_hits_total{scope=...}` / `memory_search_cache_misses_total`  
  - 缓存命中率：  
    - 示例：`sum(rate(memory_search_cache_hits_total[5m])) / (sum(rate(memory_search_cache_hits_total[5m])) + sum(rate(memory_search_cache_misses_total[5m])))`

- `memory_ann_calls_total{modality=...}`  
  - 观察各模态向量检索的调用比例（text/image/audio/clip_image/face），用于容量规划与性能分析。

### 2.3 Graph / TTL / Explain

- `memory_graph_requests_total{endpoint=...}`  
  - GraphService 请求次数（分端点），用于观察图查询负载。

- `memory_ttl_cleanup_total{status=success|error|dry_run}`  
  - TTL 清理调用结果计数。  
- `memory_ttl_cleanup_nodes_total` / `memory_ttl_cleanup_edges_total`  
  - 单位时间内清理的节点/边数量，用于判断 TTL 策略是否过激或过弱。

- `memory_explain_cache_hits_total` / `memory_explain_cache_misses_total` / `memory_explain_cache_evictions_total`  
  - Explain 场景（首次相遇/事件证据链）缓存有效性与容量压力。

### 2.4 后端与可靠性

- `memory_qdrant_http_errors_total` / `memory_qdrant_retries_total`  
- `memory_neo4j_errors_total` / `memory_neo4j_retries_total`  
  - 用于快速定位 Qdrant 或 Neo4j 不稳定/不可用问题。

- `memory_circuit_breaker_open_total`  
  - 记录熔断开启次数；高于基线时说明下游持续错误或延迟过高。

---

## 3. Grafana Dashboard 建议

可直接导入仓库中的 `grafana_memory.json`，包含基础面板：

- Searches per second：`rate(memory_searches_total[1m])`  
- Errors total：`rate(memory_errors_total[5m])`  
- Circuit breaker opens：`rate(memory_circuit_breaker_open_total[5m])`  
- Latency P95：基于 `memory_search_latency_ms_bucket` 的 P95 曲线  
- Cache hit ratio：`hits / (hits+misses)`  
- ANN calls by modality：各模态 ANN 调用统计

建议按环境添加标签（如 `env`、`tenant`），用于分环境和大租户视图。

---

## 4. Prometheus 告警建议

仓库提供了基础告警模板：`observability/alerts/memory_rules.yml`，包含：

- `MemorySearchLatencyHighP95`  
  - 条件：`P95 > 300ms` 持续 5 分钟  
  - 用途：搜索整体变慢时预警。

- `MemoryErrorsSpike`  
  - 条件：`increase(memory_errors_total[10m]) > 20`  
  - 用途：10 分钟内错误超过 20 次视为严重异常。

- `CircuitBreakerOpenSpike`  
  - 条件：`increase(memory_circuit_breaker_open_total[10m]) > 3`  
  - 用途：下游不稳定导致熔断频繁打开时发出警告。

> 建议：  
> - 在生产中可以根据实际流量/延迟情况调整阈值；  
> - 对关键租户或重要路径可以单独扩展更细粒度的告警规则。

---

## 5. 最小可监控集合（MVP）

对于小规模部署，只需确保以下几项可见：

1. 搜索 QPS + P95 延迟（判断是否“能用且不卡”）。  
2. `memory_errors_total` 和 `memory_circuit_breaker_open_total`（判断是否频繁失败）。  
3. TTL 清理调用/删除数量（避免记忆“清光”或“堆爆”）。  
4. Cache 命中率（极低时检查缓存配置是否生效）。  

其余指标可按需要逐步启用，以防仪表盘信息过载。

