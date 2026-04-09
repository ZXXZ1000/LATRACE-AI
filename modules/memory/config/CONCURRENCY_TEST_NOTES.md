# 并发与批量参数压测纪要（OpenRouter+Qdrant）

时间：2026-01-25（本地环境，Neo4j/Qdrant 本地容器，OpenRouter 转发）

## 目标
- 找到在当前架构下的最快且稳定的 Stage3（事件+知识抽取、向量写入）参数组合。
- 关注：LLM 并发、embedding 并发与批大小、切片大小（max_turns），对耗时与准确率的影响。

## 测试样本
- LoCoMo `conv-30`（369 turns，105 queries），开启 judge & with-answer。

## 关键配置说明
- LLM 并发：`MEMORY_LLM_MAX_CONCURRENT=50`（环境变量）
- 事件/知识抽取并发：`event_extract_concurrency=50`，`fact_extract_concurrency=50`
- 事件对齐向量并发：`event_alignment.embed_concurrency=50`
- 向量写入 embedding 并发：`vector_store.embedding.embed_concurrency=30`
- 批大小、切片由本次实验调参（见下表）。

## 实验结果对比
| 名称 | batch_size | align_embed_batch | max_turns | Stage3 总耗时 | extract | vector | graph | Judge 准确率 |
| ---- | ---------- | ----------------- | --------- | ------------- | ------- | ------ | ----- | ------------ |
| 基线（B） | 64 | 128 | 120 | 34.5s | 13.0s | 15.9s | 2.7s | 80% |
| 半批/半切片 | 32 | 64 | 60 | 44.8s | 13.1s | 25.0s | 3.0s | 80% |
| 大批（定版） | **128** | **128** | **120** | **31.9s** | 14.9s | **10.7s** | 2.5s | 81% |

结论：
- 抽取段数受 session 数限制，LLM 并发上限未跑满，切小 max_turns 并未提升并发反而增加调用开销 → 总体变慢。
- 向量阶段是主要耗时点。增大 embedding batch_size（64→128）显著减少请求数，向量阶段耗时从 15.9s 降到 10.7s，整体最快且准确率不降。

## 采用的最终参数（已写入 config）
- `vector_store.embedding.batch_size = 128`
- `dialog.event_alignment.embed_batch_size = 128`
- `dialog.event_segmentation.max_turns = 120`
- 并发保持高位：LLM 50，event/fact 50，alignment 50，embedding 30

## 观测指标
- Prometheus 指标：`llm_inflight_max`、`embedding_inflight_max`、`ingest_stage3_*_ms_sum/count`
- 本次最佳 run：`job_70a86c9006ae`（目录 `benchmark/outputs/run_037_20260125_170249_conv-30`）

## 后续建议
- 如果出现大样本/更长对话，可小幅上调 `embed_concurrency`（30→40），但需关注 Qdrant TPS（本地瓶颈约 1500 pts/s）。
- 如需进一步提速，可尝试 max_turns 80~120 的小范围微调，但当前参数已处于速度/准确率平衡点。
