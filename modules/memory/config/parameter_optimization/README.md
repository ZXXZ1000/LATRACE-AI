# Memory 排序参数优化

这个目录用于承载 Memory 检索排序层的离线参数优化资产，统一把脚本、默认参数空间、说明和输出放在一起，避免再散落到 benchmark 目录各处。

## 目录结构

- `scripts/offline_rerank_sweep.py`
  - 读取一份带 `debug.candidate_details` 的 `results.jsonl`
  - 在候选池内离线重算排序分
  - 批量扫描排序权重组合并输出 leaderboard
- `data/default_search_space.json`
  - 默认 baseline 与 sweep preset
- `outputs/`
  - 离线 sweep 结果输出目录

## 适用场景

这个工具只解决一类问题：

- 已经有一份 `results_conv-26.jsonl`
- 里面已经带了 `debug.candidate_details`
- 想快速验证不同排序权重对 `gold_hit_at_topk / support_recall / official_overall_recall / weighted_support_recall_at_topk / ndcg_at_topk` 的影响

## 不能离线优化的内容

这点很重要：

- `candidate_k`
- `e_vec_oversample`
- route 开关
- 真正影响候选池组成的召回参数

这些参数会改变“候选池里有什么”，而离线脚本只能在“当前候选池内部”重排，不能把原本没召回进来的候选凭空补出来。

所以这套工具最适合做：

- `rrf_k`
- route weight
- `match / recency / signal`
- score-weighted RRF blend

## 指标口径

- `gold_hit_at_topk`
  - top-k 内是否至少命中一个 gold context
- `support_recall`
  - top-k 选中候选里，命中的 gold `event_id` 占比
- `official_overall_recall`
  - 把 `event_id` 转回 LoCoMo `D1:3` 这种 context id 之后计算的 context-level recall
- `weighted_support_recall_at_topk`
  - 对每个 gold `event_id` 按其第一次出现在 top-k 的位置做折扣，越靠前分越高
- `ndcg_at_topk`
  - 候选级二元相关性 nDCG，衡量“相关候选是否被排到了更前面”

默认主排序指标是 `weighted_support_recall_at_topk`。如果你更看重“是否至少命中一次”，可以改成 `gold_hit_at_topk`。

## 快速使用

```bash
python modules/memory/config/parameter_optimization/scripts/offline_rerank_sweep.py \
  --results-jsonl benchmark/outputs/abtest_conv26_20260321_current_default/results_conv-26.jsonl \
  --preset priority_scan \
  --primary-metric weighted_support_recall_at_topk
```

指定输出目录：

```bash
python modules/memory/config/parameter_optimization/scripts/offline_rerank_sweep.py \
  --results-jsonl benchmark/outputs/abtest_conv26_20260321_current_default/results_conv-26.jsonl \
  --preset priority_scan \
  --output-dir modules/memory/config/parameter_optimization/outputs/conv26_priority_scan
```

## 预设说明

- `priority_scan`
  - 先扫你当前最关心的三个方向：`rrf_k / w_event_vec / w_recency`
- `route_balance_scan`
  - 进一步拉开五路权重差异
- `full_ranking_scan`
  - 更完整的排序层参数空间，组合数明显更多，适合离线长跑

## 输出内容

每次运行会生成：

- `leaderboard.csv`
- `leaderboard.json`
- `best_config.json`
- `best_query_results.jsonl`
- `summary.md`
- `resolved_search_space.json`

## 当前默认 baseline

对应当前代码里的 `dialog_v2` 非 reranker 默认值：

- `rrf_k = 60`
- `w_event_vec = 0.6`
- `w_vec = 0.6`
- `w_knowledge = 0.9`
- `w_entity = 0.15`
- `w_time = 0.15`
- `w_match = 1.0`
- `w_recency = 0.0`
- `w_signal = 0.0`
- `score_blend_alpha = 0.7`
- `topk = 20`
