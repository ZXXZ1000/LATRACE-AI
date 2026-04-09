# Experiment Configs

This directory stores experiment records and tuning protocols relative to
`modules/memory/config/memory.config.yaml`.

## Principles

- `memory.config.yaml` is the only formal default config.
- `runtime_overrides.json` is local-only runtime state and is intentionally not tracked.
- Each experiment file stores only the changed knobs, benchmark context, and observed
  results.
- Markdown protocol files may live beside YAML records when we need to document the
  optimization process, isolation rules, or promotion gates for a tuning phase.

## File Format

```yaml
meta:
  id: 2026-03-12_showcase_bm25-mid_top20_qwen-plus
  owner: zhaoxiang
  purpose: lexical hybrid tuning
  datasets:
    - showcase_pack_v1

overrides:
  memory.search.rerank.alpha_vector: 0.35
  memory.search.rerank.beta_bm25: 0.50
  memory.search.lexical_hybrid.enabled: true

results:
  showcase_accuracy: "14/19"
  avg_retrieval_latency_ms: 313.43
  notes: candidate only, not yet promoted
```

## Materialize A Candidate Config

```bash
python modules/memory/scripts/materialize_experiment_config.py \
  --experiment modules/memory/config/experiments/2026-03-12_showcase_bm25_mid_top20_qwen_plus.yaml \
  --output /tmp/memory.experiment.yaml
```

This renders a full merged config for inspection without mutating the formal baseline.

## Recommended Layout

- `*.yaml`: machine-readable experiment deltas, baseline snapshots, and observed results.
- `*.md`: human-readable optimization protocols, search plans, and promotion notes for the
  current tuning phase.
