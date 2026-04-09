# Dialog V2 Outer Param Optimization Protocol

## Goal

This document defines the next tuning phase after establishing the current working baseline:

- `modules/memory/retrieval.py`
- `_DIALOG_V2_DEFAULT_WEIGHTS["knowledge"] = 0.6`

The purpose of this phase is to continue improving the outer fusion layer of `dialog_v2`
without confusing retrieval gains with ingest instability or cross-tenant contamination.

## Current Working Baseline

- Baseline record:
  `2026-03-13_dialog_v2_knowledge_0_6_working_baseline.yaml`
- Current hardcoded outer weights:
  - `event_vec = 0.6`
  - `vec = 0.6`
  - `knowledge = 0.6`
  - `entity = 0.3`
  - `time = 0.3`
  - `recency = 0.35`
  - `signal = 0.0`
  - `multi = 0.03`
- Other important defaults that remain unchanged:
  - `rrf_k = 90`
  - `graph_cap = 5`
  - `candidate_k = 50`
  - `seed_topn = 15`

This baseline is a code-level anchor, not yet a promoted config baseline.

## Dataset Roles

### 1. Smoke / Regression Set

- `showcase_pack_v1`
- Use only as a fast guardrail.
- Expected behavior under the current working baseline:
  - fixed-state run should stay at `19/19`

### 2. Main Dev Set

- `conv-26`
- This is the primary dataset for parameter search.
- Use it to compare candidate settings against the current working baseline.

### 3. Validation Set

Before promoting any candidate into `memory.config.yaml`, validate on:

- `conv-30`
- plus at least `2-3` additional conversation samples with different distributions

The purpose is to prevent overfitting to `showcase_pack_v1` or a single LoCoMo sample.

## Isolation Rules

### Fresh Ingest Runs

- Always use a new unique tenant for each fresh run.
- Always record:
  - tenant id
  - user id / user tokens
  - session id or state file path

### Fixed-State A/B Runs

- Reuse the exact same ingest state with:
  - `--skip-ingest`
  - `--state-file <existing_state>`
- Do not compare candidates across separately ingested graphs.
- When comparing only ranking behavior, the ingest state must be identical.

### Why

The current memory architecture isolates primarily by:

- `tenant_id`
- `user_tokens`
- `memory_domain`

Session or run id alone is not a sufficient isolation boundary for fair A/B evaluation.

## Search Space

### First-Priority Axes

These axes currently have the strongest signal and should be explored first:

- `knowledge`: `0.50, 0.55, 0.60, 0.65, 0.70`
- `rrf_k`: `20, 25, 30, 35, 40`

### Second-Priority Axis

Only after selecting a reasonable `knowledge x rrf_k` region:

- `multi`: `0.00, 0.005, 0.01, 0.015, 0.02`

### Hold Constant for Now

Unless a later result clearly justifies reopening them, keep these fixed:

- service-layer rerank weights
- `signal`
- `graph_cap`
- `candidate_k`
- `seed_topn`
- `lexical_hybrid`

The current evidence suggests the main bottleneck is outer fusion, not the inner hybrid
retrieval formula.

## Search Procedure

### Stage 0: Baseline Freeze

- Freeze the current working baseline:
  - `knowledge = 0.6`
- Reproduce:
  - `showcase_pack_v1` fixed-state `19/19`
  - `conv-26` baseline score

### Stage 1: Coarse Sweep

- Sweep `knowledge` across the first-priority range on `conv-26`
- Keep:
  - `rrf_k = 90`
  - `multi = 0.03`
- Goal:
  - verify whether `0.6` is locally best or only one good point

### Stage 2: Joint Sweep

- On the best `knowledge` neighborhood, sweep:
  - `rrf_k`
- Goal:
  - test whether lower rank flattening improves final placement of decisive evidence

### Stage 3: Fine Sweep

- Only after identifying a stable `knowledge x rrf_k` pair, fine-tune:
  - `multi`

### Stage 4: Validation

- Re-run the candidate winner on:
  - `showcase_pack_v1` fixed-state
  - `conv-30`
  - additional held-out conversation samples

## Promotion Gate

A candidate can be promoted into `memory.config.yaml` only if:

1. It beats the current working baseline on `conv-26`.
2. It does not regress the fixed-state showcase guardrail in a meaningful way.
3. It remains competitive on the validation set.
4. The improvement is reproducible across repeated runs with clean isolation.

If any of the above is not satisfied, keep the candidate recorded but do not promote it.

## Recording Requirements

Every candidate run must produce:

- one YAML record in `config/experiments/`
- one benchmark report path
- explicit dataset scope
- explicit isolation metadata
- a short note about whether the run is:
  - fresh-ingest
  - fixed-state
  - smoke
  - dev
  - validation

## Immediate Next Step

The next search should start from the current working baseline and use:

- primary dev set: `conv-26`
- regression guard: `showcase_pack_v1`
- first sweep axes:
  - `knowledge`
  - `rrf_k`

Do not reopen service-layer alpha/beta/gamma/delta tuning until the outer-fusion search
has clearly plateaued.
