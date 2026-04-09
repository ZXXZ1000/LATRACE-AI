# LATRACE Benchmark Guide

This document is the public benchmark landing page for the open-source LATRACE release.

## What this guide is for

Use this page if you want to understand:

- which benchmark claims are currently presented publicly
- how to evaluate LATRACE in a fair first-pass setup
- what benchmark material is already published
- what still needs to be packaged for wider public reproduction

## Benchmarks highlighted by LATRACE

### LoCoMo

LATRACE uses LoCoMo to evaluate long-conversation memory, temporal tracking, and multihop retrieval quality.

### LongMemEval

LATRACE uses LongMemEval to evaluate long-term memory retrieval and answer quality over extended histories.

### Showcase flows

In addition to benchmark suites, LATRACE also maintains showcase-style evaluation flows for product-facing demos and retrieval analysis.

## Current public scope

As of `v0.1.0`, the open-source repository publicly includes:

- benchmark positioning and summary in the main [README](../README.md)
- release-level benchmark notes in [docs/releases/v0.1.0.md](releases/v0.1.0.md)
- the public API surface needed to evaluate ingestion and retrieval in [docs/api_reference.md](api_reference.md)

The full benchmark workspace is still being separated from maintainer-local tooling before broader publication. This means the public repo already explains the benchmark story, but not every internal helper script has been promoted into stable public documentation yet.

## Recommended evaluation path

1. Start with the self-host path in the main [README](../README.md).
2. Validate the ingest and retrieve loop through [docs/api_reference.md](api_reference.md).
3. Review tenant boundaries through [docs/tenant_isolation.md](tenant_isolation.md).
4. Compare the benchmark framing in [docs/releases/v0.1.0.md](releases/v0.1.0.md).
5. Open a GitHub Discussion if you want a public reproduction guide for a specific suite.

## What is likely to be published next

- a public LoCoMo reproduction walkthrough
- a public LongMemEval reproduction walkthrough
- a lightweight showcase example that can be run without internal maintainer setup

## Practical takeaway

For `v0.1.0`, LATRACE should be evaluated first as a production-ready memory service with strong temporal reasoning and multi-tenant isolation. Public benchmark packaging is underway, but the best current public entry points are the main README, API docs, tenant-isolation docs, and release notes.
