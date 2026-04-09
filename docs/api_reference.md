# LATRACE Memory Developer API Reference (HTTP)

> **Base URL**: `http://<host>:8000`  
> **Protocol**: HTTP/1.1 REST (JSON)  
> **Content-Type**: `application/json`  
> **Tenant Isolation**: When `auth.enabled=false` (development default), memory and graph requests MUST carry the `X-Tenant-ID` header. Public ops endpoints (`/health`, `/metrics`, `/metrics_prom`) are exempt. When `auth.enabled=true`, the tenant is parsed from the Token/JWT, but it is still highly recommended to include `X-Tenant-ID` for explicit alignment and debugging.

This document targets developers integrating with LATRACE Memory. The goal is to provide a comprehensive, runnable, and aligned integration contract, detailing HTTP boundaries, public vs internal endpoints, and data contracts.

**Applicability:**
- For **ADK Semantic Tools (Layer 1)**, please review `adk_integration.md` first (which covers tool schemas, runtime wiring, and agent orchestration).
- For **tenant boundaries and request scoping**, please review `tenant_isolation.md` (which covers `tenant_id`, `user_tokens`, and namespace guidance).
- This documentation strictly covers **External HTTP Contracts**; internal processes are omitted unless necessary.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Authentication & Security](#2-authentication--security)
3. [Quick Start](#3-quick-start)
4. [Core Concepts & Data Contracts](#4-core-concepts--data-contracts)
5. [API Reference (HTTP)](#5-api-reference-http)
6. [Errors & Retries](#6-errors--retries)
7. [Limits & Performance](#7-limits--performance)
8. [Versioning & Compatibility](#8-versioning--compatibility)
9. [Best Practices: Dialog Ingress (Session Write)](#9-best-practices-dialog-ingress-session-write)

---

## 1. Overview

### 1.1 Service Positioning
LATRACE Memory is a "**Retrieval-Augmented Memory Service**", segregated into two core layers:
1. **Memory Layer (Recall + Filter + Optional Graph Expansion + Rerank)**: Evaluated via `POST /search`. Built for injecting broad context into LLMs.
2. **Graph Layer (Typed TKG: Entities/Events/Evidences/TimeSlices)**: Evaluated via `/graph/v0/*` and `/graph/v1/*`. Built for strict structure, exact event tracking, and explicit timelines.

*Rule of Thumb:* `/search` handles fuzzy conceptual recall. The Graph APIs handle structured exact topological queries.

### 1.2 Minimal Integration Path
1. **Writing (Recommended: Batch upon session close)**
   - **High-Level**: `POST /ingest/dialog/v1` (the recommended session-ingest endpoint; handles the ingestion workflow automatically).
   - **Low-Level (Fallback)**: `POST /write` (Store `MemoryEntry` vectors explicitly).
2. **Retrieving (During Agent Runtime)**
   - **High-Level**: `POST /retrieval/dialog/v2` (Multi-path recall + fusion + optional Synthesis).
   - **Low-Level (Fallback)**: `POST /search` (Returns evidence `hits` + `neighbors` + `trace`).

### 1.3 The Three Retrieval Entries 
- `POST /retrieval/dialog/v2`: **Advanced Dialogue Orchestration** (Multi-channel recall + Graph interpretation + Option QA). This is the default recommendation.
- `POST /search`: **Vector Recall (Qdrant ANN)**. Optional overlays for BM25, TKG expansion, and time decay. Best for implicit natural language.
- `POST /graph/v1/search`: **Structured Event Search** (Neo4j). Explicit keyword/entity matching. It is *not* an approximation fallback.

If your query relies heavily on implicit NLP: do not force Graph-first recall. Trigger `/retrieval/dialog/v2`.

### 1.4 API System Architecture

#### System Route Topology (Text Tree)
```text
Memory Server (Service Root)
│
├── 📦 1. Write & Lifecycle
│   ├── POST /write                 (Atomic Write: Text/Vector)
│   ├── POST /update                (Deep-patch Entry)
│   ├── POST /delete                (Delete Entry)
│   ├── POST /link                  (Edge Creation)
│   ├── POST /batch_delete          (Batch Remove)
│   ├── POST /memory/v1/clear       (Tenant Cache Wipe, supports dry-run)
│   └── POST /rollback              (Version Rollback)
│
├── 🔍 2. Core Search
│   ├── POST /search                (★ Main Entry: Hybrid + Graph)
│   ├── POST /timeline_summary      (Time-Series Summary)
│   ├── POST /speech_search         (Audio-Transcription Keyword hit)
│   ├── POST /entity_event_anchor   (Spatiotemporal Anchor localization)
│   └── POST /object_search         (Visual Object Search)
│
├── 🕸️ 3. Graph TKG
│   ├── POST /graph/v1/search       (Structured Event Target Search)
│   ├── POST /graph/v0/upsert       (Node Write injection)
│   ├── GET  /graph/v0/events/{event_id}
│   ├── GET  /graph/v0/entities/{entity_id}/timeline
│   ├── GET  /graph/v0/entities/{entity_id}/evidences
│   ├── GET  /graph/v0/entities/resolve
│   └── GET  /graph/v0/explain/*    (Evidence Chain Derivation)
│
├── 🤖 4. Agent SDK Proxies
│   ├── POST /ingest/dialog/v1      (Session dialog ingestion)
│   ├── GET  /ingest/jobs/{id}      (Job state tracking)
│   └── POST /retrieval/dialog/v2   (Orchestrated inference recall)
│
└── ⚙️ 5. Infrastructure & Ops
    ├── GET  /health                (Deep Health Status)
    ├── GET  /metrics               (Telemetry)
    ├── GET  /config                (System Snapshots)
    ├── PATCH /config               (Hot-Reload Configurations)
    └── POST /config/search/*       (Dynamic Overrides)
```

---

## 2. Authentication & Security

Security operates across three tiers: **Tenant Isolation → API Token (JWT) → HMAC Operations Signature**.

### 2.1 Essential Headers
| Header | Required | Description |
| --- | --- | --- |
| `Content-Type` | Yes (POST/PATCH) | Strictly `application/json` |
| `Authorization` | Recommended | `Bearer <token>` |
| `X-API-Token` | Optional | Fallback format mapping |
| `X-Tenant-ID` | Dependent | **MANDATORY** for memory and graph routes if `auth.enabled=false`. Public ops routes (`/health`, `/metrics`, `/metrics_prom`) are exempt. If auth is enabled, the server maps by token, but explicit alignment is encouraged. |
| `X-Request-ID` | Recommended | UUID request tracing tag. |

### 2.2 Operation Signatures (HMAC-SHA256)
When `auth.signing.required=true`, mutating endpoints (`write/update/delete`) demand:
- `X-Signature-Ts`: Unix timestamp (Int). Tolerance default `±300s`.
- `X-Signature`: Hex-Digest (using tenant secret mapped to `f"{ts}.{request_path}.{raw_request_body_bytes}"`).

### 2.3 Bring Your Own Key (BYOK)
- End-user provider keys are **never** passed via raw header parameters.
- LATRACE queries `api_token -> api_key_id` locally behind the firewall.
- Unmapped keys organically fallback to platform-curated inference pools.

---

## 3. Quick Start

### 3.1 Base Health
```bash
curl -sS "http://127.0.0.1:8000/health" -H "X-Tenant-ID: t1"
```

### 3.2 Atomic Ingestion: `POST /write`
```bash
curl -sS "http://127.0.0.1:8000/write" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: t1" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "upsert": true,
    "entries": [
      {
        "kind": "semantic",
        "modality": "text",
        "contents": ["User likes Apples"],
        "metadata": {
          "user_id": ["u:1001"],
          "memory_domain": "dialog",
          "timestamp": 1734775200
        }
      }
    ]
  }'
```

### 3.3 Core Execution: `POST /search`
```bash
curl -sS "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: t1" \
  -d '{
    "query": "What does he like eating?",
    "topk": 10,
    "expand_graph": true,
    "graph_backend": "memory",
    "filters": {
      "user_id": ["u:1001"],
      "memory_domain": "dialog"
    }
  }'
```

---

## 4. Core Concepts & Data Contracts

### 4.1 MemoryEntry Standard
A `MemoryEntry` dictates atomic vector storage objects.
| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | `string` | No | Server assigns UUID if absent. |
| `kind` | `string` | Yes | `"episodic" \| "semantic"` |
| `modality` | `string` | Yes | `"text" \| "image" \| "audio" \| "structured"` |
| `contents` | `string[]` | Yes | Core payload injected to `contents[0]`. |
| `metadata` | `object` | Yes | Target routing filters (`user_id`, `run_id`, `memory_domain`, `timestamp`). |

### 4.2 SearchFilters Options
Standard JSON configuration injected into `filters` mapping blocks.
| Field | Type | Description |
| --- | --- | --- |
| `user_id` | `string[]` | Restrict search matching isolated identities. |
| `user_match` | `string` | `"any" \| "all"` |
| `memory_domain` | `string` | Workspace scopes (e.g. `dialog`, `project_alpha`). |
| `time_range` | `object` | `{gte: "start", lte: "end"}` constraints in unix or ISO logic. |
| `topic_path` | `string[]` | Node-tree explicit paths (`travel/japan`). |

---

## 5. API Reference (HTTP)

### 5.1 Telemetry (`GET /health`)
- Response: Status map checking `vectors` (Qdrant), `graph` (Neo4j), `llm_provider`, and `disk` buffers.
- Error Code Mappings: `API_KEY_MISSING`, `AUTH_FAILED`, `BALANCE_BELOW_THRESHOLD`.

### 5.2 L1: Dialog Ingest (`POST /ingest/dialog/v1`)
The safest method to process conversations recursively.

With the default self-hosted `.env.example` settings, API auth is disabled and callers must send `X-Tenant-ID`. In that mode, `user_tokens` can be omitted because the server derives a stable user token from the tenant boundary.

| Field | Type | Req | Description |
| --- | --- | --- | --- |
| `session_id` | `string` | Yes | Session lock guaranteeing identical message boundaries. |
| `user_tokens` | `string[]` | No | Optional user mappings; derived from tenant scope when omitted. |
| `memory_domain` | `string` | No | Partition workspace isolation. |
| `turns` | `object[]` | Yes | Array of `{turn_id, text/content, role(user/assistant/tool), timestamp_iso}`. |
| `commit_id` | `string` | No | Idempotent key blocking duplicate parallel execution. |
| `client_meta` | `object` | Yes | Must include at least `memory_policy` and `user_id`; can also carry model overrides. |

*Returns:* `{"ok": true, "job_id": "job_123", "status": "RECEIVED"}` (Job queues automatically).

### 5.3 Jobs State (`GET /ingest/jobs/{job_id}`)
Checks asynchronous ingestion tracks.
*Returns Status Enums:* `RECEIVED`, `STAGE2_RUNNING`, `STAGE2_FAILED`, `STAGE3_RUNNING`, `STAGE3_FAILED`, `COMPLETED`.

### 5.4 L1: Dialog Retrieval (`POST /retrieval/dialog/v2`)
Advanced orchestration for LLM inference (Hybrid Match + Graph Logic).

| Field | Type | Req | Description |
| --- | --- | --- | --- |
| `query` | `string` | Yes | Natural Language user prompt. |
| `user_tokens` | `string[]` | No | Optional user mappings; derived from tenant scope when omitted. |
| `with_answer` | `boolean` | No | Directs server to synthesize an LLM contextual answer array natively. |
| `topk` | `number` | No | Default `30`. |
| `client_meta` | `object` | Yes | Must include at least `memory_policy` and `user_id`; can also supply BYOK/provider metadata. |

*Returns Evidence Map:* Containing `tkg_event_id`, `score`, `text`, `_base_score` and optionally `tkg_explain` array chains dictating exactly where memory derivations occurred.

### 5.5 Core Search (`POST /search`)
Raw Vector + Graph logic query executing nearest-neighbor algorithms over Qdrant.
Includes `expand_graph=boolean` dictating if Neo4j neighborhoods should be loaded, and `threshold=number` to slice similarity metrics rigidly before Reranker pipelines execute.

### 5.6 Basic Write (`POST /write`, `POST /update`, `POST /delete`, `POST /link`)
Low-level operations bypassing semantic LLM generation pipelines. Requires explicit `entries: []`, `patch: {}`, or graph `links: [{src_id, dst_id, rel_type}]`.

### 5.7 Equivalencies (`/equiv/pending/*`)
Governs merges across separated identity mappings securely using approvals. Endpoints: `/equiv/pending/add`, `/confirm`, `/remove`.

### 5.8 Dynamic Configurations (`PATCH /config`)
Hot-reloads system heuristics without rebooting the memory core. Endpoints overlay `memory.search.rerank`, graph limits, and scope resolutions. Modifies `alpha_vector`, `beta_bm25`, `gamma_graph`, etc.

---

## 5.22 Semantic Memory (Memory v1 Beta)
*Deep entity extraction and parsing.*

### 5.22.1 / 5.22.2 Entity/Topic Queries
- **`GET /memory/v1/entities`**: Queries paginated Entity objects (`name`, `type`, `first_mentioned`, `mention_count`).
- **`GET /memory/v1/topics`**: Queries categorized hierarchical subjects running natively in `topic_path` structures.

### 5.22.5 Topic Timeline (`POST /memory/v1/topic-timeline`)
Validates a `{topic: "..."}` and generates chronological state changes formatting into a `timeline` JSON array holding `event_id`, `when`, and `summary` per entry.

### 5.22.6 Entity Profile (`POST /memory/v1/entity-profile`)
Pass `{"entity": "Alice"}` to extract 360-fields dictating:
1. `facts`: Raw immutable graph mappings.
2. `relations`: `co_occurs_with` relationship networks mapping to known contacts.
3. `recent_events`: Immediate temporal context hits.

### 5.22.7 Quotes (`POST /memory/v1/quotes`)
Fetches identical verbatim spoken records parsing `UtteranceEvidence` blocks natively mapped back to `speaker_id` identities to prevent hallucinated citations.

### 5.22.9 Time Since (`POST /memory/v1/time-since`)
Requests temporal drift metrics tracking exactly when an entity/topic was last observed. Calculates a direct `days_ago` float representing temporal decay.

### 5.22.10 Explain Chains (`POST /memory/v1/explain`)
Atomic tracing interface. Given an explicitly mapped `event_id="xyz"`, triggers a network dump querying interconnected `entities`, `places`, `evidences`, `utterances`, and `knowledge` tags defining the context strictly to frontend visualizers.

### 5.22.11 Clear Cache (`POST /memory/v1/clear`)
Total WIPE operation isolating against the targeted Tenant. 
Requires explicit `{"scope": "tenant", "reason": "wipe", "confirm": true}` payloads. If `confirm=false`, dry-runs generating estimated vector counts. 

---

## 5.23 State Memory Tracking (Phase 3)
Tracks shifting variable states against ISO limits.
- **`POST /memory/state/current`**: Ingests `subject_id` and `property="job"`. Returns `item: {value: "employed", valid_from: "2026-01-01"}`.
- **`POST /memory/state/changes`**: Retrieves all sequential history permutations mutating an object over time.
- **`POST /memory/state/pending/list`**: Queries pending non-approved state collisions.
- **`POST /memory/state/pending/approve`**: Human-In-The-Loop authorization pipeline.

---

## 5.24 Agentic Router Integrations
- **`GET /memory/agentic/tools?format=openai`**: Automatically compiles native Tool Schemas mapped over local Python endpoints directly to REST parsable formatting schemas.
- **`POST /memory/agentic/execute`**: Triggers explicit function maps without routing algorithms.
- **`POST /memory/agentic/query`**: Submit a `query="Natural phrase"`. LATRACE will internally prompt a secondary LLM Router, decide the exact `tool_used`, serialize `tool_args`, execute the matching system process asynchronously, and emit back a `ToolResult`!

---

## 6. Standard Error Matrices

| HTTP | Meaning | Developer Mitigation Protocol |
| --- | --- | --- |
| `201` | Async Job Created | Wait; Query `GET /ingest/jobs/{id}` for completion. |
| `400` | Payload Missing | Enforce valid JSON headers, `roles`, and required constraints. |
| `401` | Unauthorized | Refresh JWT or `X-API-Token`. Ensure HMAC signatures map correct TS. |
| `403` | Tenant Breach | Absolute blockage. `X-Tenant-ID` cross-contamination rejected. |
| `404` | Data Missing | Safely handle `null` arrays internally; object isolated inside network does not exist. |
| `409` | Conflict | Safely handle. `commit_id` matched an existing processing task. Skip. |
| `413` | Over-Capacity | Restrict JSON payloads below `10MB` bounds. |
| `429` | Depleted Ratelimits | Sleep natively using exponential random offsets, commonly `+60s`. |
| `503` | Reranker Subsystem Down | Subsystems temporarily isolated by active fault-breakers. Retry later. |
| `504` | Graph Timeout | Multi-hop derivation exhausted system capacity. Reduce `expand_graph` bounds. |

---

## 7. Best Practices: Writing Context (Dialog Ingress)

### Rules of Engagement
1. **Asynchronous Batches:** Submit entire dialogue sessions *once* they are finished or paused natively. Submitting on every single character generates unstable subgraph entities.
2. **Never Consolidate Text:** Supplying `{"role": "user", "text": "USER said X, and AI said Y"}` is architecturally forbidden. Separate each actor explicitly across index arrays so `UtteranceEvidence` extraction targets correctly.
3. **Commit Flags:** Pass highly stable `commit_id` hashes to `POST /ingest` algorithms natively to ensure that frontend network errors do not multiply entities on retry attempts. 
