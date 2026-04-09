# LATRACE ADK Integration Guide

**Version**: `v2.4`
**Applicability**: `LATRACE/modules/memory/adk` (Layer 1 ADK Tools)

The Agent Development Kit (ADK) governs the Python execution layer of the LATRACE engine. Its objective is to allow AI Agents (LangChain, AutoGen, direct OpenAI SDKs) to natively execute memory functions without writing low-level HTTP calls or complex vector search logic.

This document clearly distinguishes between the abstract internal REST APIs and the concrete Python tools exposed to the Agent layer.

---

## 1. Core Tooling Cards (The Agentic Python Library)

The ADK abstracts complex backend multi-hop retrievals into unified Python functions. These represent the entire dictionary of interactions you should expose to your LLM prompt.

### 1.1 Factual Memory Tools
- **`entity_profile(entity: str)`**
  * **Objective:** Extract the unified, 360-degree biography of an entity. Heavily utilized upon session launch.
  * **Expected Input:** `{"entity": "Alice"}`

- **`topic_timeline(topic: str)`**
  * **Objective:** Creates structured historical sequence lines on specific topics. Lightweight baseline arrays, enabling optional deeper expansions upon LLM request.
  * **Expected Input:** `{"topic": "Project Alpha"}`

- **`time_since(entity: str, topic: str)`**
  * **Objective:** Provides the explicit timestamp and lapsed duration metric since a subject was last brought up. (e.g. "I haven't talked to Alice in 3 months")
  * **Expected Input:** `{"entity": "Alice", "topic": "Project Alpha"}`

- **`quotes(entity: str, limit: int)`**
  * **Objective:** Strict literal citation fetching. Bypasses generation pipelines for perfect attribution. Errs or aborts under ambiguity to prevent hallucinations.
  * **Expected Input:** `{"entity": "Alice", "limit": 3}`

- **`relations(entity: str)`**
  * **Objective:** Deduces interpersonal or semantic graph bonds between entities (Note: 'Found' relations explicitly differs from purely 'Matched' text keywords).
  * **Expected Input:** `{"entity": "Alice"}`

- **`explain(event_id: str)`**
  * **Objective:** Atomic tool restricted from broad searching. Deduces exact evidence provenance arrays for a constrained ID. **Do not expose to LLM search tasks** unless they already hold a valid `event_id`.
  * **Expected Input:** `{"event_id": "evt_abc123"}`

- **`list_entities()` / `list_topics()`**
  * **Objective:** Exploratory browsing operations without deep fetching. Discovery mechanism.

### 1.2 State Analysis Tools
State schemas fetch dynamic properties tracking attributes mutating across timelines (e.g., job titles, addresses, emotional dispositions).
- **`entity_status(entity: str, property: str)`**
  * **Objective:** Queries current precise status or targets an exact lookup block against an ISO threshold.
  * **Expected Input:** `{"entity": "Alice", "property": "current_work_status"}`
- **`status_changes(entity: str, property: str, limit: int)`**
  * **Objective:** Emits an array payload documenting every historical instance an attribute drifted or updated.
  * **Expected Input:** `{"entity": "Alice", "property": "current_work_status", "limit": 10}`
- **`state_time_since(entity: str, property: str)`**
  * **Objective:** Returns strict durations on how long an attribute has remained frozen.
  * **Expected Input:** `{"entity": "Alice", "property": "current_work_status"}`

---

## 2. Advanced Tool Orchestration

### 2.1 Generating OpenAI/MCP Supported Tool Schemas
To ensure LLMs strictly follow typing for the exact tools above, ADK generates OpenAI JSON-schemas automatically at compile time.

```python
from modules.memory.adk.runtime import create_memory_runtime

async def main():
    async with create_memory_runtime(
        base_url="http://127.0.0.1:8000",
        tenant_id="primary_tenant",
        user_tokens=["u:user_11"],
    ) as runtime:
        
        # Selectively expose 5 critical queries to the active LLM context
        tools = runtime.get_openai_tools(
            names=["entity_profile", "topic_timeline", "time_since", "quotes", "relations"]
        )
        # You hand `tools` array directly to standard OpenAI / Chat Completions objects!
```

### 2.2 The Payload Format (`ToolResult`)

Every execution of a tool listed in **1.1** yields a predictable `ToolResult` interface, guaranteeing LLMs don't parse verbose server backend traces.

* **`matched`** (bool): Were valid DB records detected? Provide answers purely on data.
* **`needs_disambiguation`** (bool): Toggles True if the string parsed corresponds to multiple unrelated entities.
* **`message`** (str): Built-in system text prompting the LLM agent to disambiguate the request further.
* **`data`** (dict): Contains the executed Tool arrays.

### 2.3 Layer 2 Direct Invocation (Server Execution Routing)
If developers do not want to configure their own internal Python LangChain/AutoGen environment to run standard function calls, they can execute pure language against our Layer 2 HTTP Tooling endpoints. 

LATRACE parses the natural text, identifies the tool internally, and returns the payload.

```bash
curl -X POST "http://127.0.0.1:8000/memory/agentic/query" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant_demo" \
  -d '{
    "query": "What has Zhang San been busy with recently?",
    "tool_whitelist": ["entity_profile", "time_since", "quotes"],
    "include_debug": false
  }'
```

---

## 3. Strict Development Prompts (System Role Play Constraints)
If orchestrating ADK internally, paste this constraint into your Agent's root System Prompt to avoid severe logic errors:
1. `explain` is NOT a search tool; you MUST use `entity_profile` first to capture an `event_id`.
2. Do not hallucinate or supplement data when `matched=false`. Simply respond the memory does not exist.
3. If `needs_disambiguation=true`, DO NOT guess the entity structure; halt the query and clarify the entity cluster directly with the user.



# 4. ADK Tool Cards (System Prompt & Developer Reference)

> **Baseline**: Tool schemas and parameter constraints are dictated natively by `modules/memory/adk/tool_definitions.py`.

---

## 4.1. System Prompt (Clean Template)

> **Scope**: Layer 1 (Business Agents independently orchestrating tool-calling).  
> If you are invoking the Layer 2 server-side routing API (`/memory/agentic/query`), see section **4.9** instead.

```text
You are a "Memory Retrieval Agent". You possess a suite of memory tools to extract structured facts from historical dialogues and knowledge graphs.

Your objectives:
- Prioritize invoking tools if the user's inquiry relies on "what happened in the past, who said what, or what are the relations/states/changes".
- Do not forcibly invoke tools if the query does not depend on historical memory.

[Tool Invocation Protocol]
1) Invoke exactly 1 tool per turn; if true multi-step reasoning is required, execute serially (acquire results of step 1 to decide step 2).
2) Upon initiating a tool call, output exactly:
   - function.name 
   - function.arguments (strict JSON parameter mapping)
3) Strictly adhere ONLY to defined parameter fields. Do not hallucinate fields.
4) Dissect the tool response relying strictly onto these 4 output fields:
   - matched
   - needs_disambiguation
   - message
   - data
5) NEVER expose debug/trace metrics to the user.

[Result Handling Rules]
- matched=true: Hits were found. Synthesize an answer based purely on `data`.
- matched=false AND needs_disambiguation=false: State truthfully that "No related records were found currently in memory." Do not hallucinate.
- needs_disambiguation=true: You must clarify the target entity candidate with the user first. DO NOT guess.
- message != null: Absorb the system hint heavily (e.g. strict AND semantic reminders).

[Tool Selection Dictionary]
- Inquiring about a person/object's broad facts: entity_profile
- Inquiring about topic developments/history: topic_timeline
- Inquiring "when was the last time we mentioned XYZ": time_since
- Inquiring "what was the exact original quote": quotes
- Inquiring "who is connected to who": relations
- Inquiring "what is the base evidence/proof": explain (Prerequisite: You MUST possess a valid event_id)
- Inquiring "what is the current/past status": entity_status
- Inquiring "how did this status change": status_changes
- Inquiring "how long has this status been stagnant": state_time_since
- Inquiring "browse available entities/topics": list_entities / list_topics

[Critical Guardrails]
- `explain` is NOT a search tool. It ONLY accepts `event_id`.
- If you lack an `event_id`, use `entity_profile` / `topic_timeline` / `quotes` to locate the event FIRST, then chain into `explain`.
- `matched=false` is NOT a system error. Do not invent memory out of thin air.
```

---

## 4.2. Developer Integration Examples

### 4.2.1 OpenAI Function Calling Format

**Model returns a tool call (Example):**

```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "entity_profile",
    "arguments": "{\"entity\":\"Alice\",\"include\":[\"facts\",\"events\"],\"limit\":5}"
  }
}
```

**Return the payload to the LLM mapped as `role=tool` (Example):**

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"matched\":true,\"needs_disambiguation\":false,\"message\":null,\"data\":{...}}"
}
```

Constraints:
- You MUST `json.loads` `function.arguments` prior to execution.
- We aggressively recommend returning purely `ToolResult.to_llm_dict()` (the strict 4 field payload).
- Keep debug parameters in your systemic logs; NEVER reinject them into the LLM context.

---

## 4.3. Tool Matrix (Minimum Invocation References)

| Tool Name | Minimal Parameter Required | JSON Argument Example |
|---|---|---|
| `entity_profile` | `entity` OR `entity_id` | `{"entity":"Alice"}` |
| `topic_timeline` | `topic` / `topic_id` / `topic_path` / `keywords` | `{"topic":"Project Alpha"}` |
| `time_since` | `entity` OR `topic` (combining targets applies AND logic) | `{"entity":"Alice","topic":"Project Alpha"}` |
| `quotes` | `entity` OR `topic` | `{"entity":"Alice","limit":3}` |
| `relations` | `entity` OR `entity_id` | `{"entity":"Alice"}` |
| `explain` | `event_id` | `{"event_id":"evt_abc123"}` |
| `entity_status` | `entity+property` OR `entity_id+property_canonical` | `{"entity":"Bob","property":"current_work_status"}` |
| `status_changes` | `entity+property` (can bind `time_range`) | `{"entity":"Bob","property":"current_work_status","limit":10}` |
| `state_time_since` | `entity+property` | `{"entity":"Bob","property":"current_work_status"}` |
| `list_entities` | Optional string | `{"query":"Al","limit":20}` |
| `list_topics` | Optional string | `{"query":"Project","limit":20}` |

**Hard Limitations:**
- `quotes.limit`: `1..10`
- `topic_timeline.include`: Permitted Enums strictly `quotes` / `entities`
- `relations.relation_type`: Currently enforced to strictly `co_occurs_with`

---

## 4.4. Multi-Step Chaining Strategy

### The `Explain` Two-Step Protocol
1. Query `entity_profile` / `topic_timeline` / `quotes` to extract a specific `event_id`.
2. Sequentially query `explain({"event_id":"..."})`.

**DO NOT ATTEMPT to feed natural language queries natively into `explain`.**

### Proper Disambiguation Protocol
If a query returns `needs_disambiguation=true` AND `data.candidates` evaluates positively, DO NOT attempt to query again sequentially. Pause execution, clarify the target object naturally with the end user, and fire the tool again using the strictly delineated `entity_id` mapped from the candidate array.

---

## 4.5. Complete OpenAI Python Integration (End-To-End)

```python
import json
from openai import AsyncOpenAI
from modules.memory.adk import ToolResult, create_memory_runtime

async def query_alice_pipeline():
    # 1. Start the Secure Runtime
    runtime = create_memory_runtime(
        base_url="http://127.0.0.1:8000",
        tenant_id="tenant_demo",
        user_tokens=["u:alice"],
    )

    # 2. Extract tools and map executors securely
    tools = runtime.get_openai_tools()
    TOOL_EXECUTORS = {
        "entity_profile": runtime.entity_profile,
        "topic_timeline": runtime.topic_timeline,
        # ... map all active bindings
    }

    client = AsyncOpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Prompt array from section 4.1
        {"role": "user", "content": "What has Zhang San been busy with?"},
    ]

    # REQUEST #1: Model decides the tool routing
    resp1 = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg1 = resp1.choices[0].message
    messages.append(msg1.model_dump(exclude_none=True))

    # INTERCEPT: Execute matching local ADK tools safely
    for tc in (msg1.tool_calls or []):
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        fn = TOOL_EXECUTORS.get(name)
        
        result = await fn(**args) 
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result.to_llm_dict(), ensure_ascii=False),
        })

    # REQUEST #2: Generation of final answer synthesizing ToolResult content
    resp2 = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )
    return resp2.choices[0].message.content
```

---

## 4.6. Common Anti-Patterns (Avoid These)
1. Misidentifying `explain` as an unrestricted search operator vector.
2. Continuing to blind-query memory despite hitting `needs_disambiguation=true`.
3. Over-feeding HTTP payloads! Stop returning `.to_wire_dict()` arrays natively into LLMs.
4. Erroneously attributing `matched=false` to catastrophic routing failures (it's completely acceptable for memory simply not to exist yet).

---

## 4.7. Layer 2 Direct Bypass 

If creating custom LLM execution loops like `4.5` is overwhelming, execute queries completely naturally directly against the backend. LATRACE implements an internal independent router via `/memory/agentic/query`, bypassing user Agent orchestration.

```bash
curl -X POST "http://127.0.0.1:8000/memory/agentic/query" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant_demo" \
  -d '{
    "query": "What has Zhang San been busy with recently?",
    "tool_whitelist": ["entity_profile","time_since","quotes"],
    "include_debug": false
  }'
```

Returns:
- `tool_used`: The specific isolated operation executing underneath the hood.
- `tool_args`: Raw parsed payload.
- `result`: The exact clean `ToolResult` interface dictionary.

*Note: This specific internal router evaluates solely on single-tool modes natively. It does not iteratively chain tools internally without explicit instruction.*
