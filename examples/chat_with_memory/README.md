# Chat-with-Memory Demo / 带记忆的对话 Demo

A self-contained FastAPI application that demonstrates LATRACE's three core memory capabilities in a single browser window.

一个自包含的 FastAPI 应用，在单个浏览器窗口里完整展示 LATRACE 的三大核心记忆能力。

## What it demonstrates / 演示内容

| Feature | Endpoint used | Description |
|---------|--------------|-------------|
| Memory ingestion | `POST /ingest/dialog/v1` | Conversation turns are batched and submitted for async LLM extraction into Qdrant (vector) + Neo4j (TKG graph) |
| Memory retrieval | `POST /retrieval/dialog/v2` | Before every reply, relevant memories are retrieved and injected into the system prompt |
| TKG graph view | Neo4j HTTP API | Interactive D3.js force-directed graph showing Event / Entity / Knowledge nodes and their relationships |

## Prerequisites / 前置条件

1. **LATRACE Memory API** running at `http://localhost:8000`

   ```bash
   # Quickstart with Docker Compose (recommended)
   docker compose up

   # Or start only the databases and run the API locally
   docker compose up qdrant neo4j
   uv run uvicorn modules.memory.api.server:app --port 8000
   ```

2. **An OpenAI-compatible LLM API key** — used for chat replies and memory extraction

3. **Neo4j** at `bolt://localhost:7687` — required for the `/graph` visualisation page (already started via Docker Compose)

## Configuration / 配置

All settings are read from environment variables. Copy `.env.example` from the repo root, fill in your values, then source it:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | *(required)* | Your LLM API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `MEMORY_API_URL` | `http://localhost:8000` | LATRACE API base URL |
| `NEO4J_PASSWORD` | `neo4j_password` | Neo4j password (matches `.env`) |
| `DEMO_TENANT_ID` | `chat-demo` | Multi-tenant isolation key |
| `DEMO_USER_ID` | `demo-user-001` | User identity for memory scoping |
| `DEMO_PORT` | `7860` | Port for this demo app |

## Quickstart / 快速启动

```bash
# 1. Start the LATRACE stack
docker compose up -d

# 2. Install demo dependencies (already in the main pyproject.toml)
uv sync

# 3. Set your LLM key
export LLM_API_KEY=sk-...
export LLM_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
export LLM_MODEL=gpt-4o-mini

# 4. Run the demo
uv run python examples/chat_with_memory/chat_with_memory.py
```

Then open:

- **Chat window** → `http://localhost:7860`
- **Memory graph** → `http://localhost:7860/graph`

## How memory works / 记忆工作原理

```
User message
     │
     ▼
POST /retrieval/dialog/v2  ──► inject relevant memories into system prompt
     │
     ▼
LLM reply (streaming)
     │
     ▼
Buffer turns (every 2 rounds = 4 turns)
     │
     ▼
POST /ingest/dialog/v1  ──► async pipeline
                                ├── Stage 2: dedup / normalise
                                └── Stage 3: LLM extraction (~30–120 s)
                                          ├── Qdrant memory_text (vector search)
                                          └── Neo4j TKG (graph: Event/Entity/Knowledge)
```

The `/graph` page reads the Neo4j graph directly via its HTTP API and renders it with D3.js. Nodes are coloured by type:

- **Blue** — `Event` (semantic events extracted from conversations)
- **Green** — `Entity` (persons, objects, places)
- **Orange** — `Knowledge` (facts and beliefs)
- **Gray** — `UtteranceEvidence` (raw conversation evidence)
- **Teal** — `MediaSegment` (dialog segments)
- **Purple** — `MemoryNode` (legacy memory entries)

## Demo walkthrough / 演示步骤

1. Open `http://localhost:7860` and send a few messages about yourself (hobbies, plans, preferences).
2. Click **💾 保存** to manually flush the current turns, or wait for the automatic 2-round flush.
3. Wait ~30–60 seconds for Stage 3 LLM extraction to finish.
4. Ask a follow-up question — you should see the memory tag showing retrieved memories.
5. Open `http://localhost:7860/graph` to see the TKG graph grow in real time.
6. Click any node to inspect its properties in the detail panel.
