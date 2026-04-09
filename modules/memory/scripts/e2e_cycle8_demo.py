from __future__ import annotations

import asyncio
import os

from modules.memory.application.service import MemoryService
from modules.memory.adapters.mem0_adapter import build_entries_from_mem0
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application.decider_mem0 import build_mem0_decider_from_env
from dotenv import load_dotenv


async def main() -> None:
    # Load env from repo config if present
    load_dotenv()
    cfg_env = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    load_dotenv(os.path.abspath(cfg_env))
    qdr = QdrantStore({
        "host": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"},
    })
    neo = Neo4jStore({
        "uri": os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
    })
    audit = AuditStore()
    svc = MemoryService(qdr, neo, audit)
    decider = build_mem0_decider_from_env()
    if decider is not None:
        print("LLM decider: enabled from env provider")
        svc.set_update_decider(decider)
    else:
        print("LLM decider: not configured, using heuristic fallback")

    # Build a simple mem0 semantic preference
    messages = [{"role": "user", "content": "我喜欢晚上在客厅看电影"}]
    entries, edges = build_entries_from_mem0(messages, profile={"user_id": "user.owner"})

    # Write to stores
    await svc.write(entries, links=edges)

    # Search via Qdrant (text-only MVP)
    res = await svc.search("我 喜欢 客厅 电影", topk=3, expand_graph=False)
    print("Search hits:")
    for h in res.hits:
        print("-", h.entry.contents[0], "score=", h.score)


if __name__ == "__main__":
    asyncio.run(main())
