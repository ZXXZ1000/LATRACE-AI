from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from modules.memory.application.service import MemoryService
from modules.memory.adapters.mem0_adapter import build_entries_from_mem0
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application.decider_mem0 import build_mem0_decider_from_env
from modules.memory.contracts.memory_models import SearchFilters


async def main() -> None:
    # load env
    load_dotenv()
    cfg_env = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    load_dotenv(os.path.abspath(cfg_env))

    qdr = QdrantStore({
        "host": os.getenv("QDRANT_HOST", "127.0.0.1"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"},
        "embedding": {
            "provider": os.getenv("EMBEDDING_PROVIDER", ""),
            "model": os.getenv("EMBEDDING_MODEL", ""),
            "dim": int(os.getenv("EMBEDDING_DIM", "1536")),
        },
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

    # 1) 写入初始偏好（ADD）
    entries, edges = build_entries_from_mem0([
        {"role": "user", "content": "我喜欢奶酪披萨"}
    ], profile={"user_id": "user.owner"})
    await svc.write(entries, links=edges)

    # 2) 写入冲突事实（期望 LLM 决策 UPDATE/DELETE/NONE）
    entries2, _ = build_entries_from_mem0([
        {"role": "user", "content": "我不喜欢奶酪披萨"}
    ], profile={"user_id": "user.owner"})
    await svc.write(entries2)

    # 3) 搜索（带 filters），观察命中与 hints
    # Unfiltered search
    res = await svc.search("奶酪 披萨 偏好", topk=5, expand_graph=True)
    print("Search hits (unfiltered):")
    for h in res.hits:
        print(f"- {h.entry.contents[0]} | score={h.score:.3f} | v={getattr(h, 'v', '?')} g={getattr(h, 'g', '?')}")
    print("Hints:\n" + res.hints)
    print("Trace:", res.trace)

    # Filtered search
    filters = SearchFilters(modality=["text"], memory_type=["semantic"], source=["mem0"])
    res2 = await svc.search("奶酪 披萨 偏好", topk=5, filters=filters, expand_graph=True)
    print("Search hits (filtered: modality=text, kind=semantic, source=mem0):")
    for h in res2.hits:
        print(f"- {h.entry.contents[0]} | score={h.score:.3f}")
    print("Trace2:", res2.trace)


if __name__ == "__main__":
    asyncio.run(main())
