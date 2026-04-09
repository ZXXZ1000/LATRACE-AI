from __future__ import annotations

"""
Cycle11: 运行时热更新与指标导出体验脚本

前置：已启动 FastAPI 服务（api/server.py），默认端口例如 http://127.0.0.1:8000

功能：
- 写入两条样例语句
- 搜索并打印命中与邻域
- 通过 HTTP 热更新 rerank 权重 与 图关系白名单
- 再次搜索并对比
- 拉取 /metrics 与 /metrics_prom
"""

import asyncio
import os
import requests
from dotenv import load_dotenv

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore


API_BASE = os.getenv("MEMORY_API_BASE", "http://127.0.0.1:8000")


async def main() -> None:
    load_dotenv()
    # 直接走 Service 写入，便于拿到对象 id（也可改为 HTTP /write）
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

    a = MemoryEntry(kind="semantic", modality="text", contents=["我 喜欢 奶酪 披萨"], metadata={"source": "mem0"})
    b = MemoryEntry(kind="semantic", modality="text", contents=["我 不 喜欢 奶酪 披萨"], metadata={"source": "mem0"})
    await svc.write([a, b])

    # 初始搜索
    r0 = await svc.search("奶酪 披萨", topk=2, expand_graph=True)
    print("Initial hits:", [(h.entry.contents[0], h.score) for h in r0.hits])
    print("Initial neighbors keys:", list((r0.neighbors.get("neighbors") or {}).keys()))

    # 调整 rerank 权重（强调 BM25）
    payload = {"alpha_vector": 0.0, "beta_bm25": 1.0, "gamma_graph": 0.0, "delta_recency": 0.0}
    resp = requests.post(f"{API_BASE}/config/search/rerank", json=payload, timeout=5)
    print("Set rerank resp:", resp.status_code, resp.json())

    # 调整图白名单
    resp = requests.post(f"{API_BASE}/config/graph", json={"rel_whitelist": ["prefer"], "max_hops": 1, "neighbor_cap_per_seed": 5}, timeout=5)
    print("Set graph resp:", resp.status_code, resp.json())

    # 再次搜索
    r1 = await svc.search("奶酪 披萨", topk=2, expand_graph=True)
    print("After override hits:", [(h.entry.contents[0], h.score) for h in r1.hits])
    print("After override neighbors keys:", list((r1.neighbors.get("neighbors") or {}).keys()))

    # 拉取指标
    m_json = requests.get(f"{API_BASE}/metrics", timeout=5).json()
    print("/metrics:", m_json)
    m_text = requests.get(f"{API_BASE}/metrics_prom", timeout=5).text
    print("/metrics_prom:\n", m_text)


if __name__ == "__main__":
    asyncio.run(main())

