from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application.metrics import get_metrics


async def main() -> None:
    load_dotenv()
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

    # 1) control episodic + executed 边
    device = MemoryEntry(kind="semantic", modality="structured", contents=["device.light.living_main"], metadata={"source": "ctrl", "entity_type": "device"})
    epi = MemoryEntry(kind="episodic", modality="text", contents=["用户通过控制器打开客厅主灯"], metadata={"source": "ctrl"})
    edges = []
    # ids will be assigned in write; create link with later service.link
    ver = await svc.write([device, epi], links=None)
    print("WRITE version:", ver.value)
    # fetch ids from vector store dump if available (for demo we assume QdrantStore doesn't expose)
    # so we link by calling link after write using the newly stored ids in entries (ids are set in MemoryService)
    try:
        cache = getattr(svc, "_cache", {})
        print("DEBUG cache size:", len(cache), "epi_id:", epi.id)
        print("DEBUG cache keys sample:", list(cache.keys())[:5])
        print("DEBUG cache has epi:", epi.id in cache)
    except Exception:
        pass
    edges.append(Edge(src_id=device.id, dst_id=epi.id, rel_type="executed", weight=1.0))
    await svc.link(device.id, epi.id, "executed", weight=1.0)
    print("LINK executed added")

    # 2) UPDATE episodic 内容（演示审计/回滚）
    # 2) 通过搜索拿到实际存储使用的 id（若被合并则与原 id 不同）
    res = await svc.search("用户 打开 客厅 主灯", topk=1, expand_graph=False)
    actual_id = res.hits[0].id if res.hits else epi.id
    # 3) UPDATE episodic 内容（演示审计/回滚）
    upd = await svc.update(actual_id, {"contents": ["用户打开客厅主灯（更新描述）"]}, reason="refine text")
    print("UPDATE version:", upd.value)
    try:
        evt = audit.get_event(upd.value)
        print("UPDATE event payload keys:", list((evt or {}).get("payload", {}).keys()) if evt else None)
    except Exception:
        pass
    # 3) DELETE 软删（演示回滚）
    # 回滚 UPDATE（演示回滚成功）
    ok_upd = await svc.rollback_version(upd.value)
    print("ROLLBACK update ok=", ok_upd)

    # Metrics
    print("METRICS:", get_metrics())


if __name__ == "__main__":
    asyncio.run(main())
