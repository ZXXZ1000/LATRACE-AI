from __future__ import annotations

import asyncio
import os
from dotenv import load_dotenv

from modules.memory.application.service import MemoryService
from modules.memory.application.metrics import get_metrics
from modules.memory.application.config import load_memory_config
from modules.memory.adapters.mem0_adapter import build_entries_from_mem0
from modules.memory.adapters.m3_adapter import build_entries_from_m3
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application.decider_mem0 import build_mem0_decider_from_env


async def scenario_mem0_conflict(svc: MemoryService):
    print("\n[Scenario A] mem0 偏好冲突 + LLM 决策（如启用）")
    e1, _ = build_entries_from_mem0([
        {"role": "user", "content": "我喜欢奶酪披萨"},
    ], profile={"user_id": "user.owner"})
    await svc.write(e1)
    e2, _ = build_entries_from_mem0([
        {"role": "user", "content": "我不喜欢奶酪披萨"},
    ], profile={"user_id": "user.owner"})
    await svc.write(e2)
    res = await svc.search("奶酪 披萨 偏好", topk=5, expand_graph=True)
    print("Top hits:")
    for h in res.hits[:3]:
        print("-", h.entry.contents[0], "score=", round(h.score, 3))
    print("Hints:\n", res.hints)


async def scenario_control_episodic(svc: MemoryService):
    print("\n[Scenario B] Control episodic + executed 边 + UPDATE/ROLLBACK")
    device = MemoryEntry(kind="semantic", modality="structured", contents=["device.light.living_main"], metadata={"source": "ctrl", "entity_type": "device"})
    epi = MemoryEntry(kind="episodic", modality="text", contents=["用户通过控制器打开客厅主灯"], metadata={"source": "ctrl"})
    await svc.write([device, epi])
    await svc.link(device.id, epi.id, "executed", weight=1.0)
    res = await svc.search("用户 打开 客厅 主灯", topk=1, expand_graph=False)
    if res.hits:
        tgt = res.hits[0].id
        ver_upd = await svc.update(tgt, {"contents": ["用户打开客厅主灯（更新描述）"]}, reason="refine")
        ok = await svc.rollback_version(ver_upd.value)
        print("UPDATE:", ver_upd.value, "ROLLBACK:", ok)
    else:
        print("WARN: 未命中 episodic，跳过 UPDATE/ROLLBACK")
    print("METRICS:", get_metrics())


async def scenario_qdrant_filters(svc: MemoryService):
    print("\n[Scenario C] Qdrant 过滤：modality=text, kind=semantic, source=mem0")
    filters = SearchFilters(modality=["text"], memory_type=["semantic"], source=["mem0"])
    res = await svc.search("奶酪 披萨 偏好", topk=5, filters=filters, expand_graph=True)
    print("Filtered Top hits:")
    for h in res.hits:
        print("-", h.entry.contents[0], "score=", round(h.score, 3))
    print("Trace:", res.trace)


async def scenario_m3_edges(svc: MemoryService):
    print("\n[Scenario D] m3 关系构建与邻居扩展")
    parsed = {
        "clip_id": 3,
        "timestamp": "2025-09-23T12:00:00Z",
        "faces": ["<face_5>"],
        "voices": ["<voice_7>"],
        "episodic": ["<face_5> 在 <voice_7> 的请求下打开了灯"],
        "semantic": ["<face_5> 更倾向回应 <voice_7> 的请求"],
        "room": "room:living",
        "device": "device.light.living_main",
    }
    entries, edges = build_entries_from_m3(parsed)
    await svc.write(entries, links=edges)
    # pick first episodic id
    epi_ids = [e.id for e in entries if e.kind == "episodic" and e.id]
    if epi_ids:
        nbrs = await svc.graph.expand_neighbors(epi_ids[:1], rel_whitelist=["appears_in", "said_by", "located_in", "executed"], max_hops=1, neighbor_cap_per_seed=5)
        print("Neighbors for", epi_ids[0], ":", nbrs.get("neighbors", {}).get(epi_ids[0], []))
    else:
        print("WARN: 无 episodic 节点可展开")


async def main():
    load_dotenv()
    cfg_env = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    load_dotenv(os.path.abspath(cfg_env))
    cfg = load_memory_config()
    # Expand env vars in YAML-like config
    vc = cfg.get("memory", {}).get("vector_store", {})
    gc = cfg.get("memory", {}).get("graph_store", {})
    def _int_env(name: str, default: int) -> int:
        val = os.getenv(name)
        try:
            return int(val) if val and val.strip().isdigit() else default
        except Exception:
            return default

    def _int_var(v: object, default: int) -> int:
        try:
            s = str(v)
            return int(s) if s.isdigit() else default
        except Exception:
            return default

    host_default = str(vc.get("host", "127.0.0.1"))
    if host_default.strip().startswith("${"):
        host_default = "127.0.0.1"
    port_default = _int_var(vc.get("port", 6333), 6333)
    vcfg = {
        "host": os.getenv("QDRANT_HOST", host_default),
        "port": _int_env("QDRANT_PORT", port_default),
        "api_key": os.getenv("QDRANT_API_KEY", ""),
        "collections": vc.get("collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}),
        "embedding": vc.get("embedding", {}),
    }
    uri_default = str(gc.get("uri", "bolt://127.0.0.1:7687"))
    if uri_default.strip().startswith("${"):
        uri_default = "bolt://127.0.0.1:7687"
    user_default = str(gc.get("user", "neo4j"))
    if user_default.strip().startswith("${"):
        user_default = "neo4j"
    pass_default = str(gc.get("password", "password"))
    if pass_default.strip().startswith("${"):
        pass_default = "password"
    gcfg = {
        "uri": os.getenv("NEO4J_URI", uri_default),
        "user": os.getenv("NEO4J_USER", user_default),
        "password": os.getenv("NEO4J_PASSWORD", pass_default),
    }
    qdr = QdrantStore(vcfg)
    neo = Neo4jStore(gcfg)
    audit = AuditStore()
    svc = MemoryService(qdr, neo, audit)
    decider = build_mem0_decider_from_env()
    if decider:
        svc.set_update_decider(decider)

    await scenario_mem0_conflict(svc)
    await scenario_control_episodic(svc)
    await scenario_qdrant_filters(svc)
    await scenario_m3_edges(svc)


if __name__ == "__main__":
    asyncio.run(main())
