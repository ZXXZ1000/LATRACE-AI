from __future__ import annotations

"""Convert m3 VideoGraph pkl into unified MemoryEntry/Edge and write via MemoryPort.

根据样例 bedroom_12.pkl 的结构：
- 顶层类型：VideoGraph（对象），含属性：nodes(dict[int->Node])、edges(dict[tuple(int,int)->float]) ...
- Node：含属性 id:int, type:str in {episodic, semantic, voice, img}, embeddings:list, metadata:dict（含 contents/list, 可选 timestamp）
- edges：键为 (src_id, dst_id)，值为 float 权重；通常 voice↔episodic、img↔episodic、voice↔semantic、img↔semantic 成对出现（双向一对）。

映射策略：
- 节点 → MemoryEntry：
  - kind: episodic→"episodic"，其余（semantic/voice/img）→"semantic"
  - modality: episodic/semantic→"text"；voice→"audio"；img→"image"
  - contents: 从 node.metadata.contents（list[str]）取一条（或全部）
  - metadata: {source:"m3", timestamp?, vg_node_id:id}
- 关系 → Edge：
  - voice→episodic/semantic：rel_type="said_by"
  - img→episodic/semantic：rel_type="appears_in"
  - 去重：边通常双向成对，统一用 canonical 方向（voice/img 作为 src 指向 text 节点）并累加权重

写入：
- 默认使用配置的 Qdrant/Neo4j（从 memory.config.yaml + .env 读取）；
- 提供 --dry-run 与 --inmem（仅 InMem 写入用于本地快速验证）。
"""

import argparse
from typing import Any, Dict, List

from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.application.service import MemoryService
from modules.memory.application.config import load_memory_config
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
try:
    # Prefer unified mapper from Memorization Agent to ensure online/offline consistency
    from modules.memorization_agent.application.videograph_to_memory import VideoGraphMapper  # type: ignore
except Exception:
    VideoGraphMapper = None  # type: ignore


class Placeholder(object):
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class SafeUnpickler:
    @staticmethod
    def load(path: str) -> Any:
        import pickle

        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("builtins"):
                    return super().find_class(module, name)
                return type(name, (Placeholder,), {})

        with open(path, "rb") as f:
            return _U(f).load()


def build_service(use_inmem: bool) -> MemoryService:
    if use_inmem:
        return MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    cfg = load_memory_config()
    vcfg = cfg.get("memory", {}).get("vector_store", {})
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    qdr = QdrantStore({
        "host": vcfg.get("host", "127.0.0.1"),
        "port": int(vcfg.get("port", 6333)),
        "api_key": vcfg.get("api_key", ""),
        "collections": vcfg.get("collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}),
        "embedding": vcfg.get("embedding", {}),
    })
    neo = Neo4jStore({
        "uri": gcfg.get("uri", "bolt://127.0.0.1:7687"),
        "user": gcfg.get("user", "neo4j"),
        "password": gcfg.get("password", "password"),
    })
    return MemoryService(qdr, neo, AuditStore())


def run(
    input_pkl: str,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = 1000,
    use_inmem: bool = False,
    user_id: str | None = None,
    memory_domain: str | None = None,
    run_id: str | None = None,
) -> Dict[str, int]:
    vg = SafeUnpickler.load(input_pkl)
    # Try unified mapper first
    entries: List[MemoryEntry] = []
    rels: List[Edge] = []
    if VideoGraphMapper is not None:
        defaults: Dict[str, Any] = {}
        if user_id:
            defaults["user_id"] = [user_id]
        if memory_domain:
            defaults["memory_domain"] = memory_domain
        if run_id:
            defaults["run_id"] = run_id
        mapper = VideoGraphMapper()
        try:
            entries, rels = mapper.map(vg, defaults=defaults)
            if limit is not None:
                entries = entries[:limit]
        except Exception:
            entries = []
            rels = []
    # 统一映射失败：不再提供旧版回退，直接告警退出
    if not entries:
        raise RuntimeError(
            "VideoGraphMapper mapping returned no entries. Legacy ETL fallback has been removed. "
            "Please ensure VideoGraph contains valid nodes and the mapper is available."
        )

    if dry_run:
        print(f"[DRY-RUN] entries={len(entries)} relations={len(rels)}")
        return {"entries": len(entries), "relations": len(rels)}

    svc = build_service(use_inmem)
    # simple batching for large imports
    written = 0
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i : i + batch_size]
        # 仅在第一批携带全部关系，避免重复（简化处理）
        batch_links = rels if i == 0 else None
        try:
            import asyncio

            async def _one():
                return await svc.write(batch_entries, batch_links, upsert=True)

            asyncio.run(_one())
        except RuntimeError:
            # in case of nested loop
            import asyncio as _a

            _a.get_event_loop().run_until_complete(svc.write(batch_entries, batch_links, upsert=True))
        written += len(batch_entries)
    print(f"Imported entries={written} relations={len(rels)}")
    # 若 mapper 可用，尝试建立 equivalence 候选（仅在明确确认时执行）
    try:
        if VideoGraphMapper is not None and (user_id or memory_domain or run_id):
            defaults = {}
            if user_id:
                defaults["user_id"] = [user_id]
            if memory_domain:
                defaults["memory_domain"] = memory_domain
            if run_id:
                defaults["run_id"] = run_id
            mapper = VideoGraphMapper()
            # 再次调用以生成候选对（这一步无需写入，只为提取 eq）
            mapper.map(vg, defaults=defaults)
            eq_pairs = mapper.extract_equivalence_candidates(vg)
            if eq_pairs:
                print(f"[INFO] Found {len(eq_pairs)} equivalence candidates (face/voice->character). Use API /link with confirm True to establish when appropriate.")
    except Exception:
        pass
    return {"entries": written, "relations": len(rels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to VideoGraph pkl")
    parser.add_argument("--dry-run", action="store_true", help="Only parse and report counts")
    parser.add_argument("--limit", type=int, help="Limit number of nodes to import")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for write()")
    parser.add_argument("--inmem", action="store_true", help="Use InMem stores for quick validation")
    parser.add_argument("--user-id", help="Default user_id for three-key injection")
    parser.add_argument("--memory-domain", help="Default memory_domain for three-key injection")
    parser.add_argument("--run-id", help="Default run_id for three-key injection")
    args = parser.parse_args()
    run(
        args.input,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
        use_inmem=args.inmem,
        user_id=args.user_id,
        memory_domain=args.memory_domain,
        run_id=args.run_id,
    )
