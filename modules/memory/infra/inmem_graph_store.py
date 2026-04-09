from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from modules.memory.contracts.memory_models import MemoryEntry, Edge


@dataclass(frozen=True)
class _EdgeKey:
    src: str
    dst: str
    rel: str


class InMemGraphStore:
    """A minimal in-memory graph-store facade for testing and development.

    - Stores nodes (MemoryEntry) by id
    - Stores relations keyed by (src, dst, rel_type)
    """

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        self._nodes: Dict[str, MemoryEntry] = {}
        self._edges: Dict[_EdgeKey, Dict[str, Any]] = {}

    async def merge_nodes_edges(self, entries: List[MemoryEntry], edges: Optional[List[Edge]] = None) -> None:
        for e in entries:
            if not e.id:
                continue
            self._nodes[e.id] = e.model_copy(deep=True)
        if edges:
            for ed in edges:
                key = _EdgeKey(ed.src_id, ed.dst_id, ed.rel_type)
                if key in self._edges:
                    # accumulate weight (reinforce)
                    prev = self._edges[key].get("weight") or 0.0
                    add = ed.weight if ed.weight is not None else 1.0
                    self._edges[key]["weight"] = float(prev) + float(add)
                else:
                    self._edges[key] = {"weight": ed.weight if ed.weight is not None else 1.0}

    async def merge_rel(self, src_id: str, dst_id: str, rel_type: str, *, weight: Optional[float] = None) -> None:
        key = _EdgeKey(src_id, dst_id, rel_type)
        if key in self._edges:
            prev = self._edges[key].get("weight") or 0.0
            add = weight if weight is not None else 1.0
            self._edges[key]["weight"] = float(prev) + float(add)
        else:
            self._edges[key] = {"weight": weight if weight is not None else 1.0}

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok", "nodes": len(self._nodes), "edges": len(self._edges)}

    # Testing helpers
    def dump_nodes(self) -> Dict[str, MemoryEntry]:
        return dict(self._nodes)

    def _edge_rels_lower(self) -> Dict[_EdgeKey, Dict[str, Any]]:
        out: Dict[_EdgeKey, Dict[str, Any]] = {}
        for k, v in self._edges.items():
            vv = dict(v)
            vv["orig_rel"] = vv.get("orig_rel", k.rel)
            out[_EdgeKey(k.src, k.dst, k.rel.lower())] = vv
        return out

    def dump_edges(self) -> List[Tuple[str, str, str, Optional[float]]]:
        out = []
        for k, v in self._edges.items():
            out.append((k.src, k.dst, k.rel, v.get("weight")))
        return out

    # CRUD helpers for service
    async def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        # remove incident edges
        to_del = [k for k in self._edges if k.src == node_id or k.dst == node_id]
        for k in to_del:
            self._edges.pop(k, None)

    async def count_tenant_nodes(self, tenant_id: str) -> int:
        total = 0
        for node in self._nodes.values():
            metadata = dict(node.metadata or {})
            if str(metadata.get("tenant_id") or "") == str(tenant_id):
                total += 1
        return total

    async def purge_tenant(self, tenant_id: str) -> int:
        tenant = str(tenant_id or "")
        to_delete = [
            node_id
            for node_id, node in self._nodes.items()
            if str((node.metadata or {}).get("tenant_id") or "") == tenant
        ]
        for node_id in to_delete:
            await self.delete_node(node_id)
        return len(to_delete)

    async def purge_source_except_events(
        self,
        *,
        tenant_id: str,
        source_id: str,
        keep_event_ids: List[str],
    ) -> Dict[str, int]:
        # In-memory store does not track source_id/event ids; return no-op counts.
        return {"events": 0, "knowledge": 0}

    # Testing helpers
    def get_edge_weight(self, src_id: str, dst_id: str, rel_type: str) -> Optional[float]:
        key = _EdgeKey(src_id, dst_id, rel_type)
        if key in self._edges:
            return self._edges[key].get("weight")
        return None

    async def expand_neighbors(
        self,
        seed_ids: List[str],
        *,
        rel_whitelist: Optional[List[str]] = None,
        max_hops: int = 1,
        neighbor_cap_per_seed: int = 5,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        restrict_to_user: bool = True,
        restrict_to_domain: bool = True,
        memory_scope: Optional[str] = None,
        restrict_to_scope: bool = True,
    ) -> Dict[str, Any]:
        """Return neighbors per seed up to max_hops (BFS), with simple cap.

        语义约定（需要与 Neo4jStore.expand_neighbors 行为保持一致）：
        - hop=1: 双向邻居（src/dst 都视为可达）
        - hop>1: 在 BFS frontier 上继续展开，同时避免重复访问
        - 对同一 target 节点，如存在多条边，仅保留 weight 最大的一条，并记录最小 hop
        - 最终 neighbors 列表按 weight 降序排序（等权重时保持插入顺序），然后截断到 neighbor_cap_per_seed

        任何针对权重/排序的行为变更，都必须同步更新 Neo4j 侧实现与相关测试，
        否则 InMem 与真实后端之间的推理结果将出现偏差。
        """
        rels = {r.lower() for r in (rel_whitelist or [])}
        edges = self._edge_rels_lower()
        summary: Dict[str, Any] = {"neighbors": {}, "edges": []}
        if max_hops <= 0:
            return summary
        for sid in seed_ids:
            seen = set([sid])
            frontier = [sid]
            hop = 0
            scores: Dict[str, Dict[str, Any]] = {}
            while frontier and hop < max_hops:
                nxt: List[str] = []
                for nid in frontier:
                    # gather hop-1 neighbors
                    for k, v in edges.items():
                        if rels and k.rel not in rels:
                            continue
                        def _acc(target: str, rel: str, w: float) -> None:
                            if target == sid:
                                return
                            tgt = self._nodes.get(target)
                            if tgt is None or tgt.published is False:
                                return
                            # apply user/domain restrictions if requested
                            if restrict_to_user and user_ids:
                                e_users_raw = tgt.metadata.get("user_id")
                                if e_users_raw is None:
                                    return
                                if isinstance(e_users_raw, list):
                                    e_users = set(str(x) for x in e_users_raw)
                                else:
                                    e_users = {str(e_users_raw)}
                                if not e_users.intersection(set(str(x) for x in user_ids)):
                                    return
                            if restrict_to_domain and memory_domain is not None:
                                tgt = self._nodes.get(target)
                                if tgt is None or str(tgt.metadata.get("memory_domain")) != str(memory_domain):
                                    return
                            prev = scores.get(target)
                            # keep strongest edge and minimal hop seen for target
                            if prev is None or w > float(prev.get("weight", 0.0)):
                                scores[target] = {"to": target, "rel": v.get("orig_rel", rel), "weight": w, "hop": hop + 1}
                            elif prev is not None:
                                # update minimal hop if smaller
                                try:
                                    prev_hop = int(prev.get("hop", hop + 1))
                                    scores[target]["hop"] = min(prev_hop, hop + 1)
                                except Exception:
                                    scores[target]["hop"] = hop + 1
                            if target not in seen:
                                nxt.append(target)
                        if k.src == nid:
                            _acc(k.dst, k.rel, float(v.get("weight", 0.0)))
                        if k.dst == nid:
                            _acc(k.src, k.rel, float(v.get("weight", 0.0)))
                seen.update(frontier)
                frontier = [n for n in nxt if n not in seen]
                hop += 1
            # finalize neighbors list sorted by weight
            lst = sorted(scores.values(), key=lambda x: x["weight"], reverse=True)
            summary["neighbors"][sid] = lst[:neighbor_cap_per_seed]
        return summary

    async def decay_edges(self, *, factor: float = 0.9, rel_whitelist: Optional[List[str]] = None, min_weight: float = 0.0) -> None:
        rels = set(rel_whitelist or [])
        for k, v in list(self._edges.items()):
            if rels and k.rel not in rels:
                continue
            w = float(v.get("weight", 0.0))
            if w <= float(min_weight):
                continue
            self._edges[k]["weight"] = w * float(factor)
