from __future__ import annotations

"""
Memory MCP 适配（最小可用）：

提供一个轻量的“工具调用适配器”，读取 ToolSpec 并将 memory.* 路由到 MemoryService。
用于在不引入网络依赖的情况下，完成 MCP→MemoryPort 的适配逻辑与测试。

使用方式：
- adapter = MemoryMCPAdapter(service=svc) 或 MemoryMCPAdapter.from_defaults()
- adapter.tools() → 列出 tool 名称
- adapter.invoke("memory.search", {...}) → 调用并返回结果（dict）
"""

from typing import Any, Dict, List
import json
import os

from modules.memory.contracts.memory_models import MemoryEntry, Edge, SearchFilters
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


TOOLSPEC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "memory_toolspec.json")
)


class MemoryMCPAdapter:
    def __init__(self, service: MemoryService, toolspec_path: str | None = None) -> None:
        self.svc = service
        self.toolspec_path = toolspec_path or TOOLSPEC_PATH
        try:
            with open(self.toolspec_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._tools = [t.get("name") for t in data.get("tools", []) if t.get("name")]
        except Exception:
            self._tools = [
                "memory.search",
                "memory.write",
                "memory.update",
                "memory.delete",
                "memory.link",
            ]

    @classmethod
    def from_defaults(cls) -> "MemoryMCPAdapter":
        # 默认用内存存储，便于测试
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        return cls(svc)

    def tools(self) -> List[str]:
        return list(self._tools)

    async def invoke(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if name == "memory.search":
            filters = params.get("filters")
            sf = SearchFilters.model_validate(filters) if filters else None
            res = await self.svc.search(
                params.get("query", ""),
                topk=int(params.get("topk", 10)),
                filters=sf,
                expand_graph=bool(params.get("expand_graph", True)),
                threshold=params.get("threshold"),
            )
            return res.model_dump()

        if name == "memory.write":
            entries = [MemoryEntry.model_validate(e) for e in params.get("entries", [])]
            links_raw = params.get("links") or []
            links = [Edge.model_validate(link) for link in links_raw] if links_raw else None
            ver = await self.svc.write(entries, links, upsert=bool(params.get("upsert", True)))
            return {"version": ver.value}

        if name == "memory.update":
            ver = await self.svc.update(str(params.get("id")), params.get("patch") or {}, reason=params.get("reason"))
            return {"version": ver.value}

        if name == "memory.delete":
            ver = await self.svc.delete(str(params.get("id")), soft=bool(params.get("soft", True)), reason=params.get("reason"))
            return {"version": ver.value}

        if name == "memory.link":
            tenant_id = str(params.get("tenant_id") or "").strip() or None
            ok = await self.svc.link(
                str(params.get("src_id")),
                str(params.get("dst_id")),
                str(params.get("rel_type")),
                weight=params.get("weight"),
                tenant_id=tenant_id,
            )
            return {"ok": ok}

        raise ValueError(f"Unknown tool: {name}")
