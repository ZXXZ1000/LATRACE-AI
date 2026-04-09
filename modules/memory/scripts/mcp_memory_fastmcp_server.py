from __future__ import annotations

"""
FastMCP 服务器示例：在编排层注册 memory.* 工具，并直调 MemoryService。

运行：
  python -m MOYAN_Agent_Infra.modules.memory.scripts.mcp_memory_fastmcp_server

要求：已安装 fastmcp（mcp.server.fastmcp）与项目依赖，并在 `.env` 配好 Qdrant/Neo4j。
"""

import asyncio
import json
import os
from typing import Any, Dict
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, Edge, SearchFilters
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore


load_dotenv()
mcp = FastMCP("Memory Tools Server")


def _on_memory_ready(payload: Dict[str, Any]) -> None:
    print("[memory_ready]", json.dumps(payload, ensure_ascii=False))


def _publish_event(event: str, payload: Dict[str, Any]) -> None:
    if str(event) == "memory_ready":
        _on_memory_ready(payload)


def create_service() -> MemoryService:
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
    svc.set_event_publisher(_publish_event)
    return svc


svc = create_service()


@mcp.tool()
def memory_search(query: str, topk: int = 10, expand_graph: bool = True, threshold: float | None = None, filters: Dict[str, Any] | None = None) -> str:
    """Search unified memory. Returns JSON string of SearchResult."""
    async def _run():
        sf = SearchFilters.model_validate(filters) if filters else None
        res = await svc.search(query, topk=topk, expand_graph=expand_graph, threshold=threshold, filters=sf)
        return json.dumps(res.model_dump(), ensure_ascii=False)

    return asyncio.get_event_loop().run_until_complete(_run())


@mcp.tool()
def memory_write(entries_json: str, links_json: str | None = None, upsert: bool = True) -> str:
    """Write entries (and optional links). Params are JSON strings. Returns version JSON."""
    async def _run():
        entries = [MemoryEntry.model_validate(e) for e in json.loads(entries_json)]
        links = [Edge.model_validate(link) for link in json.loads(links_json)] if links_json else None
        ver = await svc.write(entries, links, upsert=upsert)
        return json.dumps(ver.model_dump(), ensure_ascii=False)

    return asyncio.get_event_loop().run_until_complete(_run())


@mcp.tool()
def memory_update(id: str, patch_json: str, reason: str | None = None) -> str:
    """Update a memory entry by id. patch_json is a JSON dict."""
    async def _run():
        patch = json.loads(patch_json)
        ver = await svc.update(id, patch, reason=reason)
        return json.dumps(ver.model_dump(), ensure_ascii=False)

    return asyncio.get_event_loop().run_until_complete(_run())


@mcp.tool()
def memory_delete(id: str, soft: bool = True, reason: str | None = None) -> str:
    """Delete a memory entry (soft by default)."""
    async def _run():
        ver = await svc.delete(id, soft=soft, reason=reason)
        return json.dumps(ver.model_dump(), ensure_ascii=False)

    return asyncio.get_event_loop().run_until_complete(_run())


@mcp.tool()
def memory_link(
    src_id: str,
    dst_id: str,
    rel_type: str,
    weight: float | None = None,
    tenant_id: str | None = None,
) -> str:
    """Create or update a relation."""
    async def _run():
        ok = await svc.link(src_id, dst_id, rel_type, weight=weight, tenant_id=tenant_id)
        return json.dumps({"ok": ok}, ensure_ascii=False)

    return asyncio.get_event_loop().run_until_complete(_run())


if __name__ == "__main__":
    # 启动 MCP 服务器
    mcp.run()
