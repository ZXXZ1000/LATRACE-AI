from __future__ import annotations

import asyncio

from modules.memory.adapters.mem0_adapter import build_entries_from_mem0
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_mem0_adapter_builds_prefer_edge_when_likes_detected():
    async def _run():
        messages = [{"role": "user", "content": "我喜欢晚上在客厅看电影"}]
        profile = {"user_id": "user.owner"}
        entries, edges = build_entries_from_mem0(messages, profile=profile)
        assert any(e.metadata.get("entity_type") == "user" for e in entries)
        assert any(ed.rel_type == "prefer" for ed in edges)

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        await svc.write(entries, links=edges)
        # ensure edge exists with weight
        pref = next(ed for ed in edges if ed.rel_type == "prefer")
        w = graph.get_edge_weight(pref.src_id, pref.dst_id, "prefer")
        assert w is not None and w >= 1.0

    asyncio.run(_run())

