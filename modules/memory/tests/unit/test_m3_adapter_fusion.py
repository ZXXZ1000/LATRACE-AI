from __future__ import annotations

import asyncio

from modules.memory.adapters.m3_adapter import build_entries_from_m3
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_m3_adapter_builds_entries_and_edges_and_reinforce():
    async def _run():
        parsed = {
            "clip_id": 3,
            "timestamp": "2025-02-12T11:12:30Z",
            "faces": ["<face_5>"],
            "voices": ["<voice_7>"],
            "episodic": ["<face_5> 在 <voice_7> 的请求下打开了灯"],
            "semantic": ["<face_5> 更倾向回应 <voice_7> 的请求"],
            "room": "room:living",
            "device": "device.light.living_main",
        }
        entries, edges = build_entries_from_m3(parsed)
        # at least one episodic + one face + one voice + optional room/device
        assert any(e.kind == "episodic" for e in entries)
        assert any(e.modality == "image" for e in entries)
        assert any(e.modality == "audio" for e in entries)
        assert len(edges) >= 3
        rels = {ed.rel_type for ed in edges}
        assert {"appears_in", "said_by"}.issubset(rels)

        vec = InMemVectorStore()
        graph = InMemGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)
        await svc.write(entries, links=edges)

        # Reinforce by writing the same edges again
        await svc.write([], links=edges)

        # fetch one appears_in edge weight and ensure >= 2.0
        # pick first appears_in
        ap = next(ed for ed in edges if ed.rel_type == "appears_in")
        w = graph.get_edge_weight(ap.src_id, ap.dst_id, "appears_in")
        assert w is not None and w >= 2.0

    asyncio.run(_run())

