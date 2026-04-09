from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters, Edge
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_search_graph_includes_occurs_at_by_default():
    async def _run():
        import os
        os.environ.setdefault("MPLCONFIGDIR", ".cache/mpl")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # two entries: episodic event and time node (assign ids explicitly to avoid write())
        e = MemoryEntry(id="e1", kind="episodic", modality="text", contents=["E"], metadata={"user_id": ["u"], "memory_domain": "d"})
        t = MemoryEntry(id="t1", kind="semantic", modality="structured", contents=["time"], metadata={"entity_type": "time", "user_id": ["u"], "memory_domain": "d"})
        await svc.graph.merge_nodes_edges([e, t], [Edge(src_id=e.id, dst_id=t.id, rel_type="OCCURS_AT", weight=1.0)])

        # graph-first query with seeds (no explicit whitelist)
        out = svc.search_graph({
            "seeds": ["e1"],
            "filters": SearchFilters(user_id=["u"], memory_domain="d"),
            "max_hops": 1,
            "neighbor_cap_per_seed": 5,
        })
        nbrs = out.get("neighbors", {}).get("e1") or []
        # expect at least one neighbor via OCCURS_AT
        assert any(n.get("rel") == "OCCURS_AT" for n in nbrs), f"expected OCCURS_AT in neighbors, got: {nbrs}"

    asyncio.run(_run())
