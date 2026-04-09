from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def _mk_service() -> MemoryService:
    vec = InMemVectorStore({})
    gra = InMemGraphStore({})
    aud = AuditStore()
    return MemoryService(vec, gra, aud)


def test_write_rewrites_placeholder_ids_and_rewires_edges():
    svc = _mk_service()

    e1 = MemoryEntry(id="tmp-0", kind="semantic", modality="text", contents=["A"], metadata={})
    e2 = MemoryEntry(id="tmp-1", kind="semantic", modality="text", contents=["B"], metadata={})
    ed = Edge(src_id="tmp-0", dst_id="tmp-1", rel_type="describes", weight=1.0)

    ver = asyncio.run(svc.write([e1, e2], [ed]))
    assert ver.value

    # vector store should not contain tmp-* ids
    dumped = svc.vectors.dump()  # type: ignore[attr-defined]
    assert all(not str(k).startswith("tmp-") for k in dumped.keys())
    # graph edges should also be rewired
    edges = svc.graph.dump_edges()  # type: ignore[attr-defined]
    assert len(edges) == 1
    src, dst, rel, w = edges[0]
    assert rel == "describes"
    assert not str(src).startswith("tmp-") and not str(dst).startswith("tmp-")

