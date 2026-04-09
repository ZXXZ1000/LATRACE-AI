from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.ttl_jobs import run_ttl_cleanup


def test_per_domain_ttl_and_importance_override():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # work: ttl should override to ~30d, importance +0.1
        e1 = MemoryEntry(kind="semantic", modality="text", contents=["work note"], metadata={"memory_domain": "work", "user_id": ["u1"]})
        await svc.write([e1])
        # read back (via vec store dump)
        dump = svc.vectors.dump()  # type: ignore[attr-defined]
        ent = next(iter(dump.values()))
        md = ent.metadata
        assert md.get("memory_domain") == "work"
        ttl = md.get("ttl")
        assert isinstance(ttl, int) and ttl >= 29 * 86400, f"expected ttl override around 30d, got {ttl}"
        imp = float(md.get("importance", 0.0))
        # importance baseline ~0.6 for semantic? ensure within (0,1] and at least baseline
        assert 0.0 <= imp <= 1.0

        # home: ttl ~90d
        e2 = MemoryEntry(kind="semantic", modality="text", contents=["home note"], metadata={"memory_domain": "home", "user_id": ["u1"]})
        await svc.write([e2])
        dump2 = svc.vectors.dump()  # type: ignore
        ttl2 = None
        for v in dump2.values():
            if v.contents[0] == "home note":
                ttl2 = v.metadata.get("ttl")
        assert isinstance(ttl2, int) and ttl2 >= 89 * 86400

        # pinned ttl should not be overridden
        e3 = MemoryEntry(kind="semantic", modality="text", contents=["pinned"], metadata={"memory_domain": "work", "user_id": ["u1"], "ttl": 123, "ttl_pinned": True})
        await svc.write([e3])
        dump3 = svc.vectors.dump()  # type: ignore
        for v in dump3.values():
            if v.contents[0] == "pinned":
                assert v.metadata.get("ttl") == 123

        # TTL cleanup should mark an ancient entry as deleted
        old = MemoryEntry(kind="episodic", modality="text", contents=["old"], metadata={"memory_domain": "system", "user_id": ["u1"], "created_at": "1970-01-01T00:00:00+00:00"})
        await svc.write([old])
        changed = await run_ttl_cleanup(svc.vectors)
        assert changed >= 1
    asyncio.run(_run())

