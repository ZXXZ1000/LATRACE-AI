from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import SearchFilters
from modules.memory.application import runtime_config as rtconf


class _VecStoreProbe:
    def __init__(self) -> None:
        self.last_filters: Dict[str, Any] | None = None

    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None = None) -> List[Dict[str, Any]]:
        self.last_filters = dict(filters)
        return []

    async def upsert_vectors(self, entries):
        return None

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok"}


def test_default_modalities_override_applied():
    vec = _VecStoreProbe()
    graph = InMemGraphStore()
    audit = AuditStore()
    svc = MemoryService(vec, graph, audit)

    # clear then set runtime override to all modalities
    rtconf.clear_ann_override()
    rtconf.set_ann_params(default_modalities=["text", "image", "audio"])  # type: ignore[list-item]

    async def _run():
        await svc.search("科幻", filters=SearchFilters(), topk=3, expand_graph=False)
        assert vec.last_filters is not None
        assert set(vec.last_filters.get("modality", [])) == {"text", "image", "audio"}

    asyncio.run(_run())

