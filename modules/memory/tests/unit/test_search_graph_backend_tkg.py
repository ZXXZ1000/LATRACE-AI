from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.audit_store import AuditStore


class _StubGraphStore:
    async def merge_nodes_edges(self, entries, edges=None) -> None:  # pragma: no cover
        return None

    async def expand_neighbors(self, *args, **kwargs):  # pragma: no cover
        return {"neighbors": {}}

    async def query_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:
        # Minimal explain payload contract.
        return {
            "event": {"id": event_id, "tenant_id": tenant_id, "summary": "demo"},
            "entities": [{"id": "ent-1", "tenant_id": tenant_id, "type": "PERSON"}],
            "places": [{"id": "pl-1", "tenant_id": tenant_id, "name": "room"}],
            "timeslices": [{"id": "ts-1", "tenant_id": tenant_id, "kind": "dialog_session"}],
            "evidences": [],
            "utterances": [{"id": "utt-1", "tenant_id": tenant_id, "raw_text": "User: hi"}],
            "knowledge": [{"id": "k-1", "tenant_id": tenant_id, "summary": "User said hi"}],
        }


def test_search_graph_backend_tkg_expands_neighbors_via_explain() -> None:
    async def _run() -> None:
        vec = InMemVectorStore()
        graph = _StubGraphStore()
        audit = AuditStore()
        svc = MemoryService(vec, graph, audit)

        e = MemoryEntry(
            id="idx-1",
            kind="semantic",
            modality="text",
            contents=["User: hi"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u1"],
                "memory_domain": "dialog",
                "run_id": "sess-1",
                "source": "tkg_dialog_utterance_index_v1",
                "tkg_event_id": "ev-1",
                "dedup_skip": True,
            },
        )
        await svc.write([e])

        res = await svc.search(
            "hi",
            topk=1,
            filters=SearchFilters(
                tenant_id="t1",
                user_id=["u1"],
                user_match="all",
                memory_domain="dialog",
                modality=["text"],
                memory_type=["semantic"],
                source=["tkg_dialog_utterance_index_v1"],
            ),
            expand_graph=True,
            graph_backend="tkg",
        )

        assert res.hits
        nbrs = (res.neighbors or {}).get("neighbors", {}).get(res.hits[0].id) or []
        rels = {n.get("rel") for n in nbrs}
        assert "SUPPORTED_BY" in rels
        assert "INVOLVES" in rels
        assert "OCCURS_AT" in rels
        assert "COVERS_EVENT" in rels
        assert "DERIVED_FROM" in rels

        assert (res.trace or {}).get("graph_backend_requested") == "tkg"
        assert (res.trace or {}).get("graph_backend_used") == "tkg"

    asyncio.run(_run())

