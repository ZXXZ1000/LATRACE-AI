from __future__ import annotations

from typing import Any, Dict, List

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.infra.audit_store import InMemAuditStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.retrieval import retrieval


class _StubGraphStore:
    def __init__(self) -> None:
        self.search_calls: List[Dict[str, Any]] = []
        self.evidence_calls: List[Dict[str, Any]] = []

    async def search_event_candidates(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int,
        source_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        self.search_calls.append(
            {"tenant_id": tenant_id, "query": query, "limit": limit, "source_id": source_id}
        )
        return [
            {
                "event_id": "e1",
                "summary": "Graph Event 1",
                "score": 2.0,
                "t_abs_start": "2024-01-01T00:00:00+00:00",
            },
            {
                "event_id": "e2",
                "summary": "Graph Event 2",
                "score": 1.6,
                "t_abs_start": "2024-01-02T00:00:00+00:00",
            },
        ]

    async def query_event_evidence(self, *, tenant_id: str, event_id: str) -> Dict[str, Any]:
        self.evidence_calls.append({"tenant_id": tenant_id, "event_id": event_id})
        return {
            "event": {"id": event_id, "summary": f"event-{event_id}"},
            "entities": [{"id": "ent-1"}],
            "places": [],
            "timeslices": [{"id": "ts-1"}],
            "evidences": [],
            "utterances": [{"id": "u-1"}],
            "knowledge": [{"id": "k-1"}],
        }


@pytest.mark.anyio
async def test_dialog_v2_graph_and_vec_parallel_fill_integration() -> None:
    vectors = InMemVectorStore({})
    graph = _StubGraphStore()
    audit = InMemAuditStore()
    svc = MemoryService(vectors, graph, audit)

    entries = [
        MemoryEntry(
            id="ev1",
            kind="semantic",
            modality="text",
            contents=["Graph Event 1"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_event_index_v1",
                "tkg_event_id": "e1",
                "event_id": "conv-1_D1_1",
                "timestamp_iso": "2024-01-01T00:00:00+00:00",
            },
        ),
        MemoryEntry(
            id="ev2",
            kind="semantic",
            modality="text",
            contents=["Graph Event 2"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_event_index_v1",
                "tkg_event_id": "e2",
                "event_id": "conv-1_D1_2",
                "timestamp_iso": "2024-01-02T00:00:00+00:00",
            },
        ),
        MemoryEntry(
            id="u1",
            kind="semantic",
            modality="text",
            contents=["Caroline talked about painting"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_utterance_index_v1",
                "tkg_event_id": "e1",
            },
        ),
        MemoryEntry(
            id="u2",
            kind="semantic",
            modality="text",
            contents=["Caroline visited the museum"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_utterance_index_v1",
                "tkg_event_id": "e3",
            },
        ),
        MemoryEntry(
            id="u3",
            kind="semantic",
            modality="text",
            contents=["Caroline met a friend"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_utterance_index_v1",
                "tkg_event_id": "e4",
            },
        ),
        MemoryEntry(
            id="u4",
            kind="semantic",
            modality="text",
            contents=["Caroline made dinner"],
            metadata={
                "tenant_id": "t1",
                "user_id": ["u:alice"],
                "memory_domain": "dialog",
                "source": "tkg_dialog_utterance_index_v1",
                "tkg_event_id": "e5",
            },
        ),
    ]
    await vectors.upsert_vectors(entries)

    res = await retrieval(
        svc,  # type: ignore[arg-type]
        tenant_id="t1",
        user_tokens=["u:alice"],
        query="Caroline",
        strategy="dialog_v2",
        candidate_k=5,
        topk=5,
        debug=True,
        seed_topn=2,
        enable_entity_route=False,
        enable_time_route=False,
    )

    evidence = res.get("evidence") or []
    assert len(evidence) == 5
    assert "conv-1_D1_1" in evidence and "conv-1_D1_2" in evidence, "E_event_vec results must be retained"
    assert "e3" in evidence and "e4" in evidence and "e5" in evidence, "E_vec should fill remaining slots"

    dbg = res.get("debug") or {}
    apis = [c.get("api") for c in (dbg.get("executed_calls") or [])]
    assert "event_search_event_vec" in apis
    assert "event_search_utterance_vec" in apis
