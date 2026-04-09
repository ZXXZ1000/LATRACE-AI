from __future__ import annotations

import pytest
from typing import Any, Dict, List

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters


class _VecStub:
    async def search_vectors(self, query: str, filters: Dict[str, Any], topk: int, threshold: float | None):
        hits: List[Dict[str, Any]] = []
        # if query contains speech keyword, return asr-like episodic entries
        if "说" in query or "对话" in query or "speak" in query:
            e = MemoryEntry(id="u1", kind="episodic", modality="text", contents=["男子说：你好"], metadata={"source": "asr", "start": 12.3, "end": 15.7, "clip_id": 3})
            hits.append({"id": "u1", "score": 0.9, "payload": e})
        # entity/action
        if "男子" in query:
            e2 = MemoryEntry(id="ev1", kind="episodic", modality="text", contents=["男子进入客厅"], metadata={"timestamp": 20.0, "clip_id": 4})
            hits.append({"id": "ev1", "score": 0.8, "payload": e2})
        return hits

    async def upsert_vectors(self, entries):
        return None


class _GraphStub:
    async def expand_neighbors(self, *args, **kwargs):
        # minimal neighbor map
        return {"neighbors": {"u1": [{"to": "ev1", "rel": "OCCURS_AT", "weight": 1.0, "hop": 1}]}}

    async def merge_nodes_edges(self, entries, edges=None):
        return None


class _AuditNoop:
    async def add_batch(self, *args, **kwargs):
        return "v1"


@pytest.mark.anyio
async def test_speech_search_basic():
    svc = MemoryService(_VecStub(), _GraphStub(), _AuditNoop())
    out = await svc.speech_search(keywords=["男子", "说"], filters=SearchFilters(memory_scope="vh::x"))
    utter = out.get("utterances") or []
    assert len(utter) >= 1
    u = utter[0]
    assert u.get("start") == 12.3 and u.get("end") == 15.7 and u.get("clip") == 3


@pytest.mark.anyio
async def test_entity_event_anchor_basic():
    svc = MemoryService(_VecStub(), _GraphStub(), _AuditNoop())
    out = await svc.entity_event_anchor(entity="男子", action="进入")
    triples = out.get("triples") or []
    assert len(triples) >= 1
    t = triples[0]
    assert t.get("entity") == "男子"
    # time_range present (from episodic metadata)
    tr = t.get("time_range") or []
    assert tr and isinstance(tr, list)

