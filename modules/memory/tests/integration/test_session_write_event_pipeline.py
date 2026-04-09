from __future__ import annotations

import asyncio

from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.contracts.memory_models import SearchFilters, SearchResult, Version
from modules.memory.session_write import session_write


class _StubStore:
    def __init__(self) -> None:
        self.graph_requests: list[GraphUpsertRequest] = []
        self.entries_written: int = 0

    async def get(self, _id: str):
        return None

    async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
        return SearchResult(hits=[])

    async def write(self, entries, links=None, *, upsert: bool = True, return_id_map: bool = False):  # type: ignore[override]
        self.entries_written += len(entries or [])
        return Version(value="v1")

    async def delete(self, memory_id: str, *, soft: bool = True, reason: str | None = None) -> Version:  # type: ignore[override]
        return Version(value="v1")

    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:  # type: ignore[override]
        self.graph_requests.append(body)


def test_session_write_event_pipeline_builds_events_and_links() -> None:
    async def _run() -> None:
        store = _StubStore()

        def _tkg_extractor(_turns):
            return {
                "events": [
                    {
                        "summary": "User shares a preference for sci-fi movies",
                        "event_type": "Atomic",
                        "source_turn_ids": ["D1:1"],
                        "evidence_status": "mapped",
                        "event_confidence": 0.8,
                    },
                    {
                        "summary": "Assistant acknowledges the preference",
                        "event_type": "Atomic",
                        "source_turn_ids": ["D1:2"],
                        "evidence_status": "mapped",
                        "event_confidence": 0.7,
                    },
                ],
                "knowledge": [],
            }

        turns = [
            {"dia_id": "D1:1", "speaker": "User", "text": "I like sci-fi movies."},
            {"dia_id": "D1:2", "speaker": "Assistant", "text": "Got it."},
        ]

        res = await session_write(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            session_id="sess-ev-1",
            turns=turns,
            extract=True,
            write_facts=False,
            write_events=True,
            graph_upsert=True,
            tkg_extractor=_tkg_extractor,
        )

        assert res["status"] == "ok"
        assert store.graph_requests, "graph_upsert_v0 should be called"
        req = store.graph_requests[0]
        assert len(req.events) == 2
        assert len(req.utterances) == 2
        rels = [e.rel_type for e in req.edges]
        assert "SUPPORTED_BY" in rels
        assert "NEXT_EVENT" in rels

    asyncio.run(_run())
