from __future__ import annotations

import asyncio

from modules.memory.contracts.memory_models import Hit, MemoryEntry, SearchFilters, SearchResult
from modules.memory.application.qa_dialog_v1 import QA_SYSTEM_PROMPT_GENERAL
from modules.memory.retrieval import retrieval


class _StubStore:
    def __init__(self) -> None:
        self.calls = []

        self._facts = [
            Hit(
                id="f1",
                score=0.9,
                entry=MemoryEntry(
                    kind="semantic",
                    modality="text",
                    contents=["Fact A"],
                    metadata={"source_turn_ids": ["D1:1"], "source_sample_id": "s1", "fact_type": "fact"},
                ),
            ),
            Hit(
                id="f2",
                score=0.6,
                entry=MemoryEntry(
                    kind="semantic",
                    modality="text",
                    contents=["Fact B"],
                    metadata={"source_turn_ids": ["D1:2"], "source_sample_id": "s1", "fact_type": "preference"},
                ),
            ),
        ]

        self._events = [
            Hit(
                id="e1",
                score=0.8,
                entry=MemoryEntry(
                    kind="episodic",
                    modality="text",
                    contents=["Event text"],
                    metadata={"event_id": "s1_D1_1"},
                ),
            )
        ]

    async def search(
        self,
        query: str,
        *,
        topk: int = 10,
        filters: SearchFilters | None = None,
        expand_graph: bool = True,
        threshold=None,
        scope=None,
    ) -> SearchResult:
        self.calls.append(
            {
                "query": query,
                "filters": filters.model_dump(exclude_none=True) if filters else None,
                "expand_graph": expand_graph,
            }
        )
        mtypes = list((filters.memory_type or []) if filters else [])
        if mtypes == ["semantic"]:
            return SearchResult(hits=list(self._facts))
        if mtypes == ["episodic"]:
            return SearchResult(hits=list(self._events))
        return SearchResult(hits=[])


def test_retrieval_dialog_v1_fixed_3way_fusion_matches_expected_contract() -> None:
    async def _run() -> None:
        store = _StubStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice", "p:prod"],
            query="What happened?",
            strategy="dialog_v1",
            topk=10,
            debug=True,
        )

        assert res["strategy"] == "dialog_v1"
        assert res["evidence"] == ["s1_D1_1", "s1_D1_2"]

        details = res["evidence_details"]
        assert len(details) == 2
        assert details[0]["source"] == "fact_search"
        assert details[0]["event_id"] == "s1_D1_1"

        # default backend is tkg-first with fallback to legacy episodic search
        assert len(store.calls) == 3
        assert store.calls[0]["expand_graph"] is False
        assert store.calls[0]["filters"]["memory_type"] == ["semantic"]
        # utterance index attempt (no usable hits in this stub, so it falls back)
        assert store.calls[1]["expand_graph"] is False
        assert store.calls[1]["filters"]["source"] == ["tkg_dialog_utterance_index_v1"]
        assert store.calls[2]["expand_graph"] is True
        assert store.calls[2]["filters"]["memory_type"] == ["episodic"]

    asyncio.run(_run())


def test_retrieval_dialog_v1_with_answer_calls_qa_generate_with_benchmark_prompt() -> None:
    async def _run() -> None:
        store = _StubStore()
        captured: dict = {}

        def _qa_generate(system_prompt: str, user_prompt: str) -> str:
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return "ANSWER"

        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice", "p:prod"],
            query="What happened?",
            strategy="dialog_v1",
            topk=10,
            debug=True,
            with_answer=True,
            task="GENERAL",
            qa_generate=_qa_generate,
        )

        assert res["answer"] == "ANSWER"
        assert captured["system"].strip() == QA_SYSTEM_PROMPT_GENERAL.strip()
        assert captured["user"] == (
            "Question: What happened?\n"
            "Task type: GENERAL\n\n"
            "Evidence from memory (Fact=summarized memory, Event=original dialogue):\n"
            "[1] (Fact) id=s1_D1_1, ts=None\n"
            "    Fact A\n"
            "[2] (Fact) id=s1_D1_2, ts=None\n"
            "    Fact B\n\n"
            "Based on the evidence above, provide the best answer. Focus on facts and details mentioned."
        )

        dbg = res.get("debug", {})
        assert dbg.get("plan", {}).get("qa_latency_ms") is not None
        assert dbg.get("plan", {}).get("total_latency_ms") is not None

        # tkg-first attempt + fallback + fact_search
        assert len(store.calls) == 3

    asyncio.run(_run())


def test_retrieval_dialog_v1_with_answer_without_llm_returns_insufficient_information_when_no_evidence(monkeypatch) -> None:
    async def _run() -> None:
        class _EmptyStore:
            async def search(self, query: str, *, topk: int = 10, filters=None, expand_graph: bool = True, threshold=None, scope=None):  # type: ignore[no-untyped-def]
                return SearchResult(hits=[])

        monkeypatch.setattr("modules.memory.application.llm_adapter.build_llm_from_env", lambda: None)
        res = await retrieval(
            _EmptyStore(),  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Anything?",
            strategy="dialog_v1",
            with_answer=True,
            llm_policy="best_effort",
        )
        assert res["answer"] == "insufficient information"

    asyncio.run(_run())


def test_retrieval_dialog_v1_rerank_reorders_candidates_and_sets_rerank_fields() -> None:
    async def _run() -> None:
        class _EventOnlyStore:
            def __init__(self) -> None:
                self.calls = []
                self._events = [
                    Hit(
                        id="e1",
                        score=0.2,
                        entry=MemoryEntry(kind="episodic", modality="text", contents=["A"], metadata={"event_id": "e1"}),
                    ),
                    Hit(
                        id="e2",
                        score=0.1,
                        entry=MemoryEntry(kind="episodic", modality="text", contents=["B"], metadata={"event_id": "e2"}),
                    ),
                ]

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                self.calls.append({"filters": filters.model_dump(exclude_none=True) if filters else None, "expand_graph": expand_graph})
                mtypes = list((filters.memory_type or []) if filters else [])
                if mtypes == ["semantic"]:
                    return SearchResult(hits=[])
                if mtypes == ["episodic"]:
                    return SearchResult(hits=list(self._events))
                return SearchResult(hits=[])

        store = _EventOnlyStore()

        def _rerank_generate(system_prompt: str, user_prompt: str) -> str:
            # Prefer passage 2 heavily.
            return '{"1": 0.0, "2": 1.0}'

        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Q",
            strategy="dialog_v1",
            topk=10,
            debug=True,
            rerank={"enabled": True, "model": "llm", "top_n": 2, "rerank_pool_size": 2},
            rerank_generate=_rerank_generate,
        )

        # rerank should move e2 (passage 2) ahead of e1
        assert [d.get("event_id") for d in res["evidence_details"]] == ["e2", "e1"]
        assert all("_rerank_score" in d for d in res["evidence_details"])
        assert all("_rank" in d for d in res["evidence_details"])

        dbg = res.get("debug", {})
        assert any(c.get("api") == "rerank" for c in dbg.get("executed_calls", []))

    asyncio.run(_run())


def test_retrieval_dialog_v1_tkg_backend_uses_utterance_index_source_and_remaps_fact_event_ids() -> None:
    async def _run() -> None:
        from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid

        class _TkgStore(_StubStore):
            def __init__(self) -> None:
                super().__init__()
                # Provide utterance index hits (semantic) that map to the same TKG event ids as facts.
                ev1 = generate_uuid("tkg.dialog.event", "t|s1|1")
                self._utterance_hits = [
                    Hit(
                        id="u1",
                        score=0.7,
                        entry=MemoryEntry(
                            kind="semantic",
                            modality="text",
                            contents=["User: I like sci-fi movies."],
                            metadata={"source": "tkg_dialog_utterance_index_v1", "tkg_event_id": ev1, "tkg_utterance_id": "utt-1"},
                        ),
                    )
                ]
                self._explains = {ev1: {"event": {"id": ev1}, "utterances": [{"id": "utt-1"}], "knowledge": []}}

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                self.calls.append(
                    {
                        "query": query,
                        "filters": filters.model_dump(exclude_none=True) if filters else None,
                        "expand_graph": expand_graph,
                    }
                )
                mtypes = list((filters.memory_type or []) if filters else [])
                srcs = list((filters.source or []) if filters else [])
                if mtypes == ["semantic"] and srcs == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(hits=list(self._utterance_hits))
                if mtypes == ["semantic"]:
                    return SearchResult(hits=list(self._facts))
                if mtypes == ["episodic"]:
                    return SearchResult(hits=list(self._events))
                return SearchResult(hits=[])

            async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:
                return dict(self._explains.get(str(event_id)) or {})

        store = _TkgStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice", "p:prod"],
            query="What happened?",
            strategy="dialog_v1",
            topk=10,
            debug=True,
            backend="tkg",
        )

        # facts remapped to TKG event ids by turn index
        ev1 = generate_uuid("tkg.dialog.event", "t|s1|1")
        ev2 = generate_uuid("tkg.dialog.event", "t|s1|2")
        assert res["evidence"] == [ev1, ev2]

        # fact_search + utterance_search_tkg
        assert len(store.calls) == 2
        assert store.calls[0]["filters"]["memory_type"] == ["semantic"]
        assert store.calls[0]["expand_graph"] is False
        assert store.calls[1]["filters"]["source"] == ["tkg_dialog_utterance_index_v1"]
        assert store.calls[1]["expand_graph"] is False

        # explain enrichment attached under separate key (should not affect QA prompt formatting)
        assert res["evidence_details"][0].get("tkg_explain", {}).get("event", {}).get("id") == ev1

    asyncio.run(_run())
