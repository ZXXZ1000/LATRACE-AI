from __future__ import annotations

import asyncio
import time
import pytest

from modules.memory.contracts.memory_models import Hit, MemoryEntry, SearchFilters, SearchResult
from modules.memory.retrieval import retrieval


class _StubStoreV2:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
        self.calls.append({"api": "search", "topk": topk, "filters": filters.model_dump(exclude_none=True) if filters else None})
        src = list((filters.source or []) if filters else [])
        if set(src) == {"tkg_dialog_event_index_v1"} or set(src) == {"tkg_dialog_event_index_v1", "dialog_session_write_v1"}:
            hits = [
                Hit(
                    id="e1",
                    score=0.95,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Graph Event 1 Fee details here"],
                        metadata={"tkg_event_id": "e1", "event_id": "conv-1_D1_1", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
                    ),
                ),
                Hit(
                    id="e2",
                    score=0.85,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Graph Event 2"],
                        metadata={"tkg_event_id": "e2", "event_id": "conv-1_D1_2", "timestamp_iso": "2025-01-02T00:00:00+00:00"},
                    ),
                ),
            ]
            return SearchResult(hits=hits)
        if src == ["tkg_dialog_utterance_index_v1"]:
            hits = [
                Hit(
                    id="u1",
                    score=0.8,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Utterance A"],
                        metadata={"tkg_event_id": "e1"},
                    ),
                ),
                Hit(
                    id="u2",
                    score=0.7,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Utterance B"],
                        metadata={"tkg_event_id": "e3"},
                    ),
                ),
                Hit(
                    id="u3",
                    score=0.6,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Utterance C"],
                        metadata={"tkg_event_id": "e3"},
                    ),
                ),
                Hit(
                    id="u4",
                    score=0.5,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Utterance D"],
                        metadata={"tkg_event_id": "e4"},
                    ),
                ),
                Hit(
                    id="u5",
                    score=0.4,
                    entry=MemoryEntry(
                        kind="semantic",
                        modality="text",
                        contents=["Utterance E"],
                        metadata={"tkg_event_id": "e5"},
                    ),
                ),
            ]
            return SearchResult(hits=hits)
        return SearchResult(hits=[])

    async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:  # type: ignore[override]
        self.calls.append({"api": "graph_explain_event_evidence", "event_id": event_id})
        return {"utterances": [{"id": "u"}], "entities": [{"id": "ent"}], "knowledge": []}

    async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
        self.calls.append({"api": "graph_list_events", "entity_id": entity_id, "limit": limit})
        return []

    async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
        self.calls.append({"api": "graph_resolve_entities", "name": name, "limit": limit})
        return []

    async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
        self.calls.append({"api": "graph_list_timeslices_range", "start": start_iso, "end": end_iso})
        return []


def test_retrieval_dialog_v2_dynamic_fill_keeps_graph_hits_and_fills_with_vec() -> None:
    async def _run() -> None:
        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
        )

        evidence = res["evidence"]
        assert len(evidence) == 5
        # event vector hits must be retained (logical ids preferred when present)
        assert "conv-1_D1_1" in evidence and "conv-1_D1_2" in evidence
        # e3 should only appear once (dedup)
        assert evidence.count("e3") == 1
        # e1 should include desc in evidence text
        e1 = next(item for item in res["evidence_details"] if item.get("event_id") == "conv-1_D1_1")
        assert "Graph Event 1" in (e1.get("text") or "")
        assert "Fee details here" in (e1.get("text") or "")

        dbg = res.get("debug", {})
        apis = [c.get("api") for c in dbg.get("executed_calls", [])]
        assert "event_search_event_vec" in apis
        assert "event_search_utterance_vec" in apis
        stats = dbg.get("candidate_stats") or {}
        assert stats.get("candidate_k") == 5
        assert stats.get("selected_count") == 5
        assert stats.get("selection_mode") == "score_order"
        assert stats.get("union_count", 0) >= 5
        candidates = dbg.get("candidate_details") or []
        assert candidates
        first = candidates[0]
        assert "route_scores" in first
        assert "route_ranks" in first
        assert "route_support_score" in first
        assert "match_fidelity_score" in first
        assert "match_details" in first
        assert "preselect_score" in first
        assert "selection_reason" in first
        assert "base_rank" in first
        e1_dbg = next(item for item in candidates if item.get("logical_event_id") == "conv-1_D1_1")
        assert e1_dbg.get("same_dialog_key") == "D1"
        assert e1_dbg.get("mapping_status") == "mapped_logical"
        assert e1_dbg.get("selected") is True
        assert e1_dbg.get("selection_reason") == "score_order"
        assert float(e1_dbg.get("final_score") or 0.0) == pytest.approx(float(e1_dbg.get("preselect_score") or 0.0))

    asyncio.run(_run())


def test_retrieval_dialog_v2_uses_public_store_embed_query_once() -> None:
    async def _run() -> None:
        class _EmbedAwareStore(_StubStoreV2):
            def __init__(self) -> None:
                super().__init__()
                self.embed_calls: list[dict[str, str | None]] = []

            async def embed_query(self, query: str, *, tenant_id: str | None = None) -> list[float]:
                self.embed_calls.append({"query": query, "tenant_id": tenant_id})
                return [0.1, 0.2, 0.3]

        store = _EmbedAwareStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
        )

        assert store.embed_calls == [{"query": "Ask about Caroline", "tenant_id": "t"}]
        apis = [c.get("api") for c in (res.get("debug", {}) or {}).get("executed_calls", [])]
        assert "event_search_event_vec" in apis
        assert "event_search_utterance_vec" in apis
        assert "fact_search" in apis

    asyncio.run(_run())


def test_retrieval_dialog_v2_score_blend_alpha_changes_default_selection() -> None:
    async def _run() -> None:
        class _ScoreBlendStore(_StubStoreV2):
            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if set(src) == {"tkg_dialog_event_index_v1", "dialog_session_write_v1"}:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="event_a",
                                score=0.51,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Event A"],
                                    metadata={"tkg_event_id": "tA", "event_id": "eA", "timestamp_iso": "2025-01-01T00:00:00+00:00"},
                                ),
                            ),
                            Hit(
                                id="event_b",
                                score=0.50,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Event B"],
                                    metadata={"tkg_event_id": "tB", "event_id": "eB", "timestamp_iso": "2025-01-02T00:00:00+00:00"},
                                ),
                            ),
                            Hit(
                                id="event_c",
                                score=0.10,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Event C"],
                                    metadata={"tkg_event_id": "tC", "event_id": "eC", "timestamp_iso": "2025-01-03T00:00:00+00:00"},
                                ),
                            ),
                        ]
                    )
                if src == ["locomo_text_pipeline"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="fact_b",
                                score=0.90,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Fact about Event B"],
                                    metadata={"source_turn_ids": ["eB"], "fact_type": "semantic"},
                                ),
                            ),
                            Hit(
                                id="fact_a",
                                score=0.10,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Fact about Event A"],
                                    metadata={"source_turn_ids": ["eA"], "fact_type": "semantic"},
                                ),
                            ),
                        ]
                    )
                return SearchResult(hits=[])

        base_weights = {
            "event_vec": 1.2,
            "vec": 0.0,
            "knowledge": 0.6,
            "entity": 0.0,
            "time": 0.0,
            "match": 0.0,
            "recency": 0.0,
            "signal": 0.0,
        }

        store = _ScoreBlendStore()
        pure_res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about the strongest memory",
            strategy="dialog_v2",
            candidate_k=3,
            topk=1,
            debug=True,
            tkg_explain=False,
            enable_evidence_route=False,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_weights={**base_weights, "score_blend_alpha": 1.0},
            rrf_k=45,
        )
        blended_res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about the strongest memory",
            strategy="dialog_v2",
            candidate_k=3,
            topk=1,
            debug=True,
            tkg_explain=False,
            enable_evidence_route=False,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_weights={**base_weights, "score_blend_alpha": 0.7},
            rrf_k=45,
        )

        assert pure_res["evidence"] == ["eA"]
        assert blended_res["evidence"] == ["eB"]

        pure_debug = pure_res.get("debug", {})
        blended_debug = blended_res.get("debug", {})
        assert pure_debug.get("plan", {}).get("score_blend_alpha") == pytest.approx(1.0)
        assert blended_debug.get("plan", {}).get("score_blend_alpha") == pytest.approx(0.7)

        blended_candidates = blended_debug.get("candidate_details") or []
        event_b = next(item for item in blended_candidates if item.get("logical_event_id") == "eB")
        assert event_b["route_norm_scores"]["knowledge"] == pytest.approx(1.0)
        assert event_b["route_norm_scores"]["event_vec"] > 0.9

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_supports_route_ablation_switches() -> None:
    async def _run() -> None:
        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_event_route=False,
            enable_evidence_route=False,
            enable_knowledge_route=False,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_test_ablation={
                "disabled_routes": ["event", "evidence", "knowledge", "entity", "time"],
                "disabled_backlinks": ["event"],
                "disabled_signals": ["event", "evidence", "time", "timestamp", "recency", "graph_signal"],
                "source_native_only": True,
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        assert res["evidence"] == []

        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        assert plan.get("strategy") == "dialog_v2_test"
        assert plan.get("route_toggles") == {
            "event": False,
            "evidence": False,
            "knowledge": False,
            "entity": False,
            "time": False,
        }
        assert plan.get("test_ablation") == {
            "disabled_routes": ["entity", "event", "evidence", "knowledge", "time"],
            "disabled_backlinks": ["event"],
            "disabled_signals": ["event", "evidence", "graph_signal", "recency", "time", "timestamp"],
            "source_native_only": True,
        }

        calls_by_api = {call.get("api"): call for call in dbg.get("executed_calls", []) if isinstance(call, dict)}
        assert calls_by_api["event_search_event_vec"]["error"] == "event_route_disabled"
        assert calls_by_api["event_search_utterance_vec"]["error"] == "evidence_route_disabled"
        assert calls_by_api["fact_search"]["error"] == "knowledge_route_disabled"
        assert calls_by_api["entity_route"]["error"] == "entity_route_disabled"
        assert calls_by_api["time_route"]["error"] == "time_route_disabled"

        assert not any(call.get("api") == "search" for call in store.calls)
        assert not any(call.get("api") == "graph_resolve_entities" for call in store.calls)
        assert not any(call.get("api") == "graph_list_timeslices_range" for call in store.calls)

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_strict_event_ablation_keeps_source_native_candidates() -> None:
    async def _run() -> None:
        class _SourceNativeStore(_StubStoreV2):
            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if src == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="u1",
                                score=0.9,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Alice: booked the sunrise tour"],
                                    metadata={
                                        "tkg_utterance_id": "utt_1",
                                        "event_id": "conv-1_D1_1",
                                        "timestamp_iso": "2025-01-01T09:00:00+00:00",
                                    },
                                ),
                            ),
                            Hit(
                                id="u2",
                                score=0.8,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Bob: packed the red backpack"],
                                    metadata={
                                        "tkg_utterance_id": "utt_2",
                                        "event_id": "conv-1_D1_1",
                                        "timestamp_iso": "2025-01-01T09:05:00+00:00",
                                    },
                                ),
                            ),
                        ]
                    )
                return await super().search(query, topk=topk, filters=filters, expand_graph=expand_graph, threshold=threshold, scope=scope)

        store = _SourceNativeStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="sunrise tour",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_event_route=False,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_test_ablation={
                "disabled_routes": ["event"],
                "disabled_backlinks": ["event"],
                "disabled_signals": ["event", "graph_signal"],
                "source_native_only": True,
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        assert res["evidence"][:2] == ["utterance::utt_1", "utterance::utt_2"]
        assert res["evaluation_evidence"] == ["conv-1_D1_1"]
        first = res["evidence_details"][0]
        assert first["event_id"] == "utterance::utt_1"
        assert first["evaluation_event_ids"] == ["conv-1_D1_1"]
        assert first["source"] == "E_vec"

        dbg = res.get("debug", {})
        candidates = dbg.get("candidate_details") or []
        assert len(candidates) >= 2
        assert all(item.get("source_native") is True for item in candidates[:2])
        assert all(item.get("mapping_status") == "source_native" for item in candidates[:2])

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_event_backlink_ablation_keeps_canonical_outputs() -> None:
    async def _run() -> None:
        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_test_ablation={
                "disabled_backlinks": ["event"],
                "source_native_only": False,
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        assert res["evidence"]
        assert not any(str(item).startswith("utterance::") for item in res["evidence"])
        assert not any(str(item).startswith("fact::") for item in res["evidence"])
        assert "conv-1_D1_1" in res["evidence"] or "e1" in res["evidence"]

        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        assert plan.get("route_toggles") == {
            "event": True,
            "evidence": True,
            "knowledge": True,
            "entity": False,
            "time": False,
        }

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_no_event_route_keeps_entity_and_time_routes_enabled() -> None:
    async def _run() -> None:
        class _RouteAwareStore(_StubStoreV2):
            async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
                self.calls.append({"api": "graph_resolve_entities", "name": name, "limit": limit})
                return [{"entity_id": "ent:caroline", "score": 0.9}]

            async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
                self.calls.append({"api": "graph_list_events", "entity_id": entity_id, "limit": limit})
                return [{"id": "e9"}]

            async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
                self.calls.append({"api": "graph_list_timeslices_range", "start": start_iso, "end": end_iso})
                return [{"id": "ts1", "event_ids": ["e8"]}]

        store = _RouteAwareStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="What happened on 2025-01-02 with Caroline?",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=5,
            debug=True,
            entity_hints=["Caroline"],
            time_hints={
                "start_iso": "2025-01-02T00:00:00+00:00",
                "end_iso": "2025-01-02T23:59:59+00:00",
            },
            dialog_v2_test_ablation={
                "disabled_routes": ["event"],
                "disabled_backlinks": ["event"],
                "disabled_signals": ["event", "graph_signal"],
                "source_native_only": True,
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        assert plan.get("route_toggles") == {
            "event": False,
            "evidence": True,
            "knowledge": True,
            "entity": True,
            "time": True,
        }
        calls_by_api = {call.get("api"): call for call in dbg.get("executed_calls", []) if isinstance(call, dict)}
        assert calls_by_api["event_search_event_vec"]["error"] == "event_route_disabled"
        assert "error" not in calls_by_api["entity_route"]
        assert "error" not in calls_by_api["time_route"]

    asyncio.run(_run())


def test_retrieval_dialog_v2_runs_routes_concurrently() -> None:
    async def _run() -> None:
        class _SlowParallelStore(_StubStoreV2):
            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                await asyncio.sleep(0.05)
                src = list((filters.source or []) if filters else [])
                if set(src) == {"tkg_dialog_event_index_v1", "dialog_session_write_v1"}:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="ev-1",
                                score=0.9,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Alice booked the tickets"],
                                    metadata={"tkg_event_id": "ev-1", "event_id": "conv-1_D1_1", "timestamp_iso": "2025-01-01T09:00:00+00:00"},
                                ),
                            )
                        ]
                    )
                if src == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="utt-1",
                                score=0.8,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Alice: booked the tickets"],
                                    metadata={"tkg_event_id": "ev-1", "event_id": "conv-1_D1_1", "timestamp_iso": "2025-01-01T09:00:00+00:00"},
                                ),
                            )
                        ]
                    )
                if src == ["locomo_text_pipeline"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="fact-1",
                                score=0.7,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Alice booked the tickets for the show."],
                                    metadata={"event_ids": ["conv-1_D1_1"], "fact_id": "fact-1"},
                                ),
                            )
                        ]
                    )
                return SearchResult(hits=[])

            async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
                await asyncio.sleep(0.05)
                return [{"entity_id": "person-1", "score": 0.95}]

            async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
                await asyncio.sleep(0.05)
                return [{"id": "ev-1"}]

            async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
                await asyncio.sleep(0.05)
                return [{"id": "ts-1", "event_ids": ["ev-1"]}]

        store = _SlowParallelStore()
        started = time.perf_counter()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="What did Alice do on 2025-01-01?",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            entity_hints=["Alice"],
        )
        elapsed = time.perf_counter() - started

        assert elapsed < 0.18
        calls_by_api = {call.get("api"): call for call in res.get("debug", {}).get("executed_calls", []) if isinstance(call, dict)}
        assert set(calls_by_api) >= {
            "event_search_event_vec",
            "event_search_utterance_vec",
            "fact_search",
            "entity_route",
            "time_route",
        }
        assert calls_by_api["entity_route"]["events"] == 1
        assert calls_by_api["time_route"]["events"] == 1
        assert "conv-1_D1_1" in res["evidence"]

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_source_native_only_does_not_disable_explain() -> None:
    async def _run() -> None:
        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=3,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            tkg_explain=True,
            dialog_v2_test_ablation={
                "source_native_only": True,
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        dbg = res.get("debug", {})
        calls_by_api = {call.get("api"): call for call in dbg.get("executed_calls", []) if isinstance(call, dict)}
        explain_call = calls_by_api.get("tkg_explain_event_evidence")
        assert explain_call is not None
        assert int(explain_call.get("count") or 0) > 0
        assert any(call.get("api") == "graph_explain_event_evidence" for call in store.calls)

    asyncio.run(_run())


def test_retrieval_dialog_v2_test_no_time_signal_strips_timestamp_and_recency() -> None:
    async def _run() -> None:
        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="What happened on Jan 2?",
            strategy="dialog_v2_test",
            candidate_k=5,
            topk=3,
            debug=True,
            enable_entity_route=False,
            dialog_v2_test_ablation={
                "disabled_routes": ["time"],
                "disabled_signals": ["time", "timestamp", "recency", "graph_signal"],
            },
        )

        assert res["strategy"] == "dialog_v2_test"
        assert all(item.get("timestamp") is None for item in res.get("evidence_details") or [])
        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        assert plan.get("test_ablation") == {
            "disabled_routes": ["time"],
            "disabled_backlinks": [],
            "disabled_signals": ["graph_signal", "recency", "time", "timestamp"],
            "source_native_only": False,
        }
        calls_by_api = {call.get("api"): call for call in dbg.get("executed_calls", []) if isinstance(call, dict)}
        assert calls_by_api["time_route"]["error"] == "time_route_disabled"
        candidate_details = dbg.get("candidate_details") or []
        assert candidate_details
        assert all(float(item.get("recency_score") or 0.0) == pytest.approx(0.0) for item in candidate_details)
        assert all(item.get("timestamp") is None for item in candidate_details)

    asyncio.run(_run())


def test_retrieval_dialog_v2_vec_supports_multi_event_ids() -> None:
    async def _run() -> None:
        class _StubStore:
            async def graph_search_v1(self, *, tenant_id: str, query: str, topk: int = 10, source_id=None, include_evidence: bool = True) -> dict:  # type: ignore[override]
                return {"items": []}

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if src == ["tkg_dialog_utterance_index_v1"]:
                    hits = [
                        Hit(
                            id="u1",
                            score=0.9,
                            entry=MemoryEntry(
                                kind="semantic",
                                modality="text",
                                contents=["Utterance A"],
                                metadata={"tkg_event_ids": ["e10", "e11"]},
                            ),
                        )
                    ]
                    return SearchResult(hits=hits)
                return SearchResult(hits=[])

            async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:  # type: ignore[override]
                return {}

            async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
                return []

            async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
                return []

            async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
                return []

        store = _StubStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="What happened",
            strategy="dialog_v2",
            candidate_k=2,
            topk=2,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
        )
        evidence = res["evidence"]
        assert "e10" in evidence
        assert "e11" in evidence

    asyncio.run(_run())


def test_retrieval_dialog_v2_entity_and_time_routes_add_candidates() -> None:
    async def _run() -> None:
        class _EntityTimeStore(_StubStoreV2):
            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                # Focus this test on entity/time routes by suppressing vector hits.
                return SearchResult(hits=[])

            async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
                return [{"entity_id": "ent1", "score": 0.9}]

            async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
                if entity_id == "ent1":
                    return [{"id": "e10"}]
                return []

            async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
                return [{"id": "ts1", "event_ids": ["e11"]}]

        store = _EntityTimeStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Caroline on 2024-01-01",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            entity_hints=["Caroline"],
            time_hints={"start_iso": "2024-01-01T00:00:00+00:00", "end_iso": "2024-01-01T23:59:59+00:00"},
        )

        evidence = res["evidence"]
        assert "e10" in evidence
        assert "e11" in evidence

    asyncio.run(_run())


def test_retrieval_dialog_v2_fact_route_adds_candidates() -> None:
    async def _run() -> None:
        class _FactStore(_StubStoreV2):
            async def graph_search_v1(self, *, tenant_id: str, query: str, topk: int = 10, source_id=None, include_evidence: bool = True) -> dict:  # type: ignore[override]
                return {"items": []}

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if src == ["locomo_text_pipeline"]:
                    hit = Hit(
                        id="fact-1",
                        score=1.0,
                        entry=MemoryEntry(
                            kind="semantic",
                            modality="text",
                            contents=["Caroline went to the support group yesterday."],
                            metadata={"source_turn_ids": ["D1:3"], "source_sample_id": "conv-26"},
                        ),
                    )
                    return SearchResult(hits=[hit])
                return SearchResult(hits=[])

        store = _FactStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="When did Caroline go to the LGBTQ support group?",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
        )

        evidence = res.get("evidence") or []
        assert "conv-26_D1_3" in evidence

    asyncio.run(_run())


def test_retrieval_dialog_v2_merges_logical_and_tkg_event_ids_across_routes() -> None:
    async def _run() -> None:
        class _MergeStore(_StubStoreV2):
            async def graph_search_v1(self, *, tenant_id: str, query: str, topk: int = 10, source_id=None, include_evidence: bool = True) -> dict:  # type: ignore[override]
                return {"items": [{"event_id": "tkg-ev-1", "score": 1.0, "summary": "Graph Event", "t_abs_start": "2025-01-01T00:00:00+00:00"}]}

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if src == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="u1",
                                score=0.9,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Utterance A"],
                                    metadata={"tkg_event_id": "tkg-ev-1", "event_id": "conv-26_D1_1"},
                                ),
                            )
                        ]
                    )
                return SearchResult(hits=[])

        store = _MergeStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about something",
            strategy="dialog_v2",
            candidate_k=5,
            topk=5,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            tkg_explain=True,
        )

        # Output should prefer the logical (benchmark-compatible) event id.
        evidence = res.get("evidence") or []
        assert "conv-26_D1_1" in evidence

        # Explain should use the TKG event id (not the logical id).
        dbg = res.get("debug", {})
        calls = dbg.get("executed_calls", [])
        explain_calls = [c for c in calls if c.get("api") == "tkg_explain_event_evidence"]
        assert explain_calls, "expected explain to run"
        # The store call records the raw graph event id.
        assert any(call.get("event_id") == "tkg-ev-1" for call in store.calls if call.get("api") == "graph_explain_event_evidence")

    asyncio.run(_run())


def test_retrieval_dialog_v2_returns_unmapped_utterance_evidence() -> None:
    async def _run() -> None:
        class _UnmappedStore(_StubStoreV2):
            async def graph_search_v1(self, *, tenant_id: str, query: str, topk: int = 10, source_id=None, include_evidence: bool = True) -> dict:  # type: ignore[override]
                return {"items": []}

            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if src == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="u1",
                                score=0.9,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Utterance only evidence"],
                                    metadata={"tkg_utterance_id": "utt-1"},
                                ),
                            )
                        ]
                    )
                return SearchResult(hits=[])

        store = _UnmappedStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about unmapped evidence",
            strategy="dialog_v2",
            candidate_k=5,
            topk=3,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
        )

        evidence_details = res.get("evidence_details") or []
        assert any(item.get("unmapped_to_event") is True for item in evidence_details)
        dbg = res.get("debug", {})
        stats = dbg.get("candidate_stats") or {}
        assert stats.get("unmapped_count") == 1
        unmapped = dbg.get("unmapped_evidence_details") or []
        assert len(unmapped) == 1
        assert unmapped[0].get("utterance_id") == "utt-1"
        assert unmapped[0].get("unmapped_to_event") is True

    asyncio.run(_run())


def test_retrieval_dialog_v2_preselect_score_promotes_question_fidelity() -> None:
    async def _run() -> None:
        class _MatchSensitiveStore:
            async def search(self, query: str, *, topk: int = 10, filters: SearchFilters | None = None, expand_graph: bool = True, threshold=None, scope=None) -> SearchResult:  # type: ignore[override]
                src = list((filters.source or []) if filters else [])
                if set(src) == {"tkg_dialog_event_index_v1", "dialog_session_write_v1"} or set(src) == {"tkg_dialog_event_index_v1"}:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="e-generic",
                                score=0.95,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Adoption agency interviews completed successfully."],
                                    metadata={"tkg_event_id": "e-generic", "event_id": "conv-1_D2_1"},
                                ),
                            )
                        ]
                    )
                if src == ["locomo_text_pipeline"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="fact-1",
                                score=0.9,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Adoption paperwork was approved."],
                                    metadata={"source_turn_ids": ["D2:1"], "source_sample_id": "conv-1"},
                                ),
                            )
                        ]
                    )
                if src == ["tkg_dialog_utterance_index_v1"]:
                    return SearchResult(
                        hits=[
                            Hit(
                                id="u-status",
                                score=0.85,
                                entry=MemoryEntry(
                                    kind="semantic",
                                    modality="text",
                                    contents=["Caroline: It'll be tough as a single parent, but I'm up for the challenge!"],
                                    metadata={"tkg_event_id": "e-status", "event_id": "conv-1_D2_14"},
                                ),
                            )
                        ]
                    )
                return SearchResult(hits=[])

            async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:  # type: ignore[override]
                return {}

            async def graph_list_events(self, *, tenant_id: str, entity_id: str | None = None, limit: int = 100, **kwargs) -> list[dict]:  # type: ignore[override]
                return []

            async def graph_resolve_entities(self, *, tenant_id: str, name: str, entity_type: str | None = None, limit: int = 20) -> list[dict]:  # type: ignore[override]
                return []

            async def graph_list_timeslices_range(self, *, tenant_id: str, start_iso: str | None, end_iso: str | None, kind: str | None = None, limit: int = 200) -> list[dict]:  # type: ignore[override]
                return []

        store = _MatchSensitiveStore()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="What is Caroline's relationship status?",
            strategy="dialog_v2",
            candidate_k=1,
            topk=1,
            graph_cap=1,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_weights={
                "event_vec": 0.2,
                "vec": 0.6,
                "knowledge": 0.2,
                "entity": 0.0,
                "time": 0.0,
                "match": 2.0,
                "recency": 0.0,
                "signal": 0.0,
                "score_blend_alpha": 0.7,
            },
        )

        evidence = res.get("evidence") or []
        assert evidence == ["conv-1_D2_14"]

        dbg = res.get("debug", {})
        candidates = dbg.get("candidate_details") or []
        exact = next(item for item in candidates if item.get("logical_event_id") == "conv-1_D2_14")
        generic = next(item for item in candidates if item.get("logical_event_id") == "conv-1_D2_1")
        assert exact.get("match_fidelity_score", 0.0) > generic.get("match_fidelity_score", 0.0)
        assert exact.get("preselect_score", 0.0) > generic.get("preselect_score", 0.0)
        assert exact.get("selection_reason") == "score_order"
        assert float(exact.get("final_score") or 0.0) == pytest.approx(float(exact.get("preselect_score") or 0.0))

    asyncio.run(_run())


def test_retrieval_dialog_v2_optional_reranker_can_reorder_final_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        class _FakeAdapter:
            kind = "fake_reranker"

            def generate(self, messages, response_format=None):  # noqa: ANN001
                return "{\"1\": 0.1, \"2\": 0.9}"

        class _FakeResult:
            def __init__(self, candidate, rerank_score: float, final_score: float, rank: int) -> None:
                self.candidate = candidate
                self.rerank_score = rerank_score
                self.final_score = final_score
                self.rank = rank

            @property
            def event_id(self) -> str:
                return self.candidate.event_id

        class _FakeService:
            def rerank(self, query, candidates):  # noqa: ANN001
                assert query == "Ask about Caroline"
                ordered = sorted(candidates, key=lambda c: 0 if c.event_id == "conv-1_D1_2" else 1)
                out = []
                for idx, candidate in enumerate(ordered, start=1):
                    rerank_score = 0.9 if candidate.event_id == "conv-1_D1_2" else 0.1
                    out.append(_FakeResult(candidate, rerank_score, 1.0 - idx * 0.01, idx))
                return out

        monkeypatch.setattr(
            "modules.memory.application.llm_adapter.build_llm_from_config",
            lambda kind="text": _FakeAdapter() if kind == "reranker" else None,
        )
        monkeypatch.setattr(
            "modules.memory.application.rerank_dialog_v1.create_rerank_service",
            lambda config, llm_client=None: _FakeService(),
        )

        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2",
            candidate_k=5,
            topk=3,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_reranker={
                "enabled": True,
                "engine": "llm",
                "stage": "final_only",
                "llm_kind": "reranker",
                "rerank_pool_size": 5,
            },
        )

        evidence = res.get("evidence") or []
        assert evidence[0] == "conv-1_D1_2"

        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        reranker = plan.get("reranker") or {}
        assert reranker.get("enabled") is True
        assert reranker.get("applied") is True
        assert reranker.get("llm_kind") == "reranker"

        calls = dbg.get("executed_calls") or []
        rerank_calls = [item for item in calls if item.get("api") == "dialog_v2_rerank"]
        assert rerank_calls
        assert rerank_calls[0].get("applied") is True

        candidate_details = dbg.get("candidate_details") or []
        reranked = next(item for item in candidate_details if item.get("logical_event_id") == "conv-1_D1_2")
        assert reranked.get("rerank_score") is not None
        assert reranked.get("final_rank") == 1

    asyncio.run(_run())


def test_retrieval_dialog_v2_native_reranker_preselect_can_reorder_topk(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        class _FakeNativeResult:
            def __init__(self, document_id: str, relevance_score: float, rank: int) -> None:
                self.document_id = document_id
                self.relevance_score = relevance_score
                self.rank = rank

        class _FakeNativeResponse:
            def __init__(self, results) -> None:  # noqa: ANN001
                self.results = results
                self.request_id = "req_native_1"
                self.usage_total_tokens = 42
                self.model = "fake-reranker"

        class _FakeNativeClient:
            def rerank(self, *, query, documents, top_n=None, instruct=None):  # noqa: ANN001
                assert query == "Ask about Caroline"
                assert top_n == len(documents)
                assert instruct
                ordered = sorted(documents, key=lambda d: 0 if d.document_id == "conv-1_D1_2" else 1)
                return _FakeNativeResponse(
                    [
                        _FakeNativeResult(doc.document_id, 1.0 - idx * 0.1, idx)
                        for idx, doc in enumerate(ordered, start=1)
                    ]
                )

        monkeypatch.setattr(
            "modules.memory.application.reranker_adapter.build_reranker_from_config",
            lambda kind="reranker": _FakeNativeClient(),
        )

        store = _StubStoreV2()
        res = await retrieval(
            store,  # type: ignore[arg-type]
            tenant_id="t",
            user_tokens=["u:alice"],
            query="Ask about Caroline",
            strategy="dialog_v2",
            candidate_k=5,
            topk=3,
            debug=True,
            enable_entity_route=False,
            enable_time_route=False,
            dialog_v2_reranker={
                "enabled": True,
                "engine": "native",
                "stage": "preselect",
                "score_mode": "rerank_only",
                "llm_kind": "reranker",
                "rerank_pool_size": 5,
            },
        )

        evidence = res.get("evidence") or []
        assert evidence[0] == "conv-1_D1_2"

        dbg = res.get("debug", {})
        plan = dbg.get("plan") or {}
        reranker = plan.get("reranker") or {}
        assert reranker.get("enabled") is True
        assert reranker.get("applied") is True
        assert reranker.get("engine") == "native"
        assert reranker.get("stage") == "preselect"
        assert reranker.get("score_mode") == "rerank_only"
        assert reranker.get("request_id") == "req_native_1"
        assert reranker.get("usage_total_tokens") == 42

        stats = dbg.get("candidate_stats") or {}
        assert stats.get("selection_mode") == "rerank_preselect"
        assert stats.get("rerank_pool_count") == 5

        calls = dbg.get("executed_calls") or []
        rerank_calls = [item for item in calls if item.get("api") == "dialog_v2_rerank"]
        assert rerank_calls
        assert rerank_calls[0].get("applied") is True
        assert rerank_calls[0].get("score_mode") == "rerank_only"

        candidate_details = dbg.get("candidate_details") or []
        reranked = next(item for item in candidate_details if item.get("logical_event_id") == "conv-1_D1_2")
        assert reranked.get("in_rerank_pool") is True
        assert reranked.get("rerank_score") is not None
        assert reranked.get("rerank_rank") == 1
        assert reranked.get("selection_score") == pytest.approx(reranked.get("rerank_score"))
        assert reranked.get("selection_reason") == "rerank_preselect"
        assert reranked.get("final_rank") == 1

    asyncio.run(_run())
