from __future__ import annotations

import asyncio

from modules.memory.application import runtime_config as rtconf
from modules.memory.application.service import MemoryService, _bm25_tokenize
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.inmem_vector_store import InMemVectorStore


class DenseMissVectorStore(InMemVectorStore):
    """Simulate a dense retriever that misses the lexical answer-bearing record."""

    async def search_vectors(self, query, filters, topk, threshold=None):
        candidates = await super().search_vectors(query, filters, topk, threshold)
        out = []
        for item in candidates:
            text = " ".join((item.get("payload").contents or [])).lower()
            if "flask" in text or "fastapi" in text:
                continue
            out.append(item)
        return out[: max(1, topk)]


def test_bm25_tokenize_keeps_english_terms_and_chinese_bigrams():
    toks = _bm25_tokenize("从Flask改成FastAPI")
    assert "flask" in toks
    assert "fastapi" in toks
    assert "从F" not in toks
    assert "改成" in toks


def test_bm25_tokenize_handles_edge_cases():
    assert _bm25_tokenize("") == []
    assert _bm25_tokenize("   ") == []
    assert _bm25_tokenize("!!! ... ，，，") == []

    digit_toks = _bm25_tokenize("版本 2025 升级到 v2")
    assert "2025" in digit_toks
    assert "v2" in digit_toks
    assert "版本" in digit_toks

    single_char_toks = _bm25_tokenize("从A到B")
    assert "a" in single_char_toks
    assert "b" in single_char_toks
    assert "从" in single_char_toks
    assert "到" in single_char_toks


def test_search_hybrid_lexical_can_recover_dense_miss(monkeypatch):
    async def _run():
        rtconf.clear_rerank_weights_override()
        rtconf.clear_lexical_hybrid_override()
        try:
            vec = DenseMissVectorStore()
            graph = InMemGraphStore()
            audit = AuditStore()
            service = MemoryService(vec, graph, audit)
            rtconf.set_rerank_weights(
                {
                    "alpha_vector": 0.40,
                    "beta_bm25": 0.50,
                    "gamma_graph": 0.05,
                    "delta_recency": 0.05,
                }
            )
            rtconf.set_lexical_hybrid_params(
                enabled=True,
                corpus_limit=100,
                lexical_topn=20,
                normalize_scores=True,
            )

            relevant = MemoryEntry(
                kind="episodic",
                modality="text",
                contents=["Mia 把项目后端从 Flask 重写成 FastAPI，以便后续接口分层更清晰。"],
                metadata={"source": "dialog", "tenant_id": "tenant-hybrid", "user_id": ["u1"]},
            )
            distractor = MemoryEntry(
                kind="episodic",
                modality="text",
                contents=["Mia 今天和 Leo 讨论了 Aurora 试点项目的排期和上线节奏。"],
                metadata={"source": "dialog", "tenant_id": "tenant-hybrid", "user_id": ["u1"]},
            )
            await service.write([relevant, distractor])

            res = await service.search(
                query="Mia 后端是从什么框架改成什么框架？",
                topk=5,
                filters=SearchFilters(modality=["text"], tenant_id="tenant-hybrid", user_id=["u1"]),
                expand_graph=False,
            )

            texts = [" ".join(hit.entry.contents) for hit in res.hits]
            lex = res.trace.get("lexical_hybrid") or {}
            assert any("Flask" in text and "FastAPI" in text for text in texts), {
                "texts": texts,
                "trace": res.trace,
            }
            assert lex.get("enabled") is True
            assert int(lex.get("lexical_candidates_added") or 0) >= 1
            assert lex.get("bm25_source") == "hybrid_corpus"
        finally:
            rtconf.clear_rerank_weights_override()
            rtconf.clear_lexical_hybrid_override()

    asyncio.run(_run())
