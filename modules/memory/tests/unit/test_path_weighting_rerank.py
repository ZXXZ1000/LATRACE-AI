from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application import runtime_config as rtconf


def test_path_weighting_prefers_hop1_over_hop2_when_graph_weight_only():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())

        # create two entries with similar tokens so vector/BM25 can be neutralized
        a = MemoryEntry(kind="semantic", modality="text", contents=["alpha 节点"], metadata={"source": "mem0"})
        b = MemoryEntry(kind="semantic", modality="text", contents=["alpha 节点"], metadata={"source": "mem0"})
        await svc.write([a, b])
        dump = vec.dump()
        idA = next(i for i, e in dump.items() if e is not None)  # first id
        idB = next(i for i in dump.keys() if i != idA)

        # monkeypatch vector search to return both ids with equal scores
        async def fake_search(query, filters, topk, threshold):
            return [
                {"id": idA, "score": 0.0, "payload": dump[idA]},
                {"id": idB, "score": 0.0, "payload": dump[idB]},
            ]

        vec.search_vectors = fake_search  # type: ignore

        # monkeypatch graph neighbors: idA has 1-hop neighbor, idB has only 2-hop neighbor (same weight)
        async def fake_neighbors(seed_ids, rel_whitelist=None, max_hops=1, neighbor_cap_per_seed=5):
            return {
                "neighbors": {
                    idA: [{"to": "nX", "rel": "prefer", "weight": 1.0, "hop": 1}],
                    idB: [{"to": "nY", "rel": "prefer", "weight": 1.0, "hop": 2}],
                }
            }

        graph.expand_neighbors = fake_neighbors  # type: ignore

        # rely only on graph score
        rtconf.set_rerank_weights({"alpha_vector": 0.0, "beta_bm25": 0.0, "gamma_graph": 1.0, "delta_recency": 0.0})
        res = await svc.search("alpha", topk=2, filters=SearchFilters(modality=["text"]))
        # top-1 should be idA due to hop=1 boost
        assert res.hits[0].id == idA
        rtconf.clear_rerank_weights_override()

    asyncio.run(_run())

