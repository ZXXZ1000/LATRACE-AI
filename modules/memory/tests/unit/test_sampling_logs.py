from __future__ import annotations

import asyncio

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def test_search_sampling_logs_emitted_with_sampler():
    async def _run():
        vec = InMemVectorStore()
        graph = InMemGraphStore()
        svc = MemoryService(vec, graph, AuditStore())

        # enable sampler with rate=1.0 for determinism
        captured = []
        svc.set_search_sampler(lambda s: captured.append(s), enabled=True, rate=1.0)

        e = MemoryEntry(kind="semantic", modality="text", contents=["采样 日志 测试"], metadata={"source": "mem0"})
        await svc.write([e])
        await svc.search("采样 日志", topk=1, filters=SearchFilters(modality=["text"]))

        assert len(captured) >= 1
        sample = captured[-1]
        assert "query" in sample and "latency_ms" in sample and "top_hits" in sample
        assert len(sample.get("top_hits", [])) >= 1

    asyncio.run(_run())

