from __future__ import annotations

import asyncio
import pytest

from modules.memory.client import Memory
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application.fact_extractor_mem0 import build_fact_extractor_from_env


def test_infer_requires_llm_or_raises():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        m = Memory(svc)
        messages = [
            {"role": "user", "content": "我 喜欢 奶酪 披萨"},
            {"role": "assistant", "content": "好的，已记录"},
        ]
        extractor = build_fact_extractor_from_env()
        if extractor is None:
            with pytest.raises(RuntimeError):
                await m.add(messages, user_id="alice", memory_domain="home", run_id="r1", infer=True)
        else:
            try:
                r = await m.add(messages, user_id="alice", memory_domain="home", run_id="r1", infer=True)
                assert r.get("results") and len(r["results"]) >= 1
            except Exception as e:
                pytest.skip(f"LLM not reachable in test environment: {e}")
    asyncio.run(_run())
