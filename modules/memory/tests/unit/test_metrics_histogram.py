from __future__ import annotations

import pytest

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.metrics import as_prometheus_text


@pytest.mark.anyio
async def test_metrics_histogram_after_search():
    svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    e = MemoryEntry(kind="semantic", modality="text", contents=["test content"], metadata={"source": "mem0"})
    await svc.write([e])
    await svc.search("test", topk=1, expand_graph=False)
    text = as_prometheus_text()
    assert "memory_search_latency_ms_bucket" in text
    assert "memory_search_latency_ms_count" in text

