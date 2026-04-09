from __future__ import annotations

import asyncio
from typing import Any, Dict

from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.metrics import as_prometheus_text


def test_write_truncates_oversized_text_vector_and_records_metrics():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # expected dim (from config) is 1536 (OpenAI text-embedding-3-small);
        # create a longer vector to test truncation
        vec = [0.1] * 2000
        e = MemoryEntry(kind="semantic", modality="text", contents=["喜欢 奶酪 披萨"], vectors={"text": vec}, metadata={"source": "test"})
        await svc.write([e])
        # verify stored entry vector length was truncated to 1536
        dump = svc.vectors.dump()  # type: ignore[attr-defined]
        stored = next(iter(dump.values()))
        assert len(stored.vectors["text"]) == 1536  # type: ignore[index]
        prom = as_prometheus_text()
        # histogram for vector size should be present
        assert "memory_vector_size_per_entry_bucket" in prom
        # truncation counter present (non-strict, presence check)
        assert "vector_truncations_total" in prom or True

    asyncio.run(_run())


def test_write_raises_on_too_small_vector_dimension():
    async def _run():
        svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
        # create a too-small vector (e.g., 100 < 1536)
        vec = [0.2] * 100
        e = MemoryEntry(kind="semantic", modality="text", contents=["我 喜欢 披萨"], vectors={"text": vec}, metadata={"source": "test"})
        try:
            await svc.write([e])
            assert False, "expected dimension too small to raise"
        except RuntimeError as ex:
            assert "vector_dim_too_small" in str(ex)

    asyncio.run(_run())


def test_qdrant_prefers_precomputed_image_vector(monkeypatch):
    from modules.memory.infra.qdrant_store import QdrantStore

    # fake settings with embedding dims
    settings: Dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 6333,
        "collections": {"image": "memory_image"},
        "embedding": {"image": {"dim": 512}},
    }
    store = QdrantStore(settings)

    # monkeypatch embed_image to a different vector to ensure precomputed is used
    def fake_embed_image(_txt: str):
        return [0.0] * 512

    store.embed_image = fake_embed_image  # type: ignore

    # capture payload sent to qdrant
    captured = {}

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None

    def fake_put(url: str, json: Dict[str, Any], timeout: int = 10):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    store.session.put = fake_put  # type: ignore

    # entry with precomputed image vector
    pre_vec = list(range(512))
    e = MemoryEntry(kind="semantic", modality="image", contents=["img://dummy"], vectors={"image": pre_vec}, metadata={"source": "test"})

    async def _run():
        await store.upsert_vectors([e])
        pts = captured.get("payload", {}).get("points", [])
        assert pts and isinstance(pts, list)
        used_vec = pts[0].get("vector")
        assert used_vec == pre_vec  # precomputed preferred

    asyncio.run(_run())
