import pytest

from modules.memory.infra.vector_store_router import VectorStoreRouter


class _FakeQdrant:
    def __init__(self, exists_result, *, embed_result=None):
        self.exists_result = exists_result
        self.called = 0
        self.embed_calls = 0
        self.embed_result = list(embed_result or [0.1, 0.2, 0.3])

    async def tenant_exists(self, tenant_id: str):
        self.called += 1
        if isinstance(self.exists_result, Exception):
            raise self.exists_result
        return self.exists_result

    async def search_vectors(self, query, filters, topk, threshold=None):
        return [{"id": "q1"}]

    def embed_text(self, query):
        self.embed_calls += 1
        return list(self.embed_result)

    async def fetch_text_corpus(self, filters, *, limit=500):
        return []

    async def upsert_vectors(self, entries):
        return None

    async def get(self, entry_id):
        return None

    async def delete_ids(self, ids):
        return None

    async def set_published(self, ids, published):
        return 0

    async def health(self):
        return {"status": "ok"}

    async def ensure_collections(self):
        return None


class _FakeMilvus(_FakeQdrant):
    async def search_vectors(self, query, filters, topk, threshold=None):
        return [{"id": "m1"}]


class _FlakyQdrant(_FakeQdrant):
    def __init__(self, results):
        self.results = list(results)
        self.called = 0

    async def tenant_exists(self, tenant_id: str):
        self.called += 1
        if not self.results:
            raise RuntimeError("no more probe results")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.asyncio
async def test_router_prefers_qdrant_when_tenant_exists():
    qdr = _FakeQdrant(True)
    mil = _FakeMilvus(True)
    router = VectorStoreRouter(qdr, mil)
    res = await router.search_vectors("q", {"tenant_id": "t1"}, 1)
    assert res[0]["id"] == "q1"
    assert qdr.called == 1


@pytest.mark.asyncio
async def test_router_falls_back_to_milvus_on_probe_error():
    qdr = _FakeQdrant(RuntimeError("boom"))
    mil = _FakeMilvus(True)
    router = VectorStoreRouter(qdr, mil)
    res = await router.search_vectors("q", {"tenant_id": "t1"}, 1)
    assert res[0]["id"] == "m1"


@pytest.mark.asyncio
async def test_router_does_not_cache_milvus_route_on_probe_error():
    qdr = _FlakyQdrant([RuntimeError("boom"), True])
    mil = _FakeMilvus(True)
    router = VectorStoreRouter(qdr, mil)

    first = await router.search_vectors("q", {"tenant_id": "t1"}, 1)
    second = await router.search_vectors("q", {"tenant_id": "t1"}, 1)

    assert first[0]["id"] == "m1"
    assert second[0]["id"] == "q1"
    assert qdr.called == 2


@pytest.mark.asyncio
async def test_router_embed_query_uses_selected_backend():
    qdr = _FakeQdrant(True, embed_result=[0.1, 0.2])
    mil = _FakeMilvus(True, embed_result=[9.0, 9.1])
    router = VectorStoreRouter(qdr, mil)

    vec = await router.embed_query("hello", tenant_id="t1")

    assert vec == [0.1, 0.2]
    assert qdr.embed_calls == 1
    assert mil.embed_calls == 0
