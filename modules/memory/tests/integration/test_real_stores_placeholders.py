import os
import pytest

from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.neo4j_store import Neo4jStore


@pytest.mark.skipif(os.getenv("QDRANT_HOST") is None, reason="Qdrant not configured")
def test_qdrant_store_health_placeholder():
    store = QdrantStore({
        "host": os.getenv("QDRANT_HOST"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
        "api_key": os.getenv("QDRANT_API_KEY"),
    })
    # placeholder: just call health()
    import asyncio
    res = asyncio.run(store.health())
    assert "status" in res


@pytest.mark.skipif(os.getenv("NEO4J_URI") is None, reason="Neo4j not configured")
def test_neo4j_store_health_placeholder():
    store = Neo4jStore({
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER"),
        "password": os.getenv("NEO4J_PASSWORD"),
    })
    import asyncio
    res = asyncio.run(store.health())
    assert "status" in res

