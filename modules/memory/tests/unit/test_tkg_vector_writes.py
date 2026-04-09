"""Test TKG vector writes to Qdrant on GraphUpsert."""
import asyncio
import hashlib
import uuid
from typing import Any, List


from modules.memory.application.graph_service import GraphService
from modules.memory.contracts.graph_models import (
    Entity,
    Event,
    GraphUpsertRequest,
)


def _expected_uuid(prefix: str, node_id: str) -> str:
    """Generate expected UUID using same logic as graph_service._make_vector_id."""
    readable_id = f"{prefix}_{node_id}"
    return str(uuid.UUID(hashlib.md5(readable_id.encode("utf-8")).hexdigest()))


class MockVectorStore:
    """Mock vector store to capture upsert calls."""

    def __init__(self):
        self.upsert_calls: List[List[Any]] = []

    async def upsert_vectors(self, entries: List[Any]) -> None:
        self.upsert_calls.append(list(entries))


class MockNeo4jStore:
    """Mock Neo4j store."""

    async def upsert_graph_v0(self, **kwargs) -> None:
        pass


def test_tkg_vector_writes_event_summary():
    """Event.summary should be written to Qdrant as text vector."""
    neo4j = MockNeo4jStore()
    vector_store = MockVectorStore()
    svc = GraphService(neo4j, vector_store=vector_store)  # type: ignore[arg-type]

    req = GraphUpsertRequest(
        events=[
            Event(
                id="evt_001",
                tenant_id="test_tenant",
                summary="Richard is cooking in the kitchen",
                source="vlm",
            )
        ],
        entities=[],
        edges=[],
    )

    asyncio.run(svc.upsert(req))

    # Should have one upsert call with one entry
    assert len(vector_store.upsert_calls) == 1
    entries = vector_store.upsert_calls[0]
    assert len(entries) == 1

    entry = entries[0]
    expected_uuid = _expected_uuid("tkg_event", "evt_001")
    assert entry.id == expected_uuid
    assert entry.modality == "text"
    assert entry.contents == ["Richard is cooking in the kitchen"]
    assert entry.metadata["node_type"] == "Event"
    assert entry.metadata["node_id"] == "evt_001"
    assert entry.metadata["vector_id_readable"] == "tkg_event_evt_001"  # Readable ID in metadata
    assert entry.metadata["tenant_id"] == "test_tenant"

    # Event should have text_vector_id set (UUID format, matches Qdrant point ID)
    assert req.events[0].text_vector_id == expected_uuid


def test_tkg_vector_writes_entity_name():
    """Entity.name should be written to Qdrant as text vector."""
    neo4j = MockNeo4jStore()
    vector_store = MockVectorStore()
    svc = GraphService(neo4j, vector_store=vector_store)  # type: ignore[arg-type]

    req = GraphUpsertRequest(
        events=[],
        entities=[
            Entity(
                id="person::test::richard",
                tenant_id="test_tenant",
                type="PERSON",
                name="Richard",
            )
        ],
        edges=[],
    )

    asyncio.run(svc.upsert(req))

    assert len(vector_store.upsert_calls) == 1
    entries = vector_store.upsert_calls[0]
    assert len(entries) == 1

    entry = entries[0]
    expected_uuid = _expected_uuid("tkg_entity_text", "person::test::richard")
    assert entry.id == expected_uuid
    assert entry.modality == "text"
    assert entry.contents == ["Richard"]
    assert entry.metadata["node_type"] == "Entity"
    assert entry.metadata["entity_type"] == "PERSON"
    assert entry.metadata["vector_id_readable"] == "tkg_entity_text_person::test::richard"

    # Entity should have text_vector_id set (UUID format, matches Qdrant point ID)
    assert req.entities[0].text_vector_id == expected_uuid


def test_tkg_vector_writes_multiple_nodes():
    """Multiple events and entities should all be written."""
    neo4j = MockNeo4jStore()
    vector_store = MockVectorStore()
    svc = GraphService(neo4j, vector_store=vector_store)  # type: ignore[arg-type]

    req = GraphUpsertRequest(
        events=[
            Event(id="evt_1", tenant_id="t", summary="Event one"),
            Event(id="evt_2", tenant_id="t", summary="Event two"),
        ],
        entities=[
            Entity(id="ent_1", tenant_id="t", type="PERSON", name="Alice"),
        ],
        edges=[],
    )

    asyncio.run(svc.upsert(req))

    assert len(vector_store.upsert_calls) == 1
    entries = vector_store.upsert_calls[0]
    # 2 events + 1 entity = 3 entries
    assert len(entries) == 3


def test_tkg_vector_writes_skipped_without_vector_store():
    """Without vector_store, no vector writes should happen."""
    neo4j = MockNeo4jStore()
    svc = GraphService(neo4j, vector_store=None)  # type: ignore[arg-type]

    req = GraphUpsertRequest(
        events=[
            Event(id="evt_1", tenant_id="t", summary="Event one"),
        ],
        entities=[],
        edges=[],
    )

    # Should not raise, and Event should not have vector_id set
    asyncio.run(svc.upsert(req))
    assert req.events[0].text_vector_id is None


def test_tkg_vector_writes_skips_empty_summary():
    """Events with empty summary should not be written."""
    neo4j = MockNeo4jStore()
    vector_store = MockVectorStore()
    svc = GraphService(neo4j, vector_store=vector_store)  # type: ignore[arg-type]

    req = GraphUpsertRequest(
        events=[
            Event(id="evt_1", tenant_id="t", summary=""),
            Event(id="evt_2", tenant_id="t", summary="   "),
        ],
        entities=[],
        edges=[],
    )

    asyncio.run(svc.upsert(req))

    # No entries should be written
    if vector_store.upsert_calls:
        entries = vector_store.upsert_calls[0]
        assert len(entries) == 0

