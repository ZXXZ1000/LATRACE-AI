from __future__ import annotations

import pytest

from modules.memory.infra.milvus_store import MilvusStore


@pytest.mark.asyncio
async def test_set_published_can_load_entry_from_non_text_collection() -> None:
    store = MilvusStore.__new__(MilvusStore)
    store.collections = {
        "text": "memory_text",
        "image": "memory_image",
        "audio": "memory_audio",
    }
    store._payload_json = True

    seen_collections: list[str] = []
    upserted_payloads: list[object] = []

    def _fake_get_sync(collection_name: str, entry_id: str):
        seen_collections.append(collection_name)
        if collection_name != "memory_image":
            return None
        return {
            store._PAYLOAD_FIELD: {
                "id": entry_id,
                "kind": "semantic",
                "modality": "image",
                "contents": ["image memory"],
                "metadata": {"tenant_id": "t1"},
                "published": False,
            }
        }

    async def _fake_upsert_vectors(entries):
        upserted_payloads.extend(entries)
        return None

    store._get_sync = _fake_get_sync
    store.upsert_vectors = _fake_upsert_vectors

    updated = await store.set_published(["img-1"], True)

    assert updated == 1
    assert seen_collections == ["memory_text", "memory_image"]
    assert len(upserted_payloads) == 1
    assert upserted_payloads[0].id == "img-1"
    assert upserted_payloads[0].published is True
