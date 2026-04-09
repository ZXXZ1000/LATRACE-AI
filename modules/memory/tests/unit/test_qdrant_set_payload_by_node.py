from __future__ import annotations

from modules.memory.infra.qdrant_store import QdrantStore


class _Resp:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.status_code = 200
        self.text = "ok"


def test_qdrant_set_payload_by_node_builds_filter(monkeypatch) -> None:
    sent = []

    async def _fake_request(method, url, *, json=None, timeout=None):  # type: ignore[no-untyped-def]
        sent.append({"method": method, "url": url, "json": json, "timeout": timeout})
        return _Resp(True)

    qs = QdrantStore({"host": "127.0.0.1", "port": 6333, "collections": {"text": "memory_text"}})
    monkeypatch.setattr(qs, "_request", _fake_request)

    import asyncio
    updated = asyncio.run(
        qs.set_payload_by_node(tenant_id="t", node_id="evt-1", payload={"topic_path": "travel/japan"})
    )
    assert updated == 1
    assert sent
    body = sent[0]["json"]
    assert body["payload"]["topic_path"] == "travel/japan"
    must = body["filter"].get("must", [])
    assert {"key": "metadata.tenant_id", "match": {"value": "t"}} in must
    assert {"key": "metadata.node_id", "match": {"value": "evt-1"}} in must
