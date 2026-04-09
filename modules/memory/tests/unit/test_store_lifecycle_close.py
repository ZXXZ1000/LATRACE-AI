from __future__ import annotations


def test_qdrant_store_close_sets_flag():
    from modules.memory.infra.qdrant_store import QdrantStore

    s = QdrantStore({"host": "127.0.0.1", "port": 6333})
    assert getattr(s, "_closed", False) is False
    s.close()
    assert getattr(s, "_closed", False) is True
    # idempotent
    s.close()
    assert getattr(s, "_closed", False) is True


def test_neo4j_store_close_idempotent():
    from modules.memory.infra.neo4j_store import Neo4jStore

    # no uri/user/password provided → driver stays None (no external connection attempted)
    n = Neo4jStore({})
    assert getattr(n, "_closed", False) is False
    n.close()
    assert getattr(n, "_closed", False) is True
    n.close()
    assert getattr(n, "_closed", False) is True


def test_memory_service_close_cascades():
    from modules.memory.application.service import MemoryService

    class _Dummy:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    v = _Dummy()
    g = _Dummy()
    a = _Dummy()
    svc = MemoryService(v, g, a)
    svc.close()
    # close once
    assert v.closed == 1
    assert g.closed == 1
    assert a.closed == 1
    # idempotent from service perspective
    svc.close()
    assert v.closed == 2  # service calls close again if invoked

