from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.contracts.memory_models import MemoryEntry, Edge


class _Tx:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def run(self, cypher: str, **params):
        self.calls.append({"cypher": cypher, "params": params})
        # emulate result iterator
        return []


class _Sess:
    def __init__(self, mode: str, calls: List[Dict[str, Any]]):
        self.mode = mode
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write_transaction(self, fn):
        tx = _Tx(self.calls)
        return fn(tx)

    def execute_write(self, fn):
        """Neo4j 5.x API, alias for write_transaction."""
        return self.write_transaction(fn)

    def read_transaction(self, fn):
        class _TxRead(_Tx):
            def run(self, cypher: str, **params):
                self.calls.append({"cypher": cypher, "params": params})
                # Return synthetic rows for path queries
                if "-[r]->(n)" in cypher:
                    return [{"nid": "n1", "rel": "prefer", "w": 0.9}]
                if "-[r1]->(m)-[r2]->(n)" in cypher:
                    return [{"nid": "n2", "rel": "appears_in", "w": 0.7}]
                return []
        tx = _TxRead(self.calls)
        return fn(tx)

    def execute_read(self, fn):
        """Neo4j 5.x API, alias for read_transaction."""
        return self.read_transaction(fn)


class _Driver:
    def __init__(self, calls: List[Dict[str, Any]]):
        self.calls = calls

    def session(self, *args, **kwargs):
        return _Sess("rw", self.calls)


def test_merge_nodes_edges_batch_builds_unwind_and_splits():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    entries = [
        MemoryEntry(kind="semantic", modality="text", contents=[f"c{i}"], metadata={"source": "t", "tenant_id": "tenant-a"}, id=f"id{i}")
        for i in range(3)
    ]
    edges = [Edge(src_id="id0", dst_id="id1", rel_type="prefer", weight=1.0)]

    asyncio.run(store.merge_nodes_edges_batch(entries, edges, chunk_size=2))

    # Expect at least two UNWIND node batches due to chunk_size=2
    node_merges = [c for c in calls if c["cypher"].startswith("UNWIND $nodes AS n")]
    assert len(node_merges) >= 2
    assert all("MERGE (e:MemoryNode" in c["cypher"] for c in node_merges), "memory nodes must use :MemoryNode label"
    assert all(
        (
            "e.tenant_id=CASE WHEN e.tenant_id IS NULL THEN n.tenant ELSE e.tenant_id END" in c["cypher"]
            or "MERGE (e:MemoryNode {id:n.id, tenant_id:$tenant})" in c["cypher"]
        )
        for c in node_merges
    )
    assert all(
        (
            all(node.get("tenant") == "tenant-a" for node in c["params"]["nodes"])
            or c["params"].get("tenant") == "tenant-a"
        )
        for c in node_merges
    )
    # Relationship MERGE statements present
    rel_merges = [c for c in calls if "MERGE (s)-[e:" in c["cypher"]]
    assert len(rel_merges) >= 1
    assert all(
        ("MERGE (s:MemoryNode" in c["cypher"] or "MATCH (s:MemoryNode" in c["cypher"])
        for c in rel_merges
    ), "memory rels must use :MemoryNode label"


def test_find_paths_returns_neighbors_map():
    calls: List[Dict[str, Any]] = []
    store = Neo4jStore({})
    store._driver = _Driver(calls)  # type: ignore[attr-defined]

    res = asyncio.run(store.find_paths(["seed"], rel_whitelist=["prefer","appears_in"], max_hops=2, min_weight=0.5, user_ids=["u"], memory_domain="home", cap=10))
    assert "neighbors" in res
    assert "seed" in res["neighbors"]
    nbrs = res["neighbors"]["seed"]
    assert any("MATCH (s:MemoryNode {id:$sid})" in c["cypher"] for c in calls), "find_paths must start from :MemoryNode"
    # From our fake run(): hop1 yields n1, hop2 yields n2
    ids = {n["to"] for n in nbrs}
    assert {"n1","n2"}.issubset(ids)
