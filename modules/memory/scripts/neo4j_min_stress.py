from __future__ import annotations

"""
Minimal Neo4j batch write stress (development only).

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/neo4j_min_stress.py --nodes 2000 --batch 500

Requires a running Neo4j and env NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD set in .env or shell.
"""

import argparse
import time
from typing import List

from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.contracts.memory_models import MemoryEntry


def build_entries(n: int) -> List[MemoryEntry]:
    out = []
    for i in range(n):
        out.append(MemoryEntry(id=f"stress-{i}", kind="semantic", modality="text", contents=[f"node {i}"], metadata={"source": "stress"}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    store = Neo4jStore({})
    entries = build_entries(args.nodes)
    t0 = time.perf_counter()
    import asyncio
    asyncio.run(store.merge_nodes_edges_batch(entries, [], chunk_size=args.batch))
    dt = (time.perf_counter() - t0) * 1000
    print(f"Inserted {args.nodes} nodes in {dt:.1f} ms (batch={args.batch})")


if __name__ == "__main__":
    main()

