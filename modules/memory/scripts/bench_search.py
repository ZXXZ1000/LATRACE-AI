from __future__ import annotations

"""
Simple search benchmark script.

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python -m modules.memory.scripts.bench_search \
      --queries "客厅 开灯,喜欢 披萨" --iters 200

It seeds a small corpus in the in-memory stores, runs repeated searches, and prints P50/P95/P99.
"""

import argparse
import asyncio
import statistics
from time import perf_counter

from modules.memory.application.service import MemoryService
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, default="客厅 开灯,喜欢 披萨", help="Comma-separated queries")
    parser.add_argument("--iters", type=int, default=200, help="Total search iterations")
    args = parser.parse_args()

    vec = InMemVectorStore()
    graph = InMemGraphStore()
    svc = MemoryService(vec, graph, AuditStore())

    # seed corpus
    await svc.write([
        MemoryEntry(kind="episodic", modality="text", contents=["客厅 打开 主灯"], metadata={"source": "ctrl"}),
        MemoryEntry(kind="episodic", modality="text", contents=["厨房 关闭 灯"], metadata={"source": "ctrl"}),
        MemoryEntry(kind="semantic", modality="text", contents=["我 喜欢 奶酪 披萨"], metadata={"source": "mem0"}),
        MemoryEntry(kind="semantic", modality="text", contents=["我 不 喜欢 奶酪 披萨"], metadata={"source": "mem0"}),
    ])

    qlist = [q.strip() for q in args.queries.split(",") if q.strip()]
    lat = []
    for i in range(args.iters):
        q = qlist[i % len(qlist)]
        t0 = perf_counter()
        await svc.search(q, topk=3, filters=SearchFilters(modality=["text"]))
        lat.append((perf_counter() - t0) * 1000.0)

    print("Count:", len(lat))
    print("P50(ms):", round(percentile(lat, 0.50), 2))
    print("P95(ms):", round(percentile(lat, 0.95), 2))
    print("P99(ms):", round(percentile(lat, 0.99), 2))
    print("Mean(ms):", round(statistics.mean(lat), 2))


if __name__ == "__main__":
    asyncio.run(main())

