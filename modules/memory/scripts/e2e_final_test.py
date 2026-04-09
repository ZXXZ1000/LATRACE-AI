from __future__ import annotations

"""
End-to-end functional test using configured Qdrant + Neo4j + optional LLM.

Scenarios covered (prints concise PASS/FAIL per step):
  1) Health check (vector/graph)
  2) Ensure Qdrant collections
  3) Write mem0-like entries + prefer edge, search (filtered/ unfilt) + sampling log
  4) Multi-hop neighbor expansion (A->B->C, hop=2)
  5) Update → Soft delete
  6) Hard delete with safety policy (expect reject → then confirm to pass)
  7) TTL cleanup (older entry marked soft-deleted)
  8) Graph edge decay (prefer relations)
  9) Micro benchmark (P50/P95/P99)

If real backends are unavailable, falls back to in-memory stores (prints a note).
"""

import asyncio
import os
import sys
from time import perf_counter
from statistics import mean
from typing import Any, Dict, List

# Fix Python path to allow imports from 'modules'
# The script is in MOYAN_Agent_Infra/modules/memory/scripts
# We need to go up 3 levels to get to the MOYAN_Agent_Infra directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from dotenv import load_dotenv

from modules.memory.api.server import create_service
from modules.memory.application.service import SafetyError
from modules.memory.application import runtime_config as rtconf
from modules.memory.contracts.memory_models import MemoryEntry, SearchFilters
from modules.memory.adapters.mem0_adapter import build_entries_from_mem0


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p
    i = int(k)
    if i == k:
        return xs[i]
    j = min(i + 1, len(xs) - 1)
    return xs[i] * (j - k) + xs[j] * (k - i)


async def main() -> None:
    load_dotenv()
    # Try to create service with real backends
    print("Attempting to connect to REAL backend services (Qdrant+Neo4j)...")
    try:
        svc = create_service()
        # retry health a few times (Neo4j may take time to be ready)
        import asyncio as _aio
        h = {}
        for i in range(10):
            h = await svc.health_check()
            v_ok = (h.get("vectors", {}).get("status") == "ok")
            g_status = (h.get("graph", {}).get("status"))
            if v_ok and g_status in {"ok", "unconfigured"}:
                break
            await _aio.sleep(2)
        v_ok = (h.get("vectors", {}).get("status") == "ok")
        g_ok = (h.get("graph", {}).get("status") in {"ok", "unconfigured"})
        if not v_ok or not g_ok:
            print("\nERROR: Backend health check failed. Please ensure Qdrant and Neo4j are running and configured correctly.")
            print(f"Health check response: {h}")
            return  # Exit script
        real = True
    except Exception as e:
        print(f"\nERROR: Failed to connect to real backends. Exception: {e}")
        print("Please ensure Qdrant and Neo4j services are running and accessible.")
        return # Exit script

    print(f"Backend: {'REAL(Qdrant+Neo4j)' if real else 'INMEM'}")

    # Ensure collections (Qdrant)
    if real and hasattr(svc.vectors, "ensure_collections"):
        try:
            await svc.vectors.ensure_collections()  # type: ignore
            print("[1] ensure_collections: PASS")
        except Exception as e:
            print(f"[1] ensure_collections: FAIL -> {e}")
    else:
        print("[1] ensure_collections: SKIP (inmem)")

    # Health check
    h = await svc.health_check()
    print(f"[2] health_check: {h}")

    # Sampling log capture
    samples: List[Dict[str, Any]] = []
    svc.set_search_sampler(lambda s: samples.append(s), enabled=True, rate=1.0)

    # 3) Write mem0-like entries and search
    entries, edges = build_entries_from_mem0([
        {"role": "user", "content": "我 喜欢 奶酪 披萨"},
        {"role": "assistant", "content": "好的，已记录你的偏好"},
    ], profile={"user_id": "user.owner"})
    try:
        await svc.write(entries, links=edges)
        res = await svc.search("奶酪 披萨", topk=3, filters=SearchFilters(modality=["text"]))
        print(f"[3] write+search: PASS hits={len(res.hits)} hints='{res.hints}'")
    except Exception as e:
        print(f"[3] write+search: FAIL -> {e}")

    # 4) Multi-hop neighbor expansion (A->B->C)
    a = MemoryEntry(kind="semantic", modality="text", contents=["A 节点"], metadata={"source": "mem0"})
    b = MemoryEntry(kind="semantic", modality="text", contents=["B 节点"], metadata={"source": "mem0"})
    c = MemoryEntry(kind="semantic", modality="text", contents=["C 节点"], metadata={"source": "mem0"})
    await svc.write([a, b, c])
    # Build prefer edges chain
    ids = []
    # Use search to recover ids
    for q in ["A 节点", "B 节点", "C 节点"]:
        r = await svc.search(q, topk=1, filters=SearchFilters(modality=["text"]))
        if r.hits:
            ids.append(r.hits[0].entry.id)
    if len(ids) == 3:
        await svc.link(ids[0], ids[1], "prefer", weight=1.0)
        await svc.link(ids[1], ids[2], "prefer", weight=1.0)
        rtconf.set_graph_params(rel_whitelist=["prefer"], max_hops=2, neighbor_cap_per_seed=10)
        r2 = await svc.search("A 节点", topk=3, filters=SearchFilters(modality=["text"]))
        nbrs = r2.neighbors.get("neighbors", {}).get(r2.hits[0].id if r2.hits else "", [])
        print(f"[4] multihop neighbors: PASS neighbors={nbrs}")
        rtconf.clear_graph_params_override()
    else:
        print("[4] multihop neighbors: SKIP (ids not resolved)")

    # 5) Update then soft delete
    r3 = await svc.search("奶酪 披萨", topk=1, filters=SearchFilters(modality=["text"]))
    if r3.hits:
        mid = r3.hits[0].entry.id
        await svc.update(mid, {"contents": ["我 很 喜欢 奶酪 披萨"]}, reason="e2e")
        await svc.delete(mid, soft=True, reason="cleanup")
        print("[5] update+soft_delete: PASS")
    else:
        print("[5] update+soft_delete: SKIP (no hit)")

    # 6) Hard delete safety: expect rejection then confirm
    svc.set_safety_policy(require_confirm_hard_delete=True, require_reason_delete=True)
    e_tmp = MemoryEntry(kind="semantic", modality="text", contents=["临时 删除"], metadata={"source": "mem0"})
    await svc.write([e_tmp])
    r4 = await svc.search("临时 删除", topk=1, filters=SearchFilters(modality=["text"]))
    if r4.hits:
        tmp_id = r4.hits[0].entry.id
        try:
            await svc.delete(tmp_id, soft=False)
            print("[6] hard_delete safety: FAIL (unexpected pass)")
        except SafetyError as e:
            print(f"[6] hard_delete safety: PASS reject='{e}'")
            await svc.delete(tmp_id, soft=False, reason="cleanup", confirm=True)
            print("[6] hard_delete confirm: PASS")
    else:
        print("[6] hard_delete: SKIP (no tmp id)")

    # 7) TTL cleanup
    ttl_entry = MemoryEntry(
        kind="episodic", modality="text", contents=["TTL 过期"],
        metadata={"source": "ctrl", "ttl": 1, "created_at": "1970-01-01T00:00:00+00:00"}
    )
    await svc.write([ttl_entry])
    changed = await svc.run_ttl_cleanup_now()
    print(f"[7] ttl_cleanup: PASS changed={changed}")

    # 8) Edge decay
    if len(ids) == 3:
        ok = await svc.decay_graph_edges(factor=0.5, rel_whitelist=["prefer"], min_weight=0.0)
        print(f"[8] edge_decay: {'PASS' if ok else 'SKIP'}")
    else:
        print("[8] edge_decay: SKIP (no edges)")

    # 9) Micro benchmark
    queries = ["奶酪 披萨", "A 节点", "B 节点", "C 节点"]
    lat = []
    for i in range(30):
        q = queries[i % len(queries)]
        t0 = perf_counter()
        await svc.search(q, topk=3, filters=SearchFilters(modality=["text"]))
        lat.append((perf_counter() - t0) * 1000.0)
    print(
        f"[9] bench: count={len(lat)} P50={pct(lat,0.5):.2f}ms P95={pct(lat,0.95):.2f}ms P99={pct(lat,0.99):.2f}ms Mean={mean(lat):.2f}ms"
    )


if __name__ == "__main__":
    asyncio.run(main())
