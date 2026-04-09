from __future__ import annotations

"""
Export a subgraph (entries/ids) by filters (user_id/memory_domain/run_id) using MemoryService.search paging.

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/export_subgraph.py --domain home --user alice --out subgraph.json

Note: This uses search paging (k=200) and collects ids/contents/metadata. For large datasets,
consider a backend-native export.
"""

import argparse
import json
from typing import Any, Dict, List

from modules.memory.api.server import create_service
from modules.memory.contracts.memory_models import SearchFilters


async def _collect(svc, filters: Dict[str, Any], topk: int = 200, max_pages: int = 100) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # naive paging by repeated calls; in real systems use after/before or id ranges
    for _ in range(max_pages):
        res = await svc.search("", filters=SearchFilters.model_validate(filters), topk=topk, expand_graph=False)
        if not res.hits:
            break
        for h in res.hits:
            out.append({
                "id": h.id,
                "kind": h.entry.kind,
                "modality": h.entry.modality,
                "text": (h.entry.contents[0] if h.entry.contents else ""),
                "metadata": h.entry.metadata,
            })
        # crude break to avoid infinite loop in placeholder paging
        if len(res.hits) < topk:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", action="append", help="user_id (repeatable)")
    ap.add_argument("--domain", help="memory_domain")
    ap.add_argument("--run", help="run_id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    filters: Dict[str, Any] = {}
    if args.user:
        filters["user_id"] = list(args.user)
    if args.domain:
        filters["memory_domain"] = args.domain
    if args.run:
        filters["run_id"] = args.run

    svc = create_service()
    import asyncio
    items = asyncio.run(_collect(svc, filters))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"count": len(items), "items": items}, f, ensure_ascii=False, indent=2)
    print(f"exported {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()

