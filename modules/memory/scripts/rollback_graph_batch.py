from __future__ import annotations

"""
Rollback a list of versions recorded by AuditStore (UPDATE/DELETE soft/hard).

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/rollback_graph_batch.py --versions versions.txt --dry-run
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/rollback_graph_batch.py --versions versions.txt

Each line of versions.txt should be an audit version string (e.g., v-UPDATE-<id> or v-DELETE-<id>).
"""

import argparse
from typing import List

from modules.memory.api.server import create_service


def load_versions(path: str) -> List[str]:
    return [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    versions = load_versions(args.versions)
    print(f"Loaded {len(versions)} versions from {args.versions}")

    svc = create_service()
    import asyncio
    async def _run():
        ok = 0
        for v in versions:
            if args.dry_run:
                print(f"DRYRUN would rollback {v}")
                ok += 1
                continue
            res = await svc.rollback_version(v)
            print(f"rollback {v}: {res}")
            ok += 1 if res else 0
        print(f"Done. attempted={len(versions)} ok={ok}")
    asyncio.run(_run())


if __name__ == "__main__":
    main()

