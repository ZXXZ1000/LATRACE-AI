from __future__ import annotations

"""
Backup Qdrant collection points to a JSONL file via /scroll.

Usage (dry-run):
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/backup_qdrant.py \
      --host 127.0.0.1 --port 6333 --collection memory_text --out backup.jsonl --dry-run

Usage (live):
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/backup_qdrant.py \
      --host 127.0.0.1 --port 6333 --collection memory_text --out backup.jsonl

Notes:
  - This script uses /collections/<name>/points/scroll and streams results to JSONL.
  - For large datasets, adjust --batch to control per-request items.
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional


def build_scroll_payload(offset: Optional[Dict[str, Any]] = None, batch: int = 512) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"with_payload": True, "limit": int(batch)}
    if offset is not None:
        payload["offset"] = offset
    return payload


def scroll_once(host: str, port: int, collection: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/collections/{collection}/points/scroll"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:  # nosec - local tool
        out = resp.read().decode("utf-8")
        return json.loads(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backup Qdrant collection via /scroll")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = build_scroll_payload(batch=args.batch)
    if args.dry_run:
        print(json.dumps({"endpoint": f"http://{args.host}:{args.port}/collections/{args.collection}/points/scroll", "payload": payload}, ensure_ascii=False, indent=2))
        return

    offset = None
    total = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        while True:
            payload = build_scroll_payload(offset=offset, batch=args.batch)
            res = scroll_once(args.host, args.port, args.collection, payload)
            pts = ((res.get("result") or {}).get("points") or [])
            for p in pts:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                total += 1
            next_page = ((res.get("result") or {}).get("next_page_offset"))
            if not next_page:
                break
            offset = next_page
    print(f"Backed up {total} points to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

