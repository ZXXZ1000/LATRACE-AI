from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from modules.memory.scripts.phase5.metrics import precision_at_k
from modules.memory.scripts.phase5.utils import build_headers, call_api, load_jsonl, write_json


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--tenant-id", default="t1")
    ap.add_argument("--api-key", default=None)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    samples = load_jsonl(Path(args.input))
    headers = build_headers(args.tenant_id, args.api_key)

    per_sample: List[Dict[str, Any]] = []
    for s in samples:
        payload = {
            "entity": s.get("entity"),
            "entity_id": s.get("entity_id"),
            "user_tokens": s.get("user_tokens"),
            "time_range": s.get("time_range"),
        }
        if args.base_url:
            pred = call_api(args.base_url, "/memory/v1/relations", payload=payload, headers=headers)
        else:
            pred = s.get("pred", {})

        pred_ids = [r.get("entity_id") or r.get("name") for r in pred.get("relations", [])]
        gt_ids = list(s.get("expected_related_entities") or [])

        per_sample.append(
            {
                "query_id": s.get("query_id"),
                "precision@k": precision_at_k(pred_ids, gt_ids),
            }
        )

    precision = sum(x["precision@k"] for x in per_sample) / len(per_sample) if per_sample else 0.0
    report = {"metric": "relations", "precision@k": precision, "samples": per_sample}
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
