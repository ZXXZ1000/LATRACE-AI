from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from modules.memory.scripts.phase5.metrics import order_consistency, precision_at_k, recall_at_k
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
            "topic": s.get("topic"),
            "topic_id": s.get("topic_id"),
            "topic_path": s.get("topic_path"),
            "user_tokens": s.get("user_tokens"),
            "time_range": s.get("time_range"),
            "limit": int(s.get("limit") or 50),
        }
        if args.base_url:
            pred = call_api(args.base_url, "/memory/v1/topic-timeline", payload=payload, headers=headers)
        else:
            pred = s.get("pred", {})

        pred_ids = [x.get("event_id") for x in pred.get("timeline", []) if x.get("event_id")]
        gt_ids = list(s.get("expected_event_ids") or [])
        exp_order = list(s.get("expected_order") or [])
        per_sample.append(
            {
                "query_id": s.get("query_id"),
                "precision@k": precision_at_k(pred_ids, gt_ids),
                "recall@k": recall_at_k(pred_ids, gt_ids),
                "order_consistency": order_consistency(pred_ids, exp_order),
                "pred_count": len(pred_ids),
                "gt_count": len(gt_ids),
            }
        )

    if per_sample:
        precision = sum(x["precision@k"] for x in per_sample) / len(per_sample)
        recall = sum(x["recall@k"] for x in per_sample) / len(per_sample)
        order = sum(x["order_consistency"] for x in per_sample) / len(per_sample)
    else:
        precision = recall = order = 0.0

    report = {
        "metric": "topic_timeline",
        "precision@k": precision,
        "recall@k": recall,
        "order_consistency": order,
        "samples": per_sample,
    }
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
