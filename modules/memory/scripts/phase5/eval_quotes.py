from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from modules.memory.scripts.phase5.metrics import precision_at_k, speaker_precision
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
        exp_quotes = list(s.get("expected_quotes") or [])
        limit = int(s.get("limit") or 0)
        if limit <= 0:
            limit = max(1, len(exp_quotes)) if exp_quotes else 10
        payload = {
            "entity": s.get("entity"),
            "entity_id": s.get("entity_id"),
            "topic": s.get("topic"),
            "topic_id": s.get("topic_id"),
            "topic_path": s.get("topic_path"),
            "user_tokens": s.get("user_tokens"),
            "limit": limit,
        }
        if args.base_url:
            pred = call_api(args.base_url, "/memory/v1/quotes", payload=payload, headers=headers)
        else:
            pred = s.get("pred", {})

        pred_quotes = list(pred.get("quotes") or [])
        gt_quotes = exp_quotes
        pred_ids = [q.get("utterance_id") for q in pred_quotes if q.get("utterance_id")]
        gt_ids = [q.get("utterance_id") for q in gt_quotes if q.get("utterance_id")]

        per_sample.append(
            {
                "query_id": s.get("query_id"),
                "quote_relevance": precision_at_k(pred_ids, gt_ids),
                "speaker_precision": speaker_precision(pred_quotes, gt_quotes),
            }
        )

    if per_sample:
        quote_rel = sum(x["quote_relevance"] for x in per_sample) / len(per_sample)
        spk_p = sum(x["speaker_precision"] for x in per_sample) / len(per_sample)
    else:
        quote_rel = spk_p = 0.0

    report = {
        "metric": "quotes",
        "quote_relevance": quote_rel,
        "speaker_precision": spk_p,
        "samples": per_sample,
    }
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
