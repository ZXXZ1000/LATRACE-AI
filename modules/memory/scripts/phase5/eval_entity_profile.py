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


def _facts_to_text(items: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for f in items or []:
        for key in ("summary", "statement"):
            val = f.get(key)
            if val:
                out.append(str(val))
                break
    return out


def main() -> int:
    args = _parse_args()
    samples = load_jsonl(Path(args.input))
    headers = build_headers(args.tenant_id, args.api_key)

    per_sample: List[Dict[str, Any]] = []
    for s in samples:
        exp_facts = list(s.get("expected_facts") or [])
        exp_rel = list(s.get("expected_relations") or [])
        exp_events = list(s.get("expected_recent_event_ids") or [])
        facts_limit = int(s.get("facts_limit") or 0) or (len(exp_facts) if exp_facts else 0)
        relations_limit = int(s.get("relations_limit") or 0) or (len(exp_rel) if exp_rel else 0)
        events_limit = int(s.get("events_limit") or 0) or (len(exp_events) if exp_events else 0)
        payload = {
            "entity": s.get("entity"),
            "entity_id": s.get("entity_id"),
            "user_tokens": s.get("user_tokens"),
            "facts_limit": facts_limit,
            "relations_limit": relations_limit,
            "events_limit": events_limit,
        }
        if args.base_url:
            pred = call_api(args.base_url, "/memory/v1/entity-profile", payload=payload, headers=headers)
        else:
            pred = s.get("pred", {})

        pred_facts = _facts_to_text(pred.get("facts") or [])
        gt_facts = exp_facts
        pred_rel = [r.get("entity_id") or r.get("name") for r in pred.get("relations", [])]
        gt_rel = exp_rel
        pred_events = [e.get("event_id") for e in pred.get("recent_events", []) if e.get("event_id")]
        gt_events = exp_events

        per_sample.append(
            {
                "query_id": s.get("query_id"),
                "facts_precision": precision_at_k(pred_facts, gt_facts),
                "relations_precision": precision_at_k(pred_rel, gt_rel),
                "recent_events_precision": precision_at_k(pred_events, gt_events),
            }
        )

    if per_sample:
        facts_p = sum(x["facts_precision"] for x in per_sample) / len(per_sample)
        rel_p = sum(x["relations_precision"] for x in per_sample) / len(per_sample)
        ev_p = sum(x["recent_events_precision"] for x in per_sample) / len(per_sample)
    else:
        facts_p = rel_p = ev_p = 0.0

    report = {
        "metric": "entity_profile",
        "facts_precision": facts_p,
        "relations_precision": rel_p,
        "recent_events_precision": ev_p,
        "samples": per_sample,
    }
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
