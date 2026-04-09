from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from modules.memory.scripts.phase5.metrics import absolute_time_error_days
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
    errors: List[float] = []
    for s in samples:
        payload = {
            "topic": s.get("topic"),
            "topic_id": s.get("topic_id"),
            "topic_path": s.get("topic_path"),
            "entity": s.get("entity"),
            "entity_id": s.get("entity_id"),
            "user_tokens": s.get("user_tokens"),
            "time_range": s.get("time_range"),
        }
        if args.base_url:
            pred = call_api(args.base_url, "/memory/v1/time-since", payload=payload, headers=headers)
        else:
            pred = s.get("pred", {})

        err = absolute_time_error_days(pred.get("last_mentioned"), s.get("expected_last_mentioned"))
        if err is not None:
            errors.append(err)
        per_sample.append(
            {
                "query_id": s.get("query_id"),
                "error_days": err,
            }
        )

    avg_err = sum(errors) / len(errors) if errors else None
    report = {"metric": "time_since", "avg_error_days": avg_err, "samples": per_sample}
    write_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
