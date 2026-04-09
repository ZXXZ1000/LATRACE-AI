from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _iter_events(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "event" in obj and isinstance(obj["event"], dict):
                yield obj["event"]
                continue
            # Qdrant point payloads: {payload:{...}} or {payload:{metadata:{...}}}
            if isinstance(obj, dict) and "payload" in obj:
                payload = obj.get("payload") or {}
                if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
                    yield payload.get("metadata")  # type: ignore[misc]
                    continue
                if isinstance(payload, dict):
                    yield payload
                    continue
            if isinstance(obj, dict):
                yield obj


def _is_event_related(ev: Dict[str, Any], sources: Sequence[str]) -> bool:
    if ev.get("node_type") == "Event":
        return True
    src = str(ev.get("source") or "").strip()
    if src and src in sources:
        return True
    if ev.get("event_id") or ev.get("tkg_event_id") or ev.get("node_id"):
        return True
    tkg_ids = ev.get("tkg_event_ids")
    if isinstance(tkg_ids, list) and len(tkg_ids) > 0:
        return True
    return False


def _parse_multi(values: List[str] | None) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _match_filters(ev: Dict[str, Any], tenants: Sequence[str], domains: Sequence[str], sources: Sequence[str]) -> bool:
    if tenants:
        tid = str(ev.get("tenant_id") or "").strip()
        if tid not in tenants:
            return False
    if domains:
        dom = str(ev.get("memory_domain") or "").strip()
        if dom not in domains:
            return False
    if sources:
        src = str(ev.get("source") or "").strip()
        if src not in sources:
            return False
    return True


def _ratio(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round(n / d, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Topic coverage report from JSONL events.")
    parser.add_argument("--input", "-i", action="append", required=True, help="JSONL file path (repeatable)")
    parser.add_argument("--event-only", action="store_true", help="Only count event-related entries")
    parser.add_argument(
        "--event-index-only",
        action="store_true",
        help="Shortcut for event index coverage (sets --event-only and source=tkg_dialog_event_index_v1)",
    )
    parser.add_argument("--tenant", action="append", help="Tenant id filter (repeatable or comma-separated)")
    parser.add_argument("--memory-domain", action="append", help="Memory domain filter (repeatable or comma-separated)")
    parser.add_argument("--source", action="append", help="Source filter (repeatable or comma-separated)")
    parser.add_argument(
        "--event-sources",
        help="Comma-separated sources treated as events (default: tkg_dialog_event_index_v1,dialog_unified)",
    )
    parser.add_argument("--output", "-o", help="Optional output JSON path")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    sources = ["tkg_dialog_event_index_v1", "dialog_unified"]
    if args.event_sources:
        sources = [s.strip() for s in args.event_sources.split(",") if s.strip()]
    if args.event_index_only:
        sources = ["tkg_dialog_event_index_v1"]
    tenants = _parse_multi(args.tenant)
    domains = _parse_multi(args.memory_domain)
    src_filter = _parse_multi(args.source)
    if args.event_index_only and not src_filter:
        src_filter = ["tkg_dialog_event_index_v1"]
        args.event_only = True
    total = 0
    has_topic = 0
    uncategorized = 0
    has_tags = 0
    has_keywords = 0
    has_time_bucket = 0

    for ev in _iter_events(inputs):
        if args.event_only and not _is_event_related(ev, sources):
            continue
        if (tenants or domains or src_filter) and not _match_filters(ev, tenants, domains, src_filter):
            continue
        total += 1
        topic_path = str(ev.get("topic_path") or "").strip()
        if topic_path:
            has_topic += 1
            if topic_path.startswith("_uncategorized/"):
                uncategorized += 1
        tags = ev.get("tags") or []
        if isinstance(tags, list) and len(tags) > 0:
            has_tags += 1
        keywords = ev.get("keywords") or []
        if isinstance(keywords, list) and len(keywords) > 0:
            has_keywords += 1
        tb = ev.get("time_bucket") or []
        if isinstance(tb, list) and len(tb) > 0:
            has_time_bucket += 1

    report = {
        "total_events": total,
        "topic_path_coverage": _ratio(has_topic, total),
        "uncategorized_ratio": _ratio(uncategorized, total),
        "tags_coverage": _ratio(has_tags, total),
        "keywords_coverage": _ratio(has_keywords, total),
        "time_bucket_coverage": _ratio(has_time_bucket, total),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
