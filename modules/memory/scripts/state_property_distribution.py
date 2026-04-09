from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _iter_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
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
            if isinstance(obj, dict):
                yield obj


def _extract_text(obj: Dict[str, Any]) -> str:
    # qdrant payload
    payload = obj.get("payload")
    if isinstance(payload, dict):
        text = payload.get("content") or payload.get("contents")
        if isinstance(text, str) and text.strip():
            return text.strip()
    # event-like dict
    summary = str(obj.get("summary") or "").strip()
    desc = str(obj.get("desc") or "").strip()
    if summary or desc:
        return " ".join([t for t in [summary, desc] if t])
    return ""


def _metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    payload = obj.get("payload") or {}
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        return meta
    if isinstance(obj.get("metadata"), dict):
        return obj.get("metadata")  # type: ignore[return-value]
    return {}


def _compile_rules() -> Dict[str, Sequence[str]]:
    return {
        "location": [
            r"\b(moved to|move to|relocated to|living in|live in|based in|located in|staying in)\b",
            r"(住在|搬到|迁到|定居|位于)[\u4e00-\u9fff]{2,6}",
            r"(去了|到达)[\u4e00-\u9fff]{2,6}",
        ],
        "job_status": [
            r"\b(hired|job|career|promotion|layoff|laid off|unemployed|offer|internship|resign|quit|fired)\b",
            r"(求职|面试|晋升|裁员|失业|入职|离职|辞职|实习|转岗|换岗)",
        ],
        "relationship_status": [
            r"\b(boyfriend|girlfriend|dating|married|divorced|breakup|relationship|engaged|wedding)\b",
            r"(恋爱|男朋友|女朋友|结婚|离婚|分手|订婚|婚礼|相亲)",
        ],
        "health": [
            r"\b(sick|ill|hospital|doctor|surgery|medicine|therapy|clinic|checkup)\b",
            r"(生病|医院|医生|手术|药物|治疗|检查|看病)",
        ],
        "mood": [
            r"\b(happy|sad|anxious|depressed|stress|stressed|overwhelmed|angry|lonely)\b",
            r"(开心|难过|焦虑|抑郁|压力|崩溃|生气|孤独|紧张)",
        ],
        "finance": [
            r"\b(budget|salary|paycheck|debt|loan|broke|savings|expense|rent)\b",
            r"(工资|薪水|预算|欠债|贷款|花费|理财|房租)",
        ],
    }


def _match_any(text: str, patterns: Sequence[str]) -> bool:
    if not text:
        return False
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate candidate State properties by heuristic triggers.")
    parser.add_argument("--input", "-i", action="append", required=True, help="JSONL input (repeatable)")
    parser.add_argument("--source", help="Filter by metadata.source (e.g. tkg_dialog_event_index_v1)")
    parser.add_argument("--top", type=int, default=6, help="Top properties to print")
    parser.add_argument("--output", "-o", help="Optional output JSON path")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    rules = _compile_rules()
    counts: Dict[str, int] = {k: 0 for k in rules}
    samples: Dict[str, List[str]] = {k: [] for k in rules}

    for obj in _iter_records(inputs):
        meta = _metadata(obj)
        if args.source:
            src = str(meta.get("source") or "").strip()
            if src != args.source:
                continue
        text = _extract_text(obj)
        if not text:
            continue
        for prop, patterns in rules.items():
            if _match_any(text, patterns):
                counts[prop] += 1
                if len(samples[prop]) < 3:
                    samples[prop].append(text[:160])

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    report = {
        "total_properties": len(rules),
        "top": ranked[: args.top],
        "counts": counts,
        "samples": samples,
        "source": args.source,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
