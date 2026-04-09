from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _norm_list(val: Any) -> tuple[str, ...]:
    if val is None:
        return tuple()
    items = val if isinstance(val, list) else [val]
    out = [str(x).strip() for x in items if str(x).strip()]
    return tuple(sorted(set(out)))


def _signature(match: Dict[str, Any]) -> Tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        _norm_list(match.get("places")),
        _norm_list(match.get("keywords_any")),
        _norm_list(match.get("keywords_all")),
    )


def test_normalization_rules_no_conflict() -> None:
    vocab_path = Path(__file__).resolve().parents[4] / "modules" / "memory" / "vocab" / "normalization_rules.yaml"
    data = yaml.safe_load(vocab_path.read_text(encoding="utf-8")) or {}
    rules = data.get("rules") or []
    seen: Dict[Tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], str] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        match = rule.get("match") or {}
        out = rule.get("output") or {}
        topic_path = str(out.get("topic_path") or "").strip()
        if not topic_path:
            continue
        sig = _signature(match)
        if sig in seen:
            # if same match signature, topic_path must be identical
            assert seen[sig] == topic_path
        else:
            seen[sig] = topic_path
