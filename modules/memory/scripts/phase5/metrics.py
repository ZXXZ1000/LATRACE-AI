from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence


def _normalize_list(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values or []:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def precision_at_k(pred_ids: Sequence[str], gt_ids: Sequence[str], k: Optional[int] = None) -> float:
    pred = _normalize_list(pred_ids)
    gt = set(_normalize_list(gt_ids))
    if not pred:
        return 0.0
    if k is not None:
        pred = pred[: max(1, int(k))]
    hits = sum(1 for p in pred if p in gt)
    return hits / float(len(pred)) if pred else 0.0


def recall_at_k(pred_ids: Sequence[str], gt_ids: Sequence[str], k: Optional[int] = None) -> float:
    pred = _normalize_list(pred_ids)
    gt = _normalize_list(gt_ids)
    if not gt:
        return 0.0
    if k is not None:
        pred = pred[: max(1, int(k))]
    hits = sum(1 for p in pred if p in set(gt))
    return hits / float(len(gt)) if gt else 0.0


def order_consistency(pred_ids: Sequence[str], expected_order: Sequence[str]) -> float:
    pred = _normalize_list(pred_ids)
    exp = _normalize_list(expected_order)
    if not exp or not pred:
        return 0.0
    index_map = {eid: i for i, eid in enumerate(pred)}
    last_idx = -1
    for eid in exp:
        if eid not in index_map:
            continue
        if index_map[eid] < last_idx:
            return 0.0
        last_idx = index_map[eid]
    return 1.0


def speaker_precision(
    pred_quotes: Sequence[Dict[str, str]],
    gt_quotes: Sequence[Dict[str, str]],
) -> float:
    if not pred_quotes:
        return 0.0
    gt_map: Dict[str, str] = {}
    for q in gt_quotes or []:
        utt = str(q.get("utterance_id") or "").strip()
        sid = str(q.get("speaker_id") or "").strip()
        if utt:
            gt_map[utt] = sid
    if not gt_map:
        return 0.0
    correct = 0
    total = 0
    for q in pred_quotes or []:
        utt = str(q.get("utterance_id") or "").strip()
        sid = str(q.get("speaker_id") or "").strip()
        if not utt:
            continue
        total += 1
        if gt_map.get(utt) == sid:
            correct += 1
    return correct / float(total) if total else 0.0


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def absolute_time_error_days(pred_iso: Optional[str], gt_iso: Optional[str]) -> Optional[float]:
    pred_dt = parse_iso(pred_iso)
    gt_dt = parse_iso(gt_iso)
    if pred_dt is None or gt_dt is None:
        return None
    if pred_dt.tzinfo is None:
        pred_dt = pred_dt.replace(tzinfo=timezone.utc)
    if gt_dt.tzinfo is None:
        gt_dt = gt_dt.replace(tzinfo=timezone.utc)
    return abs((pred_dt - gt_dt).total_seconds()) / 86400.0

