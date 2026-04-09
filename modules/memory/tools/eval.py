from __future__ import annotations

"""
Evaluation utilities for retrieval weight tuning.

Dataset format (JSONL):
  {"query": "text", "gold_ids": ["uuid1","uuid2"], "notes": "optional"}

Service adapters:
  - HttpService(query, topk) -> list[{id, score, entry?}]
  - LocalService can be added later if needed

Report: returns dict with accuracy@k, latency stats, and per-weight grid results.
"""

from typing import Any, Dict, List
import json
import time


class ServiceAdapter:
    def search(self, query: str, *, topk: int = 10) -> List[Dict[str, Any]]:  # pragma: no cover (interface)
        raise NotImplementedError

    def set_modality_weights(self, weights: Dict[str, float]) -> None:  # pragma: no cover
        pass


class HttpService(ServiceAdapter):
    def __init__(self, base_url: str) -> None:
        import requests  # local import
        self._base = base_url.rstrip("/")
        self._s = requests.Session()

    def search(self, query: str, *, topk: int = 10) -> List[Dict[str, Any]]:
        url = f"{self._base}/search"
        r = self._s.post(url, json={"query": query, "topk": int(topk), "expand_graph": True}, timeout=30)
        r.raise_for_status()
        data = r.json() or {}
        res = data.get("hits") or data.get("results") or []
        # Normalize
        out: List[Dict[str, Any]] = []
        for h in res:
            if isinstance(h, dict) and "id" in h:
                out.append(h)
            elif hasattr(h, "id"):
                out.append({"id": getattr(h, "id"), "score": getattr(h, "score", 0.0), "entry": getattr(h, "entry", None)})
        return out

    def set_modality_weights(self, weights: Dict[str, float]) -> None:
        url = f"{self._base}/config/search/modality_weights"
        self._s.post(url, json={"weights": weights}, timeout=10)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("query"):
                    items.append(obj)
            except Exception:
                continue
    return items


def accuracy_at_k(results: List[List[Dict[str, Any]]], gold_ids: List[List[str]], *, k: int) -> float:
    n = len(results)
    if n == 0:
        return 0.0
    hit = 0
    for i in range(n):
        got = results[i][:k]
        gold = set([str(x) for x in (gold_ids[i] or [])])
        if not gold:
            continue
        for h in got:
            if str(h.get("id")) in gold:
                hit += 1
                break
    return float(hit) / float(n)


def evaluate(service: ServiceAdapter, dataset: List[Dict[str, Any]], *, topk: int = 5) -> Dict[str, Any]:
    results: List[List[Dict[str, Any]]] = []
    gold: List[List[str]] = []
    latencies: List[float] = []
    for item in dataset:
        q = str(item.get("query"))
        gold.append(list(item.get("gold_ids") or []))
        t0 = time.perf_counter()
        hits = service.search(q, topk=topk)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        results.append(hits)
    acc = accuracy_at_k(results, gold, k=topk)
    lat_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        idx = int(max(0, min(len(lat_sorted) - 1, round(p * (len(lat_sorted) - 1)))))
        return float(lat_sorted[idx])
    return {
        "acc@k": acc,
        "latency_ms": {
            "p50": pct(0.5),
            "p95": pct(0.95),
            "mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        },
        "count": len(dataset),
    }


def scan_weights(service: ServiceAdapter, dataset: List[Dict[str, Any]], *, topk: int, grids: List[Dict[str, float]]) -> Dict[str, Any]:
    """Try multiple modality weight settings and compare acc/latency.

    grids example: [{"text":1.0,"clip_image":0.8},{"text":0.9,"clip_image":1.1}]
    """
    out: List[Dict[str, Any]] = []
    for w in grids:
        try:
            service.set_modality_weights(w)
        except Exception:
            pass
        res = evaluate(service, dataset, topk=topk)
        out.append({"weights": dict(w), "metrics": res})
    # choose best by acc@k then p95 latency asc
    best = None
    for item in out:
        if best is None:
            best = item
            continue
        a = item["metrics"]["acc@k"]
        b = best["metrics"]["acc@k"]
        if a > b:
            best = item
        elif a == b and item["metrics"]["latency_ms"]["p95"] < best["metrics"]["latency_ms"]["p95"]:
            best = item
    return {"best": best, "all": out}

