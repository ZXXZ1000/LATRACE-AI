from __future__ import annotations

from typing import Any, Dict, List

from modules.memory.tools.eval import ServiceAdapter, evaluate, scan_weights


class _FakeSvc(ServiceAdapter):
    def __init__(self) -> None:
        # weights affect order: if clip_image weight > text weight, we rank id 'img' first
        self._mw = {"text": 1.0, "clip_image": 0.85}

    def search(self, query: str, *, topk: int = 5) -> List[Dict[str, Any]]:
        # two candidates, IDs fixed, scores depend on weights
        t = float(self._mw.get("text", 1.0))
        c = float(self._mw.get("clip_image", 0.85))
        res = [
            {"id": "txt", "score": 0.6 * t},
            {"id": "img", "score": 0.6 * c},
        ]
        res.sort(key=lambda x: x["score"], reverse=True)
        return res[:topk]

    def set_modality_weights(self, weights: Dict[str, float]) -> None:
        self._mw = dict(weights or {})


def test_evaluate_and_scan_weights():
    svc = _FakeSvc()
    dataset = [
        {"query": "q1", "gold_ids": ["img"]},  # expecting image-first to be correct
        {"query": "q2", "gold_ids": ["img"]},
    ]
    # direct eval
    res = evaluate(svc, dataset, topk=1)
    assert isinstance(res.get("acc@k"), float)
    # scan weights, expect config with higher clip_image to win
    grids = [
        {"text": 1.0, "clip_image": 0.8},
        {"text": 0.8, "clip_image": 1.2},
    ]
    rep = scan_weights(svc, dataset, topk=1, grids=grids)
    best = rep.get("best", {})
    assert best and best["weights"]["clip_image"] > best["weights"]["text"]

