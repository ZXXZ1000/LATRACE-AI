from __future__ import annotations

from typing import Any, Dict, Tuple

from modules.memory.application.service import MemoryService


class _VecStub:
    async def health(self) -> dict:
        return {"status": "ok"}


class _AuditStub:
    def add_one(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        class V:  # minimal Version-like
            def __init__(self) -> None:
                self.value = "v1"
        return V()


class _GraphFake:
    def __init__(self) -> None:
        # map (src,dst) -> {pending: bool, score: float|None}
        self._eq: Dict[Tuple[str, str], Dict[str, Any]] = {}

    async def add_pending_equivalence(self, src: str, dst: str, *, score=None, reason=None):  # type: ignore[no-untyped-def]
        self._eq[(src, dst)] = {"pending": True, "score": score, "reason": reason}

    async def list_pending_equivalence(self, *, limit: int = 50) -> dict:
        out = []
        for (s, d), meta in list(self._eq.items()):
            if not meta.get("pending"):
                continue
            out.append({"src_id": s, "dst_id": d, "score": meta.get("score")})
            if len(out) >= limit:
                break
        return {"pending": out}

    async def confirm_equivalence(self, pairs: list[tuple[str, str]], *, weight: float | None = None) -> int:
        cnt = 0
        for p in pairs:
            meta = self._eq.get(p)
            if meta is None:
                self._eq[p] = {"pending": False, "score": None}
                cnt += 1
            else:
                meta["pending"] = False
                cnt += 1
        return cnt

    async def delete_equivalence(self, pairs: list[tuple[str, str]]) -> int:
        cnt = 0
        for p in pairs:
            if p in self._eq:
                self._eq.pop(p, None)
                cnt += 1
        return cnt


def test_equiv_pending_service_flow():
    svc = MemoryService(_VecStub(), _GraphFake(), _AuditStub())
    pairs = [("a", "b"), ("b", "c")]
    # add pending
    n = _run_async(svc.add_pending_equivalence(pairs, scores=[0.9, 0.7]))
    assert n == 2
    lst = _run_async(svc.list_pending_equivalence(limit=10))
    assert len(lst.get("pending", [])) == 2
    # confirm one
    m = _run_async(svc.confirm_pending_equivalence([pairs[0]], weight=0.1))
    assert m == 1
    lst2 = _run_async(svc.list_pending_equivalence(limit=10))
    # one remains pending
    assert len(lst2.get("pending", [])) == 1
    # remove remaining pending
    k = _run_async(svc.remove_pending_equivalence([pairs[1]]))
    assert k == 1


def _run_async(coro):  # type: ignore[no-untyped-def]
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # run in a new loop in thread to avoid nested event loop issues
    result: Dict[str, Any] = {"value": None}
    def _w():
        nl = asyncio.new_event_loop()
        asyncio.set_event_loop(nl)
        try:
            result["value"] = nl.run_until_complete(coro)
        finally:
            nl.close()
    import threading
    t = threading.Thread(target=_w)
    t.start()
    t.join()
    return result["value"]
