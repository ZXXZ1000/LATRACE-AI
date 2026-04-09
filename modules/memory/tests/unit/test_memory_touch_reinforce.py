import asyncio

from modules.memory.application.service import MemoryService


class _StubGraph:
    def __init__(self):
        self.calls = []

    async def touch(self, *, tenant_id: str, node_ids: list[str], extend_seconds=None):
        self.calls.append({"tenant": tenant_id, "ids": list(node_ids), "extend_seconds": extend_seconds})
        return {"updated": len(node_ids)}


def test_touch_filter_and_throttle():
    graph = _StubGraph()
    svc = MemoryService(None, graph, None)  # type: ignore[arg-type]
    svc.set_graph_tenant("t")
    svc._touch_min_interval_s = 10.0
    svc._touch_max_batch = 2
    svc._touch_extend_seconds = 5.0

    # first pass: enforce max_batch=2, expect a/b touched and last_access recorded
    asyncio.run(svc._touch_nodes(["a", "b", "c"], tenant_id="t"))
    assert graph.calls[0]["ids"] == ["a", "b"]
    assert graph.calls[0]["extend_seconds"] == 5.0

    # second pass immediately: "a"/"b" throttled by min_interval, only new "c" passes
    asyncio.run(svc._touch_nodes(["a", "c"], tenant_id="t"))
    assert graph.calls[-1]["ids"] == ["c"]
    assert graph.calls[-1]["tenant"] == "t"
