from __future__ import annotations

import pytest

from modules.memory.api.mcp_server import MemoryMCPAdapter


@pytest.mark.anyio
async def test_mcp_memory_tools_end_to_end():
    adapter = MemoryMCPAdapter.from_defaults()
    # write two semantic entries and one prefer link
    r_write = await adapter.invoke(
        "memory.write",
        {
            "entries": [
                {"kind": "semantic", "modality": "text", "contents": ["我 喜欢 奶酪 披萨"], "metadata": {"source": "mem0"}},
                {"kind": "semantic", "modality": "text", "contents": ["我 不 喜欢 奶酪 披萨"], "metadata": {"source": "mem0"}},
            ]
        },
    )
    assert "version" in r_write

    # search top2
    r_search = await adapter.invoke(
        "memory.search",
        {"query": "奶酪 披萨", "topk": 2, "expand_graph": False, "filters": {"modality": ["text"], "memory_type": ["semantic"], "source": ["mem0"]}},
    )
    assert len(r_search.get("hits", [])) >= 2
    id0 = r_search["hits"][0]["id"]

    # update first hit
    r_update = await adapter.invoke("memory.update", {"id": id0, "patch": {"contents": ["我 很 喜欢 奶酪 披萨"]}})
    assert "version" in r_update

    # delete soft
    r_del = await adapter.invoke("memory.delete", {"id": id0, "soft": True})
    assert "version" in r_del

    # search again: soft-deleted item should be skipped and still return results (second item)
    r_search2 = await adapter.invoke(
        "memory.search",
        {"query": "奶酪 披萨", "topk": 2, "expand_graph": False, "filters": {"modality": ["text"], "memory_type": ["semantic"], "source": ["mem0"]}},
    )
    assert len(r_search2.get("hits", [])) >= 1

