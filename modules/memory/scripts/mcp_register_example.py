from __future__ import annotations

"""
示例脚本：演示如何在编排层的 MCP 服务器里注册 memory.* 工具（最小化演示）。

注意：本脚本并非完整 MCP Server，只演示如何列出 ToolSpec 并把调用路由到 MemoryService。
在真实 MCP Server 中，应将此逻辑嵌入到工具注册与 handler 中。
"""

import asyncio
from modules.memory.api.mcp_server import MemoryMCPAdapter


async def main() -> None:
    adapter = MemoryMCPAdapter.from_defaults()  # InMem 演示；实际应注入真实 MemoryService
    print("Registered tools:", adapter.tools())

    # 演示调用 memory.write → memory.search
    write_res = await adapter.invoke(
        "memory.write",
        {
            "entries": [
                {"kind": "semantic", "modality": "text", "contents": ["我 喜欢 奶酪 披萨"], "metadata": {"source": "mem0"}},
                {"kind": "semantic", "modality": "text", "contents": ["我 不 喜欢 奶酪 披萨"], "metadata": {"source": "mem0"}},
            ]
        },
    )
    print("write version:", write_res)

    search_res = await adapter.invoke(
        "memory.search",
        {"query": "奶酪 披萨", "topk": 2, "expand_graph": False, "filters": {"modality": ["text"], "memory_type": ["semantic"], "source": ["mem0"]}},
    )
    hits = [(h["entry"]["contents"][0], h["score"]) for h in search_res.get("hits", [])]
    print("search hits:", hits)


if __name__ == "__main__":
    asyncio.run(main())

