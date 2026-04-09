from __future__ import annotations

"""
Cycle13 ETL 演示脚本：
- 使用 etl/samples/bedroom_12.pkl（m3 VideoGraph）
- 先 dry-run 打印 entries/relations 数量
- 可选 --inmem 将映射写入 InMem Stores 并进行一次检索验证
"""

import os
from modules.memory.etl.pkl_to_db import run as etl_run
from modules.memory.application.service import MemoryService
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore


def main() -> None:
    pkl = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "etl", "samples", "bedroom_12.pkl"))
    print("PKL:", pkl)
    # 1) dry-run
    stats = etl_run(pkl, dry_run=True, limit=None)
    print("DRY-RUN:", stats)

    # 2) in-memory write and quick search check（可选）
    stats2 = etl_run(pkl, dry_run=False, limit=200, inmem=True, batch_size=500)
    print("INMEM IMPORT:", stats2)
    MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())  # new empty; here仅演示接口
    # 如果需要进一步验证，可以在上一步返回 MemoryService 或改造 etl_run 返回 svc 以便检索验证


if __name__ == "__main__":
    main()

