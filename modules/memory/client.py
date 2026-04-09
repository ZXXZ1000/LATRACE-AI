from __future__ import annotations

"""
Memory SDK（P0）：提供 mem0 风格的易用门面。

目标：
- 统一“文本对话入口（可选 LLM 抽取+合并决策）”与“结构化入口（M3 视频图谱）”。
- 写入时自动补齐三键：user_id（可多值）、memory_domain、run_id。
- 检索时内置作用域（scope）与回退链路（session→domain→user），并透传到服务层。
"""

from typing import Any, Dict, List, Optional, Union

from modules.memory.contracts.memory_models import MemoryEntry, Edge, SearchFilters
from modules.memory.application.service import MemoryService
from modules.memory.application.fact_extractor_mem0 import build_fact_extractor_from_env
from modules.memory.application.decider_mem0 import build_mem0_decider_from_env


class Memory:
    def __init__(self, service: MemoryService) -> None:
        self.svc = service
        # 允许外部注入更强的抽取/合并策略；P0 走简单路径

    @classmethod
    def from_defaults(cls) -> "Memory":
        # 直接使用配置构建真实后端的 service
        from modules.memory.api.server import create_service

        svc = create_service()
        return cls(svc)

    # ---- 写入：文本/消息入口（可选 LLM 抽取） ----
    async def add(
        self,
        data: Union[str, List[Dict[str, Any]]],
        *,
        user_id: Union[str, List[str]],
        memory_domain: str,
        run_id: Optional[str] = None,
        infer: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """写入记忆。

        - data 可以是字符串（单条事实）或消息数组（role/content）。
        - infer=True 时，采用 mem0 风格的“消息→事实条目”映射（占位），并走服务层内置去重/合并逻辑。
        - 写入会自动补齐 user_id/memory_domain/run_id。
        """
        base_md = dict(metadata or {})
        # 统一三键
        base_md["user_id"] = [user_id] if isinstance(user_id, str) else list(user_id)
        base_md["memory_domain"] = str(memory_domain)
        if run_id is not None:
            base_md["run_id"] = str(run_id)

        entries: List[MemoryEntry] = []
        edges: Optional[List[Edge]] = None

        if isinstance(data, list):
            # 消息入口
            if infer:
                # 强制使用 LLM 抽取器；无可用 LLM 时直接报错
                extractor = build_fact_extractor_from_env()
                if extractor is None:
                    raise RuntimeError("LLM fact extractor is not configured. Please set LLM env (e.g., OPENROUTER_API_KEY/OPENAI_API_KEY) and model.")
                facts = extractor(data)
                entries = [
                    MemoryEntry(kind="semantic", modality="text", contents=[f], metadata={"source": "mem0-extract"})
                    for f in facts
                    if str(f).strip()
                ]
            else:
                # 直接逐条记成语义事实
                entries = [
                    MemoryEntry(kind="semantic", modality="text", contents=[str(msg.get("content", ""))], metadata={"source": msg.get("role")})
                    for msg in data
                    if str(msg.get("content", "")).strip()
                ]
        else:
            # 单条文本入口
            text = str(data).strip()
            if not text:
                return {"results": []}
            entries = [MemoryEntry(kind="semantic", modality="text", contents=[text], metadata={"source": "sdk"})]

        # 注入三键信息
        for e in entries:
            md = dict(e.metadata)
            # 合并用户传入 metadata
            md.update(base_md)
            e.metadata = md

        # 尝试注入 mem0 风格的更新决策（ADD/UPDATE/DELETE/NONE）
        if getattr(self.svc, "update_decider", None) is None:
            decider = build_mem0_decider_from_env()
            if decider is not None:
                self.svc.set_update_decider(decider)

        ver = await self.svc.write(entries, links=edges)
        return {
            "version": ver.value,
            "results": [
                {
                    "id": e.id,
                    "memory": (e.contents[0] if e.contents else ""),
                    "metadata": e.metadata,
                    "event": "ADD",
                }
                for e in entries
            ],
        }

    # ---- 写入：结构化入口（M3 视频图谱） ----
    async def add_entries(
        self,
        entries: List[Union[MemoryEntry, Dict[str, Any]]],
        links: Optional[List[Union[Edge, Dict[str, Any]]]] = None,
        *,
        user_id: Union[str, List[str]],
        memory_domain: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        base_md = {
            "user_id": [user_id] if isinstance(user_id, str) else list(user_id),
            "memory_domain": str(memory_domain),
        }
        if run_id is not None:
            base_md["run_id"] = str(run_id)

        ents: List[MemoryEntry] = [e if isinstance(e, MemoryEntry) else MemoryEntry.model_validate(e) for e in entries]
        lnks: Optional[List[Edge]] = [link if isinstance(link, Edge) else Edge.model_validate(link) for link in links] if links else None
        for e in ents:
            md = dict(e.metadata)
            md.update(base_md)
            e.metadata = md
        ver = await self.svc.write(ents, links=lnks)
        return {"version": ver.value, "count": len(ents)}

    # ---- 检索 ----
    async def search(
        self,
        query: str,
        *,
        user_id: Union[str, List[str]],
        memory_domain: Optional[str] = None,
        run_id: Optional[str] = None,
        scope: Optional[str] = None,
        user_match: str = "any",
        topk: int = 10,
        expand_graph: bool = True,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        f = dict(extra_filters or {})
        f["user_id"] = [user_id] if isinstance(user_id, str) else list(user_id)
        if memory_domain is not None:
            f["memory_domain"] = str(memory_domain)
        if run_id is not None:
            f["run_id"] = str(run_id)
        f["user_match"] = user_match
        sf = SearchFilters.model_validate(f)
        res = await self.svc.search(query, topk=topk, filters=sf, expand_graph=expand_graph, scope=scope)
        out = {
            "results": [
                {
                    "id": h.id,
                    "memory": (h.entry.contents[0] if h.entry.contents else ""),
                    "metadata": h.entry.metadata,
                    "event": "ADD",
                }
                for h in res.hits
            ],
            "trace": res.trace,
        }
        return out

    # ---- 读取/更新/历史/删除 ----
    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        if hasattr(self.svc.vectors, "get"):
            e = await self.svc.vectors.get(memory_id)  # type: ignore[attr-defined]
            if e is None:
                return None
            return {"id": memory_id, "memory": (e.contents[0] if e.contents else ""), "metadata": e.metadata}
        return None

    async def update(self, memory_id: str, data: Union[str, Dict[str, Any]], *, reason: Optional[str] = None) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}
        if isinstance(data, str):
            patch["contents"] = [data]
        else:
            patch = dict(data)
        ver = await self.svc.update(memory_id, patch, reason=reason)
        return {"version": ver.value}

    async def history(self, memory_id: str) -> List[Dict[str, Any]]:
        if hasattr(self.svc, "audit") and hasattr(self.svc.audit, "list_events_for_obj"):
            return self.svc.audit.list_events_for_obj(memory_id)  # type: ignore[attr-defined]
        return []

    async def delete(self, memory_id: str, *, soft: bool = True, reason: Optional[str] = None, confirm: Optional[bool] = None) -> Dict[str, Any]:
        ver = await self.svc.delete(memory_id, soft=soft, reason=reason, confirm=confirm)
        return {"version": ver.value}

    async def delete_all(
        self,
        *,
        user_id: Optional[Union[str, List[str]]] = None,
        memory_domain: Optional[str] = None,
        run_id: Optional[str] = None,
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """谨慎实现：按过滤条件迭代删除（小规模适用）。

        说明：当前向量后端未提供“按过滤批量删除”专用接口，P0 采用简单循环删除。
        大规模数据请改用 HTTP /batch_delete 或后端专用清理脚本。
        """
        if not confirm:
            raise ValueError("delete_all requires confirm=True")
        if user_id is None and memory_domain is None and run_id is None:
            # 防止无过滤全删
            raise ValueError("delete_all requires at least one filter (user_id/memory_domain/run_id)")
        remaining = 0
        total = 0
        # 粗略分页：每次取 200 条
        while True:
            f: Dict[str, Any] = {}
            if user_id is not None:
                f["user_id"] = [user_id] if isinstance(user_id, str) else list(user_id)
            if memory_domain is not None:
                f["memory_domain"] = str(memory_domain)
            if run_id is not None:
                f["run_id"] = str(run_id)
            sf = SearchFilters.model_validate(f)
            res = await self.svc.search("", topk=200, filters=sf, expand_graph=False)
            ids = [h.id for h in res.hits]
            if not ids:
                break
            for mid in ids:
                await self.svc.delete(mid, soft=True, reason="SDK.delete_all")
                total += 1
        return {"deleted": total, "remaining": remaining}
