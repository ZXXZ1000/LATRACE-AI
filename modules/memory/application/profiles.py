from __future__ import annotations

"""
Ingest Profiles（按来源对齐归一化与治理管道的入口封装）

目标：
- 为 m3/mem0/ctrl 等来源提供稳定的条目/连边构建入口；
- 统一落到 MemoryEntry/Edge，交由 MemoryService 完成治理补齐/去重合并/落库。

说明：
- 真实治理（importance/ttl/pinned/stability/hash 等）由 MemoryService 统筹；
- profiles 负责选择合适的 adapter 构造 entries/edges，并可按需补少量来源元数据。
"""

from typing import Any, Dict, List, Tuple

from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.adapters.m3_adapter import build_entries_from_m3
from modules.memory.adapters.mem0_adapter import build_entries_from_mem0


def profile_m3_episodic(parsed: Dict[str, Any], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """m3 视频管线产物（偏情节）→ 统一 entries/edges

    parsed 期望包含：faces/voices/episodic/semantic/clip_id/timestamp/room/device
    返回：MemoryEntry 列表与 Edge 列表
    """
    entries, edges = build_entries_from_m3(parsed, profile=profile)
    # 附加来源标记（如 adapter 未设置）
    for e in entries:
        md = dict(e.metadata)
        md.setdefault("source", "m3")
        e.metadata = md
    return entries, edges


def profile_m3_semantic(parsed: Dict[str, Any], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """m3 产物（偏语义）→ 统一 entries/edges（与 episodic 复用同一适配结构）"""
    return profile_m3_episodic(parsed, profile=profile)


def profile_mem0_fact(messages: List[Dict[str, Any]], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """mem0 抽取到的事实（或原始对话消息）→ 统一 entries/edges

    messages: [{"role": "user|assistant", "content": "..."}, ...]
    profile: 可传入 {"user_id": "..."} 用于建立 prefer 等关系。
    """
    entries, edges = build_entries_from_mem0(messages, profile=profile)
    for e in entries:
        md = dict(e.metadata)
        md.setdefault("source", "mem0")
        e.metadata = md
    return entries, edges


def profile_ctrl_event(event: Dict[str, Any], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """控制事件（设备执行/状态变化）→ 统一 entries/edges

    event 例：{
      "text": "用户通过控制器打开客厅主灯",  # 记情节
      "device": "device.light.living_main",  # 结构实体
      "room": "living_room",                # 可选
    }
    """
    text = str(event.get("text") or "").strip() or "control event"
    device = event.get("device")
    room = event.get("room")
    parsed = {
        "episodic": [text],
        "semantic": [],
        "faces": [],
        "voices": [],
        "device": device,
        "room": room,
        "clip_id": event.get("clip_id"),
        "timestamp": event.get("timestamp"),
    }
    entries, edges = build_entries_from_m3(parsed, profile=profile)
    for e in entries:
        md = dict(e.metadata)
        # 控制事件来源应标记为 ctrl（覆盖 adapter 默认）
        md["source"] = "ctrl"
        e.metadata = md
    return entries, edges
