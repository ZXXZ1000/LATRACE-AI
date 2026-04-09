from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MemoryToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    category: str
    default_enabled: bool
    trigger_keywords: Tuple[str, ...]


_TOOL_ORDER: Tuple[str, ...] = (
    "entity_profile",
    "topic_timeline",
    "time_since",
    "quotes",
    "relations",
    "list_entities",
    "list_topics",
    "explain",
    "entity_status",
    "status_changes",
    "state_time_since",
)


TOOL_DEFINITIONS: Dict[str, MemoryToolDefinition] = {
    "entity_profile": MemoryToolDefinition(
        name="entity_profile",
        description="查询实体画像（事实、关系、近期事件），支持实体名到实体ID解析。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "实体名称（如：张三）"},
                "entity_id": {"type": "string", "description": "实体ID（已知时可直传）"},
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["facts", "relations", "events", "quotes", "states"]},
                    "description": "返回模块，默认 [facts, relations, events]",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "memory_domain": {"type": "string"},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=True,
        trigger_keywords=("最近在忙", "画像", "事实", "履历"),
    ),
    "topic_timeline": MemoryToolDefinition(
        name="topic_timeline",
        description="按话题或关键词查询事件时间线。",
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "topic_id": {"type": "string"},
                "topic_path": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["quotes", "entities"]},
                    "description": "默认不展开重型字段",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                "session_id": {"type": "string"},
                "memory_domain": {"type": "string"},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=True,
        trigger_keywords=("时间线", "进展", "话题", "项目"),
    ),
    "time_since": MemoryToolDefinition(
        name="time_since",
        description="查询实体/话题最近一次提及距今多久。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "topic": {"type": "string"},
                "topic_id": {"type": "string"},
                "topic_path": {"type": "string"},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "memory_domain": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 50},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=True,
        trigger_keywords=("多久", "上次", "最近一次", "提到"),
    ),
    "quotes": MemoryToolDefinition(
        name="quotes",
        description="按实体/话题查相关原话，返回可引用的对话片段。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "topic": {"type": "string"},
                "topic_id": {"type": "string"},
                "topic_path": {"type": "string"},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "memory_domain": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=True,
        trigger_keywords=("原话", "引用", "怎么说", "说过"),
    ),
    "relations": MemoryToolDefinition(
        name="relations",
        description="查询实体关系图谱（邻居、首次见面等统计）。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "relation_type": {"type": "string", "enum": ["co_occurs_with"], "default": "co_occurs_with"},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=True,
        trigger_keywords=("关系", "相关的人", "共同出现", "联系人"),
    ),
    "list_entities": MemoryToolDefinition(
        name="list_entities",
        description="列出租户下的实体候选（发现/浏览用途，支持分页）。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "entity_type": {"type": "string"},
                "mentioned_since": {"type": "string", "description": "ISO 时间"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                "cursor": {"type": "string"},
                "auto_page": {"type": "boolean", "default": False},
                "max_pages": {"type": "integer", "minimum": 1, "maximum": 3, "default": 3},
                "memory_domain": {"type": "string"},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=False,
        trigger_keywords=("实体列表", "有哪些人", "候选实体", "浏览实体"),
    ),
    "list_topics": MemoryToolDefinition(
        name="list_topics",
        description="列出租户下的话题候选（发现/浏览用途，支持分页）。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "parent_path": {"type": "string"},
                "min_events": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                "cursor": {"type": "string"},
                "auto_page": {"type": "boolean", "default": False},
                "max_pages": {"type": "integer", "minimum": 1, "maximum": 3, "default": 3},
                "memory_domain": {"type": "string"},
            },
            "additionalProperties": False,
        },
        category="memory",
        default_enabled=False,
        trigger_keywords=("话题列表", "有哪些话题", "候选话题", "浏览话题"),
    ),
    "explain": MemoryToolDefinition(
        name="explain",
        description="根据 event_id 查询证据链；不做事件搜索。",
        input_schema={
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "事件ID（必填）"},
                "memory_domain": {"type": "string"},
            },
            "required": ["event_id"],
            "additionalProperties": False,
        },
        category="evidence",
        default_enabled=False,
        trigger_keywords=("依据", "证据链", "来源", "为什么知道"),
    ),
    "entity_status": MemoryToolDefinition(
        name="entity_status",
        description="查询实体状态（当前或指定时间点）。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "property": {"type": "string"},
                "property_canonical": {"type": "string"},
                "when": {"type": "string", "description": "ISO 时间；缺省则查当前"},
                "force_vocab_refresh": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        category="state",
        default_enabled=False,
        trigger_keywords=("状态", "现在", "当时", "情况"),
    ),
    "status_changes": MemoryToolDefinition(
        name="status_changes",
        description="查询实体状态变化历史（支持时间范围）。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "property": {"type": "string"},
                "property_canonical": {"type": "string"},
                "when": {"type": "string", "description": "自然时间范围文本（需 parser）"},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
                "force_vocab_refresh": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        category="state",
        default_enabled=False,
        trigger_keywords=("变化", "变更", "历史", "期间发生了什么"),
    ),
    "state_time_since": MemoryToolDefinition(
        name="state_time_since",
        description="查询实体状态距离上次变化多久。",
        input_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "entity_id": {"type": "string"},
                "property": {"type": "string"},
                "property_canonical": {"type": "string"},
                "when": {"type": "string", "description": "自然时间范围文本（需 parser）"},
                "time_range": {
                    "type": "object",
                    "properties": {"start_iso": {"type": "string"}, "end_iso": {"type": "string"}},
                    "additionalProperties": False,
                },
                "force_vocab_refresh": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        category="state",
        default_enabled=False,
        trigger_keywords=("多久没变", "上次变化", "状态多久"),
    ),
}


def _validate_names(names: Optional[List[str]]) -> List[str]:
    if names is None:
        return list(_TOOL_ORDER)
    normalized = [str(x).strip() for x in names if str(x).strip()]
    unknown = [x for x in normalized if x not in TOOL_DEFINITIONS]
    if unknown:
        raise ValueError(f"unknown tool names: {unknown}")
    return normalized


def get_tool_definitions(*, enabled_only: bool = True, names: Optional[List[str]] = None) -> List[MemoryToolDefinition]:
    ordered = _validate_names(names)
    out: List[MemoryToolDefinition] = []
    for name in ordered:
        item = TOOL_DEFINITIONS[name]
        if enabled_only and names is None and not item.default_enabled:
            continue
        out.append(item)
    return out


def to_openai_tools(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    defs = get_tool_definitions(enabled_only=(names is None), names=names)
    out: List[Dict[str, Any]] = []
    for item in defs:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": dict(item.input_schema),
                },
            }
        )
    return out


def to_mcp_tools(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    defs = get_tool_definitions(enabled_only=(names is None), names=names)
    out: List[Dict[str, Any]] = []
    for item in defs:
        out.append(
            {
                "name": item.name,
                "description": item.description,
                "inputSchema": dict(item.input_schema),
                "category": item.category,
            }
        )
    return out
