from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from .errors import normalize_exception, normalize_http_error
from .models import ToolDebugTrace, ToolResult
from .resolve import ResolveEntityFn, _resolve_if_needed

EntityProfileFn = Callable[..., Awaitable[Dict[str, Any]]]
TimeSinceFn = Callable[..., Awaitable[Dict[str, Any]]]
QuotesFn = Callable[..., Awaitable[Dict[str, Any]]]
TopicTimelineFn = Callable[..., Awaitable[Dict[str, Any]]]
ExplainFn = Callable[..., Awaitable[Dict[str, Any]]]
ListEntitiesFn = Callable[..., Awaitable[Dict[str, Any]]]
ListTopicsFn = Callable[..., Awaitable[Dict[str, Any]]]

_ENTITY_PROFILE_INCLUDE_ALLOWED = {"facts", "relations", "events", "quotes", "states"}
_ENTITY_PROFILE_INCLUDE_DEFAULT = ("facts", "relations", "events")
_TOPIC_TIMELINE_INCLUDE_ALLOWED = {"quotes", "entities"}
_RELATION_TYPES_ALLOWED = {"co_occurs_with", "co-occurs-with"}


def _is_http_error_payload(resp: Any) -> bool:
    return isinstance(resp, dict) and "status_code" in resp and int(resp.get("status_code") or 200) >= 400


def _norm_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _clamp_limit(value: Any, *, default: int = 10, min_value: int = 1, max_value: int = 50) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    if n < min_value:
        n = min_value
    if n > max_value:
        n = max_value
    return n


def _parse_iso_optional(value: Any) -> Optional[str]:
    text = _norm_text(value)
    if text is None:
        return None
    raw = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _normalize_discovery_cursor(value: Any) -> Tuple[Optional[str], bool]:
    text = _norm_text(value)
    if text is None:
        return None, False
    raw = text
    if raw.startswith(("c:", "o:")):
        raw = raw[2:]
    if raw.isdigit():
        return text, False
    return None, True


def _norm_nonnegative_int(value: Any) -> Tuple[Optional[int], bool]:
    if value is None:
        return None, False
    try:
        n = int(value)
    except Exception:
        return None, True
    if n < 0:
        return 0, True
    return n, False


def _normalize_time_range_dict(value: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, str]], bool]:
    if value is None:
        return None, False
    if not isinstance(value, dict):
        return None, True
    start_raw = value.get("start_iso", value.get("start"))
    end_raw = value.get("end_iso", value.get("end"))
    start_iso = _parse_iso_optional(start_raw)
    end_iso = _parse_iso_optional(end_raw)
    if (start_raw is not None and start_iso is None) or (end_raw is not None and end_iso is None):
        return None, True
    if start_iso is None and end_iso is None:
        return None, True
    out: Dict[str, str] = {}
    if start_iso is not None:
        out["start_iso"] = start_iso
    if end_iso is not None:
        out["end_iso"] = end_iso
    return out, False


def _normalize_relation_type(value: Optional[str]) -> Tuple[Optional[str], bool]:
    text = str(value or "").strip().lower()
    if not text:
        return "co_occurs_with", False
    if text not in _RELATION_TYPES_ALLOWED:
        return None, True
    return "co_occurs_with", False


def _normalize_include(include: Optional[Iterable[str]]) -> Tuple[List[str], bool]:
    if include is None:
        return list(_ENTITY_PROFILE_INCLUDE_DEFAULT), False
    out: List[str] = []
    invalid = False
    for item in include:
        val = str(item or "").strip().lower()
        if not val:
            continue
        if val not in _ENTITY_PROFILE_INCLUDE_ALLOWED:
            invalid = True
            continue
        if val not in out:
            out.append(val)
    if not out:
        out = list(_ENTITY_PROFILE_INCLUDE_DEFAULT)
    return out, invalid


def _normalize_timeline_include(include: Optional[Iterable[str]]) -> Tuple[List[str], bool]:
    if include is None:
        return [], False
    out: List[str] = []
    invalid = False
    for item in include:
        val = str(item or "").strip().lower()
        if not val:
            continue
        if val not in _TOPIC_TIMELINE_INCLUDE_ALLOWED:
            invalid = True
            continue
        if val not in out:
            out.append(val)
    return out, invalid


def _entity_payload(*, entity_input: Optional[str], entity_id: Optional[str], entity_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entity_obj, dict) and not entity_id and not entity_input:
        return None
    base: Dict[str, Any] = dict(entity_obj) if isinstance(entity_obj, dict) else {}
    if entity_id and not base.get("id"):
        base["id"] = entity_id
    if entity_input and not base.get("name"):
        base["name"] = entity_input
    return base or None


def _build_result(
    *,
    tool_name: str = "entity_profile",
    source_mode: Optional[str] = "graph_filter",
    api_route: str = "/memory/v1/entity-profile",
    matched: bool,
    message: Optional[str],
    data: Optional[Dict[str, Any]],
    resolution_meta: Dict[str, Any],
    raw_api_response_keys: Optional[List[str]] = None,
    error_type: Optional[str] = None,
    retryable: bool = False,
) -> ToolResult:
    debug = ToolDebugTrace(
        tool_name=tool_name,
        source_mode=source_mode,
        error_type=error_type,
        retryable=retryable,
        resolution_meta=resolution_meta,
        raw_api_response_keys=raw_api_response_keys,
        extras={"api_route": api_route},
    )
    return ToolResult(
        matched=matched,
        needs_disambiguation=False,
        message=message,
        data=data,
        debug=debug,
    )


async def entity_profile(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    entity_profile_api: EntityProfileFn,
    entity: str | None = None,
    entity_id: str | None = None,
    include: Optional[List[str]] = None,
    limit: int = 10,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/entity-profile."""

    include_norm, include_had_invalid = _normalize_include(include)
    limit_norm = _clamp_limit(limit, default=10, min_value=1, max_value=50)
    entity_input = _norm_text(entity)
    entity_id_input = _norm_text(entity_id)
    memory_domain_norm = _norm_text(memory_domain)

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolved = await _resolve_if_needed(
        resolver=resolver,
        tool_name="entity_profile",
        entity=entity_input,
        entity_id=entity_id_input,
        user_tokens=effective_user_tokens,
        resolve_limit=5,
    )
    if resolved.should_stop:
        term = resolved.terminal_result or ToolResult.no_match(message="查询未完成")
        dbg = term.to_debug_dict() or {}
        if include_had_invalid and dbg.get("resolution_meta") is not None:
            dbg["resolution_meta"].setdefault("query", {})
            dbg["resolution_meta"]["query"]["include_invalid_ignored"] = True
            term.debug = dbg
        return term

    assert resolved.entity_id is not None
    resolution_meta = dict(resolved.resolution_meta or {})
    resolution_meta["query"] = {
        "include": list(include_norm),
        "limit_input": limit,
        "limit": limit_norm,
        "include_invalid_ignored": bool(include_had_invalid),
    }

    try:
        resp = await entity_profile_api(
            tenant_id=str(tenant_id),
            entity_id=resolved.entity_id,
            user_tokens=list(effective_user_tokens),
            memory_domain=memory_domain_norm,
            facts_limit=limit_norm,
            relations_limit=limit_norm,
            events_limit=limit_norm,
            quotes_limit=limit_norm,
            include_quotes=("quotes" in include_norm),
            include_relations=("relations" in include_norm),
            include_events=("events" in include_norm),
            include_states=("states" in include_norm),
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        return _build_result(
            tool_name="entity_profile",
            matched=False,
            message="服务暂时不可用" if err.retryable else "实体画像查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        msg = "服务暂时不可用" if err.retryable else "实体画像查询失败"
        return _build_result(
            tool_name="entity_profile",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="entity_profile",
            matched=False,
            message="实体画像查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    if not bool(resp.get("found")):
        return _build_result(
            tool_name="entity_profile",
            matched=False,
            message="没有找到相关实体",
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        )

    entity_obj = _entity_payload(entity_input=entity_input, entity_id=resolved.entity_id, entity_obj=resp.get("entity"))
    data: Dict[str, Any] = {
        "entity": entity_obj,
        "facts": [dict(x) for x in (resp.get("facts") or []) if isinstance(x, dict)],
        "relations": [dict(x) for x in (resp.get("relations") or []) if isinstance(x, dict)],
        "recent_events": [dict(x) for x in (resp.get("recent_events") or []) if isinstance(x, dict)],
    }
    if "quotes" in include_norm:
        data["quotes"] = [dict(x) for x in (resp.get("quotes") or []) if isinstance(x, dict)]
    if "states" in include_norm:
        data["states"] = [dict(x) for x in (resp.get("states") or []) if isinstance(x, dict)]

    debug = ToolDebugTrace(
        tool_name="entity_profile",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": "/memory/v1/entity-profile"},
    )
    return ToolResult.success(data=data, debug=debug)


def _time_since_source_mode(resp: Dict[str, Any], *, entity_only: bool) -> Optional[str]:
    trace = resp.get("trace") if isinstance(resp.get("trace"), dict) else None
    source = str((trace or {}).get("source") or "").strip()
    if source in {"graph_entity_events", "graph_topic_filter"}:
        return "graph_filter"
    if source == "retrieval_dialog_v2":
        return "retrieval_rag"
    if entity_only:
        return "graph_filter"
    return None


def _quotes_source_mode(
    resp: Dict[str, Any],
    *,
    entity_only: bool,
    topic_query_present: bool,
) -> Tuple[Optional[str], bool]:
    trace = resp.get("trace") if isinstance(resp.get("trace"), dict) else None
    source = str((trace or {}).get("source") or "").strip()
    if source in {"graph_entity_events", "graph_topic_filter"}:
        return "graph_filter", False
    if source == "retrieval_dialog_v2":
        return "retrieval_rag", True
    if entity_only and not topic_query_present:
        return "graph_filter", False
    return None, False


def _topic_timeline_source_mode(resp: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    trace = resp.get("trace") if isinstance(resp.get("trace"), dict) else None
    source = str((trace or {}).get("source") or "").strip()
    if source == "graph_topic_filter":
        return "graph_filter", False
    if source == "retrieval_dialog_v2":
        return "retrieval_rag", True
    return None, False


async def time_since(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    time_since_api: TimeSinceFn,
    entity: str | None = None,
    topic: str | None = None,
    entity_id: str | None = None,
    topic_id: str | None = None,
    topic_path: str | None = None,
    user_tokens: Optional[List[str]] = None,
    time_range: Optional[Dict[str, Any]] = None,
    memory_domain: Optional[str] = None,
    limit: int = 50,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/time-since."""

    entity_input = _norm_text(entity)
    topic_input = _norm_text(topic)
    entity_id_input = _norm_text(entity_id)
    topic_id_input = _norm_text(topic_id)
    topic_path_input = _norm_text(topic_path)
    memory_domain_norm = _norm_text(memory_domain)

    if not (entity_input or entity_id_input or topic_input or topic_id_input or topic_path_input):
        return _build_result(
            tool_name="time_since",
            source_mode=None,
            api_route="/memory/v1/time-since",
            matched=False,
            message="缺少查询条件",
            data=None,
            resolution_meta={"entity": {"input": entity_input}, "topic": {"input": topic_input}},
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolved = await _resolve_if_needed(
        resolver=resolver,
        tool_name="time_since",
        entity=entity_input,
        entity_id=entity_id_input,
        user_tokens=effective_user_tokens,
        resolve_limit=5,
    ) if (entity_input or entity_id_input) else None
    if resolved and resolved.should_stop:
        return resolved.terminal_result or ToolResult.no_match(message="查询未完成")

    resolved_entity_id = (resolved.entity_id if resolved else None) or entity_id_input
    resolution_meta = dict((resolved.resolution_meta if resolved else {}) or {})

    time_range_norm, time_range_invalid = _normalize_time_range_dict(time_range)
    if time_range_invalid:
        return _build_result(
            tool_name="time_since",
            source_mode=None,
            api_route="/memory/v1/time-since",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta={
                **resolution_meta,
                "topic": {"input": topic_input, "topic_id": topic_id_input, "topic_path": topic_path_input},
                "time_range": {"input": dict(time_range or {}) if isinstance(time_range, dict) else time_range},
            },
            error_type="invalid_input",
            retryable=False,
        )

    filter_semantics = "AND" if (resolved_entity_id and (topic_input or topic_id_input or topic_path_input)) else (
        "entity_only" if resolved_entity_id else "topic_only"
    )
    limit_norm = _clamp_limit(limit, default=50, min_value=1, max_value=50)
    resolution_meta.setdefault("topic", {})
    resolution_meta["topic"].update(
        {
            "input": topic_input,
            "topic_id": topic_id_input,
            "topic_path": topic_path_input,
        }
    )
    resolution_meta["query"] = {
        "limit_input": limit,
        "limit": limit_norm,
        "filter_semantics": filter_semantics,
    }
    if time_range_norm is not None:
        resolution_meta["time_range"] = {"normalized": dict(time_range_norm)}

    try:
        resp = await time_since_api(
            tenant_id=str(tenant_id),
            entity=entity_input,
            entity_id=resolved_entity_id,
            topic=topic_input,
            topic_id=topic_id_input,
            topic_path=topic_path_input,
            user_tokens=list(effective_user_tokens),
            time_range=dict(time_range_norm) if time_range_norm is not None else None,
            memory_domain=memory_domain_norm,
            limit=limit_norm,
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        msg = "查询超时" if err.error_type == "timeout" else ("服务暂时不可用" if err.retryable else "时距查询失败")
        return _build_result(
            tool_name="time_since",
            source_mode=None,
            api_route="/memory/v1/time-since",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        if err.error_type == "timeout":
            msg = "查询超时"
        elif err.error_type == "invalid_input":
            msg = "缺少查询条件"
        else:
            msg = "服务暂时不可用" if err.retryable else "时距查询失败"
        return _build_result(
            tool_name="time_since",
            source_mode=None,
            api_route="/memory/v1/time-since",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="time_since",
            source_mode=None,
            api_route="/memory/v1/time-since",
            matched=False,
            message="时距查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    last_mentioned = _norm_text(resp.get("last_mentioned"))
    source_mode = _time_since_source_mode(resp, entity_only=(filter_semantics == "entity_only"))
    debug = ToolDebugTrace(
        tool_name="time_since",
        source_mode=source_mode,
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={
            "api_route": "/memory/v1/time-since",
            "filter_semantics": filter_semantics,
        },
    )

    if not last_mentioned:
        return ToolResult(
            matched=False,
            needs_disambiguation=False,
            message="没有找到相关记录",
            data=None,
            debug=debug,
        )

    data = {
        "entity": _entity_payload(entity_input=entity_input, entity_id=resolved_entity_id, entity_obj=resp.get("resolved_entity")),
        "topic": {
            "input": topic_input,
            "topic_id": resp.get("topic_id", topic_id_input),
            "topic_path": resp.get("topic_path", topic_path_input),
        } if (topic_input or topic_id_input or topic_path_input) else None,
        "last_mentioned": resp.get("last_mentioned"),
        "days_ago": resp.get("days_ago"),
        "summary": resp.get("summary"),
    }
    message = None
    if filter_semantics == "AND":
        name = None
        if isinstance(data.get("entity"), dict):
            name = data["entity"].get("name") or data["entity"].get("id")
        topic_label = None
        if isinstance(data.get("topic"), dict):
            topic_label = data["topic"].get("input") or data["topic"].get("topic_path") or data["topic"].get("topic_id")
        message = f"查询的是{name or '该实体'}在{topic_label or '该话题'}下的最近一次"

    return ToolResult.success(data=data, message=message, debug=debug)


async def relations(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    relations_api: Callable[..., Awaitable[Dict[str, Any]]],
    entity: str | None = None,
    entity_id: str | None = None,
    relation_type: str = "co_occurs_with",
    time_range: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    user_tokens: Optional[List[str]] = None,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/relations."""

    entity_input = _norm_text(entity)
    entity_id_input = _norm_text(entity_id)
    if not (entity_input or entity_id_input):
        return _build_result(
            tool_name="relations",
            source_mode=None,
            api_route="/memory/v1/relations",
            matched=False,
            message="缺少实体参数",
            data=None,
            resolution_meta={"entity": {"input": entity_input, "input_entity_id": entity_id_input}},
            error_type="invalid_input",
            retryable=False,
        )

    rel_type_norm, rel_type_invalid = _normalize_relation_type(relation_type)
    if rel_type_invalid or rel_type_norm is None:
        return _build_result(
            tool_name="relations",
            source_mode=None,
            api_route="/memory/v1/relations",
            matched=False,
            message="暂不支持该关系类型",
            data=None,
            resolution_meta={"relation": {"input": relation_type, "normalized": None}},
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolved = await _resolve_if_needed(
        resolver=resolver,
        tool_name="relations",
        entity=entity_input,
        entity_id=entity_id_input,
        user_tokens=effective_user_tokens,
        resolve_limit=5,
    )
    if resolved.should_stop:
        return resolved.terminal_result or ToolResult.no_match(message="查询未完成")

    assert resolved.entity_id is not None
    resolution_meta = dict(resolved.resolution_meta or {})
    resolution_meta["query"] = {
        "relation_type_input": relation_type,
        "relation_type": rel_type_norm,
        "limit_input": limit,
        "limit": _clamp_limit(limit, default=20, min_value=1, max_value=50),
    }

    time_range_norm, time_range_invalid = _normalize_time_range_dict(time_range)
    if time_range_invalid:
        return _build_result(
            tool_name="relations",
            source_mode="graph_filter",
            api_route="/memory/v1/relations",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta={**resolution_meta, "time_range": {"input": dict(time_range or {}) if isinstance(time_range, dict) else time_range}},
            error_type="invalid_input",
            retryable=False,
        )
    if time_range_norm is not None:
        resolution_meta["time_range"] = {"normalized": dict(time_range_norm)}

    limit_norm = int(resolution_meta["query"]["limit"])
    try:
        resp = await relations_api(
            tenant_id=str(tenant_id),
            entity=entity_input,
            entity_id=resolved.entity_id,
            relation_type=rel_type_norm,
            user_tokens=list(effective_user_tokens),
            time_range=dict(time_range_norm) if time_range_norm is not None else None,
            limit=limit_norm,
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        msg = "查询超时" if err.error_type == "timeout" else ("服务暂时不可用" if err.retryable else "关系查询失败")
        return _build_result(
            tool_name="relations",
            source_mode="graph_filter",
            api_route="/memory/v1/relations",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        msg = "查询超时" if err.error_type == "timeout" else ("服务暂时不可用" if err.retryable else "关系查询失败")
        return _build_result(
            tool_name="relations",
            source_mode="graph_filter",
            api_route="/memory/v1/relations",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="relations",
            source_mode="graph_filter",
            api_route="/memory/v1/relations",
            matched=False,
            message="关系查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    found = bool(resp.get("found"))
    rels = [dict(x) for x in (resp.get("relations") or []) if isinstance(x, dict)]
    resolution_meta["entity_resolved"] = bool(found)
    data = {
        "entity": _entity_payload(entity_input=entity_input, entity_id=resolved.entity_id, entity_obj=resp.get("resolved_entity")),
        "relations": rels,
        "total": int(resp.get("total") or len(rels)),
    }
    debug = ToolDebugTrace(
        tool_name="relations",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": "/memory/v1/relations", "entity_resolved": bool(found)},
    )

    if not found:
        return ToolResult(matched=False, needs_disambiguation=False, message="未找到实体", data=None, debug=debug)
    if not rels:
        return ToolResult(matched=False, needs_disambiguation=False, message="未找到关系", data=data, debug=debug)
    return ToolResult.success(data=data, debug=debug)


async def quotes(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    quotes_api: QuotesFn,
    entity: str | None = None,
    topic: str | None = None,
    entity_id: str | None = None,
    topic_id: str | None = None,
    topic_path: str | None = None,
    time_range: Optional[Dict[str, Any]] = None,
    limit: int = 5,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/quotes."""

    entity_input = _norm_text(entity)
    topic_input = _norm_text(topic)
    entity_id_input = _norm_text(entity_id)
    topic_id_input = _norm_text(topic_id)
    topic_path_input = _norm_text(topic_path)
    memory_domain_norm = _norm_text(memory_domain)

    if not (entity_input or entity_id_input or topic_input or topic_id_input or topic_path_input):
        return _build_result(
            tool_name="quotes",
            source_mode=None,
            api_route="/memory/v1/quotes",
            matched=False,
            message="缺少实体或话题参数",
            data=None,
            resolution_meta={
                "entity": {"input": entity_input, "input_entity_id": entity_id_input},
                "topic": {"input": topic_input, "input_topic_id": topic_id_input, "input_topic_path": topic_path_input},
            },
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolved_entity_id = entity_id_input
    resolution_meta: Dict[str, Any] = {
        "entity": {"input": entity_input, "input_entity_id": entity_id_input},
        "topic": {"input": topic_input, "input_topic_id": topic_id_input, "input_topic_path": topic_path_input},
    }

    if entity_input or entity_id_input:
        resolved = await _resolve_if_needed(
            resolver=resolver,
            tool_name="quotes",
            entity=entity_input,
            entity_id=entity_id_input,
            user_tokens=effective_user_tokens,
            resolve_limit=5,
        )
        if resolved.should_stop:
            return resolved.terminal_result or ToolResult.no_match(message="查询未完成")
        resolved_entity_id = resolved.entity_id
        resolution_meta["entity"] = dict((resolved.resolution_meta or {}).get("entity") or resolution_meta["entity"])

    time_range_norm, time_range_invalid = _normalize_time_range_dict(time_range)
    if time_range_invalid:
        return _build_result(
            tool_name="quotes",
            source_mode=None,
            api_route="/memory/v1/quotes",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta={
                **resolution_meta,
                "time_range": {"input": dict(time_range or {}) if isinstance(time_range, dict) else time_range},
            },
            error_type="invalid_input",
            retryable=False,
        )

    limit_norm = _clamp_limit(limit, default=5, min_value=1, max_value=10)
    resolution_meta["query"] = {
        "limit_input": limit,
        "limit": limit_norm,
    }
    if time_range_norm is not None:
        resolution_meta["time_range"] = {"normalized": dict(time_range_norm)}

    try:
        resp = await quotes_api(
            tenant_id=str(tenant_id),
            entity=entity_input,
            entity_id=resolved_entity_id,
            topic=topic_input,
            topic_id=topic_id_input,
            topic_path=topic_path_input,
            user_tokens=list(effective_user_tokens),
            memory_domain=memory_domain_norm,
            time_range=dict(time_range_norm) if time_range_norm is not None else None,
            limit=limit_norm,
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        if err.error_type == "timeout":
            msg = "原话查询超时"
        elif err.retryable:
            msg = "服务暂时不可用"
        else:
            msg = "原话查询失败"
        return _build_result(
            tool_name="quotes",
            source_mode=None,
            api_route="/memory/v1/quotes",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        if err.error_type == "timeout":
            msg = "原话查询超时"
        elif err.retryable:
            msg = "服务暂时不可用"
        else:
            msg = "原话查询失败"
        return _build_result(
            tool_name="quotes",
            source_mode=None,
            api_route="/memory/v1/quotes",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="quotes",
            source_mode=None,
            api_route="/memory/v1/quotes",
            matched=False,
            message="原话查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    quotes_out = [dict(x) for x in (resp.get("quotes") or []) if isinstance(x, dict)]
    entity_only = bool(resolved_entity_id and not (topic_input or topic_id_input or topic_path_input))
    source_mode, fallback_used = _quotes_source_mode(
        resp,
        entity_only=entity_only,
        topic_query_present=bool(topic_input or topic_id_input or topic_path_input),
    )
    data = {
        "entity": _entity_payload(
            entity_input=entity_input,
            entity_id=resolved_entity_id,
            entity_obj=resp.get("resolved_entity"),
        ),
        "topic": (
            {
                "input": topic_input,
                "topic_id": resp.get("topic_id", topic_id_input),
                "topic_path": resp.get("topic_path", topic_path_input),
            }
            if (topic_input or topic_id_input or topic_path_input)
            else None
        ),
        "quotes": quotes_out,
        "total": int(resp.get("total") or len(quotes_out)),
    }
    debug = ToolDebugTrace(
        tool_name="quotes",
        source_mode=source_mode,
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        fallback_used=bool(fallback_used),
        extras={"api_route": "/memory/v1/quotes"},
    )

    if not quotes_out:
        return ToolResult(matched=False, needs_disambiguation=False, message="未找到相关原话", data=data, debug=debug)
    return ToolResult.success(data=data, debug=debug)


async def topic_timeline(
    *,
    tenant_id: str,
    topic_timeline_api: TopicTimelineFn,
    topic: str | None = None,
    topic_id: str | None = None,
    topic_path: str | None = None,
    keywords: Optional[List[str]] = None,
    time_range: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    limit: int = 10,
    session_id: str | None = None,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/topic-timeline."""

    topic_input = _norm_text(topic)
    topic_id_input = _norm_text(topic_id)
    topic_path_input = _norm_text(topic_path)
    session_id_norm = _norm_text(session_id)
    memory_domain_norm = _norm_text(memory_domain)
    keywords_norm = []
    seen_keywords = set()
    for item in (keywords or []):
        val = _norm_text(item)
        if not val:
            continue
        key = val.casefold()
        if key in seen_keywords:
            continue
        seen_keywords.add(key)
        keywords_norm.append(val)

    if not (topic_input or topic_id_input or topic_path_input or keywords_norm):
        return _build_result(
            tool_name="topic_timeline",
            source_mode=None,
            api_route="/memory/v1/topic-timeline",
            matched=False,
            message="缺少话题或关键词",
            data=None,
            resolution_meta={"topic": {"input": topic_input, "input_topic_id": topic_id_input, "input_topic_path": topic_path_input}},
            error_type="invalid_input",
            retryable=False,
        )

    include_norm, include_had_invalid = _normalize_timeline_include(include)
    heavy_expand = bool(("quotes" in include_norm) or ("entities" in include_norm))
    limit_norm = _clamp_limit(limit, default=10, min_value=1, max_value=20)

    time_range_norm, time_range_invalid = _normalize_time_range_dict(time_range)
    if time_range_invalid:
        return _build_result(
            tool_name="topic_timeline",
            source_mode=None,
            api_route="/memory/v1/topic-timeline",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta={
                "topic": {"input": topic_input, "input_topic_id": topic_id_input, "input_topic_path": topic_path_input},
                "time_range": {"input": dict(time_range or {}) if isinstance(time_range, dict) else time_range},
            },
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolution_meta: Dict[str, Any] = {
        "topic": {"input": topic_input, "input_topic_id": topic_id_input, "input_topic_path": topic_path_input},
        "query": {
            "keywords": list(keywords_norm),
            "include": list(include_norm),
            "include_invalid_ignored": bool(include_had_invalid),
            "limit_input": limit,
            "limit": limit_norm,
            "heavy_expand": heavy_expand,
            "session_id": session_id_norm,
        },
    }
    if time_range_norm is not None:
        resolution_meta["time_range"] = {"normalized": dict(time_range_norm)}

    try:
        resp = await topic_timeline_api(
            tenant_id=str(tenant_id),
            topic=topic_input,
            topic_id=topic_id_input,
            topic_path=topic_path_input,
            keywords=list(keywords_norm) if keywords_norm else None,
            user_tokens=list(effective_user_tokens),
            time_range=dict(time_range_norm) if time_range_norm is not None else None,
            session_id=session_id_norm,
            memory_domain=memory_domain_norm,
            limit=limit_norm,
            with_quotes=("quotes" in include_norm),
            with_entities=("entities" in include_norm),
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        if err.error_type == "timeout":
            msg = "时间线查询超时"
        elif err.retryable:
            msg = "服务暂时不可用"
        else:
            msg = "时间线查询失败"
        return _build_result(
            tool_name="topic_timeline",
            source_mode=None,
            api_route="/memory/v1/topic-timeline",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        if err.error_type == "timeout":
            msg = "时间线查询超时"
        elif err.retryable:
            msg = "服务暂时不可用"
        else:
            msg = "时间线查询失败"
        return _build_result(
            tool_name="topic_timeline",
            source_mode=None,
            api_route="/memory/v1/topic-timeline",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="topic_timeline",
            source_mode=None,
            api_route="/memory/v1/topic-timeline",
            matched=False,
            message="时间线查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    source_mode, fallback_used = _topic_timeline_source_mode(resp)
    timeline = [dict(x) for x in (resp.get("timeline") or []) if isinstance(x, dict)]
    data = {
        "topic": {
            "input": topic_input,
            "topic_id": resp.get("topic_id", topic_id_input),
            "topic_path": resp.get("topic_path", topic_path_input),
            "keywords": list(keywords_norm) if keywords_norm else None,
        },
        "status": resp.get("status"),
        "timeline": timeline,
        "total": int(resp.get("total") or len(timeline)),
    }
    debug = ToolDebugTrace(
        tool_name="topic_timeline",
        source_mode=source_mode,
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        fallback_used=bool(fallback_used),
        extras={"api_route": "/memory/v1/topic-timeline", "heavy_expand": heavy_expand},
    )

    if not timeline:
        return ToolResult(matched=False, needs_disambiguation=False, message="未找到相关时间线记录", data=data, debug=debug)
    return ToolResult.success(data=data, debug=debug)


async def explain(
    *,
    tenant_id: str,
    explain_api: ExplainFn,
    event_id: str,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
) -> ToolResult:
    """ADK semantic wrapper for /memory/v1/explain (atomic: event_id only)."""

    event_id_norm = _norm_text(event_id)
    memory_domain_norm = _norm_text(memory_domain)
    resolution_meta: Dict[str, Any] = {"event": {"input_event_id": event_id_norm}}
    if not event_id_norm:
        return _build_result(
            tool_name="explain",
            source_mode="graph_filter",
            api_route="/memory/v1/explain",
            matched=False,
            message="缺少事件ID",
            data=None,
            resolution_meta=resolution_meta,
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]
    resolution_meta["scope"] = {
        "user_tokens": list(effective_user_tokens),
        "memory_domain": memory_domain_norm,
    }

    try:
        resp = await explain_api(
            tenant_id=str(tenant_id),
            event_id=event_id_norm,
            user_tokens=list(effective_user_tokens) if effective_user_tokens else None,
            memory_domain=memory_domain_norm,
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        msg = "服务暂时不可用" if err.retryable else "证据链查询失败"
        return _build_result(
            tool_name="explain",
            source_mode="graph_filter",
            api_route="/memory/v1/explain",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        msg = "服务暂时不可用" if err.retryable else "证据链查询失败"
        return _build_result(
            tool_name="explain",
            source_mode="graph_filter",
            api_route="/memory/v1/explain",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
        )

    if not isinstance(resp, dict):
        return _build_result(
            tool_name="explain",
            source_mode="graph_filter",
            api_route="/memory/v1/explain",
            matched=False,
            message="证据链查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )

    event_obj = resp.get("event") if isinstance(resp.get("event"), dict) else None
    found = bool(resp.get("found")) and event_obj is not None
    event_id_out = _norm_text((event_obj or {}).get("id")) or _norm_text(resp.get("event_id")) or event_id_norm
    if event_id_out:
        resolution_meta["event"]["resolved_event_id"] = event_id_out

    data = {
        "event_id": event_id_out,
        "event": dict(event_obj) if event_obj else None,
        "entities": [dict(x) for x in (resp.get("entities") or []) if isinstance(x, dict)],
        "places": [dict(x) for x in (resp.get("places") or []) if isinstance(x, dict)],
        "timeslices": [dict(x) for x in (resp.get("timeslices") or []) if isinstance(x, dict)],
        "evidences": [dict(x) for x in (resp.get("evidences") or []) if isinstance(x, dict)],
        "utterances": [dict(x) for x in (resp.get("utterances") or []) if isinstance(x, dict)],
        "utterance_speakers": [dict(x) for x in (resp.get("utterance_speakers") or []) if isinstance(x, dict)],
        "knowledge": [dict(x) for x in (resp.get("knowledge") or []) if isinstance(x, dict)],
    }
    debug = ToolDebugTrace(
        tool_name="explain",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": "/memory/v1/explain"},
    )

    if not found:
        return ToolResult(matched=False, needs_disambiguation=False, message="未找到相关事件证据", data=data, debug=debug)
    return ToolResult.success(data=data, debug=debug)


async def list_entities(
    *,
    tenant_id: str,
    list_entities_api: ListEntitiesFn,
    query: str | None = None,
    entity_type: str | None = None,
    mentioned_since: str | None = None,
    limit: int = 20,
    cursor: str | None = None,
    auto_page: bool = False,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
    max_pages: int = 3,
) -> ToolResult:
    """ADK discovery wrapper for /memory/v1/entities."""

    query_norm = _norm_text(query)
    entity_type_norm = _norm_text(entity_type)
    mentioned_since_input = _norm_text(mentioned_since)
    mentioned_since_norm = _parse_iso_optional(mentioned_since_input)
    cursor_norm, cursor_invalid = _normalize_discovery_cursor(cursor)
    limit_norm = _clamp_limit(limit, default=20, min_value=1, max_value=50)
    max_pages_norm = _clamp_limit(max_pages, default=3, min_value=1, max_value=3)
    memory_domain_norm = _norm_text(memory_domain)

    if mentioned_since_input is not None and mentioned_since_norm is None:
        return _build_result(
            tool_name="list_entities",
            source_mode="graph_filter",
            api_route="/memory/v1/entities",
            matched=False,
            message="时间格式无效",
            data=None,
            resolution_meta={"query": {"mentioned_since_input": mentioned_since_input}},
            error_type="invalid_input",
            retryable=False,
        )

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolution_meta: Dict[str, Any] = {
        "query": {
            "query": query_norm,
            "entity_type": entity_type_norm,
            "mentioned_since_input": mentioned_since_input,
            "mentioned_since": mentioned_since_norm,
            "limit_input": limit,
            "limit": limit_norm,
            "cursor_input": _norm_text(cursor),
            "cursor": cursor_norm,
            "cursor_invalid": bool(cursor_invalid),
            "auto_page": bool(auto_page),
            "max_pages": max_pages_norm,
        },
        "scope": {"memory_domain": memory_domain_norm},
    }

    entities_out: List[Dict[str, Any]] = []
    total_out = 0
    has_more_out = False
    next_cursor_out: Optional[str] = None
    pages_fetched = 0
    raw_keys_last: Optional[List[str]] = None

    current_cursor = cursor_norm
    if cursor_invalid:
        current_cursor = None

    try:
        while True:
            resp = await list_entities_api(
                tenant_id=str(tenant_id),
                user_tokens=list(effective_user_tokens) if effective_user_tokens else None,
                type=entity_type_norm,
                query=query_norm,
                mentioned_since=mentioned_since_norm,
                limit=limit_norm,
                cursor=current_cursor,
                memory_domain=memory_domain_norm,
            )

            if _is_http_error_payload(resp):
                err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
                msg = "实体发现服务暂时不可用" if err.retryable else "实体发现查询失败"
                return _build_result(
                    tool_name="list_entities",
                    source_mode="graph_filter",
                    api_route="/memory/v1/entities",
                    matched=False,
                    message=msg,
                    data=None,
                    resolution_meta=resolution_meta,
                    raw_api_response_keys=sorted(str(k) for k in resp.keys()),
                    error_type=err.error_type,
                    retryable=err.retryable,
                )
            if not isinstance(resp, dict):
                return _build_result(
                    tool_name="list_entities",
                    source_mode="graph_filter",
                    api_route="/memory/v1/entities",
                    matched=False,
                    message="实体发现查询失败",
                    data=None,
                    resolution_meta=resolution_meta,
                    error_type="internal_error",
                    retryable=False,
                )

            pages_fetched += 1
            raw_keys_last = sorted(str(k) for k in resp.keys())
            page_entities = [dict(x) for x in (resp.get("entities") or []) if isinstance(x, dict)]
            entities_out.extend(page_entities)
            total_out = int(resp.get("total") or total_out or 0)
            has_more_out = bool(resp.get("has_more"))
            next_cursor_out = _norm_text(resp.get("next_cursor"))

            if not auto_page:
                break
            if not has_more_out or not next_cursor_out:
                break
            if pages_fetched >= max_pages_norm:
                break
            current_cursor = next_cursor_out
    except Exception as exc:
        err = normalize_exception(exc)
        msg = "实体发现服务暂时不可用" if err.retryable else "实体发现查询失败"
        return _build_result(
            tool_name="list_entities",
            source_mode="graph_filter",
            api_route="/memory/v1/entities",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    data = {
        "entities": entities_out,
        "total": total_out,
        "has_more": bool(has_more_out),
        "next_cursor": next_cursor_out,
    }
    debug = ToolDebugTrace(
        tool_name="list_entities",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=raw_keys_last,
        extras={
            "api_route": "/memory/v1/entities",
            "pages_fetched": pages_fetched,
            "cursor_invalid": bool(cursor_invalid),
        },
    )

    if not entities_out:
        msg = "未找到实体候选"
        if cursor_invalid:
            msg = "分页游标无效，已回到首页"
        return ToolResult(matched=False, needs_disambiguation=False, message=msg, data=data, debug=debug)
    msg_ok = "分页游标无效，已回到首页" if cursor_invalid else None
    return ToolResult.success(data=data, message=msg_ok, debug=debug)


async def list_topics(
    *,
    tenant_id: str,
    list_topics_api: ListTopicsFn,
    query: str | None = None,
    parent_path: str | None = None,
    min_events: int | None = None,
    limit: int = 20,
    cursor: str | None = None,
    auto_page: bool = False,
    user_tokens: Optional[List[str]] = None,
    memory_domain: Optional[str] = None,
    max_pages: int = 3,
) -> ToolResult:
    """ADK discovery wrapper for /memory/v1/topics."""

    query_norm = _norm_text(query)
    parent_path_norm = _norm_text(parent_path)
    min_events_norm, min_events_adjusted = _norm_nonnegative_int(min_events)
    if min_events is not None and min_events_norm is None:
        return _build_result(
            tool_name="list_topics",
            source_mode="graph_filter",
            api_route="/memory/v1/topics",
            matched=False,
            message="参数格式无效",
            data=None,
            resolution_meta={"query": {"min_events_input": min_events}},
            error_type="invalid_input",
            retryable=False,
        )

    cursor_norm, cursor_invalid = _normalize_discovery_cursor(cursor)
    limit_norm = _clamp_limit(limit, default=20, min_value=1, max_value=50)
    max_pages_norm = _clamp_limit(max_pages, default=3, min_value=1, max_value=3)
    memory_domain_norm = _norm_text(memory_domain)

    effective_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not effective_user_tokens and tenant_id:
        effective_user_tokens = [f"u:{str(tenant_id).strip()}"]

    resolution_meta: Dict[str, Any] = {
        "query": {
            "query": query_norm,
            "parent_path": parent_path_norm,
            "min_events_input": min_events,
            "min_events": min_events_norm,
            "min_events_clamped": bool(min_events_adjusted and (min_events is not None)),
            "limit_input": limit,
            "limit": limit_norm,
            "cursor_input": _norm_text(cursor),
            "cursor": cursor_norm,
            "cursor_invalid": bool(cursor_invalid),
            "auto_page": bool(auto_page),
            "max_pages": max_pages_norm,
        },
        "scope": {"memory_domain": memory_domain_norm},
    }

    topics_out: List[Dict[str, Any]] = []
    total_out = 0
    has_more_out = False
    next_cursor_out: Optional[str] = None
    status_thresholds: Optional[Dict[str, Any]] = None
    pages_fetched = 0
    raw_keys_last: Optional[List[str]] = None

    current_cursor = cursor_norm
    if cursor_invalid:
        current_cursor = None

    try:
        while True:
            resp = await list_topics_api(
                tenant_id=str(tenant_id),
                user_tokens=list(effective_user_tokens) if effective_user_tokens else None,
                query=query_norm,
                parent_path=parent_path_norm,
                min_events=min_events_norm,
                limit=limit_norm,
                cursor=current_cursor,
                memory_domain=memory_domain_norm,
            )

            if _is_http_error_payload(resp):
                err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
                msg = "话题发现服务暂时不可用" if err.retryable else "话题发现查询失败"
                return _build_result(
                    tool_name="list_topics",
                    source_mode="graph_filter",
                    api_route="/memory/v1/topics",
                    matched=False,
                    message=msg,
                    data=None,
                    resolution_meta=resolution_meta,
                    raw_api_response_keys=sorted(str(k) for k in resp.keys()),
                    error_type=err.error_type,
                    retryable=err.retryable,
                )
            if not isinstance(resp, dict):
                return _build_result(
                    tool_name="list_topics",
                    source_mode="graph_filter",
                    api_route="/memory/v1/topics",
                    matched=False,
                    message="话题发现查询失败",
                    data=None,
                    resolution_meta=resolution_meta,
                    error_type="internal_error",
                    retryable=False,
                )

            pages_fetched += 1
            raw_keys_last = sorted(str(k) for k in resp.keys())
            page_topics = [dict(x) for x in (resp.get("topics") or []) if isinstance(x, dict)]
            topics_out.extend(page_topics)
            total_out = int(resp.get("total") or total_out or 0)
            has_more_out = bool(resp.get("has_more"))
            next_cursor_out = _norm_text(resp.get("next_cursor"))
            status_thresholds = dict(resp.get("status_thresholds") or {}) if isinstance(resp.get("status_thresholds"), dict) else status_thresholds

            if not auto_page:
                break
            if not has_more_out or not next_cursor_out:
                break
            if pages_fetched >= max_pages_norm:
                break
            current_cursor = next_cursor_out
    except Exception as exc:
        err = normalize_exception(exc)
        msg = "话题发现服务暂时不可用" if err.retryable else "话题发现查询失败"
        return _build_result(
            tool_name="list_topics",
            source_mode="graph_filter",
            api_route="/memory/v1/topics",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )

    display_name_unavailable = bool(topics_out) and all((_norm_text(it.get("display_name")) is None) for it in topics_out if isinstance(it, dict))
    resolution_meta["display_name_unavailable"] = bool(display_name_unavailable)
    if status_thresholds is not None:
        resolution_meta["status_thresholds"] = dict(status_thresholds)

    data = {
        "topics": topics_out,
        "total": total_out,
        "has_more": bool(has_more_out),
        "next_cursor": next_cursor_out,
    }
    debug = ToolDebugTrace(
        tool_name="list_topics",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=raw_keys_last,
        extras={
            "api_route": "/memory/v1/topics",
            "pages_fetched": pages_fetched,
            "cursor_invalid": bool(cursor_invalid),
        },
    )

    if not topics_out:
        msg = "未找到话题候选"
        if cursor_invalid:
            msg = "分页游标无效，已回到首页"
        return ToolResult(matched=False, needs_disambiguation=False, message=msg, data=data, debug=debug)
    msg_ok = "分页游标无效，已回到首页" if cursor_invalid else None
    return ToolResult.success(data=data, message=msg_ok, debug=debug)
