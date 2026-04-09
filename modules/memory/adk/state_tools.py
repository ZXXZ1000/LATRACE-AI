from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .errors import normalize_exception, normalize_http_error
from .models import ToolDebugTrace, ToolResult
from .resolve import ResolveEntityFn
from .state_preflight import prepare_state_query_preflight
from .state_property_vocab import StatePropertyVocabManager

StateCurrentFn = Callable[..., Awaitable[Dict[str, Any]]]
StateAtTimeFn = Callable[..., Awaitable[Dict[str, Any]]]
StateChangesFn = Callable[..., Awaitable[Dict[str, Any]]]
StateTimeSinceFn = Callable[..., Awaitable[Dict[str, Any]]]
WhenParserFn = Callable[[str], Any]
TimeRangeParserFn = Callable[[str], Any]


def _is_http_error_payload(resp: Any) -> bool:
    return isinstance(resp, dict) and "status_code" in resp and int(resp.get("status_code") or 200) >= 400


def _norm_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _default_parse_when_to_iso(when_text: str) -> tuple[Optional[str], str]:
    text = _norm_text(when_text)
    if not text:
        return None, "empty"
    raw = text
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None, "invalid"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(), "iso_direct"


def _parse_optional_iso(value: Any) -> tuple[Optional[str], str]:
    text = _norm_text(value)
    if text is None:
        return None, "empty"
    iso, source = _default_parse_when_to_iso(text)
    return iso, source


async def _parse_when_to_iso(
    *,
    when: str,
    when_parser: Optional[WhenParserFn],
) -> tuple[Optional[str], Dict[str, Any]]:
    if when_parser is None:
        t_iso, source = _default_parse_when_to_iso(when)
        return t_iso, {"input": when, "t_iso": t_iso, "parse_source": source}

    try:
        parsed = when_parser(when)
        if inspect.isawaitable(parsed):
            parsed = await parsed
    except Exception:
        return None, {"input": when, "t_iso": None, "parse_source": "parser_error"}

    if isinstance(parsed, str):
        t_iso, _ = _default_parse_when_to_iso(parsed)
        return t_iso, {"input": when, "t_iso": t_iso, "parse_source": "parser_string"}
    if isinstance(parsed, dict):
        t_iso = _norm_text(parsed.get("t_iso") or parsed.get("value"))
        if t_iso:
            norm, _ = _default_parse_when_to_iso(t_iso)
            return norm, {"input": when, "t_iso": norm, "parse_source": str(parsed.get("parse_source") or "parser_dict")}
        return None, {"input": when, "t_iso": None, "parse_source": str(parsed.get("parse_source") or "parser_dict")}
    return None, {"input": when, "t_iso": None, "parse_source": "parser_invalid"}


async def _parse_time_range(
    *,
    when: Optional[str],
    time_range: Optional[Dict[str, Any]],
    time_range_parser: Optional[TimeRangeParserFn],
) -> tuple[Optional[Dict[str, Optional[str]]], Dict[str, Any]]:
    """Normalize `when` / `time_range` into start/end ISO pair.

    Returns:
      ({start_iso,end_iso} | None, debug_meta)
    """
    explicit = dict(time_range or {})
    when_input = _norm_text(when)

    if explicit:
        start_raw = explicit.get("start_iso", explicit.get("start"))
        end_raw = explicit.get("end_iso", explicit.get("end"))
        start_iso, start_src = _parse_optional_iso(start_raw)
        end_iso, end_src = _parse_optional_iso(end_raw)
        if (start_raw is not None and start_iso is None) or (end_raw is not None and end_iso is None):
            return None, {
                "when_input": when_input,
                "time_range_input": {"start": start_raw, "end": end_raw},
                "start_iso": start_iso,
                "end_iso": end_iso,
                "parse_source": "time_range_invalid",
                "start_parse_source": start_src,
                "end_parse_source": end_src,
            }
        if start_iso is None and end_iso is None:
            return None, {
                "when_input": when_input,
                "time_range_input": {"start": start_raw, "end": end_raw},
                "start_iso": None,
                "end_iso": None,
                "parse_source": "time_range_empty",
            }
        return (
            {"start_iso": start_iso, "end_iso": end_iso},
            {
                "when_input": when_input,
                "time_range_input": {"start": start_raw, "end": end_raw},
                "start_iso": start_iso,
                "end_iso": end_iso,
                "parse_source": "time_range_explicit",
                "explicit_overrides_when": when_input is not None,
            },
        )

    if when_input is None:
        return (
            {"start_iso": None, "end_iso": None},
            {
                "when_input": None,
                "time_range_input": None,
                "start_iso": None,
                "end_iso": None,
                "parse_source": "none",
            },
        )

    if time_range_parser is None:
        return None, {
            "when_input": when_input,
            "time_range_input": None,
            "start_iso": None,
            "end_iso": None,
            "parse_source": "missing_time_range_parser",
        }

    try:
        parsed = time_range_parser(when_input)
        if inspect.isawaitable(parsed):
            parsed = await parsed
    except Exception:
        return None, {
            "when_input": when_input,
            "time_range_input": None,
            "start_iso": None,
            "end_iso": None,
            "parse_source": "parser_error",
        }

    if not isinstance(parsed, dict):
        return None, {
            "when_input": when_input,
            "time_range_input": None,
            "start_iso": None,
            "end_iso": None,
            "parse_source": "parser_invalid",
        }

    start_iso_raw = _norm_text(parsed.get("start_iso") or parsed.get("start"))
    end_iso_raw = _norm_text(parsed.get("end_iso") or parsed.get("end"))
    start_iso, start_src = _parse_optional_iso(start_iso_raw)
    end_iso, end_src = _parse_optional_iso(end_iso_raw)
    if (start_iso_raw and start_iso is None) or (end_iso_raw and end_iso is None) or (start_iso is None and end_iso is None):
        return None, {
            "when_input": when_input,
            "time_range_input": None,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "parse_source": str(parsed.get("parse_source") or "parser_dict_invalid"),
            "start_parse_source": start_src,
            "end_parse_source": end_src,
        }
    return (
        {"start_iso": start_iso, "end_iso": end_iso},
        {
            "when_input": when_input,
            "time_range_input": None,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "parse_source": str(parsed.get("parse_source") or "parser_dict"),
            "start_parse_source": start_src,
            "end_parse_source": end_src,
        },
    )


def _normalize_order(value: Optional[str]) -> tuple[str, bool]:
    text = str(value or "").strip().lower()
    if text in {"asc", "desc"}:
        return text, False
    return "desc", True


def _clamp_limit(value: Any, *, default: int, min_value: int = 1, max_value: int = 50) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    if n < min_value:
        n = min_value
    if n > max_value:
        n = max_value
    return n


def _entity_payload(*, entity_input: Optional[str], entity_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not entity_id and not entity_input:
        return None
    out: Dict[str, Any] = {}
    if entity_id:
        out["id"] = entity_id
    if entity_input:
        out["name"] = entity_input
    return out


def _build_state_tool_result(
    *,
    tool_name: str,
    matched: bool,
    message: Optional[str],
    data: Optional[Dict[str, Any]],
    resolution_meta: Dict[str, Any],
    raw_api_response_keys: Optional[List[str]] = None,
    error_type: Optional[str] = None,
    retryable: bool = False,
    api_route: Optional[str] = None,
) -> ToolResult:
    debug = ToolDebugTrace(
        tool_name=tool_name,
        source_mode="graph_filter",
        error_type=error_type,
        retryable=retryable,
        resolution_meta=resolution_meta,
        raw_api_response_keys=raw_api_response_keys,
        extras={"api_route": api_route} if api_route else {},
    )
    return ToolResult(
        matched=matched,
        needs_disambiguation=False,
        message=message,
        data=data,
        debug=debug,
    )


async def entity_status(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    vocab_manager: StatePropertyVocabManager,
    state_current: StateCurrentFn,
    state_at_time: StateAtTimeFn,
    entity: str | None = None,
    property: str | None = None,
    when: str | None = None,
    entity_id: str | None = None,
    property_canonical: str | None = None,
    user_tokens: Optional[List[str]] = None,
    when_parser: Optional[WhenParserFn] = None,
    force_vocab_refresh: bool = False,
) -> ToolResult:
    """ADK semantic wrapper for /memory/state/current and /memory/state/at_time."""

    preflight = await prepare_state_query_preflight(
        tool_name="entity_status",
        tenant_id=tenant_id,
        vocab_manager=vocab_manager,
        resolver=resolver,
        entity=entity,
        entity_id=entity_id,
        property_text=property,
        property_canonical=property_canonical,
        user_tokens=user_tokens,
        force_vocab_refresh=force_vocab_refresh,
    )
    if preflight.should_stop:
        return preflight.terminal_result or ToolResult.no_match(message="查询未完成")

    assert preflight.entity_id is not None
    assert preflight.property_canonical is not None

    entity_input = _norm_text(entity)
    property_input = _norm_text(property)
    when_input = _norm_text(when)
    resolution_meta = dict(preflight.resolution_meta or {})

    api_route = "/memory/state/current"
    t_iso: Optional[str] = None
    if when_input is not None:
        t_iso, when_meta = await _parse_when_to_iso(when=when_input, when_parser=when_parser)
        resolution_meta["time"] = when_meta
        if not t_iso:
            return _build_state_tool_result(
                tool_name="entity_status",
                matched=False,
                message="时间格式无效",
                data=None,
                resolution_meta=resolution_meta,
                error_type="invalid_input",
                retryable=False,
                api_route="/memory/state/at_time",
            )
        api_route = "/memory/state/at_time"

    try:
        if t_iso is None:
            resp = await state_current(
                tenant_id=str(tenant_id),
                subject_id=preflight.entity_id,
                property=preflight.property_canonical,
                user_tokens=list(user_tokens or []),
            )
        else:
            resp = await state_at_time(
                tenant_id=str(tenant_id),
                subject_id=preflight.entity_id,
                property=preflight.property_canonical,
                t_iso=t_iso,
                user_tokens=list(user_tokens or []),
            )
    except Exception as exc:
        err = normalize_exception(exc)
        return _build_state_tool_result(
            tool_name="entity_status",
            matched=False,
            message="服务暂时不可用" if err.retryable else "状态查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        if err.error_type == "not_found":
            msg = "未找到该状态"
        elif err.error_type == "invalid_input":
            msg = "时间格式无效" if t_iso is not None else "请求参数无效"
        else:
            msg = "服务暂时不可用" if err.retryable else "状态查询失败"
        return _build_state_tool_result(
            tool_name="entity_status",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if not isinstance(resp, dict):
        return _build_state_tool_result(
            tool_name="entity_status",
            matched=False,
            message="状态查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
            api_route=api_route,
        )

    item = resp.get("item") if isinstance(resp.get("item"), dict) else None
    if not item:
        return _build_state_tool_result(
            tool_name="entity_status",
            matched=False,
            message="未找到该状态",
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            api_route=api_route,
        )

    data = {
        "entity": _entity_payload(entity_input=entity_input, entity_id=preflight.entity_id),
        "property": {
            "input": property_input,
            "canonical": preflight.property_canonical,
        },
        "when": t_iso,
        "item": dict(item),
    }
    debug = ToolDebugTrace(
        tool_name="entity_status",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": api_route},
    )
    return ToolResult.success(data=data, debug=debug)


async def status_changes(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    vocab_manager: StatePropertyVocabManager,
    state_what_changed: StateChangesFn,
    entity: str | None = None,
    property: str | None = None,
    when: str | None = None,
    time_range: Optional[Dict[str, Any]] = None,
    entity_id: str | None = None,
    property_canonical: str | None = None,
    user_tokens: Optional[List[str]] = None,
    time_range_parser: Optional[TimeRangeParserFn] = None,
    order: str = "desc",
    limit: int = 20,
    force_vocab_refresh: bool = False,
) -> ToolResult:
    """ADK semantic wrapper for /memory/state/what-changed."""

    preflight = await prepare_state_query_preflight(
        tool_name="status_changes",
        tenant_id=tenant_id,
        vocab_manager=vocab_manager,
        resolver=resolver,
        entity=entity,
        entity_id=entity_id,
        property_text=property,
        property_canonical=property_canonical,
        user_tokens=user_tokens,
        force_vocab_refresh=force_vocab_refresh,
    )
    if preflight.should_stop:
        return preflight.terminal_result or ToolResult.no_match(message="查询未完成")

    assert preflight.entity_id is not None
    assert preflight.property_canonical is not None

    resolution_meta = dict(preflight.resolution_meta or {})
    entity_input = _norm_text(entity)
    property_input = _norm_text(property)
    api_route = "/memory/state/what-changed"

    normalized_limit = _clamp_limit(limit, default=20, min_value=1, max_value=50)
    normalized_order, order_normalized = _normalize_order(order)
    resolution_meta["query"] = {
        "limit_input": limit,
        "limit": normalized_limit,
        "order_input": order,
        "order": normalized_order,
        "order_normalized": bool(order_normalized),
    }

    range_out, range_meta = await _parse_time_range(
        when=when,
        time_range=time_range,
        time_range_parser=time_range_parser,
    )
    resolution_meta["time_range"] = range_meta
    if range_out is None:
        return _build_state_tool_result(
            tool_name="status_changes",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta=resolution_meta,
            error_type="invalid_input",
            retryable=False,
            api_route=api_route,
        )

    try:
        resp = await state_what_changed(
            tenant_id=str(tenant_id),
            subject_id=preflight.entity_id,
            property=preflight.property_canonical,
            start_iso=range_out.get("start_iso"),
            end_iso=range_out.get("end_iso"),
            limit=normalized_limit,
            order=normalized_order,
            user_tokens=list(user_tokens or []),
        )
    except Exception as exc:
        err = normalize_exception(exc)
        return _build_state_tool_result(
            tool_name="status_changes",
            matched=False,
            message="服务暂时不可用" if err.retryable else "状态变化查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        return _build_state_tool_result(
            tool_name="status_changes",
            matched=False,
            message="服务暂时不可用" if err.retryable else "状态变化查询失败",
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if not isinstance(resp, dict):
        return _build_state_tool_result(
            tool_name="status_changes",
            matched=False,
            message="状态变化查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
            api_route=api_route,
        )

    items_raw = resp.get("items")
    items: List[Dict[str, Any]] = [dict(x) for x in (items_raw or []) if isinstance(x, dict)]
    data = {
        "entity": _entity_payload(entity_input=entity_input, entity_id=preflight.entity_id),
        "property": {"input": property_input, "canonical": preflight.property_canonical},
        "time_range": dict(range_out),
        "items": items,
    }

    debug = ToolDebugTrace(
        tool_name="status_changes",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": api_route, "order": normalized_order, "limit": normalized_limit},
    )
    if not items:
        return ToolResult(
            matched=False,
            needs_disambiguation=False,
            message="未找到状态变化记录",
            data=data,
            debug=debug,
        )
    return ToolResult.success(data=data, debug=debug)


async def state_time_since(
    *,
    tenant_id: str,
    resolver: ResolveEntityFn,
    vocab_manager: StatePropertyVocabManager,
    state_time_since_api: StateTimeSinceFn,
    entity: str | None = None,
    property: str | None = None,
    when: str | None = None,
    time_range: Optional[Dict[str, Any]] = None,
    entity_id: str | None = None,
    property_canonical: str | None = None,
    user_tokens: Optional[List[str]] = None,
    time_range_parser: Optional[TimeRangeParserFn] = None,
    force_vocab_refresh: bool = False,
) -> ToolResult:
    """ADK semantic wrapper for /memory/state/time-since."""

    preflight = await prepare_state_query_preflight(
        tool_name="state_time_since",
        tenant_id=tenant_id,
        vocab_manager=vocab_manager,
        resolver=resolver,
        entity=entity,
        entity_id=entity_id,
        property_text=property,
        property_canonical=property_canonical,
        user_tokens=user_tokens,
        force_vocab_refresh=force_vocab_refresh,
    )
    if preflight.should_stop:
        return preflight.terminal_result or ToolResult.no_match(message="查询未完成")

    assert preflight.entity_id is not None
    assert preflight.property_canonical is not None

    resolution_meta = dict(preflight.resolution_meta or {})
    entity_input = _norm_text(entity)
    property_input = _norm_text(property)
    api_route = "/memory/state/time-since"

    range_out, range_meta = await _parse_time_range(
        when=when,
        time_range=time_range,
        time_range_parser=time_range_parser,
    )
    resolution_meta["time_range"] = range_meta
    if range_out is None:
        return _build_state_tool_result(
            tool_name="state_time_since",
            matched=False,
            message="时间范围格式无效",
            data=None,
            resolution_meta=resolution_meta,
            error_type="invalid_input",
            retryable=False,
            api_route=api_route,
        )

    try:
        resp = await state_time_since_api(
            tenant_id=str(tenant_id),
            subject_id=preflight.entity_id,
            property=preflight.property_canonical,
            start_iso=range_out.get("start_iso"),
            end_iso=range_out.get("end_iso"),
            user_tokens=list(user_tokens or []),
        )
    except Exception as exc:
        err = normalize_exception(exc)
        return _build_state_tool_result(
            tool_name="state_time_since",
            matched=False,
            message="服务暂时不可用" if err.retryable else "状态变化时距查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if _is_http_error_payload(resp):
        err = normalize_http_error(status_code=int(resp["status_code"]), body=resp.get("body") or resp.get("detail") or resp)
        msg = "未找到状态变化记录" if err.error_type == "not_found" else ("服务暂时不可用" if err.retryable else "状态变化时距查询失败")
        return _build_state_tool_result(
            tool_name="state_time_since",
            matched=False,
            message=msg,
            data=None,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted(str(k) for k in resp.keys()),
            error_type=err.error_type,
            retryable=err.retryable,
            api_route=api_route,
        )

    if not isinstance(resp, dict):
        return _build_state_tool_result(
            tool_name="state_time_since",
            matched=False,
            message="状态变化时距查询失败",
            data=None,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
            api_route=api_route,
        )

    last_changed_at = _norm_text(resp.get("last_changed_at"))
    data = {
        "entity": _entity_payload(entity_input=entity_input, entity_id=preflight.entity_id),
        "property": {"input": property_input, "canonical": preflight.property_canonical},
        "time_range": dict(range_out),
        "subject_id": resp.get("subject_id", preflight.entity_id),
        "last_changed_at": resp.get("last_changed_at"),
        "value": resp.get("value"),
        "seconds_ago": resp.get("seconds_ago"),
    }
    if last_changed_at is None:
        resolution_meta["state_time_missing"] = True
    debug = ToolDebugTrace(
        tool_name="state_time_since",
        source_mode="graph_filter",
        resolution_meta=resolution_meta,
        raw_api_response_keys=sorted(str(k) for k in resp.keys()),
        extras={"api_route": api_route},
    )
    msg = "时间信息不完整，结果仅供参考" if last_changed_at is None else None
    return ToolResult.success(data=data, message=msg, debug=debug)
