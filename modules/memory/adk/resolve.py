from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .errors import normalize_exception
from .models import ToolDebugTrace, ToolResult


ResolveEntityFn = Callable[..., Awaitable[Dict[str, Any]]]


@dataclass
class ResolveIfNeededOutcome:
    entity_id: Optional[str]
    resolved_entity: Optional[Dict[str, Any]]
    terminal_result: Optional[ToolResult] = None
    resolution_meta: Optional[Dict[str, Any]] = None

    @property
    def should_stop(self) -> bool:
        return self.terminal_result is not None


def _bounded_resolve_limit(limit: int) -> int:
    try:
        return max(1, min(int(limit), 5))
    except Exception:
        return 5


def _base_entity_resolution_meta(
    *,
    entity_input: Optional[str],
    entity_id_input: Optional[str],
) -> Dict[str, Any]:
    return {
        "entity": {
            "input": entity_input,
            "input_entity_id": entity_id_input,
        }
    }


def _build_terminal_result(
    *,
    matched: bool,
    needs_disambiguation: bool,
    message: str,
    data: Optional[Dict[str, Any]],
    tool_name: str,
    resolution_meta: Dict[str, Any],
    error_type: Optional[str] = None,
    retryable: bool = False,
    raw_api_response_keys: Optional[List[str]] = None,
) -> ToolResult:
    debug = ToolDebugTrace(
        tool_name=tool_name,
        source_mode="graph_filter",
        error_type=error_type,
        retryable=retryable,
        resolution_meta=resolution_meta,
        raw_api_response_keys=raw_api_response_keys,
    )
    if needs_disambiguation:
        return ToolResult(
            matched=False,
            needs_disambiguation=True,
            message=message,
            data=data,
            debug=debug,
        )
    return ToolResult(
        matched=matched,
        needs_disambiguation=False,
        message=message,
        data=data,
        debug=debug,
    )


async def _resolve_if_needed(
    *,
    resolver: ResolveEntityFn,
    tool_name: str,
    entity: Optional[str] = None,
    entity_id: Optional[str] = None,
    entity_type: Optional[str] = None,
    user_tokens: Optional[List[str]] = None,
    resolve_limit: int = 5,
) -> ResolveIfNeededOutcome:
    """Shared entity resolution helper for ADK tools.

    Rules:
    - If `entity_id` is provided, bypass resolver.
    - If neither `entity` nor `entity_id` is provided, stop with invalid_input.
    - If resolver returns candidates, stop with needs_disambiguation=true.
    - If resolver not found, stop with matched=false.
    """

    entity_id_norm = str(entity_id or "").strip() or None
    entity_norm = str(entity or "").strip() or None
    resolution_meta = _base_entity_resolution_meta(entity_input=entity_norm, entity_id_input=entity_id_norm)

    if entity_id_norm:
        resolution_meta["entity"].update(
            {
                "resolved_id": entity_id_norm,
                "match_source": "direct_id",
            }
        )
        return ResolveIfNeededOutcome(
            entity_id=entity_id_norm,
            resolved_entity=None,
            resolution_meta=resolution_meta,
        )

    if not entity_norm:
        terminal = _build_terminal_result(
            matched=False,
            needs_disambiguation=False,
            message="缺少实体参数",
            data=None,
            tool_name=tool_name,
            resolution_meta=resolution_meta,
            error_type="invalid_input",
            retryable=False,
        )
        return ResolveIfNeededOutcome(
            entity_id=None,
            resolved_entity=None,
            terminal_result=terminal,
            resolution_meta=resolution_meta,
        )

    bounded_limit = _bounded_resolve_limit(resolve_limit)
    try:
        resp = await resolver(
            name=entity_norm,
            type=(str(entity_type).strip() or None) if entity_type is not None else None,
            user_tokens=list(user_tokens or []),
            limit=bounded_limit,
            debug=False,
        )
    except Exception as exc:
        err = normalize_exception(exc)
        terminal = _build_terminal_result(
            matched=False,
            needs_disambiguation=False,
            message="实体解析失败，请稍后重试" if err.retryable else "实体解析失败",
            data=None,
            tool_name=tool_name,
            resolution_meta=resolution_meta,
            error_type=err.error_type,
            retryable=err.retryable,
        )
        return ResolveIfNeededOutcome(
            entity_id=None,
            resolved_entity=None,
            terminal_result=terminal,
            resolution_meta=resolution_meta,
        )

    if not isinstance(resp, dict):
        terminal = _build_terminal_result(
            matched=False,
            needs_disambiguation=False,
            message="实体解析失败",
            data=None,
            tool_name=tool_name,
            resolution_meta=resolution_meta,
            error_type="internal_error",
            retryable=False,
        )
        return ResolveIfNeededOutcome(
            entity_id=None,
            resolved_entity=None,
            terminal_result=terminal,
            resolution_meta=resolution_meta,
        )

    candidates = [dict(x) for x in (resp.get("candidates") or []) if isinstance(x, dict)]
    resolved = resp.get("resolved_entity") if isinstance(resp.get("resolved_entity"), dict) else None
    found = bool(resp.get("found"))
    resolution_meta["entity"]["match_source"] = "resolve_entity"
    if candidates:
        resolution_meta["entity"]["candidates"] = candidates
        if resolved and resolved.get("id"):
            resolution_meta["entity"]["resolved_id"] = str(resolved.get("id"))
        terminal = _build_terminal_result(
            matched=False,
            needs_disambiguation=True,
            message=f"需要确认你说的是哪位{entity_norm}",
            data={"candidates": candidates},
            tool_name=tool_name,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted([str(k) for k in resp.keys()]),
        )
        return ResolveIfNeededOutcome(
            entity_id=None,
            resolved_entity=resolved,
            terminal_result=terminal,
            resolution_meta=resolution_meta,
        )

    resolved_id = str((resolved or {}).get("id") or "").strip() if resolved else ""
    if not found or not resolved_id:
        terminal = _build_terminal_result(
            matched=False,
            needs_disambiguation=False,
            message="没有找到相关实体",
            data=None,
            tool_name=tool_name,
            resolution_meta=resolution_meta,
            raw_api_response_keys=sorted([str(k) for k in resp.keys()]),
        )
        return ResolveIfNeededOutcome(
            entity_id=None,
            resolved_entity=resolved,
            terminal_result=terminal,
            resolution_meta=resolution_meta,
        )

    resolution_meta["entity"]["resolved_id"] = resolved_id
    if resolved:
        for key in ("match_source", "matched_by", "score"):
            if key in resolved:
                resolution_meta["entity"][key] = resolved.get(key)
    return ResolveIfNeededOutcome(
        entity_id=resolved_id,
        resolved_entity=resolved,
        resolution_meta=resolution_meta,
    )

