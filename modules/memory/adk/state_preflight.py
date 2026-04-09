from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import ToolDebugTrace, ToolResult
from .resolve import ResolveEntityFn, ResolveIfNeededOutcome, _resolve_if_needed
from .state_property_vocab import (
    PropertyResolutionResult,
    StatePropertyVocabLoadError,
    StatePropertyVocabManager,
    map_state_property,
)


@dataclass
class StateQueryPreflightOutcome:
    entity_id: Optional[str]
    property_canonical: Optional[str]
    terminal_result: Optional[ToolResult]
    resolution_meta: Dict[str, Any]

    @property
    def should_stop(self) -> bool:
        return self.terminal_result is not None


def _merge_resolution_meta(
    *,
    entity_meta: Optional[Dict[str, Any]],
    property_meta: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if entity_meta:
        out.update(dict(entity_meta))
    if property_meta:
        out["property"] = dict(property_meta)
    if extra:
        out.update(dict(extra))
    return out


def _terminal(
    *,
    tool_name: str,
    message: str,
    matched: bool = False,
    needs_disambiguation: bool = False,
    data: Optional[Dict[str, Any]] = None,
    resolution_meta: Optional[Dict[str, Any]] = None,
    error_type: Optional[str] = None,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        matched=matched,
        needs_disambiguation=needs_disambiguation,
        message=message,
        data=data,
        debug=ToolDebugTrace(
            tool_name=tool_name,
            source_mode="graph_filter",
            error_type=error_type,
            retryable=retryable,
            resolution_meta=resolution_meta or {},
        ),
    )


async def prepare_state_query_preflight(
    *,
    tool_name: str,
    tenant_id: str,
    vocab_manager: StatePropertyVocabManager,
    resolver: ResolveEntityFn,
    entity: Optional[str] = None,
    entity_id: Optional[str] = None,
    property_text: Optional[str] = None,
    property_canonical: Optional[str] = None,
    user_tokens: Optional[List[str]] = None,
    resolve_limit: int = 5,
    force_vocab_refresh: bool = False,
) -> StateQueryPreflightOutcome:
    """Shared preflight for state tools: resolve entity + map property."""

    ent_out: ResolveIfNeededOutcome = await _resolve_if_needed(
        resolver=resolver,
        tool_name=tool_name,
        entity=entity,
        entity_id=entity_id,
        user_tokens=user_tokens,
        resolve_limit=resolve_limit,
    )
    if ent_out.should_stop:
        return StateQueryPreflightOutcome(
            entity_id=None,
            property_canonical=None,
            terminal_result=ent_out.terminal_result,
            resolution_meta=dict(ent_out.resolution_meta or {}),
        )

    prop_in = str(property_text or "").strip() or None
    prop_direct = str(property_canonical or "").strip() or None
    entity_meta = dict((ent_out.resolution_meta or {}).get("entity") or {})

    if prop_direct:
        prop_meta = {
            "input": prop_in,
            "canonical": prop_direct,
            "match_source": "direct_canonical",
            "vocab_version": None,
        }
        return StateQueryPreflightOutcome(
            entity_id=ent_out.entity_id,
            property_canonical=prop_direct,
            terminal_result=None,
            resolution_meta=_merge_resolution_meta(entity_meta={"entity": entity_meta}, property_meta=prop_meta),
        )

    if not prop_in:
        terminal = _terminal(
            tool_name=tool_name,
            message="缺少状态属性参数",
            matched=False,
            resolution_meta=_merge_resolution_meta(entity_meta={"entity": entity_meta}, property_meta={"input": None}),
            error_type="invalid_input",
            retryable=False,
        )
        return StateQueryPreflightOutcome(
            entity_id=ent_out.entity_id,
            property_canonical=None,
            terminal_result=terminal,
            resolution_meta=dict((terminal.to_debug_dict() or {}).get("resolution_meta") or {}),
        )

    try:
        vocab = await vocab_manager.load_state_property_vocab(
            tenant_id=tenant_id,
            user_tokens=user_tokens,
            force_refresh=force_vocab_refresh,
        )
    except StatePropertyVocabLoadError as exc:
        msg = "状态属性词表暂不可用，请稍后重试" if exc.info.retryable else "状态属性词表不可用"
        terminal = _terminal(
            tool_name=tool_name,
            message=msg,
            matched=False,
            resolution_meta=_merge_resolution_meta(
                entity_meta={"entity": entity_meta},
                property_meta={"input": prop_in},
            ),
            error_type=exc.info.error_type,
            retryable=exc.info.retryable,
        )
        return StateQueryPreflightOutcome(
            entity_id=ent_out.entity_id,
            property_canonical=None,
            terminal_result=terminal,
            resolution_meta=dict((terminal.to_debug_dict() or {}).get("resolution_meta") or {}),
        )

    prop_res: PropertyResolutionResult = map_state_property(prop_in, vocab=vocab)
    prop_meta = prop_res.to_debug_meta(input_text=prop_in)
    prop_meta.update(vocab.debug_meta())
    merged_meta = _merge_resolution_meta(entity_meta={"entity": entity_meta}, property_meta=prop_meta)

    if prop_res.needs_disambiguation:
        terminal = _terminal(
            tool_name=tool_name,
            message="需要确认你查询的状态属性",
            matched=False,
            needs_disambiguation=True,
            data={"property_candidates": list(prop_res.candidates)},
            resolution_meta=merged_meta,
        )
        return StateQueryPreflightOutcome(
            entity_id=ent_out.entity_id,
            property_canonical=None,
            terminal_result=terminal,
            resolution_meta=merged_meta,
        )

    if not prop_res.matched or not prop_res.canonical:
        terminal = _terminal(
            tool_name=tool_name,
            message="无法识别状态属性",
            matched=False,
            data=({"property_candidates": list(prop_res.candidates)} if prop_res.candidates else None),
            resolution_meta=merged_meta,
        )
        return StateQueryPreflightOutcome(
            entity_id=ent_out.entity_id,
            property_canonical=None,
            terminal_result=terminal,
            resolution_meta=merged_meta,
        )

    return StateQueryPreflightOutcome(
        entity_id=ent_out.entity_id,
        property_canonical=prop_res.canonical,
        terminal_result=None,
        resolution_meta=merged_meta,
    )

