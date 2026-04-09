from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolDebugTrace:
    """Debug/observability payload for ADK tools (not sent to LLM context)."""

    tool_name: str
    source_mode: Optional[str] = None
    error_type: Optional[str] = None
    retryable: bool = False
    latency_ms: Optional[int] = None
    resolution_meta: Optional[Dict[str, Any]] = None
    raw_api_response_keys: Optional[List[str]] = None
    fallback_used: bool = False
    api_calls: Optional[List[Dict[str, Any]]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tool_name": str(self.tool_name),
            "source_mode": self.source_mode,
            "error_type": self.error_type,
            "retryable": bool(self.retryable),
            "latency_ms": self.latency_ms,
            "resolution_meta": dict(self.resolution_meta or {}) if self.resolution_meta else None,
            "raw_api_response_keys": list(self.raw_api_response_keys or []) if self.raw_api_response_keys else None,
            "fallback_used": bool(self.fallback_used),
            "api_calls": [dict(x) for x in (self.api_calls or [])] if self.api_calls else None,
        }
        if self.extras:
            out.update(dict(self.extras))
        return out


@dataclass
class ToolResult:
    """ADK tool result wrapper with strict LLM-visible surface and separate debug payload."""

    matched: bool
    needs_disambiguation: bool = False
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    debug: Optional[ToolDebugTrace | Dict[str, Any]] = None

    @classmethod
    def success(
        cls,
        *,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        debug: Optional[ToolDebugTrace | Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(matched=True, needs_disambiguation=False, message=message, data=data, debug=debug)

    @classmethod
    def no_match(
        cls,
        *,
        message: Optional[str] = None,
        debug: Optional[ToolDebugTrace | Dict[str, Any]] = None,
    ) -> "ToolResult":
        return cls(matched=False, needs_disambiguation=False, message=message, data=None, debug=debug)

    @classmethod
    def disambiguation(
        cls,
        *,
        candidates: List[Dict[str, Any]],
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        debug: Optional[ToolDebugTrace | Dict[str, Any]] = None,
    ) -> "ToolResult":
        payload = dict(data or {})
        payload.setdefault("candidates", [dict(x) for x in candidates])
        return cls(matched=False, needs_disambiguation=True, message=message, data=payload, debug=debug)

    def to_llm_dict(self) -> Dict[str, Any]:
        """Strictly expose the 4-field business payload to the LLM layer."""
        return {
            "matched": bool(self.matched),
            "needs_disambiguation": bool(self.needs_disambiguation),
            "message": self.message,
            "data": dict(self.data or {}) if self.data is not None else None,
        }

    def to_debug_dict(self) -> Optional[Dict[str, Any]]:
        if self.debug is None:
            return None
        if isinstance(self.debug, ToolDebugTrace):
            return self.debug.to_dict()
        return dict(self.debug)

    def to_wire_dict(self, *, include_debug: bool = False, debug_key: str = "debug") -> Dict[str, Any]:
        out = self.to_llm_dict()
        if include_debug:
            out[debug_key] = self.to_debug_dict()
        return out

