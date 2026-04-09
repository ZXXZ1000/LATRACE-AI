"""Memory module public API.

Import only from this entry in other modules to keep internals swappable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Public contracts
from modules.memory.contracts.memory_models import (
    MemoryEntry,
    Edge,
    SearchFilters,
    SearchResult,
    Version,
)
from modules.memory.contracts.graph_models import (
    GraphUpsertRequest,
    GraphEdge as GraphRel,
    MediaSegment as GraphMediaSegment,
    Evidence as GraphEvidence,
    UtteranceEvidence as GraphUtteranceEvidence,
    Entity as GraphEntity,
    SpatioTemporalRegion as GraphSpatioTemporalRegion,
    State as GraphState,
    Knowledge as GraphKnowledge,
    PendingEquiv as GraphPendingEquiv,
    Event as GraphEvent,
    Place as GraphPlace,
    TimeSlice as GraphTimeSlice,
)

# Primary service
from modules.memory.application.service import MemoryService

# High-level client pipeline API (dialog ingestion)
from modules.memory.session_write import session_write
from modules.memory.retrieval import retrieval
from modules.memory.adapters.http_memory_port import HttpMemoryPort
from modules.memory.application.llm_adapter import (
    LLMAdapter,
    LLMUsageContext,
    build_llm_from_byok,
    build_llm_from_config,
    build_llm_from_env,
    reset_llm_usage_context,
    reset_llm_usage_hook,
    set_llm_usage_context,
    set_llm_usage_hook,
)

# NOTE: GraphService and create_service have heavy optional deps (neo4j / fastapi).
# Keep `modules.memory` importable in minimal environments by lazy-loading them.
if TYPE_CHECKING:
    from modules.memory.application.graph_service import GraphService as GraphService
    from modules.memory.api.server import create_service as create_service

__all__ = [
    "MemoryEntry",
    "Edge",
    "SearchFilters",
    "SearchResult",
    "Version",
    "GraphUpsertRequest",
    "GraphRel",
    "GraphMediaSegment",
    "GraphEvidence",
    "GraphUtteranceEvidence",
    "GraphEntity",
    "GraphSpatioTemporalRegion",
    "GraphState",
    "GraphKnowledge",
    "GraphPendingEquiv",
    "GraphEvent",
    "GraphPlace",
    "GraphTimeSlice",
    "MemoryService",
    "session_write",
    "retrieval",
    "HttpMemoryPort",
    "LLMAdapter",
    "LLMUsageContext",
    "build_llm_from_byok",
    "build_llm_from_env",
    "build_llm_from_config",
    "set_llm_usage_context",
    "reset_llm_usage_context",
    "set_llm_usage_hook",
    "reset_llm_usage_hook",
    "GraphService",
    "create_service",
]


def __getattr__(name: str) -> Any:
    if name == "GraphService":
        try:
            from modules.memory.application.graph_service import GraphService as _GraphService
            return _GraphService
        except ModuleNotFoundError as exc:  # pragma: no cover
            # Keep `modules.memory` importable in minimal envs.
            if getattr(exc, "name", None) in ("neo4j", "fastapi"):
                missing_exc = exc

                def _missing(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError("GraphService requires optional dependencies (neo4j backend).") from missing_exc

                return _missing
            raise
    if name == "create_service":
        try:
            from modules.memory.api.server import create_service as _create_service
            return _create_service
        except ModuleNotFoundError as exc:  # pragma: no cover
            if getattr(exc, "name", None) == "fastapi":
                missing_exc = exc

                def _missing(*_args: Any, **_kwargs: Any) -> Any:
                    raise RuntimeError("create_service requires optional dependency 'fastapi'.") from missing_exc

                return _missing
            raise
    raise AttributeError(name)
