from __future__ import annotations

from typing import Any, Dict, Mapping

from modules.media_graph_compiler.application.semantic_provider import (
    RichBatchSemanticProvider,
    SemanticAdapter,
)


class LocalSemanticStage:
    """Default semantic stage backed by the migrated rich-batch provider shell."""

    def __init__(self, *, adapter: SemanticAdapter | None = None) -> None:
        self._provider = RichBatchSemanticProvider(adapter=adapter)

    def run(self, ctx: Mapping[str, Any]) -> Dict[str, Any]:
        return self._provider.generate_window_digests(ctx)


__all__ = ["LocalSemanticStage"]
