from __future__ import annotations

from typing import Protocol, List, Optional
from modules.memory.contracts.memory_models import (
    MemoryEntry,
    Edge,
    SearchFilters,
    SearchResult,
    Version,
)
from modules.memory.contracts.graph_models import GraphUpsertRequest


class MemoryPort(Protocol):
    """Unified memory store (Qdrant + Neo4j) public interface.

    All sources (m3/mem0/ctrl) should go through this Port. Implementation lives in application/service.py
    with concrete infra stores injected (qdrant_store, neo4j_store, audit_store).
    """

    async def search(
        self,
        query: str,
        *,
        topk: int = 10,
        filters: Optional[SearchFilters] = None,
        expand_graph: bool = True,
        threshold: Optional[float] = None,
        scope: Optional[str] = None,
    ) -> SearchResult:
        """ANN → graph neighborhood expansion → re-rank → hints."""

    async def embed_query(self, query: str, *, tenant_id: Optional[str] = None) -> Optional[List[float]]:
        """Compute a reusable query embedding through the store's public interface."""

    async def write(
        self,
        entries: List[MemoryEntry],
        links: Optional[List[Edge]] = None,
        *,
        upsert: bool = True,
        return_id_map: bool = False,
    ) -> Version | tuple[Version, dict[str, str]]:
        """Batch write (vectors + graph relations).

        - Default: returns Version (backward compatible).
        - If return_id_map=True: returns (Version, id_map) where id_map maps placeholder IDs (tmp-*/dev-*/loc-*/char-*) to persisted UUIDs.
        """

    async def update(self, memory_id: str, patch: dict, *, reason: Optional[str] = None) -> Version:
        """Update contents/metadata (importance/ttl/pinned/...). Returns new version."""

    async def delete(self, memory_id: str, *, soft: bool = True, reason: Optional[str] = None) -> Version:
        """Soft delete by default (with audit). Returns version."""

    async def link(self, src_id: str, dst_id: str, rel_type: str, *, weight: Optional[float] = None) -> bool:
        """Create/update/delete relations in Neo4j (MERGE/DELETE)."""

    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:
        """Write into TKG via Graph API v0 (schema aligned to TKG v1.0)."""

    async def graph_list_events(
        self,
        *,
        tenant_id: str,
        segment_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        place_id: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Query events filtered by segment/entity/place/source."""

    async def graph_list_places(
        self,
        *,
        tenant_id: str,
        name: Optional[str] = None,
        segment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Query places with fuzzy name/segment filters."""

    async def graph_event_detail(self, *, tenant_id: str, event_id: str) -> dict:
        """Fetch an event with linked segments/entities/places."""

    async def graph_place_detail(self, *, tenant_id: str, place_id: str) -> dict:
        """Fetch a place with linked events/segments."""

    async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:
        """Fetch an explain-style evidence chain for an event (TKG-first)."""

    async def graph_search_v1(
        self,
        *,
        tenant_id: str,
        query: str,
        topk: int = 10,
        source_id: Optional[str] = None,
        include_evidence: bool = True,
    ) -> dict:
        """Graph-first search over typed TKG events (returns items with event_id + optional evidence)."""

    async def graph_resolve_entities(
        self,
        *,
        tenant_id: str,
        name: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Resolve entities by name/alias (best-effort)."""

    async def graph_list_timeslices_range(
        self,
        *,
        tenant_id: str,
        start_iso: Optional[str],
        end_iso: Optional[str],
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        """List TimeSlice nodes overlapping an absolute time window."""
