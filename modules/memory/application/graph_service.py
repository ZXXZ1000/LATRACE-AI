from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Any
from dataclasses import dataclass
import os
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
import threading

from modules.memory.contracts.graph_models import (
    GraphEdge,
    GraphUpsertRequest,
    MediaSegment,
    TimeSlice,
)
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.equiv_store import EquivStore
from modules.memory.application.metrics import inc
from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.domain.dialog_tkg_vector_index_v1 import TKG_DIALOG_EVENT_INDEX_SOURCE_V1


class GraphValidationError(ValueError):
    """Raised when graph input violates schema-level invariants."""


@dataclass
class GatingConfig:
    confidence_threshold: float = 0.0
    importance_threshold: float = 0.0
    rel_topk: int = 100
    enabled: bool = True


class GraphService:
    """Orchestration layer for TKG (Typed Knowledge Graph) - the PRIMARY memory storage system.

    TKG is the preferred way to store and retrieve memories. It provides:
    - Typed nodes: Event, Entity, Evidence, MediaSegment, Place, TimeSlice
    - Evidence chains: traceable provenance from raw media to semantic facts
    - Entity resolution: face/voice clustering -> Person entities
    - Vector search: TKG nodes are indexed in Qdrant with node_type/node_id payload

    Usage:
        svc = GraphService(neo4j_store, vector_store=qdrant_store)
        await svc.upsert(GraphUpsertRequest(events=[...], entities=[...], edges=[...]))

    Note: The legacy MemoryEntry/MemoryNode system is deprecated. Use TKG for all new code.
    See: modules/memory/contracts/graph_models.py for TKG data models.
    """

    def __init__(
        self,
        store: Neo4jStore,
        gating: Optional[Dict[str, Any]] = None,
        vector_store: Optional[Any] = None,  # QdrantStore for TKG vector writes
    ):
        self.store = store
        self.vector_store = vector_store  # Optional: enables TKG vector writes to Qdrant
        self.equiv_store = EquivStore(store)
        self.gating = self._load_gating(gating or {})
        gating = gating or {}
        self.ttl_default_days = float(os.getenv("GRAPH_TTL_DEFAULT_DAYS", 0) or gating.get("ttl_default_days", 0) or 0.0)
        self.decay_half_life_days = float(os.getenv("GRAPH_DECAY_HALF_LIFE_DAYS", 1.0) or 1.0)
        # Explain cache (LRU, in-memory, optional)
        try:
            enabled_raw = os.getenv("GRAPH_EXPLAIN_CACHE_ENABLED", "true").strip().lower()
            self._explain_cache_enabled: bool = enabled_raw not in ("0", "false", "no")
        except Exception:
            self._explain_cache_enabled = True
        try:
            self._explain_cache_ttl_s: float = float(os.getenv("GRAPH_EXPLAIN_CACHE_TTL_SECONDS", 600) or 600)
        except Exception:
            self._explain_cache_ttl_s = 600.0
        try:
            self._explain_cache_max: int = int(os.getenv("GRAPH_EXPLAIN_CACHE_MAX_ENTRIES", 256) or 256)
        except Exception:
            self._explain_cache_max = 256
        self._explain_cache: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._explain_cache_lock = threading.RLock()

    def _load_gating(self, gating: Dict[str, Any]) -> GatingConfig:
        def _get(name: str, default: Any, cast=float):
            env_key = f"GRAPH_GATING_{name.upper()}"
            raw = os.getenv(env_key, None)
            if raw is not None:
                try:
                    return cast(raw)
                except Exception:
                    return default
            return cast(gating.get(name, default))

        return GatingConfig(
            confidence_threshold=_get("confidence_threshold", 0.0),
            importance_threshold=_get("importance_threshold", 0.0),
            rel_topk=int(_get("rel_topk", 100, int)),
            enabled=bool(gating.get("enabled", True)),
        )

    def _apply_gating(self, req: GraphUpsertRequest) -> GraphUpsertRequest:
        if not self.gating.enabled:
            return req

        # Filter events by importance
        events = []
        for ev in req.events:
            if ev.importance is not None and ev.importance < self.gating.importance_threshold:
                continue
            events.append(ev)

        # Filter edges by confidence and top-K per (src, rel_type)
        edges_by_key: Dict[tuple[str, str], List[GraphEdge]] = {}
        for e in req.edges:
            if e.confidence is not None and e.confidence < self.gating.confidence_threshold:
                continue
            key = (e.src_id, e.rel_type)
            edges_by_key.setdefault(key, []).append(e)

        filtered_edges: List[GraphEdge] = []
        topk = max(1, int(self.gating.rel_topk))
        for group in edges_by_key.values():
            group_sorted = sorted(
                group,
                key=lambda x: (x.weight if x.weight is not None else x.confidence if x.confidence is not None else 0.0),
                reverse=True,
            )
            filtered_edges.extend(group_sorted[:topk])

        return GraphUpsertRequest(
            segments=req.segments,
            evidences=req.evidences,
            utterances=getattr(req, "utterances", []),
            entities=req.entities,
            events=events,
            places=req.places,
            time_slices=req.time_slices,
            regions=getattr(req, "regions", []),
            states=getattr(req, "states", []),
            knowledge=getattr(req, "knowledge", []),
            pending_equivs=getattr(req, "pending_equivs", []),
            edges=filtered_edges,
        )

    def _apply_ttl_defaults(self, req: GraphUpsertRequest) -> GraphUpsertRequest:
        """Backfill ttl/expires_at/created_at defaults for nodes if ttl_default_days is set."""
        # Even when TTL defaults are disabled, many Cypher queries apply soft TTL filters on `expires_at`.
        # If the property key never appears in the DB, Neo4j will emit noisy UnknownPropertyKeyWarning.
        # We avoid that by backfilling a stable far-future expires_at (only when missing),
        # without touching created_at/ttl (so repeated upserts remain stable).
        if self.ttl_default_days <= 0:
            far_future = datetime(9999, 1, 1, tzinfo=timezone.utc)

            def _touch(node: Any) -> Any:
                if getattr(node, "expires_at", None) is None:
                    try:
                        node.expires_at = far_future
                    except Exception:
                        pass
                return node

            return GraphUpsertRequest(
                segments=[_touch(s) for s in req.segments],
                evidences=[_touch(e) for e in req.evidences],
                utterances=[_touch(u) for u in getattr(req, "utterances", [])],
                entities=[_touch(e) for e in req.entities],
                events=[_touch(e) for e in req.events],
                places=[_touch(p) for p in req.places],
                time_slices=[_touch(t) for t in req.time_slices],
                regions=[_touch(r) for r in getattr(req, "regions", [])],
                states=[_touch(s) for s in getattr(req, "states", [])],
                knowledge=[_touch(k) for k in getattr(req, "knowledge", [])],
                pending_equivs=getattr(req, "pending_equivs", []),
                edges=req.edges,
            )

        def _touch(node: Any) -> Any:
            if getattr(node, "ttl", None) is None:
                try:
                    node.ttl = float(self.ttl_default_days * 86400.0)
                except Exception:
                    node.ttl = None
            if getattr(node, "created_at", None) is None:
                try:
                    node.created_at = datetime.now(timezone.utc)
                except Exception:
                    pass
            if getattr(node, "expires_at", None) is None and getattr(node, "ttl", None) is not None and getattr(node, "created_at", None):
                try:
                    node.expires_at = node.created_at + timedelta(seconds=float(node.ttl))
                except Exception:
                    pass
            return node

        return GraphUpsertRequest(
            segments=[_touch(s) for s in req.segments],
            evidences=[_touch(e) for e in req.evidences],
            utterances=[_touch(u) for u in getattr(req, "utterances", [])],
            entities=[_touch(e) for e in req.entities],
            events=[_touch(e) for e in req.events],
            places=[_touch(p) for p in req.places],
            time_slices=[_touch(t) for t in req.time_slices],
            regions=[_touch(r) for r in getattr(req, "regions", [])],
            states=[_touch(s) for s in getattr(req, "states", [])],
            knowledge=[_touch(k) for k in getattr(req, "knowledge", [])],
            pending_equivs=getattr(req, "pending_equivs", []),
            edges=req.edges,
        )

    def _decay_score(self, it: Dict[str, Any], *, now: datetime, time_fields: tuple[str, str] | None = None) -> float:
        importance = float(it.get("importance") or 1.0)
        last = it.get("last_accessed_at") or it.get("t_abs_start")
        if isinstance(last, str):
            try:
                last = datetime.fromisoformat(last)
            except Exception:
                last = None
        if last is None and time_fields:
            for f in time_fields:
                v = it.get(f)
                if v:
                    try:
                        last = datetime.fromisoformat(v) if isinstance(v, str) else v
                        break
                    except Exception:
                        continue
        if last is None:
            last = now
        strength = float(it.get("memory_strength") or 1.0)
        denom = max(1.0, strength * self.decay_half_life_days * 86400.0)
        return importance * pow(2.718281828, -((now - last).total_seconds() / denom))

    def _apply_decay(self, items: List[Dict[str, Any]], key: str, time_fields: tuple[str, str] | None = None) -> List[Dict[str, Any]]:
        """Apply a simple decay-based rerank on already-filtered items."""
        if not items:
            return items
        now = datetime.now(timezone.utc)
        scored = sorted(items, key=lambda it: self._decay_score(it, now=now, time_fields=time_fields), reverse=True)
        return scored

    # --- write paths ---
    async def upsert(self, req: GraphUpsertRequest) -> None:
        self._ensure_same_tenant(req)
        self._ensure_same_scope(req)
        self._validate_segments(req.segments)
        self._validate_time_slices(req.time_slices)
        enriched = self._apply_ttl_defaults(req)
        gated = self._apply_gating(enriched)
        
        # Phase 1: Prepare TKG vector entries and set vector_ids on objects
        # (IDs are set before Neo4j write so they can be persisted to the graph)
        pending_vector_entries: List[MemoryEntry] = []
        if self.vector_store is not None:
            pending_vector_entries = self._prepare_tkg_vectors(gated)
        
        # Phase 2: Write to Neo4j (includes vector_ids set in Phase 1)
        await self.store.upsert_graph_v0(
            segments=gated.segments,
            evidences=gated.evidences,
            utterances=getattr(gated, "utterances", []),
            entities=gated.entities,
            events=gated.events,
            places=gated.places,
            time_slices=gated.time_slices,
            regions=getattr(gated, "regions", []),
            states=getattr(gated, "states", []),
            knowledge=getattr(gated, "knowledge", []),
            pending_equivs=getattr(gated, "pending_equivs", []),
            edges=gated.edges,
        )
        
        # Phase 3: Write vectors to Qdrant ONLY after Neo4j succeeds
        # This prevents orphaned vectors if graph write fails
        if pending_vector_entries and self.vector_store is not None:
            await self._write_tkg_vectors(pending_vector_entries)
        
        if getattr(gated, "pending_equivs", []):
            self.equiv_store.upsert_pending(tenant_id=gated.pending_equivs[0].tenant_id, records=gated.pending_equivs)

    def _prepare_tkg_vectors(self, req: GraphUpsertRequest) -> List[MemoryEntry]:
        """Prepare TKG vector entries and set vector_ids on Event/Entity objects.
        
        This is Phase 1 of the two-phase vector write:
        - Builds MemoryEntry objects for each node that needs vectorization
        - Sets text_vector_id/face_vector_id/voice_vector_id on the objects
        - Returns the entries list for later Qdrant write
        
        The vector_ids are set before Neo4j write so they can be persisted to the graph.
        
        Vector mapping:
        - Event.summary -> memory_text collection (text embedding)
        - Entity(PERSON) with face embedding -> memory_face collection
        - Entity(PERSON) with voice embedding -> memory_audio collection
        """
        import hashlib
        import uuid as _uuid
        
        def _make_vector_id(prefix: str, node_id: str) -> tuple[str, str]:
            """Generate a UUID vector ID for TKG nodes.
            
            Uses the same deterministic MD5-to-UUID conversion as qdrant_store.py
            to ensure the IDs stored in Neo4j match the actual Qdrant point IDs.
            
            Returns:
                (uuid_id, readable_id) - UUID for Qdrant/Neo4j, readable for debugging
            """
            readable_id = f"{prefix}_{node_id}"
            # Same logic as qdrant_store.py line 345
            uuid_id = str(_uuid.UUID(hashlib.md5(readable_id.encode("utf-8")).hexdigest()))
            return uuid_id, readable_id
        
        entries: List[MemoryEntry] = []
        tenant_id = None
        
        # Extract tenant_id from first available node
        for ev in req.events:
            if ev.tenant_id:
                tenant_id = ev.tenant_id
                break
        if tenant_id is None:
            for ent in req.entities:
                if ent.tenant_id:
                    tenant_id = ent.tenant_id
                    break
        
        # Build MemoryEntry for each Event (text vector from summary + desc)
        for ev in req.events:
            if not ev.summary or not ev.summary.strip():
                continue
            text_parts: List[str] = [str(ev.summary).strip()]
            if ev.desc and str(ev.desc).strip():
                text_parts.append(str(ev.desc).strip())
            content = " ".join(text_parts).strip()
            vector_id, readable_id = _make_vector_id("tkg_event", ev.id)
            src = ev.source or "tkg"
            if src == "dialog_session_write_v1":
                src = TKG_DIALOG_EVENT_INDEX_SOURCE_V1
            entry = MemoryEntry(
                id=vector_id,
                kind="semantic",
                modality="text",
                contents=[content],
                metadata={
                    "node_type": "Event",
                    "node_id": ev.id,
                    "vector_id_readable": readable_id,  # For debugging
                    "tenant_id": ev.tenant_id or tenant_id,
                    "source": src,
                    "event_type": ev.event_type,
                    "actor_id": ev.actor_id,
                    "t_start": ev.t_abs_start.isoformat() if ev.t_abs_start else None,
                    "t_end": ev.t_abs_end.isoformat() if ev.t_abs_end else None,
                    "topic_id": ev.topic_id,
                    "topic_path": ev.topic_path,
                    "tags": list(ev.tags) if ev.tags else None,
                    "keywords": list(ev.keywords) if ev.keywords else None,
                    "time_bucket": list(ev.time_bucket) if ev.time_bucket else None,
                    "tags_vocab_version": ev.tags_vocab_version,
                    # Scoping metadata for search filtering
                    "user_id": ev.user_id or [],
                    "memory_domain": ev.memory_domain,
                    "tkg_event_id": ev.id,
                },
                published=ev.published if ev.published is not None else True,
            )
            if ev.logical_event_id:
                entry.metadata["event_id"] = ev.logical_event_id
            if ev.source_turn_ids:
                entry.metadata["source_turn_ids"] = list(ev.source_turn_ids)
            if ev.t_abs_start:
                entry.metadata["timestamp_iso"] = ev.t_abs_start.isoformat()
            elif ev.t_abs_end:
                entry.metadata["timestamp_iso"] = ev.t_abs_end.isoformat()
            # Set text_vector_id on the Event object for Neo4j storage
            ev.text_vector_id = vector_id
            entries.append(entry)
        
        # Build MemoryEntry for each Entity with name (text vector)
        for ent in req.entities:
            try:
                ent_tenant = ent.tenant_id or tenant_id
                
                # Text vector from name
                if ent.name and ent.name.strip():
                    vector_id, readable_id = _make_vector_id("tkg_entity_text", ent.id)
                    entry = MemoryEntry(
                        id=vector_id,
                        kind="semantic",
                        modality="text",
                        contents=[ent.name],
                        metadata={
                            "node_type": "Entity",
                            "node_id": ent.id,
                            "vector_id_readable": readable_id,  # For debugging
                            "tenant_id": ent_tenant,
                            "entity_type": ent.type,
                            # Scoping metadata for search filtering
                            "user_id": ent.user_id or [],
                            "memory_domain": ent.memory_domain,
                        },
                        published=ent.published if ent.published is not None else True,
                    )
                    ent.text_vector_id = vector_id
                    entries.append(entry)
                
                # Face vector from embedding (for PERSON entities)
                _has_face = getattr(ent, 'face_embedding', None) is not None and len(ent.face_embedding) > 0
                if _has_face:
                    vector_id, readable_id = _make_vector_id("tkg_entity_face", ent.id)
                    entry = MemoryEntry(
                        id=vector_id,
                        kind="semantic",
                        modality="image",
                        contents=[ent.cluster_label or ent.name or ent.id],
                        vectors={"face": ent.face_embedding},
                        metadata={
                            "node_type": "Entity",
                            "node_id": ent.id,
                            "vector_id_readable": readable_id,  # For debugging
                            "tenant_id": ent_tenant,
                            "entity_type": ent.type,
                            "cluster_label": ent.cluster_label,
                            # Scoping metadata for search filtering
                            "user_id": ent.user_id or [],
                            "memory_domain": ent.memory_domain,
                        },
                        published=ent.published if ent.published is not None else True,
                    )
                    ent.face_vector_id = vector_id
                    entries.append(entry)
                
                # Voice vector from embedding (for PERSON entities)
                _has_voice = getattr(ent, 'voice_embedding', None) is not None and len(ent.voice_embedding) > 0
                if _has_voice:
                    vector_id, readable_id = _make_vector_id("tkg_entity_voice", ent.id)
                    entry = MemoryEntry(
                        id=vector_id,
                        kind="semantic",
                        modality="audio",
                        contents=[ent.cluster_label or ent.name or ent.id],
                        vectors={"audio": ent.voice_embedding},
                        metadata={
                            "node_type": "Entity",
                            "node_id": ent.id,
                            "vector_id_readable": readable_id,  # For debugging
                            "tenant_id": ent_tenant,
                            "entity_type": ent.type,
                            "cluster_label": ent.cluster_label,
                            # Scoping metadata for search filtering
                            "user_id": ent.user_id or [],
                            "memory_domain": ent.memory_domain,
                        },
                        published=ent.published if ent.published is not None else True,
                    )
                    ent.voice_vector_id = vector_id
                    entries.append(entry)
            except Exception:
                raise
        
        return entries

    async def _write_tkg_vectors(self, entries: List[MemoryEntry]) -> None:
        """Write prepared TKG vector entries to Qdrant.
        
        This is Phase 2 of the two-phase vector write, called AFTER Neo4j succeeds.
        This prevents orphaned vectors if the graph write fails.
        """
        if not entries or self.vector_store is None:
            return
        
        try:
            await self.vector_store.upsert_vectors(entries)
            inc("tkg_vector_upserts_total", len(entries))
        except Exception as e:
            # Log but don't fail - vector writes are best-effort after graph succeeds
            # Vectors can be rebuilt from graph data if needed
            import logging
            logging.warning(f"TKG vector upsert failed (graph already persisted): {e}")

    async def touch(self, *, tenant_id: str, node_ids: List[str], extend_seconds: Optional[float] = None) -> Dict[str, int]:
        return await self.store.touch_nodes(tenant_id=tenant_id, node_ids=node_ids, extend_seconds=extend_seconds)

    # --- read paths ---
    async def list_segments(
        self,
        *,
        tenant_id: str,
        source_id: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        modality: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        res = await self.store.query_segments_by_time(
            tenant_id=tenant_id,
            source_id=source_id,
            start=start,
            end=end,
            modality=modality,
            limit=limit,
        )
        return self._apply_decay(res, key="id")

    async def entity_timeline(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 200,
    ) -> list[dict]:
        return await self.store.query_entity_timeline(
            tenant_id=tenant_id,
            entity_id=entity_id,
            limit=limit,
        )

    async def list_entity_evidences(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        subtype: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        return await self.store.query_entity_evidences(
            tenant_id=tenant_id,
            entity_id=entity_id,
            subtype=subtype,
            source_id=source_id,
            limit=limit,
        )

    async def list_events(
        self,
        *,
        tenant_id: str,
        segment_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        place_id: Optional[str] = None,
        source_id: Optional[str] = None,
        relation: Optional[str] = None,
        layer: Optional[str] = None,
        status: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        res = await self.store.query_events(
            tenant_id=tenant_id,
            segment_id=segment_id,
            entity_id=entity_id,
            place_id=place_id,
            source_id=source_id,
            relation=relation,
            layer=layer,
            status=status,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
        )
        return self._apply_decay(res, key="id", time_fields=("t_abs_start", "t_abs_end"))

    async def list_events_by_ids(
        self,
        *,
        tenant_id: str,
        event_ids: List[str],
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        return await self.store.query_events_by_ids(
            tenant_id=tenant_id,
            event_ids=event_ids,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
        )

    async def list_entities_by_ids(
        self,
        *,
        tenant_id: str,
        entity_ids: List[str],
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        query = getattr(self.store, "query_entities_by_ids", None)
        if not callable(query):
            return []
        return await query(
            tenant_id=tenant_id,
            entity_ids=entity_ids,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
        )

    async def expand_neighbors(
        self,
        *,
        seed_ids: List[str],
        rel_whitelist: Optional[List[str]] = None,
        max_hops: int = 1,
        neighbor_cap_per_seed: int = 5,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        memory_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.store.expand_neighbors(
            seed_ids=list(seed_ids or []),
            rel_whitelist=rel_whitelist,
            max_hops=max_hops,
            neighbor_cap_per_seed=neighbor_cap_per_seed,
            user_ids=user_ids,
            memory_domain=memory_domain,
            memory_scope=memory_scope,
            restrict_to_user=True,
            restrict_to_domain=True,
            restrict_to_scope=True,
        )

    async def entity_detail(
        self,
        *,
        tenant_id: str,
        entity_id: str,
    ) -> dict:
        return await self.store.query_entity_detail(tenant_id=tenant_id, entity_id=entity_id)

    async def entity_facts(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 100,
    ) -> list[dict]:
        return await self.store.query_entity_knowledge(tenant_id=tenant_id, entity_id=entity_id, limit=limit)

    async def entity_relations(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 50,
    ) -> list[dict]:
        return await self.store.query_entity_relations(tenant_id=tenant_id, entity_id=entity_id, limit=limit)

    async def entity_relations_by_events(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        return await self.store.query_entity_relations_by_events(
            tenant_id=tenant_id,
            entity_id=entity_id,
            start=start,
            end=end,
            limit=limit,
        )

    async def topic_timeline(
        self,
        *,
        tenant_id: str,
        topic_id: Optional[str] = None,
        topic_path: Optional[str] = None,
        tags: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 200,
        event_ids: Optional[List[str]] = None,
    ) -> list[dict]:
        if event_ids:
            return await self.store.query_events_by_ids(
                tenant_id=tenant_id,
                event_ids=list(event_ids),
                start=start,
                end=end,
                user_ids=user_ids,
                memory_domain=memory_domain,
                limit=limit,
            )
        return await self.store.query_events_by_topic(
            tenant_id=tenant_id,
            topic_id=topic_id,
            topic_path=topic_path,
            tags=tags,
            keywords=keywords,
            start=start,
            end=end,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
        )

    async def search_events_v1(
        self,
        *,
        tenant_id: str,
        query: str,
        topk: int = 10,
        source_id: Optional[str] = None,
        include_evidence: bool = True,
    ) -> Dict[str, Any]:
        """Graph-first search: query -> candidate Events -> (optional) evidence chain per Event."""

        q = (query or "").strip()
        if not q:
            return {"query": "", "items": []}
        bounded_topk = max(1, min(int(topk), 50))

        hits = await self.store.search_event_candidates(
            tenant_id=tenant_id,
            query=q,
            limit=bounded_topk,
            source_id=source_id,
        )
        items: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            event_id = str(hit.get("event_id") or "")
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            if include_evidence:
                bundle = await self.explain_event_evidence(tenant_id=tenant_id, event_id=event_id)
                items.append({**dict(hit), **bundle})
            else:
                items.append(dict(hit))
            if len(items) >= bounded_topk:
                break

        return {"query": q, "items": items}

    async def list_places(
        self,
        *,
        tenant_id: str,
        name: Optional[str] = None,
        segment_id: Optional[str] = None,
        covers_timeslice: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        res = await self.store.query_places(
            tenant_id=tenant_id,
            name=name,
            segment_id=segment_id,
            covers_timeslice=covers_timeslice,
            limit=limit,
        )
        return self._apply_decay(res, key="id")

    async def resolve_entities(
        self,
        *,
        tenant_id: str,
        name: str,
        entity_type: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        return await self.store.query_entities_by_name(
            tenant_id=tenant_id,
            name=name,
            entity_type=entity_type,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
        )

    async def list_entities_overview(
        self,
        *,
        tenant_id: str,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        mentioned_since: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        return await self.store.query_entities_overview(
            tenant_id=tenant_id,
            entity_type=entity_type,
            query=query,
            mentioned_since=mentioned_since,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
            offset=offset,
        )

    async def count_entities_overview(
        self,
        *,
        tenant_id: str,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        mentioned_since: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> int:
        return await self.store.query_entities_overview_count(
            tenant_id=tenant_id,
            entity_type=entity_type,
            query=query,
            mentioned_since=mentioned_since,
            user_ids=user_ids,
            memory_domain=memory_domain,
        )

    async def list_topics_overview(
        self,
        *,
        tenant_id: str,
        query: Optional[str] = None,
        parent_path: Optional[str] = None,
        min_events: Optional[int] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        return await self.store.query_topics_overview(
            tenant_id=tenant_id,
            query=query,
            parent_path=parent_path,
            min_events=min_events,
            user_ids=user_ids,
            memory_domain=memory_domain,
            limit=limit,
            offset=offset,
        )

    async def count_topics_overview(
        self,
        *,
        tenant_id: str,
        query: Optional[str] = None,
        parent_path: Optional[str] = None,
        min_events: Optional[int] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> int:
        return await self.store.query_topics_overview_count(
            tenant_id=tenant_id,
            query=query,
            parent_path=parent_path,
            min_events=min_events,
            user_ids=user_ids,
            memory_domain=memory_domain,
        )

    async def event_detail(
        self,
        *,
        tenant_id: str,
        event_id: str,
    ) -> dict:
        res = await self.store.query_event_detail(
            tenant_id=tenant_id,
            event_id=event_id,
        )
        # detail includes nested timeslices/relations; decay not applied here to preserve detail
        return res

    async def event_id_by_logical_id(
        self,
        *,
        tenant_id: str,
        logical_event_id: str,
    ) -> Optional[str]:
        query = getattr(self.store, "query_event_id_by_logical_id", None)
        if not callable(query):
            return None
        return await query(tenant_id=tenant_id, logical_event_id=logical_event_id)

    async def place_detail(
        self,
        *,
        tenant_id: str,
        place_id: str,
    ) -> dict:
        return await self.store.query_place_detail(
            tenant_id=tenant_id,
            place_id=place_id,
        )

    async def list_time_slices(
        self,
        *,
        tenant_id: str,
        kind: Optional[str] = None,
        covers_segment: Optional[str] = None,
        covers_event: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        return await self.store.query_time_slices(
            tenant_id=tenant_id,
            kind=kind,
            covers_segment=covers_segment,
            covers_event=covers_event,
            limit=limit,
        )

    async def list_time_slices_by_range(
        self,
        *,
        tenant_id: str,
        start_iso: Optional[str],
        end_iso: Optional[str],
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict]:
        return await self.store.query_time_slices_by_range(
            tenant_id=tenant_id,
            start_iso=start_iso,
            end_iso=end_iso,
            kind=kind,
            limit=limit,
        )

    async def build_event_relations(
        self,
        *,
        tenant_id: str,
        source_id: Optional[str] = None,
        place_id: Optional[str] = None,
        limit: int = 1000,
        create_causes: bool = True,
    ) -> dict:
        return await self.store.build_event_relations(
            tenant_id=tenant_id,
            source_id=source_id,
            place_id=place_id,
            limit=limit,
            create_causes=create_causes,
        )

    async def build_time_slices_from_segments(
        self,
        *,
        tenant_id: str,
        window_seconds: float = 3600.0,
        source_id: Optional[str] = None,
        modality: Optional[str] = None,
        modes: Optional[list[str]] = None,
    ) -> dict:
        return await self.store.build_time_slices_from_segments(
            tenant_id=tenant_id,
            window_seconds=window_seconds,
            source_id=source_id,
            modality=modality,
            modes=modes,
        )

    async def build_cooccurs_from_timeslices(
        self,
        *,
        tenant_id: str,
        min_weight: float = 1.0,
    ) -> dict:
        return await self.store.build_cooccurs_from_timeslices(tenant_id=tenant_id, min_weight=min_weight)

    async def build_cooccurs_from_events(
        self,
        *,
        tenant_id: str,
        min_weight: float = 1.0,
    ) -> dict:
        return await self.store.build_cooccurs_from_events(tenant_id=tenant_id, min_weight=min_weight)

    async def build_first_meetings(
        self,
        *,
        tenant_id: str,
        limit: int = 5000,
    ) -> dict:
        """Generate FIRST_MEET edges from earliest co-involved events per entity pair."""
        return await self.store.build_first_meetings(tenant_id=tenant_id, limit=limit)

    async def purge_source(
        self,
        *,
        tenant_id: str,
        source_id: str,
        delete_orphans: bool = False,
    ) -> dict:
        src = str(source_id or "").strip()
        if not src:
            raise GraphValidationError("source_id_required")
        with self._explain_cache_lock:
            self._explain_cache.clear()
        return await self.store.purge_source(
            tenant_id=tenant_id,
            source_id=src,
            delete_orphan_entities=delete_orphans,
        )

    async def purge_source_except_events(
        self,
        *,
        tenant_id: str,
        source_id: str,
        keep_event_ids: List[str],
    ) -> dict:
        """Delete old events (and derived knowledge) for a source_id, keeping the provided event ids."""
        src = str(source_id or "").strip()
        if not src:
            raise GraphValidationError("source_id_required")
        purge_fn = getattr(self.store, "purge_source_except_events", None)
        if not callable(purge_fn):
            raise GraphValidationError("purge_source_except_events_unsupported")
        return await purge_fn(
            tenant_id=tenant_id,
            source_id=src,
            keep_event_ids=list(keep_event_ids or []),
        )

    # --- explain cache helpers ---
    def _explain_cache_key_first_meeting(self, tenant_id: str, me_id: str, other_id: str) -> str:
        return f"first_meeting|tenant={tenant_id}|me={me_id}|other={other_id}"

    def _explain_cache_key_event(
        self,
        tenant_id: str,
        event_id: str,
        *,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> str:
        uid_part = ",".join(sorted({str(x).strip() for x in (user_ids or []) if str(x).strip()})) or "*"
        domain_part = str(memory_domain or "").strip() or "*"
        return f"event_evidence|tenant={tenant_id}|event={event_id}|uids={uid_part}|domain={domain_part}"

    def _explain_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        import time as _t
        with self._explain_cache_lock:
            item = self._explain_cache.get(key)
            if not item:
                return None
            exp, res = item
            if _t.time() > exp:
                # expired entry
                self._explain_cache.pop(key, None)
                return None
            try:
                # LRU: move to end
                self._explain_cache.move_to_end(key)
            except Exception:
                pass
            return dict(res)

    def _explain_cache_put(self, key: str, value: Dict[str, Any]) -> None:
        import time as _t
        with self._explain_cache_lock:
            # evict if overflow (LRU: popitem(last=False))
            while len(self._explain_cache) >= max(1, self._explain_cache_max):
                try:
                    self._explain_cache.popitem(last=False)
                    inc("explain_cache_evictions_total", 1)
                except Exception:
                    break
            ttl = max(1.0, float(self._explain_cache_ttl_s))
            self._explain_cache[key] = (_t.time() + ttl, dict(value))

    async def explain_first_meeting(
        self,
        *,
        tenant_id: str,
        me_id: str,
        other_id: str,
    ) -> dict:
        """Explain the first meeting event between two entities within a tenant.

        Returns a stable structure even when no common event exists to keep the API contract predictable.
        """
        cache_key: Optional[str] = None
        if self._explain_cache_enabled:
            cache_key = self._explain_cache_key_first_meeting(tenant_id, me_id, other_id)
            cached = self._explain_cache_get(cache_key)
            if cached is not None:
                inc("explain_cache_hits_total", 1)
                return cached
            inc("explain_cache_misses_total", 1)

        res = await self.store.query_first_meeting(
            tenant_id=tenant_id,
            me_id=me_id,
            other_id=other_id,
        )
        if not res:
            result = {
                "found": False,
                "event_id": None,
                "t_abs_start": None,
                "place_id": None,
                "summary": None,
                "evidence_ids": [],
            }
        else:
            result = {
                "found": True,
                "event_id": res.get("event_id"),
                "t_abs_start": res.get("t_abs_start"),
                "place_id": res.get("place_id"),
                "summary": res.get("summary"),
                "evidence_ids": list(res.get("evidence_ids") or []),
            }
        if self._explain_cache_enabled and cache_key is not None:
            self._explain_cache_put(cache_key, result)
        return result

    async def explain_event_evidence(
        self,
        *,
        tenant_id: str,
        event_id: str,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> dict:
        """Return a structured evidence chain for a given event."""
        cache_key: Optional[str] = None
        if self._explain_cache_enabled:
            cache_key = self._explain_cache_key_event(
                tenant_id,
                event_id,
                user_ids=user_ids,
                memory_domain=memory_domain,
            )
            cached = self._explain_cache_get(cache_key)
            if cached is not None:
                inc("explain_cache_hits_total", 1)
                return cached
            inc("explain_cache_misses_total", 1)

        query_kwargs: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "event_id": event_id,
        }
        if user_ids:
            query_kwargs["user_ids"] = list(user_ids)
        if memory_domain:
            query_kwargs["memory_domain"] = memory_domain
        res = await self.store.query_event_evidence(**query_kwargs)
        if not res:
            result = {
                "event": None,
                "entities": [],
                "places": [],
                "timeslices": [],
                "evidences": [],
                "utterances": [],
                "utterance_speakers": [],
                "knowledge": [],
            }
        else:
            result = {
                "event": res.get("event"),
                "entities": list(res.get("entities") or []),
                "places": list(res.get("places") or []),
                "timeslices": list(res.get("timeslices") or []),
                "evidences": list(res.get("evidences") or []),
                "utterances": list(res.get("utterances") or []),
                "utterance_speakers": list(res.get("utterance_speakers") or []),
                "knowledge": list(res.get("knowledge") or []),
            }
        if self._explain_cache_enabled and cache_key is not None:
            self._explain_cache_put(cache_key, result)
        return result

    async def apply_state_update(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
        value: str,
        valid_from: Optional[datetime] = None,
        raw_value: Optional[str] = None,
        confidence: Optional[float] = None,
        source_event_id: Optional[str] = None,
        user_id: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        status: Optional[str] = None,
        pending_reason: Optional[str] = None,
        extractor_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.store.apply_state_update(
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            value=value,
            valid_from=valid_from,
            raw_value=raw_value,
            confidence=confidence,
            source_event_id=source_event_id,
            user_id=user_id,
            memory_domain=memory_domain,
            status=status,
            pending_reason=pending_reason,
            extractor_version=extractor_version,
        )

    async def get_current_state(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.store.get_current_state(
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
        )

    async def get_state_at_time(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
        t: datetime,
    ) -> Optional[Dict[str, Any]]:
        return await self.store.get_state_at_time(
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            t=t,
        )

    async def get_state_changes(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 200,
        order: str = "asc",
    ) -> List[Dict[str, Any]]:
        return await self.store.get_state_changes(
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            start=start,
            end=end,
            limit=limit,
            order=order,
        )

    async def list_pending_states(
        self,
        *,
        tenant_id: str,
        subject_id: Optional[str] = None,
        property: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return await self.store.list_pending_states(
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            status=status,
            limit=limit,
        )

    async def approve_pending_state(
        self,
        *,
        tenant_id: str,
        pending_id: str,
        apply: bool = True,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pending = await self.store.get_pending_state(tenant_id=tenant_id, pending_id=pending_id)
        if not pending:
            return None
        updated = await self.store.update_pending_state_status(
            tenant_id=tenant_id,
            pending_id=pending_id,
            status="approved",
            note=note,
        )
        applied = False
        apply_result: Optional[Dict[str, Any]] = None
        if apply and str(pending.get("pending_reason") or "") != "out_of_order":
            vf = pending.get("valid_from")
            if isinstance(vf, str):
                try:
                    if vf.endswith("Z"):
                        vf = vf[:-1] + "+00:00"
                    vf = datetime.fromisoformat(vf)
                except Exception:
                    vf = None
            apply_result = await self.store.apply_state_update(
                tenant_id=tenant_id,
                subject_id=str(pending.get("subject_id") or ""),
                property=str(pending.get("property") or ""),
                value=str(pending.get("value") or ""),
                raw_value=pending.get("raw_value"),
                confidence=pending.get("confidence"),
                valid_from=vf,
                source_event_id=pending.get("source_event_id"),
                user_id=pending.get("user_id"),
                memory_domain=pending.get("memory_domain"),
                status=None,
                extractor_version=pending.get("extractor_version"),
            )
            applied = bool(apply_result and apply_result.get("applied"))
        return {"pending": updated or pending, "applied": applied, "apply_result": apply_result}

    async def reject_pending_state(
        self,
        *,
        tenant_id: str,
        pending_id: str,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pending = await self.store.update_pending_state_status(
            tenant_id=tenant_id,
            pending_id=pending_id,
            status="rejected",
            note=note,
        )
        if not pending:
            return None
        return {"pending": pending}

    # --- validation helpers ---
    def _ensure_same_tenant(self, req: GraphUpsertRequest) -> None:
        tenants: list[str] = []
        for seq in (
            req.segments,
            req.evidences,
            getattr(req, "utterances", []),
            req.entities,
            req.events,
            req.places,
            req.time_slices,
            getattr(req, "regions", []),
            getattr(req, "states", []),
            getattr(req, "knowledge", []),
            getattr(req, "pending_equivs", []),
            req.edges,
        ):
            for item in seq:
                tenants.append(getattr(item, "tenant_id", None))
        tenants = [t for t in tenants if t is not None]
        if not tenants:
            raise GraphValidationError("tenant_id missing on graph payload")
        base = tenants[0]
        if any(t != base for t in tenants):
            raise GraphValidationError("tenant_id must be identical across all nodes/edges in one request")

    def _ensure_same_scope(self, req: GraphUpsertRequest) -> None:
        """Enforce user/domain isolation within a single upsert request when provided.

        If user_id / memory_domain are absent across the request (e.g., legacy tests),
        skip the check to remain backward compatible. When present, all non-empty values
        must be identical to avoid cross-user/domain merges within one upsert.
        """

        def _norm_user_ids(vals: Optional[Iterable[str]]) -> Optional[tuple[str, ...]]:
            if vals is None:
                return None
            norm = tuple(sorted({str(v).strip() for v in vals if str(v).strip()}))
            return norm if norm else None

        user_scopes: list[tuple[str, ...]] = []
        domains: list[str] = []
        for seq in (
            req.segments,
            req.evidences,
            getattr(req, "utterances", []),
            req.entities,
            req.events,
            req.places,
            req.time_slices,
            getattr(req, "regions", []),
            getattr(req, "states", []),
            getattr(req, "knowledge", []),
            getattr(req, "pending_equivs", []),
            req.edges,
        ):
            for item in seq:
                uid = _norm_user_ids(getattr(item, "user_id", None))
                if uid is not None:
                    user_scopes.append(uid)
                domain = str(getattr(item, "memory_domain", "") or "").strip()
                if domain:
                    domains.append(domain)

        if not user_scopes and not domains:
            return
        if user_scopes:
            base_user = user_scopes[0]
            if any(u != base_user for u in user_scopes):
                raise GraphValidationError("user_id must be identical across all nodes/edges in one request")
        if domains:
            base_domain = domains[0]
            if any(d != base_domain for d in domains):
                raise GraphValidationError("memory_domain must be identical across all nodes/edges in one request")

    def _validate_segments(self, segments: Iterable[MediaSegment]) -> None:
        for seg in segments:
            if seg.t_media_end <= seg.t_media_start:
                raise GraphValidationError("segment end must be greater than start")

    def _validate_time_slices(self, slices: Iterable[TimeSlice]) -> None:
        for sl in slices:
            if sl.t_media_start is not None and sl.t_media_end is not None:
                if float(sl.t_media_end) <= float(sl.t_media_start):
                    raise GraphValidationError("timeslice media end must be greater than start")
            if sl.t_abs_start is not None and sl.t_abs_end is not None:
                if sl.t_abs_end <= sl.t_abs_start:
                    raise GraphValidationError("timeslice absolute end must be greater than start")


__all__ = ["GraphService", "GraphValidationError"]
