from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable
import asyncio
from datetime import datetime, timezone, timedelta
from modules.memory.contracts.memory_models import MemoryEntry, Edge
from modules.memory.contracts.graph_models import (
    MediaSegment,
    Evidence as GraphEvidence,
    UtteranceEvidence,
    Entity as GraphEntity,
    Event as GraphEvent,
    Place as GraphPlace,
    TimeSlice as GraphTimeSlice,
    SpatioTemporalRegion,
    State as GraphState,
    Knowledge as GraphKnowledge,
    GraphEdge,
)
import logging


# DEPRECATED: MemoryNode is being phased out in favor of TKG (Typed Knowledge Graph).
# New code should use GraphUpsertRequest with Event/Entity/Evidence nodes instead.
# MemoryNode graph will be kept for backward compatibility but no longer actively developed.
# See: modules/memory/contracts/graph_models.py for TKG models.
MEMORY_NODE_LABEL = "MemoryNode"


class Neo4jStore:
    """Neo4j 图存储门面（接口雏形，待实现）。

    预期 settings：
    {
      "uri": "bolt://127.0.0.1:7687", "user": "neo4j", "password": "...",
      "vector_index": {"enabled": True, "metric": "cosine", "dims": {"text": 1536}}
    }

    TODO：实现连接、节点/关系 MERGE、向量属性/索引配置与 health。
    当前为占位实现。
    """

    @staticmethod
    def _as_bool(val: Any, default: bool) -> bool:
        if isinstance(val, bool):
            return val
        if val is None:
            return bool(default)
        s = str(val).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        self._driver = None
        self._database = str(self.settings.get("database", "neo4j"))
        self._strict_tenant_mode = self._as_bool(self.settings.get("strict_tenant_mode"), False)
        self._enable_legacy_memory_node = self._as_bool(self.settings.get("enable_legacy_memory_node"), True)
        self._closed: bool = False
        # reliability/circuit-breaker
        rel = (self.settings.get("reliability") or {}) if isinstance(self.settings.get("reliability"), dict) else {}
        self._cb_failure_threshold = int(rel.get("circuit_breaker", {}).get("failure_threshold", 5))
        self._cb_cooldown_s = int(rel.get("circuit_breaker", {}).get("cooldown_seconds", 30))
        self._cb_fail_count = 0
        self._cb_open_until = 0.0
        self._logger = logging.getLogger(__name__)
        self._tz_utc = timezone.utc
        try:
            import os
            from neo4j import GraphDatabase  # type: ignore

            uri = os.path.expandvars(str(self.settings.get("uri") or ""))
            user = os.path.expandvars(str(self.settings.get("user") or ""))
            password = os.path.expandvars(str(self.settings.get("password") or ""))

            if uri and user and password:
                def _make(uri_str: str):
                    try:
                        encrypted = bool(self.settings.get("encrypted", False))
                        timeout_raw = self.settings.get("connection_timeout_s") or os.getenv("NEO4J_CONNECTION_TIMEOUT_S") or 5.0
                        try:
                            timeout_s = float(timeout_raw)
                        except Exception:
                            timeout_s = 5.0
                        kwargs: Dict[str, Any] = {"auth": (user, password), "encrypted": encrypted}
                        if timeout_s and timeout_s > 0:
                            kwargs["connection_timeout"] = timeout_s
                        try:
                            return GraphDatabase.driver(uri_str, **kwargs)
                        except TypeError:
                            # Older driver: connection_timeout not supported
                            kwargs.pop("connection_timeout", None)
                            return GraphDatabase.driver(uri_str, **kwargs)
                    except Exception:
                        return None
                drv = _make(uri)
                # fallback: switch scheme between bolt:// and neo4j://
                if drv is None:
                    if uri.startswith("bolt://"):
                        drv = _make("neo4j://" + uri[len("bolt://"):])
                    elif uri.startswith("neo4j://"):
                        drv = _make("bolt://" + uri[len("neo4j://"):])
                self._driver = drv
        except Exception:
            # keep unconfigured state
            self._driver = None

    def _validate_tenant_context(self, tenant_id: Optional[str], operation: str) -> str:
        tid = str(tenant_id or "").strip()
        if not tid:
            raise ValueError(
                f"tenant_id is required for {operation}. Cross-tenant operations are forbidden."
            )
        return tid

    def _require_legacy_memory_node_enabled(self, operation: str) -> None:
        if self._enable_legacy_memory_node:
            return
        raise RuntimeError(
            f"legacy_memory_node_disabled: operation={operation} is blocked by graph_store.enable_legacy_memory_node=false"
        )

    @staticmethod
    def _collect_tenant_values(items: Iterable[Dict[str, Any]]) -> List[str]:
        vals: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("tenant_id") or "").strip()
            if tid:
                vals.add(tid)
        return sorted(vals)

    def _assert_payload_tenant(self, *, payload: Iterable[Dict[str, Any]], tenant_id: str, operation: str) -> None:
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_tid = str(item.get("tenant_id") or "").strip()
            if item_tid != tenant_id:
                raise ValueError(
                    f"{operation} tenant_id mismatch: expected={tenant_id} got={item_tid or '<empty>'}"
                )

    @staticmethod
    def _extract_applied_count(result: Any) -> Optional[int]:
        """Best-effort extraction for `RETURN count(*) AS applied` rows.

        Some tests use lightweight stubs that return plain lists without row data.
        In that case, return None so callers can skip false-positive validation failures.
        """
        row: Any = None
        try:
            if hasattr(result, "single"):
                row = result.single()
            elif isinstance(result, list):
                row = result[0] if result else None
            elif isinstance(result, dict):
                row = result
        except Exception:
            row = None

        if row is None:
            return None
        try:
            if isinstance(row, dict):
                if "applied" not in row:
                    return None
                return int(row.get("applied", 0) or 0)
            if hasattr(row, "get"):
                v = row.get("applied")  # type: ignore[attr-defined]
                if v is None:
                    return None
                return int(v or 0)
        except Exception:
            return None
        return None

    @staticmethod
    def _infer_entry_tenant(entries: List[MemoryEntry]) -> Optional[str]:
        vals: set[str] = set()
        for e in entries or []:
            try:
                md = dict(e.metadata or {})
            except Exception:
                md = {}
            tid = str(md.get("tenant_id") or "").strip()
            if tid:
                vals.add(tid)
        if len(vals) == 1:
            return next(iter(vals))
        return None

    def ensure_schema_v0(self) -> None:
        """Create minimal constraints/indexes for Graph API v0 (TKG schema baseline).

        Best-effort: failures are swallowed to avoid blocking startup.
        """

        if not self._driver:
            return
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                statements = [
                    # P0: Separate MemoryEntry projection graph from typed TKG graph labels.
                    # Legacy memory nodes were stored as :Entity without tenant_id, which conflicts with
                    # Graph v0.x node-key constraints on :Entity(tenant_id,id). Migrate them first.
                    (
                        "MATCH (n:Entity) "
                        "WHERE n.tenant_id IS NULL AND ("
                        "n.kind IS NOT NULL OR n.modality IS NOT NULL OR n.memory_domain IS NOT NULL OR "
                        "n.run_id IS NOT NULL OR n.user_id IS NOT NULL OR n.memory_scope IS NOT NULL OR "
                        "n.timestamp IS NOT NULL OR n.clip_id IS NOT NULL OR n.text IS NOT NULL"
                        ") "
                        f"SET n:{MEMORY_NODE_LABEL} "
                        "REMOVE n:Entity"
                    ),
                    "CREATE CONSTRAINT media_segment_id IF NOT EXISTS FOR (n:MediaSegment) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (n:Evidence) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT place_id IF NOT EXISTS FOR (n:Place) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT timeslice_id IF NOT EXISTS FOR (n:TimeSlice) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (n:UtteranceEvidence) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:SpatioTemporalRegion) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT state_id IF NOT EXISTS FOR (n:State) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT pending_state_id IF NOT EXISTS FOR (n:PendingState) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT knowledge_id IF NOT EXISTS FOR (n:Knowledge) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE CONSTRAINT pending_equiv_id IF NOT EXISTS FOR (n:PendingEquiv) REQUIRE (n.tenant_id, n.id) IS UNIQUE",
                    "CREATE INDEX media_segment_tenant_time IF NOT EXISTS FOR (n:MediaSegment) ON (n.tenant_id, n.source_id, n.t_media_start, n.t_media_end)",
                    "CREATE INDEX state_subject_property IF NOT EXISTS FOR (n:State) ON (n.tenant_id, n.subject_id, n.property)",
                    "CREATE INDEX pending_state_subject_property IF NOT EXISTS FOR (n:PendingState) ON (n.tenant_id, n.subject_id, n.property)",
                    "CREATE CONSTRAINT state_key_unique IF NOT EXISTS FOR (n:StateKey) REQUIRE (n.tenant_id, n.subject_id, n.property) IS UNIQUE",
                    "CREATE INDEX state_key_subject_property IF NOT EXISTS FOR (n:StateKey) ON (n.tenant_id, n.subject_id, n.property)",
                    "CREATE INDEX event_topic_id IF NOT EXISTS FOR (n:Event) ON (n.tenant_id, n.topic_id)",
                    "CREATE INDEX event_topic_path IF NOT EXISTS FOR (n:Event) ON (n.tenant_id, n.topic_path)",
                    f"CREATE CONSTRAINT memory_node_id IF NOT EXISTS FOR (n:{MEMORY_NODE_LABEL}) REQUIRE n.id IS UNIQUE",
                    # P0: Graph-first retrieval wants fast text lookup. Best-effort fulltext indexes (Neo4j 5 syntax).
                    "CREATE FULLTEXT INDEX tkg_event_summary_v1 IF NOT EXISTS FOR (n:Event) ON EACH [n.summary]",
                    "CREATE FULLTEXT INDEX tkg_event_summary_desc_v1 IF NOT EXISTS FOR (n:Event) ON EACH [n.summary, n.desc]",
                    "CREATE FULLTEXT INDEX tkg_utterance_text_v1 IF NOT EXISTS FOR (n:UtteranceEvidence) ON EACH [n.raw_text]",
                    "CREATE FULLTEXT INDEX tkg_evidence_text_v1 IF NOT EXISTS FOR (n:Evidence) ON EACH [n.text]",
                    "CREATE FULLTEXT INDEX tkg_entity_name_v1 IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.manual_name, n.cluster_label]",
                ]
                for stmt in statements:
                    try:
                        sess.run(stmt)
                    except Exception as exc:
                        self._logger.warning(
                            "neo4j.schema.ensure.statement_failed",
                            extra={
                                "event": "neo4j.schema.ensure",
                                "statement": stmt,
                                "status": "error",
                                "reason": str(exc),
                            },
                        )
                        # If Neo4j is down / auth is wrong, do not spin on every statement.
                        try:
                            from neo4j.exceptions import AuthError, ServiceUnavailable, SessionExpired  # type: ignore

                            if isinstance(exc, (AuthError, ServiceUnavailable, SessionExpired)):
                                break
                        except Exception:
                            pass
                        name = exc.__class__.__name__
                        msg = str(exc).lower()
                        if name in {"AuthError", "ServiceUnavailable", "SessionExpired"}:
                            break
                        if "failed to establish connection" in msg or "service unavailable" in msg or "handshake" in msg:
                            break
                        continue
        except Exception as exc:
            self._logger.warning(
                "neo4j.schema.ensure.failed",
                extra={"event": "neo4j.schema.ensure", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return

    # ---- Lifecycle management ----
    def close(self) -> None:
        """Close underlying Neo4j driver if present; idempotent."""
        if getattr(self, "_closed", False):
            return
        try:
            if self._driver is not None:
                try:
                    self._driver.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _sanitize_neo4j_props(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested dict/list to JSON strings for Neo4j property constraints."""
        import json

        sanitized = {}
        for k, v in props.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                sanitized[k] = json.dumps(v, ensure_ascii=False)
            else:
                sanitized[k] = v
        return sanitized

    # ---- Graph v0.1 primitives ----
    async def upsert_graph_v0(
        self,
        *,
        segments: List[MediaSegment],
        evidences: List[GraphEvidence],
        utterances: Optional[List[UtteranceEvidence]] = None,
        entities: List[GraphEntity] = None,  # type: ignore[assignment]
        events: List[GraphEvent] = None,  # type: ignore[assignment]
        places: List[GraphPlace] = None,  # type: ignore[assignment]
        time_slices: List[GraphTimeSlice] | None = None,
        regions: List[SpatioTemporalRegion] | None = None,
        states: List[GraphState] | None = None,
        knowledge: List[GraphKnowledge] | None = None,
        pending_equivs: List[Dict[str, Any]] | List[Any] | None = None,
        edges: List[GraphEdge],
    ) -> None:
        await asyncio.to_thread(
            self._upsert_graph_v0_sync,
            segments=segments,
            evidences=evidences,
            utterances=utterances,
            entities=entities,
            events=events,
            places=places,
            time_slices=time_slices,
            regions=regions,
            states=states,
            knowledge=knowledge,
            pending_equivs=pending_equivs,
            edges=edges,
        )
        return None

    def _upsert_graph_v0_sync(
        self,
        *,
        segments: List[MediaSegment],
        evidences: List[GraphEvidence],
        utterances: Optional[List[UtteranceEvidence]] = None,
        entities: List[GraphEntity] = None,  # type: ignore[assignment]
        events: List[GraphEvent] = None,  # type: ignore[assignment]
        places: List[GraphPlace] = None,  # type: ignore[assignment]
        time_slices: List[GraphTimeSlice] | None = None,
        regions: List[SpatioTemporalRegion] | None = None,
        states: List[GraphState] | None = None,
        knowledge: List[GraphKnowledge] | None = None,
        pending_equivs: List[Dict[str, Any]] | List[Any] | None = None,
        edges: List[GraphEdge],
    ) -> None:
        """Batch MERGE nodes/edges via Graph API v0 (TKG schema aligned to v1.0).

        - Uses typed labels per schema.
        - Edges are grouped by rel_type to enforce source/target labels.
        """

        import time as _t
        import json as _json

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return None
        if not self._driver:
            return None

        segments = segments or []
        evidences = evidences or []
        utterances = utterances or []
        entities = entities or []
        events = events or []
        places = places or []
        time_slices = time_slices or []
        regions = regions or []
        states = states or []
        knowledge = knowledge or []
        pending_equivs = pending_equivs or []

        def _sanitize_props(d: Dict[str, Any]) -> Dict[str, Any]:
            """Neo4j properties cannot be maps; encode nested dict/list-of-dict as JSON strings.

            GraphUpsertRequest keeps flexible objects (e.g. `provenance`, `extras`, `data`), but Neo4j only
            accepts primitive or arrays-of-primitive values as properties.
            """

            out: Dict[str, Any] = {}
            for k, v in (d or {}).items():
                if v is None:
                    out[k] = None
                    continue
                if isinstance(v, (str, int, float, bool)):
                    out[k] = v
                    continue
                if isinstance(v, (list, tuple)):
                    if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                        out[k] = list(v)
                        continue
                    try:
                        out[f"{k}_json"] = _json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    except Exception:
                        out[f"{k}_json"] = str(v)
                    continue
                if isinstance(v, dict):
                    json_key = "extras_json" if k == "extras" else f"{k}_json"
                    try:
                        out[json_key] = _json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                    except Exception:
                        out[json_key] = str(v)
                    continue
                # datetime and neo4j temporal types are OK as-is; for unknown objects, coerce to str.
                try:
                    out[k] = v
                except Exception:
                    out[k] = str(v)
            return out

        def _dump_and_sanitize(model: Any) -> Dict[str, Any]:
            try:
                raw = model.model_dump(mode="python") if hasattr(model, "model_dump") else dict(model)
            except Exception:
                raw = {}
            return _sanitize_props(raw)

        # Convert models to plain dicts for Neo4j driver, preserving native datetime objects where applicable
        seg_dicts = [_dump_and_sanitize(s) for s in segments]
        evid_dicts = [_dump_and_sanitize(e) for e in evidences]
        utt_dicts = [_dump_and_sanitize(u) for u in utterances]
        ent_dicts = [_dump_and_sanitize(e) for e in entities]
        evt_dicts = [_dump_and_sanitize(e) for e in events]
        plc_dicts = [_dump_and_sanitize(p) for p in places]
        ts_dicts = [_dump_and_sanitize(t) for t in (time_slices or [])]
        region_dicts = [_dump_and_sanitize(r) for r in (regions or [])]
        state_dicts = [_dump_and_sanitize(s) for s in (states or [])]
        knowledge_dicts = [_dump_and_sanitize(k) for k in (knowledge or [])]
        peq_dicts = []
        for peq in pending_equivs:
            if hasattr(peq, "model_dump"):
                peq_dicts.append(_dump_and_sanitize(peq))  # type: ignore[arg-type]
            elif isinstance(peq, dict):
                peq_dicts.append(_sanitize_props(peq))
        edge_dicts = [_dump_and_sanitize(e) for e in edges]

        payload_has_items = any(
            [
                bool(seg_dicts),
                bool(evid_dicts),
                bool(utt_dicts),
                bool(ent_dicts),
                bool(evt_dicts),
                bool(plc_dicts),
                bool(ts_dicts),
                bool(region_dicts),
                bool(state_dicts),
                bool(knowledge_dicts),
                bool(peq_dicts),
                bool(edge_dicts),
            ]
        )
        tenant_values = self._collect_tenant_values(
            seg_dicts
            + evid_dicts
            + utt_dicts
            + ent_dicts
            + evt_dicts
            + plc_dicts
            + ts_dicts
            + region_dicts
            + state_dicts
            + knowledge_dicts
            + peq_dicts
            + edge_dicts
        )
        if len(tenant_values) > 1:
            raise ValueError(f"upsert_graph_v0 mixed tenant_id payload is forbidden: {tenant_values}")
        expected_tenant = tenant_values[0] if tenant_values else ""
        if self._strict_tenant_mode and payload_has_items:
            expected_tenant = self._validate_tenant_context(expected_tenant, "upsert_graph_v0")
            self._assert_payload_tenant(payload=seg_dicts, tenant_id=expected_tenant, operation="segments")
            self._assert_payload_tenant(payload=evid_dicts, tenant_id=expected_tenant, operation="evidences")
            self._assert_payload_tenant(payload=utt_dicts, tenant_id=expected_tenant, operation="utterances")
            self._assert_payload_tenant(payload=ent_dicts, tenant_id=expected_tenant, operation="entities")
            self._assert_payload_tenant(payload=evt_dicts, tenant_id=expected_tenant, operation="events")
            self._assert_payload_tenant(payload=plc_dicts, tenant_id=expected_tenant, operation="places")
            self._assert_payload_tenant(payload=ts_dicts, tenant_id=expected_tenant, operation="time_slices")
            self._assert_payload_tenant(payload=region_dicts, tenant_id=expected_tenant, operation="regions")
            self._assert_payload_tenant(payload=state_dicts, tenant_id=expected_tenant, operation="states")
            self._assert_payload_tenant(payload=knowledge_dicts, tenant_id=expected_tenant, operation="knowledge")
            self._assert_payload_tenant(payload=peq_dicts, tenant_id=expected_tenant, operation="pending_equivs")
            self._assert_payload_tenant(payload=edge_dicts, tenant_id=expected_tenant, operation="edges")

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                # Nodes
                if seg_dicts:
                    sess.run(
                        "UNWIND $segments AS s "
                        "MERGE (n:MediaSegment {id: s.id, tenant_id: s.tenant_id}) "
                        "SET n += s ",
                        segments=seg_dicts,
                    )
                if evid_dicts:
                    sess.run(
                        "UNWIND $evidences AS e "
                        "MERGE (n:Evidence {id: e.id, tenant_id: e.tenant_id}) "
                        "SET n += e ",
                        evidences=evid_dicts,
                    )
                if utt_dicts:
                    sess.run(
                        "UNWIND $utterances AS u "
                        "MERGE (n:UtteranceEvidence {id: u.id, tenant_id: u.tenant_id}) "
                        "SET n += u ",
                        utterances=utt_dicts,
                    )
                if ent_dicts:
                    sess.run(
                        "UNWIND $entities AS e "
                        "MERGE (n:Entity {id: e.id, tenant_id: e.tenant_id}) "
                        "SET n += e ",
                        entities=ent_dicts,
                    )
                if evt_dicts:
                    sess.run(
                        "UNWIND $events AS e "
                        "MERGE (n:Event {id: e.id, tenant_id: e.tenant_id}) "
                        "SET n += e ",
                        events=evt_dicts,
                    )
                if plc_dicts:
                    sess.run(
                        "UNWIND $places AS p "
                        "MERGE (n:Place {id: p.id, tenant_id: p.tenant_id}) "
                        "SET n += p ",
                        places=plc_dicts,
                    )
                if ts_dicts:
                    sess.run(
                        "UNWIND $slices AS t "
                        "MERGE (n:TimeSlice {id: t.id, tenant_id: t.tenant_id}) "
                        "SET n += t ",
                        slices=ts_dicts,
                    )
                if region_dicts:
                    sess.run(
                        "UNWIND $regions AS r "
                        "MERGE (n:SpatioTemporalRegion {id: r.id, tenant_id: r.tenant_id}) "
                        "SET n += r ",
                        regions=region_dicts,
                    )
                if state_dicts:
                    sess.run(
                        "UNWIND $states AS s "
                        "MERGE (n:State {id: s.id, tenant_id: s.tenant_id}) "
                        "SET n += s ",
                        states=state_dicts,
                    )
                if knowledge_dicts:
                    sess.run(
                        "UNWIND $knowledge AS k "
                        "MERGE (n:Knowledge {id: k.id, tenant_id: k.tenant_id}) "
                        "SET n += k ",
                        knowledge=knowledge_dicts,
                    )
                if peq_dicts:
                    sess.run(
                        "UNWIND $peq AS p "
                        "MERGE (n:PendingEquiv {id: p.id, tenant_id: p.tenant_id}) "
                        "SET n += p ",
                        peq=peq_dicts,
                    )

                if edge_dicts:
                    grouped: Dict[str, List[Dict[str, Any]]] = {}
                    for e in edge_dicts:
                        grouped.setdefault(str(e.get("rel_type")), []).append(e)

                    rel_label = {
                        "NEXT_SEGMENT": ("MediaSegment", "MediaSegment"),
                        "CONTAINS_EVIDENCE": ("MediaSegment", "Evidence"),
                        "BELONGS_TO_ENTITY": ("Evidence", "Entity"),
                        "SUMMARIZES": ("Event", "MediaSegment"),
                        "INVOLVES": ("Event", "Entity"),
                        "OCCURS_AT": ("Event", "Place"),
                        # v0.2
                        "NEXT_EVENT": ("Event", "Event"),
                        "CO_OCCURS_WITH": ("Entity", "Entity"),
                        "CAUSES": ("Event", "Event"),
                        "SUPPORTED_BY": ("Event", "Evidence"),  # 可被 src/dst type 覆盖
                        "COVERS_SEGMENT": ("TimeSlice", "MediaSegment"),
                        "COVERS_EVENT": ("TimeSlice", "Event"),
                        # v0.3
                        "SPOKEN_BY": ("UtteranceEvidence", "Entity"),
                        "TEMPORALLY_CONTAINS": ("TimeSlice", "Event"),
                        "SPATIALLY_CONTAINS": ("SpatioTemporalRegion", "Entity"),
                        "HAS_STATE": ("Entity", "State"),
                        "TRANSITIONS_TO": ("State", "State"),
                        "DERIVED_FROM": ("Knowledge", "Event"),
                        "EQUIV": ("Entity", "Entity"),
                    }

                    for rel, batch in grouped.items():
                        src_label_default, dst_label_default = rel_label.get(rel, (None, None))
                        by_label: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
                        for e in batch:
                            src_label = e.get("src_type") or src_label_default
                            dst_label = e.get("dst_type") or dst_label_default
                            if not src_label or not dst_label:
                                continue
                            by_label.setdefault((str(src_label), str(dst_label)), []).append(e)
                        if not by_label:
                            continue
                        for (src_label, dst_label), edge_group in by_label.items():
                            self._logger.info(f"Upserting edges: rel={rel}, src={src_label}, dst={dst_label}, count={len(edge_group)}")
                            cypher = (
                                "UNWIND $edges AS e "
                                f"MATCH (s:{src_label} {{id: e.src_id, tenant_id: e.tenant_id}}) "
                                f"MATCH (d:{dst_label} {{id: e.dst_id, tenant_id: e.tenant_id}}) "
                                f"MERGE (s)-[r:{rel}]->(d) "
                                "SET r.tenant_id = e.tenant_id "
                                "SET r.confidence = coalesce(e.confidence, r.confidence) "
                                "SET r.role = coalesce(e.role, r.role) "
                                "SET r.layer = coalesce(e.layer, r.layer) "
                                "SET r.kind = coalesce(e.kind, r.kind) "
                                "SET r.source = coalesce(e.source, r.source) "
                                "SET r.status = coalesce(e.status, r.status) "
                                "SET r.weight = coalesce(e.weight, r.weight) "
                                "SET r.first_seen_at = coalesce(e.first_seen_at, r.first_seen_at) "
                                "SET r.last_seen_at = coalesce(e.last_seen_at, r.last_seen_at) "
                                "SET r.provenance = coalesce(e.provenance, r.provenance) "
                                "SET r.time_origin = coalesce(e.time_origin, r.time_origin) "
                                "SET r.ttl = coalesce(e.ttl, r.ttl) "
                                "SET r.importance = coalesce(e.importance, r.importance) "
                                "RETURN count(r) AS applied "
                            )
                            _res = sess.run(cypher, edges=edge_group)
                            applied = self._extract_applied_count(_res)
                            if applied is not None and applied != len(edge_group):
                                raise RuntimeError(
                                    f"neo4j_edge_tenant_violation: rel={rel} src={src_label} dst={dst_label} expected={len(edge_group)} applied={applied}"
                                )

            self._cb_fail_count = 0
        except Exception as e:
            self._cb_fail_count += 1
            try:
                self._logger.error(
                    "neo4j.upsert_graph_v0.error",
                    extra={
                        "event": "neo4j.upsert_graph_v0.error",
                        "entity": "graph",
                        "verb": "merge",
                        "status": "error",
                        "reason": str(e),
                    },
                    exc_info=True,
                )
            except Exception:
                pass
            if self._cb_fail_count >= max(1, self._cb_failure_threshold):
                self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
        return None

    async def query_segments_by_time(
        self,
        *,
        tenant_id: str,
        source_id: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        modality: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []
        filters = ["s.tenant_id = $tenant"]
        params: Dict[str, Any] = {"tenant": tenant_id, "limit": int(limit)}
        if source_id:
            filters.append("s.source_id = $source_id")
            params["source_id"] = source_id
        if start is not None:
            filters.append("s.t_media_start >= $start")
            params["start"] = float(start)
        if end is not None:
            filters.append("s.t_media_end <= $end")
            params["end"] = float(end)
        if modality:
            filters.append("s.modality = $modality")
            params["modality"] = modality
        filters.append("(s.expires_at IS NULL OR s.expires_at > datetime())")
        filters.append("(s.published IS NULL OR s.published = true)")

        where_clause = " AND ".join(filters)
        cypher = (
            "MATCH (s:MediaSegment) "
            f"WHERE {where_clause} "
            "RETURN s.id AS id, s.source_id AS source_id, s.t_media_start AS t_media_start, s.t_media_end AS t_media_end, s.modality AS modality, s.has_physical_time AS has_physical_time, s.time_origin AS time_origin "
            "ORDER BY s.t_media_start ASC LIMIT $limit"
        )

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, **params)
                return [dict(row) for row in rows]
        except Exception:
            return []

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
        return await asyncio.to_thread(
            self._apply_state_update_sync,
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

    def _apply_state_update_sync(
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
        import uuid

        if not self._driver:
            return {"status": "skipped", "applied": False}

        def _coerce_dt(dt: Optional[datetime]) -> datetime:
            if dt is None:
                return datetime.now(self._tz_utc)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=self._tz_utc)
            return dt.astimezone(self._tz_utc)

        t = _coerce_dt(valid_from)
        t_minus = t - timedelta(milliseconds=1)
        now = datetime.now(self._tz_utc)
        state_id = f"state::{subject_id}::{property}::{int(t.timestamp()*1000)}::{uuid.uuid4().hex[:6]}"
        pending_id = f"pstate::{subject_id}::{property}::{int(t.timestamp()*1000)}::{uuid.uuid4().hex[:6]}"

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                tx = sess.begin_transaction()
                if status == "pending" or pending_reason:
                    tx.run(
                        "MERGE (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                        "SET k.updated_at = $now, k.lock = coalesce(k.lock, 0) + 1 ",
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        now=now,
                    )
                    tx.run(
                        "MERGE (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "SET e.user_id = coalesce(e.user_id, $user_id), e.memory_domain = coalesce(e.memory_domain, $memory_domain) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        user_id=user_id,
                        memory_domain=memory_domain,
                    )
                    tx.run(
                        "CREATE (p:PendingState {id: $id, tenant_id: $tenant, subject_id: $subject, property: $prop, value: $value, "
                        "raw_value: $raw_value, confidence: $confidence, valid_from: $t, source_event_id: $source_event_id, "
                        "extractor_version: $extractor_version, status: $status, pending_reason: $pending_reason, "
                        "user_id: $user_id, memory_domain: $memory_domain, created_at: $now, updated_at: $now}) ",
                        id=pending_id,
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        value=value,
                        raw_value=raw_value,
                        confidence=confidence,
                        t=t,
                        source_event_id=source_event_id,
                        extractor_version=extractor_version,
                        status=status or "pending",
                        pending_reason=pending_reason or "manual",
                        user_id=user_id,
                        memory_domain=memory_domain,
                        now=now,
                    )
                    tx.run(
                        "MATCH (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "MATCH (p:PendingState {tenant_id: $tenant, id: $id}) "
                        "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                        "MERGE (e)-[:HAS_PENDING_STATE]->(p) "
                        "MERGE (k)-[:PENDING]->(p) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        id=pending_id,
                    )
                    tx.commit()
                    return {"status": "pending", "pending": True, "pending_id": pending_id, "applied": False}
                tx.run(
                    "MERGE (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                    "SET k.updated_at = $now, k.lock = coalesce(k.lock, 0) + 1 ",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    now=now,
                )
                cur = tx.run(
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[:CURRENT]->(cur:State) "
                    "RETURN cur.id AS id, cur.value AS value, cur.valid_from AS valid_from",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                ).single()

                if not cur:
                    tx.run(
                        "MERGE (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "SET e.user_id = coalesce(e.user_id, $user_id), e.memory_domain = coalesce(e.memory_domain, $memory_domain) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        user_id=user_id,
                        memory_domain=memory_domain,
                    )
                    tx.run(
                        "CREATE (s:State {id: $id, tenant_id: $tenant, subject_id: $subject, property: $prop, value: $value, "
                        "raw_value: $raw_value, confidence: $confidence, valid_from: $t, valid_to: null, last_seen_at: $t, "
                        "source_event_id: $source_event_id, extractor_version: $extractor_version, status: $status, "
                        "user_id: $user_id, memory_domain: $memory_domain, created_at: $now, updated_at: $now}) ",
                        id=state_id,
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        value=value,
                        raw_value=raw_value,
                        confidence=confidence,
                        t=t,
                        source_event_id=source_event_id,
                        extractor_version=extractor_version,
                        status=status,
                        user_id=user_id,
                        memory_domain=memory_domain,
                        now=now,
                    )
                    tx.run(
                        "MATCH (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "MATCH (s:State {tenant_id: $tenant, id: $id}) "
                        "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                        "MERGE (e)-[:HAS_STATE]->(s) "
                        "MERGE (k)-[:HAS_STATE]->(s) "
                        "MERGE (k)-[:CURRENT]->(s) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        id=state_id,
                    )
                    tx.commit()
                    return {"status": "created", "state_id": state_id, "applied": True}

                cur_valid_from = cur.get("valid_from")
                if cur_valid_from and t < cur_valid_from:
                    tx.run(
                        "MERGE (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "SET e.user_id = coalesce(e.user_id, $user_id), e.memory_domain = coalesce(e.memory_domain, $memory_domain) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        user_id=user_id,
                        memory_domain=memory_domain,
                    )
                    tx.run(
                        "CREATE (p:PendingState {id: $id, tenant_id: $tenant, subject_id: $subject, property: $prop, value: $value, "
                        "raw_value: $raw_value, confidence: $confidence, valid_from: $t, source_event_id: $source_event_id, "
                        "extractor_version: $extractor_version, status: $status, pending_reason: $pending_reason, "
                        "user_id: $user_id, memory_domain: $memory_domain, created_at: $now, updated_at: $now}) ",
                        id=pending_id,
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        value=value,
                        raw_value=raw_value,
                        confidence=confidence,
                        t=t,
                        source_event_id=source_event_id,
                        extractor_version=extractor_version,
                        status="pending",
                        pending_reason="out_of_order",
                        user_id=user_id,
                        memory_domain=memory_domain,
                        now=now,
                    )
                    tx.run(
                        "MATCH (e:Entity {tenant_id: $tenant, id: $subject}) "
                        "MATCH (p:PendingState {tenant_id: $tenant, id: $id}) "
                        "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                        "MERGE (e)-[:HAS_PENDING_STATE]->(p) "
                        "MERGE (k)-[:PENDING]->(p) ",
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        id=pending_id,
                    )
                    tx.commit()
                    return {
                        "status": "pending",
                        "pending": True,
                        "pending_id": pending_id,
                        "reason": "out_of_order",
                        "applied": False,
                    }

                if str(cur.get("value")) == str(value):
                    tx.run(
                        "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[:CURRENT]->(cur:State) "
                        "SET cur.last_seen_at = $t, cur.updated_at = $now ",
                        tenant=tenant_id,
                        subject=subject_id,
                        prop=property,
                        t=t,
                        now=now,
                    )
                    tx.commit()
                    return {"status": "touched", "state_id": cur.get("id"), "applied": True}

                tx.run(
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[:CURRENT]->(cur:State) "
                    "SET cur.valid_to = $t_minus, cur.updated_at = $now ",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    t_minus=t_minus,
                    now=now,
                )
                tx.run(
                    "CREATE (s:State {id: $id, tenant_id: $tenant, subject_id: $subject, property: $prop, value: $value, "
                    "raw_value: $raw_value, confidence: $confidence, valid_from: $t, valid_to: null, last_seen_at: $t, "
                    "source_event_id: $source_event_id, extractor_version: $extractor_version, status: $status, "
                    "user_id: $user_id, memory_domain: $memory_domain, created_at: $now, updated_at: $now}) ",
                    id=state_id,
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    value=value,
                    raw_value=raw_value,
                    confidence=confidence,
                    t=t,
                    source_event_id=source_event_id,
                    extractor_version=extractor_version,
                    status=status,
                    user_id=user_id,
                    memory_domain=memory_domain,
                    now=now,
                )
                tx.run(
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[:CURRENT]->(cur:State) "
                    "MATCH (s:State {tenant_id: $tenant, id: $id}) "
                    "MERGE (cur)-[r:TRANSITIONS_TO]->(s) "
                    "SET r.at = $t "
                    "SET r.by_event_id = coalesce($source_event_id, r.by_event_id) ",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    id=state_id,
                    t=t,
                    source_event_id=source_event_id,
                )
                tx.run(
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[r:CURRENT]->(cur:State) "
                    "MATCH (s:State {tenant_id: $tenant, id: $id}) "
                    "DELETE r "
                    "MERGE (k)-[:CURRENT]->(s) ",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    id=state_id,
                )
                tx.run(
                    "MERGE (e:Entity {tenant_id: $tenant, id: $subject}) "
                    "SET e.user_id = coalesce(e.user_id, $user_id), e.memory_domain = coalesce(e.memory_domain, $memory_domain) ",
                    tenant=tenant_id,
                    subject=subject_id,
                    user_id=user_id,
                    memory_domain=memory_domain,
                )
                tx.run(
                    "MATCH (e:Entity {tenant_id: $tenant, id: $subject}) "
                    "MATCH (s:State {tenant_id: $tenant, id: $id}) "
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                    "MERGE (e)-[:HAS_STATE]->(s) "
                    "MERGE (k)-[:HAS_STATE]->(s) ",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    id=state_id,
                )
                tx.commit()
                return {"status": "updated", "state_id": state_id, "previous_state_id": cur.get("id"), "applied": True}
        except Exception as exc:
            try:
                self._logger.error("neo4j.apply_state_update.failed", extra={"status": "error", "reason": str(exc)})
            except Exception:
                pass
            return {"status": "error", "applied": False, "error": str(exc)}

    async def get_current_state(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_current_state_sync,
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
        )

    def _get_current_state_sync(self, *, tenant_id: str, subject_id: str, property: str) -> Optional[Dict[str, Any]]:
        if not self._driver:
            return None
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    "MATCH (k:StateKey {tenant_id: $tenant, subject_id: $subject, property: $prop})-[:CURRENT]->(s:State) "
                    "RETURN s AS state LIMIT 1",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                ).single()
                if not row:
                    return None
                node = row.get("state")
                if not node:
                    return None
                props = dict(node)
                props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                props["valid_to"] = self._dt_to_iso(props.get("valid_to"))
                props["last_seen_at"] = self._dt_to_iso(props.get("last_seen_at"))
                return props
        except Exception:
            return None

    async def get_state_at_time(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
        t: datetime,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_state_at_time_sync,
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            t=t,
        )

    def _get_state_at_time_sync(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        property: str,
        t: datetime,
    ) -> Optional[Dict[str, Any]]:
        if not self._driver:
            return None
        try:
            tt = t
            if tt.tzinfo is None:
                tt = tt.replace(tzinfo=self._tz_utc)
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    "MATCH (s:State {tenant_id: $tenant, subject_id: $subject, property: $prop}) "
                    "WHERE s.valid_from <= $t AND (s.valid_to IS NULL OR s.valid_to > $t) "
                    "RETURN s AS state ORDER BY s.valid_from DESC LIMIT 1",
                    tenant=tenant_id,
                    subject=subject_id,
                    prop=property,
                    t=tt,
                ).single()
                if not row:
                    return None
                node = row.get("state")
                if not node:
                    return None
                props = dict(node)
                props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                props["valid_to"] = self._dt_to_iso(props.get("valid_to"))
                props["last_seen_at"] = self._dt_to_iso(props.get("last_seen_at"))
                return props
        except Exception:
            return None

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
        return await asyncio.to_thread(
            self._get_state_changes_sync,
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            start=start,
            end=end,
            limit=limit,
            order=order,
        )

    def _get_state_changes_sync(
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
        if not self._driver:
            return []
        try:
            st = start
            if st and st.tzinfo is None:
                st = st.replace(tzinfo=self._tz_utc)
            ed = end
            if ed and ed.tzinfo is None:
                ed = ed.replace(tzinfo=self._tz_utc)
            filters = [
                "s.tenant_id = $tenant",
                "s.subject_id = $subject",
                "s.property = $prop",
            ]
            params: Dict[str, Any] = {
                "tenant": tenant_id,
                "subject": subject_id,
                "prop": property,
                "limit": int(limit),
            }
            if st is not None:
                filters.append("s.valid_from >= $start")
                params["start"] = st
            if ed is not None:
                filters.append("s.valid_from <= $end")
                params["end"] = ed
            order_dir = "DESC" if str(order).lower().startswith("desc") else "ASC"
            cypher = (
                "MATCH (s:State) WHERE " + " AND ".join(filters) +
                f" RETURN s AS state ORDER BY s.valid_from {order_dir} LIMIT $limit"
            )
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, **params)
                out: List[Dict[str, Any]] = []
                for row in rows:
                    node = row.get("state")
                    if not node:
                        continue
                    props = dict(node)
                    props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                    props["valid_to"] = self._dt_to_iso(props.get("valid_to"))
                    props["last_seen_at"] = self._dt_to_iso(props.get("last_seen_at"))
                    out.append(props)
                return out
        except Exception:
            return []

    async def get_pending_state(
        self,
        *,
        tenant_id: str,
        pending_id: str,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._get_pending_state_sync,
            tenant_id=tenant_id,
            pending_id=pending_id,
        )

    def _get_pending_state_sync(self, *, tenant_id: str, pending_id: str) -> Optional[Dict[str, Any]]:
        if not self._driver:
            return None
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    "MATCH (p:PendingState {tenant_id: $tenant, id: $id}) RETURN p AS pending LIMIT 1",
                    tenant=tenant_id,
                    id=pending_id,
                ).single()
                if not row:
                    return None
                node = row.get("pending")
                if not node:
                    return None
                props = dict(node)
                props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                return props
        except Exception:
            return None

    async def list_pending_states(
        self,
        *,
        tenant_id: str,
        subject_id: Optional[str] = None,
        property: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._list_pending_states_sync,
            tenant_id=tenant_id,
            subject_id=subject_id,
            property=property,
            status=status,
            limit=limit,
        )

    def _list_pending_states_sync(
        self,
        *,
        tenant_id: str,
        subject_id: Optional[str] = None,
        property: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not self._driver:
            return []
        try:
            filters = ["p.tenant_id = $tenant"]
            params: Dict[str, Any] = {"tenant": tenant_id, "limit": int(limit)}
            if subject_id:
                filters.append("p.subject_id = $subject")
                params["subject"] = subject_id
            if property:
                filters.append("p.property = $prop")
                params["prop"] = property
            if status:
                filters.append("p.status = $status")
                params["status"] = status
            cypher = (
                "MATCH (p:PendingState) WHERE " + " AND ".join(filters) +
                " RETURN p AS pending ORDER BY p.created_at DESC LIMIT $limit"
            )
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, **params)
                out: List[Dict[str, Any]] = []
                for row in rows:
                    node = row.get("pending")
                    if not node:
                        continue
                    props = dict(node)
                    props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                    out.append(props)
                return out
        except Exception:
            return []

    async def update_pending_state_status(
        self,
        *,
        tenant_id: str,
        pending_id: str,
        status: str,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._update_pending_state_status_sync,
            tenant_id=tenant_id,
            pending_id=pending_id,
            status=status,
            note=note,
        )

    def _update_pending_state_status_sync(
        self,
        *,
        tenant_id: str,
        pending_id: str,
        status: str,
        note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._driver:
            return None
        try:
            now = datetime.now(self._tz_utc)
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    "MATCH (p:PendingState {tenant_id: $tenant, id: $id}) "
                    "SET p.status = $status, p.resolution_note = $note, p.resolved_at = $now, p.updated_at = $now "
                    "RETURN p AS pending",
                    tenant=tenant_id,
                    id=pending_id,
                    status=status,
                    note=note,
                    now=now,
                ).single()
                if not row:
                    return None
                node = row.get("pending")
                if not node:
                    return None
                props = dict(node)
                props["valid_from"] = self._dt_to_iso(props.get("valid_from"))
                return props
        except Exception:
            return None


    def _dt_to_iso(self, val: Any) -> Optional[str]:
        try:
            if val is None:
                return None
            to_native = getattr(val, "to_native", None)
            if callable(to_native):
                val = to_native()
            if isinstance(val, datetime):
                if val.tzinfo is None:
                    val = val.replace(tzinfo=self._tz_utc)
                return val.isoformat()
            if isinstance(val, str):
                return val
            return str(val)
        except Exception:
            return None

    async def query_events(
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
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ev)-[:INVOLVES]->(ent:Entity {tenant_id: $tenant})
  WHERE (ent.published IS NULL OR ent.published = true)
OPTIONAL MATCH (ev)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
  WHERE (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev)-[rel:NEXT_EVENT|CAUSES]->(ev2:Event {tenant_id: $tenant})
  WHERE (ev2.published IS NULL OR ev2.published = true)
WITH ev, seg, collect(DISTINCT ent.id) AS entity_ids, collect(DISTINCT pl.id) AS place_ids,
     collect(DISTINCT rel) AS rels,
     collect(
       DISTINCT {
         type: type(rel),
         target_event_id: ev2.id,
         layer: rel.layer,
         status: rel.status,
         kind: rel.kind
       }
     ) AS relations
WHERE ($segment_id IS NULL OR (seg IS NOT NULL AND seg.id = $segment_id))
  AND ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
  AND ($entity_id IS NULL OR ANY(x IN entity_ids WHERE x = $entity_id))
  AND ($place_id IS NULL OR ANY(x IN place_ids WHERE x = $place_id))
  AND ($relation IS NULL OR ANY(r IN rels WHERE type(r) = $relation))
  AND ($layer IS NULL OR ANY(r IN rels WHERE coalesce(r.layer, 'fact') = $layer))
  AND ($status IS NULL OR ANY(r IN rels WHERE coalesce(r.status, '') = $status))
RETURN ev AS event,
       seg.id AS segment_id,
       seg.source_id AS source_id,
       entity_ids,
       place_ids,
       relations
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    segment_id=segment_id,
                    entity_id=entity_id,
                    place_id=place_id,
                    source_id=source_id,
                    relation=relation,
                    layer=layer,
                    status=status,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                    limit=int(limit),
                )
                rows_list = list(rows)
                self._logger.info(f"query_events: tenant={tenant_id}, params={segment_id, entity_id}, rows={len(rows_list)}")
                items: List[Dict[str, Any]] = []
                for row in rows_list:
                    event = row.get("event")
                    seg_id = row.get("segment_id")
                    src_id = row.get("source_id")
                    entity_ids = row.get("entity_ids") or []
                    place_ids = row.get("place_ids") or []
                    data: Dict[str, Any] = {
                        "segment_id": seg_id,
                        "source_id": src_id,
                        "entity_ids": list(entity_ids),
                        "place_ids": list(place_ids),
                    }
                    if event:
                        props = dict(event)
                        data.update(
                            {
                                "id": props.get("id"),
                                "summary": props.get("summary"),
                                "time_origin": props.get("time_origin"),
                                "importance": props.get("importance"),
                                "source": props.get("source"),
                                "t_abs_start": self._dt_to_iso(props.get("t_abs_start")),
                                "t_abs_end": self._dt_to_iso(props.get("t_abs_end")),
                                "event_type": props.get("event_type"),
                                "action": props.get("action"),
                                "actor_id": props.get("actor_id"),
                            }
                        )
                    relations = []
                    for rel in row.get("relations") or []:
                        if not isinstance(rel, dict):
                            continue
                        target_id = rel.get("target_event_id")
                        rel_type = rel.get("type")
                        if not target_id or not rel_type:
                            continue
                        relations.append(
                            {
                                "type": rel_type,
                                "target_event_id": target_id,
                                "layer": rel.get("layer"),
                                "status": rel.get("status"),
                                "kind": rel.get("kind"),
                            }
                        )
                    if relations:
                        data["relations"] = relations
                    items.append(data)
                return items
        except Exception as e:
            self._logger.error(f"query_events failed: {e}", exc_info=True)
            return []

    async def query_events_by_topic(
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
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        topic_id = str(topic_id or "").strip() or None
        topic_path = str(topic_path or "").strip() or None
        if not topic_id and not topic_path and not tags and not keywords:
            return []

        start_iso = self._dt_to_iso(start)
        end_iso = self._dt_to_iso(end)
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($topic_id IS NULL OR ev.topic_id = $topic_id)
  AND ($topic_path IS NULL OR (ev.topic_path IS NOT NULL AND ev.topic_path STARTS WITH $topic_path))
  AND ($tags IS NULL OR ANY(t IN coalesce(ev.tags, []) WHERE t IN $tags))
  AND ($keywords IS NULL OR ANY(k IN coalesce(ev.keywords, []) WHERE k IN $keywords))
  AND ($start IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start >= datetime($start)))
  AND ($end IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start <= datetime($end)))
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
RETURN ev AS event
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    topic_id=topic_id,
                    topic_path=topic_path,
                    tags=list(tags) if tags else None,
                    keywords=list(keywords) if keywords else None,
                    start=start_iso,
                    end=end_iso,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                    limit=int(limit),
                )
                out: List[Dict[str, Any]] = []
                for row in rows:
                    ev = row.get("event")
                    if not ev:
                        continue
                    props = dict(ev)
                    props["t_abs_start"] = self._dt_to_iso(props.get("t_abs_start"))
                    props["t_abs_end"] = self._dt_to_iso(props.get("t_abs_end"))
                    out.append(props)
                return out
        except Exception:
            return []

    async def query_events_by_ids(
        self,
        *,
        tenant_id: str,
        event_ids: List[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        ids = [str(x) for x in (event_ids or []) if str(x).strip()]
        if not ids:
            return []

        start_iso = self._dt_to_iso(start)
        end_iso = self._dt_to_iso(end)
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE ev.id IN $event_ids
  AND (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($start IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start >= datetime($start)))
  AND ($end IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start <= datetime($end)))
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
RETURN ev AS event
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    event_ids=ids,
                    start=start_iso,
                    end=end_iso,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                    limit=int(limit),
                )
                out: List[Dict[str, Any]] = []
                for row in rows:
                    ev = row.get("event")
                    if not ev:
                        continue
                    props = dict(ev)
                    props["t_abs_start"] = self._dt_to_iso(props.get("t_abs_start"))
                    props["t_abs_end"] = self._dt_to_iso(props.get("t_abs_end"))
                    out.append(props)
                return out
        except Exception:
            return []

    async def query_event_id_by_logical_id(
        self,
        *,
        tenant_id: str,
        logical_event_id: str,
    ) -> Optional[str]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return None
        if not self._driver:
            return None

        logical_id = str(logical_event_id or "").strip()
        if not logical_id:
            return None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant, logical_event_id: $logical_event_id})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
RETURN ev.id AS event_id
ORDER BY coalesce(ev.updated_at, ev.created_at, datetime({epochMillis: 0})) DESC
LIMIT 1
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    logical_event_id=logical_id,
                ).single()
                if not row:
                    return None
                event_id = str(row.get("event_id") or "").strip()
                return event_id or None
        except Exception:
            return None

    async def search_event_candidates(
        self,
        *,
        tenant_id: str,
        query: str,
        limit: int = 10,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return ranked Event candidates for a free-text query.

        Strategy (best-effort):
        - Prefer Neo4j fulltext indexes over Event.summary / UtteranceEvidence.raw_text / Evidence.text.
        - Fallback to `toLower(..) CONTAINS toLower($q)` when fulltext is unavailable.
        """
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        q = (query or "").strip()
        if not q:
            return []
        bounded_limit = max(1, min(int(limit), 200))

        def _rows_to_hits(rows: Iterable[Any], *, matched: str) -> List[Dict[str, Any]]:
            hits: List[Dict[str, Any]] = []
            for row in rows:
                ev = row.get("event")
                if not ev:
                    continue
                props = dict(ev)
                hits.append(
                    {
                        "event_id": props.get("id"),
                        "logical_event_id": props.get("logical_event_id"),
                        "summary": props.get("summary"),
                        "desc": props.get("desc"),
                        "t_abs_start": self._dt_to_iso(props.get("t_abs_start")),
                        "t_abs_end": self._dt_to_iso(props.get("t_abs_end")),
                        "source_id": row.get("source_id"),
                        "score": float(row.get("score") or 0.0),
                        "matched": matched,
                    }
                )
            return hits

        def _merge(best: Dict[str, Dict[str, Any]], incoming: List[Dict[str, Any]]) -> None:
            for hit in incoming:
                event_id = hit.get("event_id")
                if not event_id:
                    continue
                prev = best.get(str(event_id))
                if prev is None:
                    best[str(event_id)] = hit
                    continue
                if float(hit.get("score") or 0.0) > float(prev.get("score") or 0.0):
                    best[str(event_id)] = hit
                    continue
                if not prev.get("source_id") and hit.get("source_id"):
                    prev["source_id"] = hit.get("source_id")

        hits_by_id: Dict[str, Dict[str, Any]] = {}

        event_fulltext_new = """
CALL db.index.fulltext.queryNodes('tkg_event_summary_desc_v1', $q) YIELD node, score
WITH node AS ev, score
WHERE ev.tenant_id = $tenant AND (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, score, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, score AS score, seg.source_id AS source_id
ORDER BY score DESC
LIMIT $limit
"""
        event_fulltext_old = """
CALL db.index.fulltext.queryNodes('tkg_event_summary_v1', $q) YIELD node, score
WITH node AS ev, score
WHERE ev.tenant_id = $tenant AND (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, score, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, score AS score, seg.source_id AS source_id
ORDER BY score DESC
LIMIT $limit
"""
        event_contains = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND toLower(ev.summary) CONTAINS toLower($q)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, 1.0 AS score, seg.source_id AS source_id
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""

        utt_fulltext = """
CALL db.index.fulltext.queryNodes('tkg_utterance_text_v1', $q) YIELD node, score
WITH node AS utt, score
WHERE utt.tenant_id = $tenant AND (utt.expires_at IS NULL OR utt.expires_at > datetime())
  AND (utt.published IS NULL OR utt.published = true)
MATCH (ev:Event {tenant_id: $tenant})-[:SUPPORTED_BY]->(utt)
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, max(score) AS score, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, score AS score, seg.source_id AS source_id
ORDER BY score DESC
LIMIT $limit
"""
        utt_contains = """
MATCH (utt:UtteranceEvidence {tenant_id: $tenant})
WHERE (utt.expires_at IS NULL OR utt.expires_at > datetime())
  AND (utt.published IS NULL OR utt.published = true)
  AND toLower(utt.raw_text) CONTAINS toLower($q)
MATCH (ev:Event {tenant_id: $tenant})-[:SUPPORTED_BY]->(utt)
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, 0.8 AS score, seg.source_id AS source_id
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""

        evd_fulltext = """
CALL db.index.fulltext.queryNodes('tkg_evidence_text_v1', $q) YIELD node, score
WITH node AS evd, score
WHERE evd.tenant_id = $tenant AND (evd.expires_at IS NULL OR evd.expires_at > datetime())
  AND (evd.published IS NULL OR evd.published = true)
MATCH (ev:Event {tenant_id: $tenant})-[:SUPPORTED_BY]->(evd)
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
WITH ev, max(score) AS score, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, score AS score, seg.source_id AS source_id
ORDER BY score DESC
LIMIT $limit
"""
        evd_contains = """
MATCH (evd:Evidence {tenant_id: $tenant})
WHERE (evd.expires_at IS NULL OR evd.expires_at > datetime())
  AND (evd.published IS NULL OR evd.published = true)
  AND evd.text IS NOT NULL
  AND toLower(evd.text) CONTAINS toLower($q)
MATCH (ev:Event {tenant_id: $tenant})-[:SUPPORTED_BY]->(evd)
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
WITH ev, seg
WHERE ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
RETURN ev AS event, 0.6 AS score, seg.source_id AS source_id
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                try:
                    rows = sess.run(
                        event_fulltext_new,
                        tenant=tenant_id,
                        q=q,
                        source_id=source_id,
                        limit=bounded_limit,
                    )
                    _merge(hits_by_id, _rows_to_hits(rows, matched="summary"))
                except Exception:
                    try:
                        rows = sess.run(
                            event_fulltext_old,
                            tenant=tenant_id,
                            q=q,
                            source_id=source_id,
                            limit=bounded_limit,
                        )
                        _merge(hits_by_id, _rows_to_hits(rows, matched="summary"))
                    except Exception:
                        rows = sess.run(
                            event_contains,
                            tenant=tenant_id,
                            q=q,
                            source_id=source_id,
                            limit=bounded_limit,
                        )
                        _merge(hits_by_id, _rows_to_hits(rows, matched="summary"))

                try:
                    rows = sess.run(
                        utt_fulltext,
                        tenant=tenant_id,
                        q=q,
                        source_id=source_id,
                        limit=bounded_limit,
                    )
                    _merge(hits_by_id, _rows_to_hits(rows, matched="utterance"))
                except Exception:
                    rows = sess.run(
                        utt_contains,
                        tenant=tenant_id,
                        q=q,
                        source_id=source_id,
                        limit=bounded_limit,
                    )
                    _merge(hits_by_id, _rows_to_hits(rows, matched="utterance"))

                try:
                    rows = sess.run(
                        evd_fulltext,
                        tenant=tenant_id,
                        q=q,
                        source_id=source_id,
                        limit=bounded_limit,
                    )
                    _merge(hits_by_id, _rows_to_hits(rows, matched="evidence"))
                except Exception:
                    rows = sess.run(
                        evd_contains,
                        tenant=tenant_id,
                        q=q,
                        source_id=source_id,
                        limit=bounded_limit,
                    )
                    _merge(hits_by_id, _rows_to_hits(rows, matched="evidence"))
        except Exception as exc:
            self._logger.error(
                "neo4j.search_event_candidates.failed",
                extra={
                    "event": "neo4j.search_event_candidates",
                    "tenant_id": tenant_id,
                    "status": "error",
                    "reason": str(exc),
                },
                exc_info=True,
            )
            return []

        items = list(hits_by_id.values())
        items.sort(
            key=lambda it: (float(it.get("score") or 0.0), it.get("t_abs_start") or ""),
            reverse=True,
        )
        return items[:bounded_limit]

    async def query_entities_by_name(
        self,
        *,
        tenant_id: str,
        name: str,
        entity_type: Optional[str] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Resolve entities by name/alias using fulltext (preferred) with CONTAINS fallback."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        q = str(name or "").strip()
        if not q:
            return []
        bounded_limit = max(1, min(int(limit), 200))
        etype = str(entity_type).strip() if entity_type is not None else None
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        def _rows_to_hits(rows: Iterable[Any], *, matched: str) -> List[Dict[str, Any]]:
            hits: List[Dict[str, Any]] = []
            for row in rows:
                ent = row.get("entity")
                if not ent:
                    continue
                props = dict(ent)
                ent_id = props.get("id")
                if not ent_id:
                    continue
                hits.append(
                    {
                        "entity_id": ent_id,
                        "name": props.get("name") or props.get("manual_name") or props.get("cluster_label"),
                        "type": props.get("type"),
                        "score": float(row.get("score") or 0.0),
                        "matched": matched,
                    }
                )
            return hits

        def _query_fulltext(sess) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
            cypher = """
CALL db.index.fulltext.queryNodes('tkg_entity_name_v1', $q) YIELD node, score
WITH node AS ent, score
WHERE ent.tenant_id = $tenant
  AND (ent.published IS NULL OR ent.published = true)
  AND ($etype IS NULL OR toLower(ent.type) = toLower($etype))
  AND ($domain IS NULL OR ent.memory_domain = $domain)
  AND ($uids IS NULL OR ANY(u IN coalesce(ent.user_id, []) WHERE u IN $uids))
RETURN ent AS entity, score AS score
ORDER BY score DESC
LIMIT $limit
"""
            rows = sess.run(
                cypher,
                tenant=tenant_id,
                q=q,
                etype=etype,
                domain=memory_domain,
                uids=user_ids_norm,
                limit=bounded_limit,
            )
            return _rows_to_hits(rows, matched="fulltext")

        def _query_contains(sess) -> List[Dict[str, Any]]:  # type: ignore[no-untyped-def]
            cypher = """
MATCH (ent:Entity {tenant_id: $tenant})
WHERE ($etype IS NULL OR toLower(ent.type) = toLower($etype))
  AND (ent.published IS NULL OR ent.published = true)
  AND ($domain IS NULL OR ent.memory_domain = $domain)
  AND ($uids IS NULL OR ANY(u IN coalesce(ent.user_id, []) WHERE u IN $uids))
  AND toLower(coalesce(ent.name, ent.manual_name, ent.cluster_label, "")) CONTAINS toLower($q)
RETURN ent AS entity, 1.0 AS score
ORDER BY coalesce(ent.updated_at, ent.created_at, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""
            rows = sess.run(
                cypher,
                tenant=tenant_id,
                q=q,
                etype=etype,
                domain=memory_domain,
                uids=user_ids_norm,
                limit=bounded_limit,
            )
            return _rows_to_hits(rows, matched="contains")

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                try:
                    hits = _query_fulltext(sess)
                    if hits:
                        return hits
                except Exception:
                    pass
                return _query_contains(sess)
        except Exception:
            return []

    async def query_entities_by_ids(
        self,
        *,
        tenant_id: str,
        entity_ids: List[str],
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        ids = [str(x).strip() for x in (entity_ids or []) if str(x).strip()]
        if not ids:
            return []
        bounded_limit = max(1, min(int(limit), 500))
        user_ids_norm = [str(x).strip() for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ent:Entity {tenant_id: $tenant})
WHERE ent.id IN $ids
  AND (ent.published IS NULL OR ent.published = true)
  AND ($memory_domain IS NULL OR ent.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(u IN coalesce(ent.user_id, []) WHERE u IN $user_ids))
RETURN ent AS entity
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    ids=ids,
                    user_ids=user_ids_norm,
                    memory_domain=domain,
                    limit=bounded_limit,
                )
                hits: List[Dict[str, Any]] = []
                for row in rows:
                    ent = row.get("entity")
                    if not ent:
                        continue
                    props = dict(ent)
                    ent_id = props.get("id")
                    if not ent_id:
                        continue
                    if not props.get("name"):
                        props["name"] = props.get("manual_name") or props.get("cluster_label")
                    hits.append(props)

                by_id: Dict[str, Dict[str, Any]] = {}
                for item in hits:
                    eid = str(item.get("id") or "").strip()
                    if eid:
                        by_id.setdefault(eid, item)
                ordered: List[Dict[str, Any]] = []
                for cid in ids:
                    item = by_id.get(cid)
                    if item and item not in ordered:
                        ordered.append(item)
                return ordered[:bounded_limit]
        except Exception:
            return []

    async def query_entities_overview(
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
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        q = str(query or "").strip().lower() or None
        etype = str(entity_type or "").strip() or None
        start_iso = self._dt_to_iso(mentioned_since)
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ent:Entity {tenant_id: $tenant})
WHERE (ent.published IS NULL OR ent.published = true)
  AND ($type IS NULL OR ent.type = $type)
  AND (
    $q IS NULL OR
    toLower(coalesce(ent.name, '')) CONTAINS $q OR
    toLower(coalesce(ent.manual_name, '')) CONTAINS $q OR
    toLower(coalesce(ent.cluster_label, '')) CONTAINS $q
  )
MATCH (ent)<-[:INVOLVES]-(ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($start IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start >= datetime($start)))
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
WITH ent,
     count(DISTINCT ev) AS mention_count,
     min(ev.t_abs_start) AS first_ts,
     max(ev.t_abs_start) AS last_ts
WHERE mention_count > 0
RETURN ent AS entity, mention_count, first_ts, last_ts
ORDER BY last_ts DESC, mention_count DESC
SKIP $offset
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    type=etype,
                    q=q,
                    start=start_iso,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                    offset=max(0, int(offset)),
                    limit=max(1, int(limit)),
                )
                out: List[Dict[str, Any]] = []
                for row in rows:
                    ent = row.get("entity")
                    if not ent:
                        continue
                    props = dict(ent)
                    out.append(
                        {
                            "entity": props,
                            "mention_count": int(row.get("mention_count") or 0),
                            "first_mentioned": self._dt_to_iso(row.get("first_ts")),
                            "last_mentioned": self._dt_to_iso(row.get("last_ts")),
                        }
                    )
                return out
        except Exception:
            return []

    async def query_entities_overview_count(
        self,
        *,
        tenant_id: str,
        entity_type: Optional[str] = None,
        query: Optional[str] = None,
        mentioned_since: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> int:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return 0
        if not self._driver:
            return 0

        q = str(query or "").strip().lower() or None
        etype = str(entity_type or "").strip() or None
        start_iso = self._dt_to_iso(mentioned_since)
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ent:Entity {tenant_id: $tenant})
WHERE (ent.published IS NULL OR ent.published = true)
  AND ($type IS NULL OR ent.type = $type)
  AND (
    $q IS NULL OR
    toLower(coalesce(ent.name, '')) CONTAINS $q OR
    toLower(coalesce(ent.manual_name, '')) CONTAINS $q OR
    toLower(coalesce(ent.cluster_label, '')) CONTAINS $q
  )
MATCH (ent)<-[:INVOLVES]-(ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($start IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start >= datetime($start)))
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
WITH ent, count(DISTINCT ev) AS mention_count
WHERE mention_count > 0
RETURN count(DISTINCT ent) AS total
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    type=etype,
                    q=q,
                    start=start_iso,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                )
                row = rows.single()
                return int(row.get("total") or 0) if row else 0
        except Exception:
            return 0

    async def query_topics_overview(
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
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        q = str(query or "").strip().lower() or None
        parent = str(parent_path or "").strip() or None
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None
        min_events_val = int(min_events) if min_events is not None else None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND (ev.topic_path IS NOT NULL AND ev.topic_path <> '')
  AND ($q IS NULL OR toLower(ev.topic_path) CONTAINS $q)
  AND ($parent IS NULL OR ev.topic_path STARTS WITH $parent)
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
WITH ev.topic_path AS topic_path,
     count(DISTINCT ev) AS event_count,
     min(ev.t_abs_start) AS first_ts,
     max(ev.t_abs_start) AS last_ts
WHERE ($min_events IS NULL OR event_count >= $min_events)
RETURN topic_path, event_count, first_ts, last_ts
ORDER BY last_ts DESC, event_count DESC
SKIP $offset
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    q=q,
                    parent=parent,
                    min_events=min_events_val,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                    offset=max(0, int(offset)),
                    limit=max(1, int(limit)),
                )
                out: List[Dict[str, Any]] = []
                for row in rows:
                    out.append(
                        {
                            "topic_path": row.get("topic_path"),
                            "event_count": int(row.get("event_count") or 0),
                            "first_mentioned": self._dt_to_iso(row.get("first_ts")),
                            "last_mentioned": self._dt_to_iso(row.get("last_ts")),
                        }
                    )
                return out
        except Exception:
            return []

    async def query_topics_overview_count(
        self,
        *,
        tenant_id: str,
        query: Optional[str] = None,
        parent_path: Optional[str] = None,
        min_events: Optional[int] = None,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> int:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return 0
        if not self._driver:
            return 0

        q = str(query or "").strip().lower() or None
        parent = str(parent_path or "").strip() or None
        user_ids_norm = [str(x) for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None
        min_events_val = int(min_events) if min_events is not None else None

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND (ev.topic_path IS NOT NULL AND ev.topic_path <> '')
  AND ($q IS NULL OR toLower(ev.topic_path) CONTAINS $q)
  AND ($parent IS NULL OR ev.topic_path STARTS WITH $parent)
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
WITH ev.topic_path AS topic_path,
     count(DISTINCT ev) AS event_count
WHERE ($min_events IS NULL OR event_count >= $min_events)
RETURN count(DISTINCT topic_path) AS total
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=str(tenant_id),
                    q=q,
                    parent=parent,
                    min_events=min_events_val,
                    user_ids=list(user_ids_norm) if user_ids_norm else None,
                    memory_domain=memory_domain,
                )
                row = rows.single()
                return int(row.get("total") or 0) if row else 0
        except Exception:
            return 0

    async def query_entity_detail(
        self,
        *,
        tenant_id: str,
        entity_id: str,
    ) -> Dict[str, Any]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {}
        if not self._driver:
            return {}

        eid = str(entity_id or "").strip()
        if not eid:
            return {}

        cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
WHERE (ent.expires_at IS NULL OR ent.expires_at > datetime())
  AND (ent.published IS NULL OR ent.published = true)
RETURN ent AS entity
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, entity_id=eid).single()
                if not row:
                    return {}
                ent = row.get("entity")
                if not ent:
                    return {}
                props = dict(ent)
                for key in ("created_at", "updated_at", "expires_at"):
                    if key in props:
                        props[key] = self._dt_to_iso(props.get(key))
                return props
        except Exception:
            return {}

    async def query_entity_knowledge(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        eid = str(entity_id or "").strip()
        if not eid:
            return []

        cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (k:Knowledge {tenant_id: $tenant})-[:STATED_BY]->(ent)
WHERE (k.expires_at IS NULL OR k.expires_at > datetime())
  AND (k.published IS NULL OR k.published = true)
OPTIONAL MATCH (k)-[:DERIVED_FROM]->(ev:Event {tenant_id: $tenant})
  WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
    AND (ev.published IS NULL OR ev.published = true)
RETURN k AS knowledge,
       collect(DISTINCT ev.id) AS event_ids
ORDER BY coalesce(k.updated_at, k.created_at, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, tenant=tenant_id, entity_id=eid, limit=int(limit))
                out: List[Dict[str, Any]] = []
                for row in rows:
                    k = row.get("knowledge")
                    if not k:
                        continue
                    props = dict(k)
                    for key in ("created_at", "updated_at", "expires_at"):
                        if key in props:
                            props[key] = self._dt_to_iso(props.get(key))
                    props["event_ids"] = list(row.get("event_ids") or [])
                    out.append(props)
                return out
        except Exception:
            return []

    async def query_entity_relations(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        eid = str(entity_id or "").strip()
        if not eid:
            return []

        cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (ent)-[r:CO_OCCURS_WITH]-(other:Entity {tenant_id: $tenant})
WHERE (other.published IS NULL OR other.published = true)
RETURN other AS entity, r AS rel
ORDER BY coalesce(r.weight, 0.0) DESC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, tenant=tenant_id, entity_id=eid, limit=int(limit))
                out: List[Dict[str, Any]] = []
                for row in rows:
                    ent = row.get("entity")
                    if not ent:
                        continue
                    rel = row.get("rel")
                    props = dict(ent)
                    out.append(
                        {
                            "entity_id": props.get("id"),
                            "name": props.get("name") or props.get("manual_name") or props.get("cluster_label"),
                            "type": props.get("type"),
                            "weight": float(rel.get("weight") or 0.0) if hasattr(rel, "get") else 0.0,
                        }
                    )
                return out
        except Exception:
            return []

    async def query_entity_relations_by_events(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        eid = str(entity_id or "").strip()
        if not eid:
            return []

        start_iso = self._dt_to_iso(start)
        end_iso = self._dt_to_iso(end)
        bounded_limit = max(1, min(int(limit), 200))

        cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (ent)<-[:INVOLVES]-(ev:Event {tenant_id: $tenant})-[:INVOLVES]->(other:Entity {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND (other.published IS NULL OR other.published = true)
  AND other.id <> ent.id
  AND ($start IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start >= datetime($start)))
  AND ($end IS NULL OR (ev.t_abs_start IS NOT NULL AND ev.t_abs_start <= datetime($end)))
WITH other,
     collect(DISTINCT ev.id) AS event_ids,
     count(DISTINCT ev) AS strength,
     min(ev.t_abs_start) AS first_ts,
     max(ev.t_abs_start) AS last_ts
RETURN other AS entity,
       event_ids,
       strength,
       first_ts,
       last_ts
ORDER BY strength DESC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    entity_id=eid,
                    start=start_iso,
                    end=end_iso,
                    limit=bounded_limit,
                )
                out: List[Dict[str, Any]] = []
                for row in rows:
                    ent = row.get("entity")
                    if not ent:
                        continue
                    props = dict(ent)
                    event_ids = list(row.get("event_ids") or [])
                    out.append(
                        {
                            "entity_id": props.get("id"),
                            "name": props.get("name") or props.get("manual_name") or props.get("cluster_label"),
                            "type": props.get("type"),
                            "strength": int(row.get("strength") or 0),
                            "first_mentioned": self._dt_to_iso(row.get("first_ts")),
                            "last_mentioned": self._dt_to_iso(row.get("last_ts")),
                            "evidence_event_ids": event_ids[:10],
                        }
                    )
                return out
        except Exception:
            return []

    async def query_time_slices_by_range(
        self,
        *,
        tenant_id: str,
        start_iso: Optional[str],
        end_iso: Optional[str],
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Query TimeSlice nodes that overlap a [start,end] time window (absolute time only)."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        start_raw = str(start_iso or "").strip()
        end_raw = str(end_iso or "").strip()
        if not start_raw and not end_raw:
            return []
        bounded_limit = max(1, min(int(limit), 500))

        cypher = """
MATCH (ts:TimeSlice {tenant_id: $tenant})
WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
  AND (ts.published IS NULL OR ts.published = true)
  AND ($kind IS NULL OR ts.kind = $kind)
  AND ($start IS NULL OR ts.t_abs_end >= datetime($start))
  AND ($end IS NULL OR ts.t_abs_start <= datetime($end))
OPTIONAL MATCH (ts)-[:COVERS_SEGMENT]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ts)-[:COVERS_EVENT]->(ev:Event {tenant_id: $tenant})
  WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
    AND (ev.published IS NULL OR ev.published = true)
WITH ts,
     collect(DISTINCT seg.id) AS segment_ids,
     collect(DISTINCT ev.id) AS event_ids
RETURN ts AS timeslice,
       segment_ids,
       event_ids
ORDER BY ts.t_abs_start ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    kind=kind,
                    start=(start_raw if start_raw else None),
                    end=(end_raw if end_raw else None),
                    limit=int(bounded_limit),
                )
                items: List[Dict[str, Any]] = []
                for row in rows:
                    ts = row.get("timeslice")
                    if not ts:
                        continue
                    props = dict(ts)
                    items.append(
                        {
                            "id": props.get("id"),
                            "kind": props.get("kind"),
                            "t_abs_start": self._dt_to_iso(props.get("t_abs_start")),
                            "t_abs_end": self._dt_to_iso(props.get("t_abs_end")),
                            "t_media_start": props.get("t_media_start"),
                            "t_media_end": props.get("t_media_end"),
                            "time_origin": props.get("time_origin"),
                            "granularity_level": props.get("granularity_level"),
                            "segment_ids": list(row.get("segment_ids") or []),
                            "event_ids": list(row.get("event_ids") or []),
                        }
                    )
                return items
        except Exception:
            return []

    async def query_first_meeting(
        self,
        *,
        tenant_id: str,
        me_id: str,
        other_id: str,
    ) -> Dict[str, Any]:
        """Return the earliest event where two entities co-occur."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {}
        if not self._driver:
            return {}

        # NOTE: upsert_graph_v0 writes INVOLVES edges as (Event)-[:INVOLVES]->(Entity),
        # so we must traverse from Event to Entity here instead of the other way around.
        cypher = """
MATCH (ev:Event {tenant_id: $tenant})
MATCH (ev)-[:INVOLVES]->(me:Entity {tenant_id: $tenant, id: $me_id})
MATCH (ev)-[:INVOLVES]->(other:Entity {tenant_id: $tenant, id: $other_id})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
  WHERE (pl.expires_at IS NULL OR pl.expires_at > datetime())
    AND (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(evd:Evidence {tenant_id: $tenant})
  WHERE (evd.expires_at IS NULL OR evd.expires_at > datetime())
    AND (evd.published IS NULL OR evd.published = true)
WITH ev, pl, collect(DISTINCT evd.id) AS evidence_ids
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
LIMIT 1
RETURN ev AS event,
       pl.id AS place_id,
       evidence_ids
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    cypher,
                    tenant=tenant_id,
                    me_id=me_id,
                    other_id=other_id,
                ).single()
                if not row:
                    return {}
                ev = row.get("event")
                if not ev:
                    return {}
                props = dict(ev)
                return {
                    "event_id": props.get("id"),
                    "summary": props.get("summary"),
                    "t_abs_start": self._dt_to_iso(props.get("t_abs_start")),
                    "place_id": row.get("place_id"),
                    "evidence_ids": list(row.get("evidence_ids") or []),
                }
        except Exception as exc:
            self._logger.error(
                "neo4j.query_first_meeting.failed",
                extra={
                    "event": "neo4j.query_first_meeting",
                    "tenant_id": tenant_id,
                    "status": "error",
                    "reason": str(exc),
                },
                exc_info=True,
            )
            return {}

    async def query_places(
        self,
        *,
        tenant_id: str,
        name: Optional[str] = None,
        segment_id: Optional[str] = None,
        covers_timeslice: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        cypher = """
MATCH (pl:Place {tenant_id: $tenant})
WHERE (pl.expires_at IS NULL OR pl.expires_at > datetime())
  AND (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev:Event {tenant_id: $tenant})-[:OCCURS_AT]->(pl)
  WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
    AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_EVENT]->(ev)
  WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
    AND (ts.published IS NULL OR ts.published = true)
WITH pl,
     collect(DISTINCT seg.id) AS segment_ids,
     collect(DISTINCT seg.source_id) AS source_ids,
     collect(DISTINCT ts.id) AS timeslice_ids
WHERE ($name IS NULL OR toLower(pl.name) CONTAINS toLower($name))
  AND ($segment_id IS NULL OR ANY(x IN segment_ids WHERE x = $segment_id))
  AND ($covers_timeslice IS NULL OR ANY(x IN timeslice_ids WHERE x = $covers_timeslice))
RETURN pl AS place,
       segment_ids,
       source_ids,
       timeslice_ids
ORDER BY pl.name ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    name=name,
                    segment_id=segment_id,
                    covers_timeslice=covers_timeslice,
                    limit=int(limit),
                )
                rows_list = list(rows)
                self._logger.info(f"query_places: tenant={tenant_id}, params={name, segment_id, covers_timeslice}, rows={len(rows_list)}")
                items: List[Dict[str, Any]] = []
                for row in rows_list:
                    place = row.get("place")
                    seg_ids = [s for s in (row.get("segment_ids") or []) if s is not None]
                    src_ids = [s for s in (row.get("source_ids") or []) if s is not None]
                    ts_ids = [s for s in (row.get("timeslice_ids") or []) if s is not None]
                    data: Dict[str, Any] = {
                        "segment_ids": list(seg_ids),
                        "source_ids": src_ids,
                        "timeslice_ids": ts_ids,
                    }
                    if place:
                        props = dict(place)
                        data.update(
                            {
                                "id": props.get("id"),
                                "name": props.get("name"),
                                "geo_location": props.get("geo_location"),
                                "area_type": props.get("area_type"),
                            }
                        )
                    items.append(data)
                return items
        except Exception:
            return []

    async def query_event_detail(
        self,
        *,
        tenant_id: str,
        event_id: str,
    ) -> Dict[str, Any]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {}
        if not self._driver:
            return {}

        cypher = """
MATCH (ev:Event {id: $event_id, tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ev)-[:INVOLVES]->(ent:Entity {tenant_id: $tenant})
  WHERE (ent.published IS NULL OR ent.published = true)
OPTIONAL MATCH (ev)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
  WHERE (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(evd:Evidence {tenant_id: $tenant})
  WHERE (evd.published IS NULL OR evd.published = true)
OPTIONAL MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_EVENT]->(ev)
  WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
    AND (ts.published IS NULL OR ts.published = true)
OPTIONAL MATCH (ev)-[rel:NEXT_EVENT|CAUSES]->(ev2:Event {tenant_id: $tenant})
  WHERE (ev2.published IS NULL OR ev2.published = true)
RETURN ev AS event,
       collect(DISTINCT seg) AS segments,
       collect(DISTINCT ent.id) AS entity_ids,
       collect(DISTINCT pl) AS places,
       collect(DISTINCT evd.id) AS evidence_ids,
       collect(DISTINCT ts.id) AS timeslice_ids,
       collect(DISTINCT rel) AS rels
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, event_id=event_id).single()
                if not row:
                    return {}
                ev = row.get("event")
                if not ev:
                    return {}
                data: Dict[str, Any] = dict(ev)
                segs = []
                for seg in row.get("segments") or []:
                    if seg:
                        props = dict(seg)
                        segs.append(
                            {
                                "id": props.get("id"),
                                "source_id": props.get("source_id"),
                                "t_media_start": props.get("t_media_start"),
                                "t_media_end": props.get("t_media_end"),
                            }
                        )
                places = []
                for place in row.get("places") or []:
                    if place:
                        props = dict(place)
                        places.append(
                            {
                                "id": props.get("id"),
                                "name": props.get("name"),
                                "area_type": props.get("area_type"),
                            }
                        )
                data.update(
                    {
                        "segments": segs,
                        "entity_ids": list(row.get("entity_ids") or []),
                        "places": places,
                        "evidence_ids": list(row.get("evidence_ids") or []),
                        "timeslice_ids": list(row.get("timeslice_ids") or []),
                        "relations": [
                            {
                                "type": rel.type,
                                "target_event_id": rel.end_node.get("id") if hasattr(rel, "end_node") else None,  # type: ignore[attr-defined]
                                "layer": rel.get("layer") if hasattr(rel, "get") else None,  # type: ignore[attr-defined]
                                "status": rel.get("status") if hasattr(rel, "get") else None,  # type: ignore[attr-defined]
                                "kind": rel.get("kind") if hasattr(rel, "get") else None,  # type: ignore[attr-defined]
                            }
                            for rel in (row.get("rels") or [])
                        ],
                    }
                )
                return data
        except Exception:
            return {}

    async def query_event_evidence(
        self,
        *,
        tenant_id: str,
        event_id: str,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a structured evidence chain for a given event."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {}
        if not self._driver:
            return {}
        user_ids_norm = [str(x).strip() for x in (user_ids or []) if str(x).strip()]
        if not user_ids_norm:
            user_ids_norm = None
        memory_domain = str(memory_domain or "").strip() or None

        cypher = """
MATCH (ev:Event {id: $event_id, tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND ($memory_domain IS NULL OR ev.memory_domain = $memory_domain)
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
OPTIONAL MATCH (ev)-[:INVOLVES]->(ent:Entity {tenant_id: $tenant})
  WHERE (ent.published IS NULL OR ent.published = true)
OPTIONAL MATCH (ev)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
  WHERE (pl.expires_at IS NULL OR pl.expires_at > datetime())
    AND (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(evd:Evidence {tenant_id: $tenant})
  WHERE (evd.expires_at IS NULL OR evd.expires_at > datetime())
    AND (evd.published IS NULL OR evd.published = true)
OPTIONAL MATCH (ev)-[:SUPPORTED_BY]->(utt:UtteranceEvidence {tenant_id: $tenant})
  WHERE (utt.expires_at IS NULL OR utt.expires_at > datetime())
    AND (utt.published IS NULL OR utt.published = true)
OPTIONAL MATCH (utt)-[:SUPPORTED_BY]->(ve:Evidence {tenant_id: $tenant})
  WHERE (ve.expires_at IS NULL OR ve.expires_at > datetime())
    AND (ve.published IS NULL OR ve.published = true)
OPTIONAL MATCH (utt)-[:SPOKEN_BY]->(utt_ent:Entity {tenant_id: $tenant})
  WHERE (utt_ent.published IS NULL OR utt_ent.published = true)
OPTIONAL MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_EVENT]->(ev)
  WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
    AND (ts.published IS NULL OR ts.published = true)
OPTIONAL MATCH (k:Knowledge {tenant_id: $tenant})-[:DERIVED_FROM]->(ev)
  WHERE (k.expires_at IS NULL OR k.expires_at > datetime())
    AND (k.published IS NULL OR k.published = true)
OPTIONAL MATCH (k)-[:STATED_BY]->(kent:Entity {tenant_id: $tenant})
  WHERE (kent.published IS NULL OR kent.published = true)
OPTIONAL MATCH (k)-[:SUPPORTED_BY]->(kutt:UtteranceEvidence {tenant_id: $tenant})
  WHERE (kutt.published IS NULL OR kutt.published = true)
RETURN ev AS event,
       collect(DISTINCT ent) AS entities,
       collect(DISTINCT pl) AS places,
       collect(DISTINCT evd) AS evidences,
       collect(DISTINCT utt) AS utterances,
       collect(DISTINCT {utterance_id: utt.id, entity_id: utt_ent.id}) AS utterance_speakers,
       collect(DISTINCT ve) AS voice_evidences,
       collect(DISTINCT ts) AS timeslices,
       collect(DISTINCT k) AS knowledge,
       collect(DISTINCT kent) AS knowledge_entities,
       collect(DISTINCT kutt) AS knowledge_utterances
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    cypher,
                    tenant=tenant_id,
                    event_id=event_id,
                    user_ids=(list(user_ids_norm) if user_ids_norm else None),
                    memory_domain=memory_domain,
                ).single()
                if not row:
                    return {}
                ev = row.get("event")
                if not ev:
                    return {}
                event_props = dict(ev)
                event_props["t_abs_start"] = self._dt_to_iso(event_props.get("t_abs_start"))
                event_props["t_abs_end"] = self._dt_to_iso(event_props.get("t_abs_end"))

                def _serialize_nodes(nodes: Iterable[Any]) -> List[Dict[str, Any]]:
                    out: List[Dict[str, Any]] = []
                    for n in nodes or []:
                        if not n:
                            continue
                        try:
                            props = dict(n)
                        except Exception:
                            continue
                        # Decode JSON-encoded nested properties written as `*_json` for Neo4j compatibility.
                        try:
                            for k in list(props.keys()):
                                if not k.endswith("_json"):
                                    continue
                                raw = props.get(k)
                                if not isinstance(raw, str) or not raw:
                                    continue
                                base = k[: -len("_json")]
                                if base in props:
                                    continue
                                try:
                                    import json as _json

                                    props[base] = _json.loads(raw)
                                except Exception:
                                    props[base] = raw
                            # Hide internal fields from API payloads.
                            for k in [x for x in list(props.keys()) if x.endswith("_json")]:
                                props.pop(k, None)
                        except Exception:
                            pass
                        # Normalize common datetime fields if present
                        for key in ("t_abs_start", "t_abs_end", "created_at", "expires_at"):
                            if key in props:
                                props[key] = self._dt_to_iso(props.get(key))
                        out.append(props)
                    return out

                entities = _serialize_nodes(row.get("entities") or [])
                places = _serialize_nodes(row.get("places") or [])
                evidences = _serialize_nodes(row.get("evidences") or [])
                voice_evidences = _serialize_nodes(row.get("voice_evidences") or [])
                utterances = _serialize_nodes(row.get("utterances") or [])
                timeslices = _serialize_nodes(row.get("timeslices") or [])
                knowledge = _serialize_nodes(row.get("knowledge") or [])
                utterance_speakers_raw = row.get("utterance_speakers") or []
                utterance_speakers: List[Dict[str, str]] = []
                for item in utterance_speakers_raw:
                    try:
                        if not isinstance(item, dict):
                            continue
                        utt_id = str(item.get("utterance_id") or "")
                        ent_id = str(item.get("entity_id") or "")
                        if not utt_id or not ent_id:
                            continue
                        utterance_speakers.append({"utterance_id": utt_id, "entity_id": ent_id})
                    except Exception:
                        continue
                # merge stated-by entities and knowledge utterances into the same lists (dedup by id)
                k_entities = _serialize_nodes(row.get("knowledge_entities") or [])
                k_utts = _serialize_nodes(row.get("knowledge_utterances") or [])

                def _dedup_by_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    seen: set[str] = set()
                    out: List[Dict[str, Any]] = []
                    for it in items:
                        iid = str(it.get("id") or "")
                        if not iid or iid in seen:
                            continue
                        seen.add(iid)
                        out.append(it)
                    return out

                utterances = _dedup_by_id(list(utterances) + list(k_utts))
                entities = _dedup_by_id(list(entities) + list(k_entities))
                evidences = _dedup_by_id(list(evidences) + list(voice_evidences))

                return {
                    "event": event_props,
                    "entities": entities,
                    "places": places,
                    "evidences": evidences,
                    "utterances": utterances,
                    "utterance_speakers": utterance_speakers,
                    "timeslices": timeslices,
                    "knowledge": knowledge,
                }
        except Exception as exc:
            self._logger.error(
                "neo4j.query_event_evidence.failed",
                extra={
                    "event": "neo4j.query_event_evidence",
                    "tenant_id": tenant_id,
                    "status": "error",
                    "reason": str(exc),
                },
                exc_info=True,
            )
            return {}

    async def query_entity_evidences(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        subtype: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List evidences that BELONG to an entity (best-effort, limited).

        Notes:
        - This is for UI sampling (e.g., face thumbnails). Do not use for bulk export.
        - `source_id` filters by Evidence.source_id to keep results scoped to one run/video.
        """
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []
        eid = (entity_id or "").strip()
        if not eid:
            return []

        cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (evd:Evidence {tenant_id: $tenant})-[:BELONGS_TO_ENTITY]->(ent)
WHERE (evd.expires_at IS NULL OR evd.expires_at > datetime())
  AND (ent.published IS NULL OR ent.published = true)
  AND (evd.published IS NULL OR evd.published = true)
  AND ($subtype IS NULL OR evd.subtype = $subtype)
  AND ($source_id IS NULL OR evd.source_id = $source_id)
OPTIONAL MATCH (seg:MediaSegment {tenant_id: $tenant})-[:CONTAINS_EVIDENCE]->(evd)
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
RETURN evd AS evidence,
       seg.id AS segment_id,
       seg.t_media_start AS segment_t_media_start,
       seg.t_media_end AS segment_t_media_end
ORDER BY coalesce(evd.confidence, 0.0) DESC, coalesce(segment_t_media_start, 0.0) ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    entity_id=eid,
                    subtype=subtype,
                    source_id=source_id,
                    limit=int(limit),
                )
                items: List[Dict[str, Any]] = []
                for row in rows:
                    evd = row.get("evidence")
                    if not evd:
                        continue
                    try:
                        props = dict(evd)
                    except Exception:
                        continue
                    # Decode JSON-encoded nested properties written as `*_json` for Neo4j compatibility.
                    try:
                        for k in list(props.keys()):
                            if not k.endswith("_json"):
                                continue
                            raw = props.get(k)
                            if not isinstance(raw, str) or not raw:
                                continue
                            base = k[: -len("_json")]
                            if base in props:
                                continue
                            try:
                                import json as _json

                                props[base] = _json.loads(raw)
                            except Exception:
                                props[base] = raw
                        # Hide internal fields from API payloads.
                        for k in [x for x in list(props.keys()) if x.endswith("_json")]:
                            props.pop(k, None)
                    except Exception:
                        pass
                    # Normalize datetime fields if present
                    for key in ("created_at", "updated_at", "expires_at"):
                        if key in props:
                            props[key] = self._dt_to_iso(props.get(key))

                    props["segment_id"] = row.get("segment_id")
                    props["segment_t_media_start"] = row.get("segment_t_media_start")
                    props["segment_t_media_end"] = row.get("segment_t_media_end")
                    items.append(props)
                return items
        except Exception:
            return []

    async def query_place_detail(
        self,
        *,
        tenant_id: str,
        place_id: str,
    ) -> Dict[str, Any]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {}
        if not self._driver:
            return {}

        cypher = """
MATCH (pl:Place {id: $place_id, tenant_id: $tenant})
WHERE (pl.expires_at IS NULL OR pl.expires_at > datetime())
  AND (pl.published IS NULL OR pl.published = true)
OPTIONAL MATCH (ev:Event {tenant_id: $tenant})-[:OCCURS_AT]->(pl)
  WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
    AND (ev.published IS NULL OR ev.published = true)
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_EVENT]->(ev)
  WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
    AND (ts.published IS NULL OR ts.published = true)
RETURN pl AS place,
       collect(DISTINCT ev.id) AS event_ids,
       collect(DISTINCT seg) AS segments,
       collect(DISTINCT ts.id) AS timeslice_ids
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, place_id=place_id).single()
                if not row:
                    return {}
                place = row.get("place")
                if not place:
                    return {}
                data = dict(place)
                segs = []
                for seg in row.get("segments") or []:
                    if seg:
                        props = dict(seg)
                        segs.append(
                            {
                                "id": props.get("id"),
                                "source_id": props.get("source_id"),
                                "t_media_start": props.get("t_media_start"),
                                "t_media_end": props.get("t_media_end"),
                            }
                        )
                data.update(
                    {
                        "event_ids": list(row.get("event_ids") or []),
                        "segments": segs,
                        "timeslice_ids": list(row.get("timeslice_ids") or []),
                    }
                )
                return data
        except Exception:
            return {}

    async def query_entity_timeline(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []
        eid = (entity_id or "").strip()
        if not eid:
            return []
        try:
            lim = int(limit)
        except Exception:
            lim = 200
        if lim <= 0:
            return []

        cypher_evidence = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (evd:Evidence {tenant_id: $tenant})-[:BELONGS_TO_ENTITY]->(ent)
WHERE (ent.published IS NULL OR ent.published = true)
  AND (evd.expires_at IS NULL OR evd.expires_at > datetime())
  AND (evd.published IS NULL OR evd.published = true)
OPTIONAL MATCH (seg:MediaSegment {tenant_id: $tenant})-[:CONTAINS_EVIDENCE]->(evd)
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
RETURN seg.id AS segment_id,
       coalesce(seg.source_id, evd.source_id) AS source_id,
       seg.t_media_start AS t_media_start,
       seg.t_media_end AS t_media_end,
       evd.id AS evidence_id,
       evd.subtype AS evidence_subtype,
       evd.confidence AS confidence,
       evd.text AS text,
       evd.offset_in_segment AS offset_in_segment,
       evd.utterance_id AS utterance_id
ORDER BY coalesce(seg.t_media_start, 0.0) ASC, coalesce(evd.confidence, 0.0) DESC
LIMIT $limit
"""

        cypher_utterance = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (utt:UtteranceEvidence {tenant_id: $tenant})-[:SPOKEN_BY]->(ent)
WHERE (ent.published IS NULL OR ent.published = true)
  AND (utt.expires_at IS NULL OR utt.expires_at > datetime())
  AND (utt.published IS NULL OR utt.published = true)
OPTIONAL MATCH (seg:MediaSegment {tenant_id: $tenant, id: utt.segment_id})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
RETURN coalesce(seg.id, utt.segment_id) AS segment_id,
       seg.source_id AS source_id,
       coalesce(seg.t_media_start, utt.t_media_start) AS t_media_start,
       coalesce(seg.t_media_end, utt.t_media_end) AS t_media_end,
       utt.id AS utterance_id,
       utt.raw_text AS raw_text,
       utt.speaker_track_id AS speaker_track_id,
       utt.asr_model_version AS asr_model_version,
       utt.lang AS lang
ORDER BY coalesce(seg.t_media_start, utt.t_media_start, 0.0) ASC
LIMIT $limit
"""

        def _float_or_none(val: Any) -> Optional[float]:
            try:
                if val is None:
                    return None
                return float(val)
            except Exception:
                return None

        def _sort_key(item: Dict[str, Any]) -> tuple:
            t_start = _float_or_none(item.get("t_media_start"))
            conf = _float_or_none(item.get("confidence")) or 0.0
            return (t_start is None, t_start or 0.0, -conf)

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                items: List[Dict[str, Any]] = []
                ev_rows = sess.run(cypher_evidence, tenant=tenant_id, entity_id=eid, limit=lim)
                for row in ev_rows:
                    items.append(
                        {
                            "segment_id": row.get("segment_id"),
                            "source_id": row.get("source_id"),
                            "t_media_start": _float_or_none(row.get("t_media_start")),
                            "t_media_end": _float_or_none(row.get("t_media_end")),
                            "evidence_id": row.get("evidence_id"),
                            "evidence_subtype": row.get("evidence_subtype"),
                            "confidence": _float_or_none(row.get("confidence")),
                            "text": row.get("text"),
                            "offset_in_segment": _float_or_none(row.get("offset_in_segment")),
                            "utterance_id": row.get("utterance_id"),
                            "kind": "evidence",
                        }
                    )

                utt_rows = sess.run(cypher_utterance, tenant=tenant_id, entity_id=eid, limit=lim)
                for row in utt_rows:
                    t_start = _float_or_none(row.get("t_media_start"))
                    t_end = _float_or_none(row.get("t_media_end"))
                    utt_id = row.get("utterance_id")
                    items.append(
                        {
                            "segment_id": row.get("segment_id"),
                            "source_id": row.get("source_id"),
                            "t_media_start": t_start,
                            "t_media_end": t_end,
                            "evidence_id": utt_id,
                            "utterance_id": utt_id,
                            "text": row.get("raw_text"),
                            "speaker_track_id": row.get("speaker_track_id"),
                            "asr_model_version": row.get("asr_model_version"),
                            "lang": row.get("lang"),
                            "evidence_subtype": "utterance",
                            "kind": "utterance",
                        }
                    )

            items.sort(key=_sort_key)
            return items[:lim]
        except Exception as exc:
            self._logger.error(
                "neo4j.query_entity_timeline.failed",
                extra={
                    "event": "neo4j.query_entity_timeline",
                    "tenant_id": tenant_id,
                    "entity_id": eid,
                    "status": "error",
                    "reason": str(exc),
                },
                exc_info=True,
            )
            return []

    async def build_event_relations(
        self,
        *,
        tenant_id: str,
        source_id: Optional[str] = None,
        place_id: Optional[str] = None,
        limit: int = 1000,
        create_causes: bool = True,
    ) -> Dict[str, int]:
        """Generate NEXT_EVENT (observed) and optional CAUSES (candidate) edges.

        - Orders events by t_abs_start; links adjacent events with NEXT_EVENT(kind="observed", layer="fact").
        - Adds CAUSES(status="candidate", layer="hypothesis") when adjacent events share the same place_id.
        """

        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"next_event": 0, "causes": 0}
        if not self._driver:
            return {"next_event": 0, "causes": 0}

        cypher_fetch = """
MATCH (ev:Event {tenant_id: $tenant})
OPTIONAL MATCH (ev)-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant})
OPTIONAL MATCH (ev)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
WITH ev, seg, pl
WHERE (ev.published IS NULL OR ev.published = true)
  AND ($source_id IS NULL OR (seg IS NOT NULL AND seg.source_id = $source_id))
  AND ($place_id IS NULL OR (pl IS NOT NULL AND pl.id = $place_id))
RETURN ev.id AS id,
       ev.t_abs_start AS t_abs_start,
       ev.t_abs_end AS t_abs_end,
       seg.source_id AS source_id,
       pl.id AS place_id
ORDER BY ev.t_abs_start ASC
LIMIT $limit
"""

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher_fetch,
                    tenant=tenant_id,
                    source_id=source_id,
                    place_id=place_id,
                    limit=int(limit),
                )
                events: List[Dict[str, Any]] = []
                for row in rows:
                    rid = row.get("id")
                    ts = row.get("t_abs_start")
                    if rid is None or ts is None:
                        continue
                    events.append(
                        {
                            "id": rid,
                            "t_abs_start": ts,
                            "t_abs_end": row.get("t_abs_end"),
                            "source_id": row.get("source_id"),
                            "place_id": row.get("place_id"),
                        }
                    )

                if len(events) < 2:
                    return {"next_event": 0, "causes": 0}

                edges: List[Dict[str, Any]] = []
                total_causes = 0
                for i in range(len(events) - 1):
                    a, b = events[i], events[i + 1]
                    edges.append(
                        {
                            "src_id": a["id"],
                            "dst_id": b["id"],
                            "tenant_id": tenant_id,
                            "rel_type": "NEXT_EVENT",
                            "kind": "observed",
                            "layer": "fact",
                            "confidence": 1.0,
                        }
                    )
                    if create_causes and a.get("place_id") and a.get("place_id") == b.get("place_id"):
                        total_causes += 1
                        edges.append(
                            {
                                "src_id": a["id"],
                                "dst_id": b["id"],
                                "tenant_id": tenant_id,
                                "rel_type": "CAUSES",
                                "status": "candidate",
                                "layer": "hypothesis",
                                "confidence": 0.3,
                                "source": "rule_adjacent_same_place",
                            }
                        )

                grouped: Dict[str, List[Dict[str, Any]]] = {}
                for e in edges:
                    grouped.setdefault(e["rel_type"], []).append(e)

                for rel, batch in grouped.items():
                    cypher = (
                        "UNWIND $edges AS e "
                        "MATCH (s:Event {id: e.src_id, tenant_id: e.tenant_id}) "
                        "MATCH (d:Event {id: e.dst_id, tenant_id: e.tenant_id}) "
                        f"MERGE (s)-[r:{rel}]->(d) "
                        "SET r.tenant_id = e.tenant_id "
                        "SET r.kind = coalesce(e.kind, r.kind) "
                        "SET r.layer = coalesce(e.layer, r.layer) "
                        "SET r.status = coalesce(e.status, r.status) "
                        "SET r.confidence = coalesce(e.confidence, r.confidence) "
                        "SET r.source = coalesce(e.source, r.source) "
                    )
                    sess.run(cypher, edges=batch)

                return {"next_event": len(grouped.get("NEXT_EVENT", [])), "causes": total_causes}
        except Exception as exc:
            try:
                self._logger.error(
                    "neo4j.build_event_relations.error",
                    extra={"event": "neo4j.build_event_relations", "status": "error", "reason": str(exc)},
                    exc_info=True,
                )
            except Exception:
                pass
            return {"next_event": 0, "causes": 0}
    async def query_time_slices(
        self,
        *,
        tenant_id: str,
        kind: Optional[str] = None,
        covers_segment: Optional[str] = None,
        covers_event: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return []
        if not self._driver:
            return []

        cypher = """
MATCH (ts:TimeSlice {tenant_id: $tenant})
WHERE (ts.expires_at IS NULL OR ts.expires_at > datetime())
  AND (ts.published IS NULL OR ts.published = true)
OPTIONAL MATCH (ts)-[:COVERS_SEGMENT]->(seg:MediaSegment {tenant_id: $tenant})
  WHERE (seg.expires_at IS NULL OR seg.expires_at > datetime())
    AND (seg.published IS NULL OR seg.published = true)
OPTIONAL MATCH (ts)-[:COVERS_EVENT]->(ev:Event {tenant_id: $tenant})
  WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
    AND (ev.published IS NULL OR ev.published = true)
WITH ts,
     collect(DISTINCT seg.id) AS segment_ids,
     collect(DISTINCT ev.id) AS event_ids
WHERE ($kind IS NULL OR ts.kind = $kind)
  AND ($covers_segment IS NULL OR ANY(x IN segment_ids WHERE x = $covers_segment))
  AND ($covers_event IS NULL OR ANY(x IN event_ids WHERE x = $covers_event))
RETURN ts AS timeslice,
       segment_ids,
       event_ids
ORDER BY ts.t_abs_start ASC
LIMIT $limit
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    kind=kind,
                    covers_segment=covers_segment,
                    covers_event=covers_event,
                    limit=int(limit),
                )
                items: List[Dict[str, Any]] = []
                for row in rows:
                    ts = row.get("timeslice")
                    if not ts:
                        continue
                    props = dict(ts)
                    items.append(
                        {
                            "id": props.get("id"),
                            "kind": props.get("kind"),
                            "t_abs_start": self._dt_to_iso(props.get("t_abs_start")),
                            "t_abs_end": self._dt_to_iso(props.get("t_abs_end")),
                            "t_media_start": props.get("t_media_start"),
                            "t_media_end": props.get("t_media_end"),
                            "time_origin": props.get("time_origin"),
                            "granularity_level": props.get("granularity_level"),
                            "segment_ids": list(row.get("segment_ids") or []),
                            "event_ids": list(row.get("event_ids") or []),
                        }
                    )
                return items
        except Exception:
            return []

    async def build_time_slices_from_segments(
        self,
        *,
        tenant_id: str,
        window_seconds: float = 3600.0,
        source_id: Optional[str] = None,
        modality: Optional[str] = None,
        modes: Optional[List[str]] = None,  # ["media_window","day","hour"]
    ) -> Dict[str, int]:
        """Create TimeSlice nodes by bucketing MediaSegments along media time and link via COVERS_SEGMENT."""

        import math
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"timeslices": 0, "edges": 0}
        if not self._driver:
            return {"timeslices": 0, "edges": 0}

        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                cypher = (
                    "MATCH (s:MediaSegment {tenant_id: $tenant}) "
                    "WHERE ($source_id IS NULL OR s.source_id = $source_id) "
                    "AND ($modality IS NULL OR s.modality = $modality) "
                    "RETURN s.id AS id, s.source_id AS source_id, s.t_media_start AS t_start, s.t_media_end AS t_end"
                )
                rows = sess.run(
                    cypher,
                    tenant=tenant_id,
                    source_id=source_id,
                    modality=modality,
                )
                modes = modes or ["media_window", "day", "hour"]
                buckets: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
                covers: List[Dict[str, Any]] = []
                for row in rows:
                    seg_id = row.get("id")
                    src = row.get("source_id") or "unknown"
                    t_start = float(row.get("t_start") or 0.0)
                    t_end = float(row.get("t_end") or 0.0)
                    if t_end <= t_start:
                        continue
                    for mode in modes:
                        if mode == "media_window":
                            idx = int(math.floor(t_start / max(1e-6, window_seconds)))
                            bucket_start = idx * window_seconds
                            bucket_end = bucket_start + window_seconds
                            key = (mode, src, idx)
                            if key not in buckets:
                                buckets[key] = {
                                    "id": f"{src}#ts_media_{idx}",
                                    "tenant_id": tenant_id,
                                    "kind": "media_window",
                                    "t_media_start": bucket_start,
                                    "t_media_end": bucket_end,
                                    "time_origin": "media",
                                    "granularity_level": int(window_seconds),
                                }
                            ts_id = buckets[key]["id"]
                        elif mode == "hour":
                            hour_idx = int(math.floor(t_start / 3600.0))
                            bucket_start = hour_idx * 3600.0
                            bucket_end = bucket_start + 3600.0
                            key = (mode, src, hour_idx)
                            if key not in buckets:
                                buckets[key] = {
                                    "id": f"{src}#ts_hour_{hour_idx}",
                                    "tenant_id": tenant_id,
                                    "kind": "media_hour",
                                    "t_media_start": bucket_start,
                                    "t_media_end": bucket_end,
                                    "time_origin": "media",
                                    "granularity_level": 3600,
                                }
                            ts_id = buckets[key]["id"]
                        elif mode == "day":
                            day_idx = int(math.floor(t_start / 86400.0))
                            bucket_start = day_idx * 86400.0
                            bucket_end = bucket_start + 86400.0
                            key = (mode, src, day_idx)
                            if key not in buckets:
                                buckets[key] = {
                                    "id": f"{src}#ts_day_{day_idx}",
                                    "tenant_id": tenant_id,
                                    "kind": "media_day",
                                    "t_media_start": bucket_start,
                                    "t_media_end": bucket_end,
                                    "time_origin": "media",
                                    "granularity_level": 86400,
                                }
                            ts_id = buckets[key]["id"]
                        else:
                            continue
                        covers.append(
                            {
                                "src_id": ts_id,
                                "dst_id": seg_id,
                                "rel_type": "COVERS_SEGMENT",
                                "tenant_id": tenant_id,
                                "src_type": "TimeSlice",
                                "dst_type": "MediaSegment",
                                "layer": "fact",
                            }
                        )

                ts_models = [GraphTimeSlice(**ts) for ts in buckets.values()]
                edge_models = [GraphEdge(**edge) for edge in covers]
                if ts_models or edge_models:
                    await self.upsert_graph_v0(
                        segments=[],
                        evidences=[],
                        utterances=[],
                        entities=[],
                        events=[],
                        places=[],
                        time_slices=ts_models,
                        regions=[],
                        states=[],
                        knowledge=[],
                        edges=edge_models,
                    )
                return {"timeslices": len(ts_models), "edges": len(edge_models)}
        except Exception as exc:
            try:
                self._logger.error(
                    "neo4j.build_time_slices.error",
                    extra={"event": "neo4j.build_time_slices", "status": "error", "reason": str(exc)},
                    exc_info=True,
                )
            except Exception:
                pass
            return {"timeslices": 0, "edges": 0}

    async def build_cooccurs_from_timeslices(
        self,
        *,
        tenant_id: str,
        min_weight: float = 1.0,
    ) -> Dict[str, int]:
        """Aggregate CO_OCCURS_WITH weights based on TimeSlice coverage."""

        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"co_occurs": 0}
        if not self._driver:
            return {"co_occurs": 0}

        cypher = """
MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_SEGMENT]->(seg:MediaSegment {tenant_id: $tenant})
MATCH (seg)-[:CONTAINS_EVIDENCE]->(ev:Evidence {tenant_id: $tenant})-[:BELONGS_TO_ENTITY]->(ent:Entity {tenant_id: $tenant})
WITH ts, collect(DISTINCT ent.id) AS ents
UNWIND ents AS a
UNWIND ents AS b
WITH a, b
WHERE a < b
WITH a, b, count(*) AS w
WHERE w >= $min_weight
MATCH (ea:Entity {id: a, tenant_id: $tenant})
MATCH (eb:Entity {id: b, tenant_id: $tenant})
MERGE (ea)-[r:CO_OCCURS_WITH]->(eb)
SET r.tenant_id = $tenant
SET r.weight = coalesce(r.weight, 0) + w
SET r.layer = coalesce(r.layer, 'semantic')
SET r.kind = coalesce(r.kind, 'timeslice')
RETURN count(*) AS updated
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, min_weight=float(min_weight)).single()
                val = row.get("updated") if row else 0
                return {"co_occurs": int(val or 0)}
        except Exception as exc:
            self._logger.error(
                "neo4j.build_cooccurs.error",
                extra={"event": "neo4j.build_cooccurs", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def build_cooccurs_from_events(
        self,
        *,
        tenant_id: str,
        min_weight: float = 1.0,
    ) -> Dict[str, int]:
        """Aggregate CO_OCCURS_WITH weights based on Event co-involvement."""

        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"co_occurs": 0}
        if not self._driver:
            return {"co_occurs": 0}

        cypher = """
MATCH (ev:Event {tenant_id: $tenant})-[:INVOLVES]->(a:Entity {tenant_id: $tenant})
MATCH (ev)-[:INVOLVES]->(b:Entity {tenant_id: $tenant})
WHERE a.id < b.id
  AND (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
WITH a, b,
     count(DISTINCT ev) AS w,
     min(ev.t_abs_start) AS first_ts,
     max(ev.t_abs_start) AS last_ts
WHERE w >= $min_weight
MERGE (a)-[r:CO_OCCURS_WITH]->(b)
SET r.tenant_id = $tenant
SET r.weight = w
SET r.layer = coalesce(r.layer, 'semantic')
SET r.kind = 'event'
SET r.first_seen_at = coalesce(r.first_seen_at, first_ts)
SET r.last_seen_at = last_ts
RETURN count(*) AS updated
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, min_weight=float(min_weight)).single()
                val = row.get("updated") if row else 0
                return {"co_occurs": int(val or 0)}
        except Exception as exc:
            self._logger.error(
                "neo4j.build_cooccurs.event.error",
                extra={"event": "neo4j.build_cooccurs.event", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def build_first_meetings(
        self,
        *,
        tenant_id: str,
        limit: int = 5000,
    ) -> Dict[str, int]:
        """Create FIRST_MEET edges using earliest co-involved events for each entity pair."""

        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"first_meet": 0}
        if not self._driver:
            return {"first_meet": 0}

        cypher = """
// For each entity pair (a,b) that co-occur in an Event, find the earliest such Event and store as FIRST_MEET.
MATCH (ev:Event {tenant_id: $tenant})-[:INVOLVES]->(a:Entity {tenant_id: $tenant})
MATCH (ev)-[:INVOLVES]->(b:Entity {tenant_id: $tenant})
WHERE a.id < b.id
  AND (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
WITH a, b, ev
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
WITH a, b, collect(ev)[0] AS first
WHERE first IS NOT NULL
WITH a, b, first
OPTIONAL MATCH (first)-[:OCCURS_AT]->(pl:Place {tenant_id: $tenant})
  WHERE (pl.expires_at IS NULL OR pl.expires_at > datetime())
    AND (pl.published IS NULL OR pl.published = true)
WITH a, b, first, pl
LIMIT $limit
MERGE (a)-[r:FIRST_MEET]->(b)
SET r.tenant_id = $tenant,
    r.event_id = first.id,
    r.place_id = coalesce(pl.id, r.place_id),
    r.t_abs_start = coalesce(first.t_abs_start, r.t_abs_start),
    r.layer = coalesce(r.layer, 'fact'),
    r.kind = coalesce(r.kind, 'derived'),
    r.confidence = coalesce(r.confidence, 1.0)
RETURN count(*) AS created
"""
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, tenant=tenant_id, limit=int(limit)).single()
                val = row.get("created") if row else 0
                return {"first_meet": int(val or 0)}
        except Exception as exc:
            try:
                self._logger.error(
                    "neo4j.build_first_meetings.error",
                    extra={"event": "neo4j.build_first_meetings", "status": "error", "reason": str(exc)},
                    exc_info=True,
                )
            except Exception:
                pass
            return {"first_meet": 0}

    async def cleanup_expired(self, *, tenant_id: str, buffer_hours: float = 24.0, limit: int = 500, dry_run: bool = False) -> Dict[str, int]:
        """Delete (or preview) expired nodes/edges based on ttl + created_at with a safety buffer.

        - `limit` applies separately to nodes和edges，防止单次删除过大。
        """
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"nodes": 0, "edges": 0, "dry_run": dry_run}
        if not self._driver:
            return {"nodes": 0, "edges": 0, "dry_run": dry_run}
        buffer_s = float(buffer_hours) * 3600.0
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                node_cypher = """
                    MATCH (n)
                    WHERE n.tenant_id = $tenant
                      AND n.ttl IS NOT NULL
                      AND n.created_at IS NOT NULL
                      AND datetime() > n.created_at + duration({seconds: n.ttl + $buffer_s})
                    WITH n LIMIT $limit
                    {action}
                """
                edge_cypher = """
                    MATCH ()-[r]->()
                    WHERE r.tenant_id = $tenant
                      AND r.ttl IS NOT NULL
                      AND r.first_seen_at IS NOT NULL
                      AND datetime() > r.first_seen_at + duration({seconds: r.ttl + $buffer_s})
                    WITH r LIMIT $limit
                    {action}
                """
                node_action = "RETURN count(n) AS deleted" if dry_run else "DETACH DELETE n RETURN count(*) AS deleted"
                edge_action = "RETURN count(r) AS deleted" if dry_run else "DELETE r RETURN count(*) AS deleted"
                node_res = sess.run(
                    node_cypher.format(action=node_action),
                    tenant=tenant_id,
                    buffer_s=buffer_s,
                    limit=int(limit),
                ).single()
                edge_res = sess.run(
                    edge_cypher.format(action=edge_action),
                    tenant=tenant_id,
                    buffer_s=buffer_s,
                    limit=int(limit),
                ).single()
                return {
                    "nodes": int((node_res or {}).get("deleted", 0) or 0),
                    "edges": int((edge_res or {}).get("deleted", 0) or 0),
                    "dry_run": dry_run,
                }
        except Exception as exc:
            self._logger.error(
                "neo4j.cleanup_expired.error",
                extra={"event": "neo4j.cleanup_expired", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return {"nodes": 0, "edges": 0, "dry_run": dry_run}

    async def count_tenant_nodes(self, tenant_id: str) -> int:
        if not self._driver:
            return 0
        tid = str(tenant_id or "").strip()
        if not tid:
            return 0
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    """
                    MATCH (n {tenant_id: $tenant})
                    RETURN count(n) AS total
                    """,
                    tenant=tid,
                ).single()
                return int((row or {}).get("total", 0) or 0)
        except Exception as exc:
            self._logger.error(
                "neo4j.count_tenant_nodes.error",
                extra={"event": "neo4j.count_tenant_nodes", "tenant_id": tid, "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def count_legacy_memory_nodes_by_ids(self, ids: List[str], *, chunk_size: int = 512) -> int:
        if not self._driver:
            return 0
        item_ids = [str(item).strip() for item in (ids or []) if str(item).strip()]
        if not item_ids:
            return 0
        total = 0
        chunk = max(1, int(chunk_size))
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                for start in range(0, len(item_ids), chunk):
                    row = sess.run(
                        f"""
                        MATCH (n:{MEMORY_NODE_LABEL})
                        WHERE n.id IN $ids AND n.tenant_id IS NULL
                        RETURN count(n) AS total
                        """,
                        ids=item_ids[start : start + chunk],
                    ).single()
                    total += int((row or {}).get("total", 0) or 0)
            return total
        except Exception as exc:
            self._logger.error(
                "neo4j.count_legacy_memory_nodes_by_ids.error",
                extra={"event": "neo4j.count_legacy_memory_nodes_by_ids", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def purge_legacy_memory_nodes_by_ids(self, ids: List[str], *, chunk_size: int = 512) -> int:
        if not self._driver:
            return 0
        item_ids = [str(item).strip() for item in (ids or []) if str(item).strip()]
        if not item_ids:
            return 0
        total = 0
        chunk = max(1, int(chunk_size))
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                for start in range(0, len(item_ids), chunk):
                    row = sess.run(
                        f"""
                        MATCH (n:{MEMORY_NODE_LABEL})
                        WHERE n.id IN $ids AND n.tenant_id IS NULL
                        WITH collect(n) AS nodes
                        UNWIND nodes AS n
                        DETACH DELETE n
                        RETURN count(n) AS deleted
                        """,
                        ids=item_ids[start : start + chunk],
                    ).single()
                    total += int((row or {}).get("deleted", 0) or 0)
            return total
        except Exception as exc:
            self._logger.error(
                "neo4j.purge_legacy_memory_nodes_by_ids.error",
                extra={"event": "neo4j.purge_legacy_memory_nodes_by_ids", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def purge_tenant(self, tenant_id: str, *, batch_size: int = 10_000) -> int:
        if not self._driver:
            return 0
        tid = str(tenant_id or "").strip()
        if not tid:
            return 0
        batch = max(1, int(batch_size))
        total = 0
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                while True:
                    row = sess.run(
                        """
                        MATCH (n {tenant_id: $tenant})
                        WITH collect(n)[0..$batch_size] AS nodes
                        UNWIND nodes AS n
                        DETACH DELETE n
                        RETURN count(n) AS deleted
                        """,
                        tenant=tid,
                        batch_size=batch,
                    ).single()
                    deleted = int((row or {}).get("deleted", 0) or 0)
                    total += deleted
                    if deleted < batch:
                        break
            return total
        except Exception as exc:
            self._logger.error(
                "neo4j.purge_tenant.error",
                extra={"event": "neo4j.purge_tenant", "tenant_id": tid, "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            raise

    async def purge_source(self, *, tenant_id: str, source_id: str, delete_orphan_entities: bool = False) -> Dict[str, int]:
        """Delete all graph nodes related to a specific source_id (demo/testing helper)."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"segments": 0, "events": 0, "utterances": 0, "evidences": 0, "timeslices": 0, "knowledge": 0}
        if not self._driver:
            return {"segments": 0, "events": 0, "utterances": 0, "evidences": 0, "timeslices": 0, "knowledge": 0}
        src = str(source_id or "").strip()
        if not src:
            return {"segments": 0, "events": 0, "utterances": 0, "evidences": 0, "timeslices": 0, "knowledge": 0}
        seg_prefix = f"{src}#"
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                counts: Dict[str, int] = {}

                knowledge_cypher = """
MATCH (k:Knowledge {tenant_id: $tenant})-[:DERIVED_FROM]->(ev:Event {tenant_id: $tenant})-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT k) AS ks
UNWIND ks AS k
DETACH DELETE k
RETURN count(k) AS deleted
"""
                ts_seg_cypher = """
MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_SEGMENT]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT ts) AS tss
UNWIND tss AS ts
DETACH DELETE ts
RETURN count(ts) AS deleted
"""
                ts_evt_cypher = """
MATCH (ts:TimeSlice {tenant_id: $tenant})-[:COVERS_EVENT]->(ev:Event {tenant_id: $tenant})-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT ts) AS tss
UNWIND tss AS ts
DETACH DELETE ts
RETURN count(ts) AS deleted
"""
                event_cypher = """
MATCH (ev:Event {tenant_id: $tenant})-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT ev) AS evs
UNWIND evs AS ev
DETACH DELETE ev
RETURN count(ev) AS deleted
"""
                utterance_cypher = """
MATCH (utt:UtteranceEvidence {tenant_id: $tenant})
WHERE utt.segment_id STARTS WITH $seg_prefix
WITH collect(DISTINCT utt) AS utts
UNWIND utts AS utt
DETACH DELETE utt
RETURN count(utt) AS deleted
"""
                evidence_cypher = """
MATCH (evd:Evidence {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT evd) AS evds
UNWIND evds AS evd
DETACH DELETE evd
RETURN count(evd) AS deleted
"""
                segment_cypher = """
MATCH (seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WITH collect(DISTINCT seg) AS segs
UNWIND segs AS seg
DETACH DELETE seg
RETURN count(seg) AS deleted
"""
                orphan_entity_cypher = """
MATCH (ent:Entity {tenant_id: $tenant})
WHERE NOT (ent)--()
WITH collect(ent) AS ents
UNWIND ents AS ent
DETACH DELETE ent
RETURN count(ent) AS deleted
"""
                orphan_place_cypher = """
MATCH (pl:Place {tenant_id: $tenant})
WHERE NOT (pl)--()
WITH collect(pl) AS pls
UNWIND pls AS pl
DETACH DELETE pl
RETURN count(pl) AS deleted
"""

                knowledge_row = sess.run(knowledge_cypher, tenant=tenant_id, source=src).single()
                counts["knowledge"] = int((knowledge_row or {}).get("deleted", 0) or 0)

                ts_seg_row = sess.run(ts_seg_cypher, tenant=tenant_id, source=src).single()
                ts_evt_row = sess.run(ts_evt_cypher, tenant=tenant_id, source=src).single()
                counts["timeslices"] = int((ts_seg_row or {}).get("deleted", 0) or 0) + int((ts_evt_row or {}).get("deleted", 0) or 0)

                event_row = sess.run(event_cypher, tenant=tenant_id, source=src).single()
                counts["events"] = int((event_row or {}).get("deleted", 0) or 0)

                utt_row = sess.run(utterance_cypher, tenant=tenant_id, seg_prefix=seg_prefix).single()
                counts["utterances"] = int((utt_row or {}).get("deleted", 0) or 0)

                evidence_row = sess.run(evidence_cypher, tenant=tenant_id, source=src).single()
                counts["evidences"] = int((evidence_row or {}).get("deleted", 0) or 0)

                seg_row = sess.run(segment_cypher, tenant=tenant_id, source=src).single()
                counts["segments"] = int((seg_row or {}).get("deleted", 0) or 0)

                if delete_orphan_entities:
                    ent_row = sess.run(orphan_entity_cypher, tenant=tenant_id).single()
                    pl_row = sess.run(orphan_place_cypher, tenant=tenant_id).single()
                    counts["orphan_entities"] = int((ent_row or {}).get("deleted", 0) or 0)
                    counts["orphan_places"] = int((pl_row or {}).get("deleted", 0) or 0)

                return counts
        except Exception as exc:
            self._logger.error(
                "neo4j.purge_source.error",
                extra={"event": "neo4j.purge_source", "tenant_id": tenant_id, "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return {"segments": 0, "events": 0, "utterances": 0, "evidences": 0, "timeslices": 0, "knowledge": 0}

    async def purge_source_except_events(
        self,
        *,
        tenant_id: str,
        source_id: str,
        keep_event_ids: List[str],
    ) -> Dict[str, int]:
        """Delete Event/Knowledge nodes for a source_id except the provided event ids."""
        import time as _t

        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"events": 0, "knowledge": 0}
        if not self._driver:
            return {"events": 0, "knowledge": 0}
        src = str(source_id or "").strip()
        if not src:
            return {"events": 0, "knowledge": 0}
        keep_ids = [str(x) for x in (keep_event_ids or []) if str(x).strip()]
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                counts: Dict[str, int] = {}
                knowledge_cypher = """
MATCH (k:Knowledge {tenant_id: $tenant})-[:DERIVED_FROM]->(ev:Event {tenant_id: $tenant})-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WHERE (size($keep_ids) = 0 OR NOT ev.id IN $keep_ids)
WITH collect(DISTINCT k) AS ks
UNWIND ks AS k
DETACH DELETE k
RETURN count(k) AS deleted
"""
                event_cypher = """
MATCH (ev:Event {tenant_id: $tenant})-[:SUMMARIZES]->(seg:MediaSegment {tenant_id: $tenant, source_id: $source})
WHERE (size($keep_ids) = 0 OR NOT ev.id IN $keep_ids)
WITH collect(DISTINCT ev) AS evs
UNWIND evs AS ev
DETACH DELETE ev
RETURN count(ev) AS deleted
"""
                k_row = sess.run(knowledge_cypher, tenant=tenant_id, source=src, keep_ids=keep_ids).single()
                e_row = sess.run(event_cypher, tenant=tenant_id, source=src, keep_ids=keep_ids).single()
                counts["knowledge"] = int((k_row or {}).get("deleted", 0) or 0)
                counts["events"] = int((e_row or {}).get("deleted", 0) or 0)
                return counts
        except Exception as exc:
            self._logger.error(
                "neo4j.purge_source_except_events.error",
                extra={"event": "neo4j.purge_source_except_events", "tenant_id": tenant_id, "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return {"events": 0, "knowledge": 0}

    async def export_srot(
        self,
        *,
        tenant_id: str,
        rel_types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        limit: int = 1000,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export (s,r,o,t) view with optional relation filter and simple cursor.

        - Cursor is a plain subject id to resume from (exclusive).
        """
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"items": [], "next_cursor": None}
        if not self._driver:
            return {"items": [], "next_cursor": None}
        rel_filter = ""
        params: Dict[str, Any] = {"tenant": tenant_id, "limit": int(limit)}
        if rel_types:
            rel_filter = "AND type(r) IN $rels"
            params["rels"] = rel_types
        conf_filter = ""
        if min_confidence is not None:
            conf_filter = "AND coalesce(r.confidence, 1.0) >= $min_conf"
            params["min_conf"] = float(min_confidence)
        cursor_filter = ""
        if cursor:
            cursor_filter = "AND s.id > $cursor"
            params["cursor"] = str(cursor)

        cypher = (
            "MATCH (s)-[r]->(o) "
            "WHERE r.tenant_id = $tenant AND s.tenant_id = $tenant AND o.tenant_id = $tenant "
            f"{rel_filter} {conf_filter} {cursor_filter} "
            "RETURN s.id AS subject, type(r) AS relation, o.id AS object, "
            "coalesce(r.time_origin, s.time_origin, o.time_origin) AS time_origin, "
            "coalesce(r.last_seen_at, r.first_seen_at, s.updated_at, o.updated_at, s.created_at, o.created_at) AS t_ref "
            "ORDER BY subject ASC "
            "LIMIT $limit"
        )
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, **params)
                items = [dict(row) for row in rows]
                next_cursor: Optional[str] = None
                if len(items) == int(limit):
                    # simple cursor: last subject id in this page
                    last = items[-1].get("subject")
                    if last is not None:
                        next_cursor = str(last)
                return {"items": items, "next_cursor": next_cursor}
        except Exception as exc:
            self._logger.error(
                "neo4j.export_srot.error",
                extra={"event": "neo4j.export_srot", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return []

    async def touch_nodes(
        self,
        *,
        tenant_id: str,
        node_ids: List[str],
        extend_seconds: Optional[float] = None,
    ) -> Dict[str, int]:
        """Update last_accessed_at and optionally extend expires_at for given nodes."""
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"updated": 0}
        if not self._driver or not node_ids:
            return {"updated": 0}
        try:
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(
                    """
                    MATCH (n {tenant_id:$tenant})
                    WHERE n.id IN $ids
                    SET n.last_accessed_at = datetime()
                    FOREACH (_ IN CASE WHEN $extend IS NULL OR n.ttl IS NULL THEN [] ELSE [1] END |
                        SET n.expires_at = datetime() + duration({seconds: $extend})
                    )
                    RETURN count(n) AS updated
                    """,
                    tenant=tenant_id,
                    ids=node_ids,
                    extend=extend_seconds,
                ).single()
                return {"updated": int((row or {}).get("updated", 0) or 0)}
        except Exception as exc:
            self._logger.error(
                "neo4j.touch_nodes.error",
                extra={"event": "neo4j.touch_nodes", "status": "error", "reason": str(exc)},
                exc_info=True,
            )
            return {"updated": 0}
    async def merge_nodes_edges(
        self,
        entries: List[MemoryEntry],
        edges: Optional[List[Edge]] = None,
        *,
        tenant_id: Optional[str] = None,
    ) -> None:
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return None
        if not self._driver:
            return None
        self._require_legacy_memory_node_enabled("merge_nodes_edges")

        effective_tenant = str(tenant_id or "").strip() or str(self._infer_entry_tenant(entries) or "").strip()
        if self._strict_tenant_mode:
            effective_tenant = self._validate_tenant_context(effective_tenant, "merge_nodes_edges")

        def _label_for(e: MemoryEntry) -> str:
            if e.kind == "episodic":
                return "Episodic"
            if e.kind == "semantic":
                if e.modality == "image":
                    return "Image"
                if e.modality == "audio":
                    return "Voice"
                if e.modality == "structured":
                    return "Structured"
                return "Semantic"
            return "Node"

        try:
            with self._driver.session(database=self._database) as sess:
                did_work = False
                # Merge nodes
                for e in entries:
                    if not e.id:
                        continue
                    did_work = True
                    _label_for(e)
                    text = e.contents[0] if e.contents else None
                    md = dict(e.metadata)
                    md_tenant = str(md.get("tenant_id") or "").strip()
                    if effective_tenant:
                        if md_tenant and md_tenant != effective_tenant:
                            raise ValueError(
                                f"merge_nodes_edges entry tenant mismatch: expected={effective_tenant} got={md_tenant}"
                            )
                        if not md_tenant:
                            md["tenant_id"] = effective_tenant
                            e.metadata = md
                            md_tenant = effective_tenant
                    if self._strict_tenant_mode and not md_tenant:
                        raise ValueError("merge_nodes_edges strict_tenant_mode requires entry.metadata.tenant_id")
                    # Merge MemoryEntry projection node (separate from typed TKG graph :Entity label).
                    if effective_tenant:
                        sess.run(
                            f"MERGE (n:{MEMORY_NODE_LABEL} {{id: $id, tenant_id: $tenant}}) "
                            "SET n.kind=$kind, n.modality=$modality, n.text=$text, n.source=$source, n.timestamp=$ts, n.clip_id=$clip, "
                            "n.tenant_id=$tenant, n.memory_domain=$domain, n.run_id=$run, n.user_id=$uids, n.memory_scope=$scope "
                            # label backfill for node
                            "FOREACH (_ IN CASE WHEN $kind = 'episodic' THEN [1] ELSE [] END | SET n:Episodic) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'image' THEN [1] ELSE [] END | SET n:Image) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'audio' THEN [1] ELSE [] END | SET n:Voice) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'structured' THEN [1] ELSE [] END | SET n:Structured) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND NOT $modality IN ['image','audio','structured'] THEN [1] ELSE [] END | SET n:Semantic) ",
                            id=e.id,
                            tenant=effective_tenant,
                            kind=e.kind,
                            modality=e.modality,
                            text=text,
                            source=md.get("source"),
                            ts=md.get("timestamp"),
                            clip=md.get("clip_id"),
                            domain=md.get("memory_domain"),
                            run=md.get("run_id"),
                            uids=md.get("user_id"),
                            scope=md.get("memory_scope"),
                        )
                    else:
                        sess.run(
                            f"MERGE (n:{MEMORY_NODE_LABEL} {{id: $id}}) SET n.kind=$kind, n.modality=$modality, n.text=$text, n.source=$source, n.timestamp=$ts, n.clip_id=$clip, "
                            "n.tenant_id=CASE WHEN n.tenant_id IS NULL THEN $tenant ELSE n.tenant_id END, "
                            "n.memory_domain=$domain, n.run_id=$run, n.user_id=$uids, n.memory_scope=$scope "
                            # label backfill for node
                            "FOREACH (_ IN CASE WHEN $kind = 'episodic' THEN [1] ELSE [] END | SET n:Episodic) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'image' THEN [1] ELSE [] END | SET n:Image) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'audio' THEN [1] ELSE [] END | SET n:Voice) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND $modality = 'structured' THEN [1] ELSE [] END | SET n:Structured) "
                            "FOREACH (_ IN CASE WHEN $kind = 'semantic' AND NOT $modality IN ['image','audio','structured'] THEN [1] ELSE [] END | SET n:Semantic) ",
                            id=e.id,
                            kind=e.kind,
                            modality=e.modality,
                            text=text,
                            source=md.get("source"),
                            ts=md.get("timestamp"),
                            clip=md.get("clip_id"),
                            tenant=md.get("tenant_id"),
                            domain=md.get("memory_domain"),
                            run=md.get("run_id"),
                            uids=md.get("user_id"),
                            scope=md.get("memory_scope"),
                        )
                # Merge relations
                if edges:
                    for ed in edges:
                        did_work = True
                        w = ed.weight if ed.weight is not None else 1.0
                        rel = ed.rel_type.upper()
                        if effective_tenant:
                            result = sess.run(
                                f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$src, tenant_id:$tenant}}) "
                                f"MATCH (d:{MEMORY_NODE_LABEL} {{id:$dst, tenant_id:$tenant}}) "
                                f"MERGE (s)-[r:{rel}]->(d) "
                                "SET r.tenant_id = $tenant "
                                "SET r.weight = coalesce(r.weight, 0.0) + $w "
                                "RETURN count(r) AS applied",
                                src=ed.src_id,
                                dst=ed.dst_id,
                                tenant=effective_tenant,
                                w=float(w),
                            )
                            applied = self._extract_applied_count(result)
                            if applied is not None and applied != 1:
                                raise RuntimeError(
                                    f"merge_nodes_edges edge tenant validation failed: src={ed.src_id} dst={ed.dst_id} tenant={effective_tenant}"
                                )
                        else:
                            sess.run(
                                f"MERGE (s:{MEMORY_NODE_LABEL} {{id:$src}}) MERGE (d:{MEMORY_NODE_LABEL} {{id:$dst}}) "
                                f"MERGE (s)-[r:{rel}]->(d) "
                                "SET r.weight = coalesce(r.weight, 0.0) + $w",
                                src=ed.src_id,
                                dst=ed.dst_id,
                                w=float(w),
                            )
                # If nothing was written (e.g., all entries missing id and no edges), run a no-op ping
                # to surface driver/session errors for observability.
                if not did_work:
                    sess.run("RETURN 1 AS ok")
            self._cb_fail_count = 0
        except Exception as e:
            self._cb_fail_count += 1
            # Always emit a plain root error for test harness capture
            try:
                import logging as _rootlog
                _rootlog.error("neo4j.merge_nodes_edges.error")
            except Exception:
                pass
            # Structured logs (module + root) best-effort
            try:
                self._logger.error(
                    "neo4j.merge_nodes_edges.error",
                    extra={
                        "event": "neo4j.merge_nodes_edges.error",
                        "entity": "graph",
                        "verb": "merge",
                        "status": "error",
                        "reason": str(e),
                    },
                    exc_info=True,
                )
                import logging as _rootlog
                try:
                    _rootlog.error(
                        "neo4j.merge_nodes_edges.error",
                        extra={
                            "event": "neo4j.merge_nodes_edges.error",
                            "entity": "graph",
                            "verb": "merge",
                            "status": "error",
                            "reason": str(e),
                        },
                        exc_info=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
            if self._cb_fail_count >= max(1, self._cb_failure_threshold):
                self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
        return None

    async def merge_rel(
        self,
        src_id: str,
        dst_id: str,
        rel_type: str,
        *,
        weight: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return None
        if not self._driver:
            return None
        self._require_legacy_memory_node_enabled("merge_rel")
        w = weight if weight is not None else 1.0
        rel = rel_type.upper()
        try:
            with self._driver.session(database=self._database) as sess:
                if self._strict_tenant_mode:
                    tenant = self._validate_tenant_context(tenant_id, "merge_rel")
                    result = sess.run(
                        f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$src, tenant_id:$tenant}}) "
                        f"MATCH (d:{MEMORY_NODE_LABEL} {{id:$dst, tenant_id:$tenant}}) "
                        f"MERGE (s)-[r:{rel}]->(d) "
                        "SET r.tenant_id = $tenant "
                        "SET r.weight = coalesce(r.weight, 0.0) + $w "
                        "RETURN count(r) AS applied",
                        src=src_id,
                        dst=dst_id,
                        tenant=tenant,
                        w=float(w),
                    )
                    applied = self._extract_applied_count(result)
                    if applied is not None and applied != 1:
                        raise RuntimeError(
                            f"merge_rel tenant validation failed: src={src_id} dst={dst_id} tenant={tenant}"
                        )
                else:
                    # Merge relation and ensure source/target nodes carry proper type labels if kind/modality present
                    q = (
                        f"MERGE (s:{MEMORY_NODE_LABEL} {{id:$src}}) "
                        f"MERGE (d:{MEMORY_NODE_LABEL} {{id:$dst}}) "
                        f"MERGE (s)-[r:{rel}]->(d) "
                        "SET r.weight = coalesce(r.weight, 0.0) + $w "
                        # label backfill for s
                        "FOREACH (_ IN CASE WHEN coalesce(s.kind,'') = 'episodic' THEN [1] ELSE [] END | SET s:Episodic) "
                        "FOREACH (_ IN CASE WHEN coalesce(s.kind,'') = 'semantic' AND coalesce(s.modality,'') = 'image' THEN [1] ELSE [] END | SET s:Image) "
                        "FOREACH (_ IN CASE WHEN coalesce(s.kind,'') = 'semantic' AND coalesce(s.modality,'') = 'audio' THEN [1] ELSE [] END | SET s:Voice) "
                        "FOREACH (_ IN CASE WHEN coalesce(s.kind,'') = 'semantic' AND coalesce(s.modality,'') = 'structured' THEN [1] ELSE [] END | SET s:Structured) "
                        "FOREACH (_ IN CASE WHEN coalesce(s.kind,'') = 'semantic' AND NOT coalesce(s.modality,'') IN ['image','audio','structured'] THEN [1] ELSE [] END | SET s:Semantic) "
                        # label backfill for d
                        "FOREACH (_ IN CASE WHEN coalesce(d.kind,'') = 'episodic' THEN [1] ELSE [] END | SET d:Episodic) "
                        "FOREACH (_ IN CASE WHEN coalesce(d.kind,'') = 'semantic' AND coalesce(d.modality,'') = 'image' THEN [1] ELSE [] END | SET d:Image) "
                        "FOREACH (_ IN CASE WHEN coalesce(d.kind,'') = 'semantic' AND coalesce(d.modality,'') = 'audio' THEN [1] ELSE [] END | SET d:Voice) "
                        "FOREACH (_ IN CASE WHEN coalesce(d.kind,'') = 'semantic' AND coalesce(d.modality,'') = 'structured' THEN [1] ELSE [] END | SET d:Structured) "
                        "FOREACH (_ IN CASE WHEN coalesce(d.kind,'') = 'semantic' AND NOT coalesce(d.modality,'') IN ['image','audio','structured'] THEN [1] ELSE [] END | SET d:Semantic) "
                    )
                    sess.run(q, src=src_id, dst=dst_id, w=float(w))
            self._cb_fail_count = 0
        except Exception as e:
            self._cb_fail_count += 1
            # Always emit a plain root error for test harness capture
            try:
                import logging as _rootlog
                _rootlog.error("neo4j.merge_rel.error")
            except Exception:
                pass
            # Structured logs (module + root) best-effort
            try:
                self._logger.error(
                    "neo4j.merge_rel.error",
                    extra={
                        "event": "neo4j.merge_rel.error",
                        "entity": "graph",
                        "verb": "merge_rel",
                        "status": "error",
                        "src": src_id,
                        "dst": dst_id,
                        "rel": rel_type,
                        "reason": str(e),
                    },
                    exc_info=True,
                )
                import logging as _rootlog
                try:
                    _rootlog.error(
                        "neo4j.merge_rel.error",
                        extra={
                            "event": "neo4j.merge_rel.error",
                            "entity": "graph",
                            "verb": "merge_rel",
                            "status": "error",
                            "src": src_id,
                            "dst": dst_id,
                            "rel": rel_type,
                            "reason": str(e),
                        },
                        exc_info=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
            if self._cb_fail_count >= max(1, self._cb_failure_threshold):
                self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
        return None

    @staticmethod
    def _label_from_props(kind: Optional[str], modality: Optional[str]) -> str:
        """Pure helper for tests: decide label name from kind/modality."""
        k = (kind or "").strip().lower()
        m = (modality or "").strip().lower()
        if k == "episodic":
            return "Episodic"
        if k == "semantic":
            if m == "image":
                return "Image"
            if m == "audio":
                return "Voice"
            if m == "structured":
                return "Structured"
            return "Semantic"
        return "Node"

    async def repair_node_labels(self) -> Dict[str, int]:
        """修复节点标签缺失问题：确保所有 MemoryNode 节点都有正确的类型标签

        Returns:
            修复统计：每个标签类型的修复数量
        """
        if not self._driver:
            return {}
        stats = {"Episodic": 0, "Image": 0, "Voice": 0, "Semantic": 0, "Structured": 0, "Node": 0}
        try:
            with self._driver.session(database=self._database) as sess:
                # 查询所有 MemoryNode 但缺少类型标签的节点
                nodes = sess.run("""
                    MATCH (n:MemoryNode)
                    WHERE NOT n:Episodic
                      AND NOT n:Image
                      AND NOT n:Voice
                      AND NOT n:Semantic
                      AND NOT n:Structured
                      AND NOT n:Node
                    RETURN n.id as id, n.kind as kind, n.modality as modality
                """)
                for node in nodes:
                    nid = node["id"]
                    kind = node["kind"]
                    modality = node["modality"]
                    label = self._label_from_props(kind, modality)
                    # 添加对应的类型标签
                    sess.run(
                        f"MATCH (n:{MEMORY_NODE_LABEL} {{id: $id}}) SET n:{label}",
                        id=nid
                    )
                    if label in stats:
                        stats[label] += 1
                    else:
                        stats["Node"] += 1
        except Exception:
            pass
        return stats

    # ---- Equivalence pending workflow helpers ----
    async def add_pending_equivalence(self, src_id: str, dst_id: str, *, score: Optional[float] = None, reason: Optional[str] = None) -> None:
        """Create or mark an equivalence relation as pending between two nodes.

        Implementation: MERGE (s)-[r:EQUIVALENCE]->(d) SET r.pending=true, r.score=?, r.reason=?, r.updated_at=timestamp()
        """
        if not self._driver:
            return None
        try:
            with self._driver.session(database=self._database) as sess:
                sess.run(
                    f"MERGE (s:{MEMORY_NODE_LABEL} {{id:$src}}) MERGE (d:{MEMORY_NODE_LABEL} {{id:$dst}}) "
                    "MERGE (s)-[r:EQUIVALENCE]->(d) SET r.pending=true, r.score = $score, r.reason = $reason, r.updated_at = timestamp()",
                    src=src_id,
                    dst=dst_id,
                    score=(float(score) if score is not None else None),
                    reason=(str(reason) if reason is not None else None),
                )
        except Exception:
            pass

    async def list_pending_equivalence(self, *, limit: int = 50) -> dict:
        if not self._driver:
            return {"pending": []}
        try:
            with self._driver.session(database=self._database) as sess:
                q = (
                    f"MATCH (s:{MEMORY_NODE_LABEL})-[r:EQUIVALENCE]->(d:{MEMORY_NODE_LABEL}) "
                    "WHERE coalesce(r.pending,false) = true "
                    "RETURN s.id AS src, d.id AS dst, coalesce(r.score,0.0) AS score, r.reason AS reason "
                    "LIMIT $lim"
                )
                rows = sess.run(q, lim=int(max(1, limit)))
                out = [{"src_id": rec["src"], "dst_id": rec["dst"], "score": float(rec.get("score", 0.0) or 0.0), "reason": rec.get("reason")} for rec in rows]
                return {"pending": out}
        except Exception:
            return {"pending": []}

    async def confirm_equivalence(self, pairs: list[tuple[str, str]], *, weight: float | None = None) -> int:
        """Confirm pending equivalence edges by clearing pending and setting confirmed_at.

        Returns number of edges affected.
        """
        if not self._driver or not pairs:
            return 0
        updated = 0
        try:
            with self._driver.session(database=self._database) as sess:
                for (src, dst) in pairs:
                    w = float(weight) if weight is not None else None
                    # Ensure relation exists, then clear pending and set timestamps/weights
                    sess.run(
                        f"MERGE (s:{MEMORY_NODE_LABEL} {{id:$src}}) MERGE (d:{MEMORY_NODE_LABEL} {{id:$dst}}) "
                        "MERGE (s)-[r:EQUIVALENCE]->(d) "
                        "SET r.pending = false, r.confirmed_at = timestamp(), r.weight = coalesce(r.weight, 0.0) + coalesce($w, 0.0)",
                        src=src, dst=dst, w=w,
                    )
                    updated += 1
        except Exception:
            pass
        return updated

    async def delete_equivalence(self, pairs: list[tuple[str, str]]) -> int:
        """Delete equivalence relations for the given pairs (pending or confirmed)."""
        if not self._driver or not pairs:
            return 0
        deleted = 0
        try:
            with self._driver.session(database=self._database) as sess:
                for (src, dst) in pairs:
                    sess.run(
                        f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$src}})-[r:EQUIVALENCE]->(d:{MEMORY_NODE_LABEL} {{id:$dst}}) DELETE r",
                        src=src, dst=dst,
                    )
                    deleted += 1
        except Exception:
            pass
        return deleted

    async def health(self) -> Dict[str, Any]:
        if not self._driver:
            return {"status": "unconfigured"}
        
        uri = self.settings.get("uri", "unknown")
        try:
            from neo4j.exceptions import AuthError, ServiceUnavailable
            with self._driver.session(database=self._database) as sess:
                res = sess.run("RETURN 1 AS ok").single()
                return {
                    "status": "ok" if res and res.get("ok") == 1 else "unknown",
                    "endpoint": uri,
                }
        except AuthError:
            return {
                "status": "auth_error", 
                "message": "Authentication failed. Check NEO4J_USER and NEO4J_PASSWORD in your .env file.",
                "endpoint": uri,
            }
        except ServiceUnavailable:
            return {
                "status": "unavailable", 
                "message": "Cannot connect to Neo4j. Check if the service is running and the NEO4J_URI is correct.",
                "endpoint": uri,
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e), "endpoint": uri}

    async def set_nodes_published(self, *, tenant_id: str, node_ids: List[str], published: bool = True) -> int:
        if not self._driver or not node_ids:
            return 0
        updated = 0
        try:
            ids = [str(x) for x in node_ids if str(x).strip()]
            if not ids:
                return 0
            with self._driver.session(database=self._database) as sess:  # type: ignore[attr-defined]
                res = sess.run(
                    "MATCH (n {tenant_id: $tenant}) WHERE n.id IN $ids "
                    "SET n.published = $pub RETURN count(n) AS cnt",
                    tenant=str(tenant_id),
                    ids=ids,
                    pub=bool(published),
                )
                row = res.single()
                if row and row.get("cnt") is not None:
                    updated = int(row.get("cnt") or 0)
        except Exception:
            return 0
        return updated

    async def expand_neighbors(
        self,
        seed_ids: List[str],
        *,
        rel_whitelist: Optional[List[str]] = None,
        max_hops: int = 1,
        neighbor_cap_per_seed: int = 5,
        user_ids: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        memory_scope: Optional[str] = None,
        restrict_to_user: bool = True,
        restrict_to_domain: bool = True,
        restrict_to_scope: bool = True,
    ) -> Dict[str, Any]:
        import time as _t
        if self._cb_open_until and _t.time() < self._cb_open_until:
            return {"neighbors": {}, "edges": []}
        if not self._driver:
            return {"neighbors": {}, "edges": []}
        rels = rel_whitelist or []
        result: Dict[str, Any] = {"neighbors": {}, "edges": []}
        try:
            with self._driver.session(database=self._database) as sess:
                for sid in seed_ids:
                    if max_hops <= 0:
                        result["neighbors"][sid] = []
                        continue
                    nbrs_map: Dict[str, Dict[str, Any]] = {}
                    # hop=1
                    where_parts = []
                    params = {"sid": sid}
                    where_parts.append("(n.published IS NULL OR n.published = true)")
                    if rels:
                        where_parts.append("type(r) IN $rels")
                        params["rels"] = rels
                    if restrict_to_user and user_ids:
                        where_parts.append("ANY(u IN n.user_id WHERE u IN $uids)")
                        params["uids"] = user_ids
                    if restrict_to_domain and memory_domain is not None:
                        where_parts.append("n.memory_domain = $domain")
                        params["domain"] = memory_domain
                    if restrict_to_scope and memory_scope is not None:
                        where_parts.append("n.memory_scope = $mscope")
                        params["mscope"] = memory_scope
                    where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
                    q1 = (
                        f"MATCH (s {{id:$sid}})-[r]->(n){where_clause} "
                        "RETURN n.id AS nid, type(r) AS rel, coalesce(r.weight,0.0) AS w"
                    )
                    recs1 = sess.run(q1, **params)
                    for rec in recs1:
                        nid = rec["nid"]
                        w = float(rec["w"] or 0.0)
                        # keep strongest edge and minimal hop
                        if nid not in nbrs_map or w > float(nbrs_map[nid].get("weight", 0.0)):
                            nbrs_map[nid] = {"to": nid, "rel": rec["rel"], "weight": w, "hop": 1}
                        else:
                            try:
                                prev_hop = int(nbrs_map[nid].get("hop", 1))
                                nbrs_map[nid]["hop"] = min(prev_hop, 1)
                            except Exception:
                                nbrs_map[nid]["hop"] = 1
                    # hop=2 (only if requested)
                    if max_hops >= 2:
                        params2 = {"sid": sid}
                        where_parts2 = []
                        where_parts2.append("(n.published IS NULL OR n.published = true)")
                        if rels:
                            where_parts2.append("type(r1) IN $rels AND type(r2) IN $rels")
                            params2["rels"] = rels
                        if restrict_to_user and user_ids:
                            where_parts2.append("ANY(u IN n.user_id WHERE u IN $uids)")
                            params2["uids"] = user_ids
                        if restrict_to_domain and memory_domain is not None:
                            where_parts2.append("n.memory_domain = $domain")
                            params2["domain"] = memory_domain
                        if restrict_to_scope and memory_scope is not None:
                            where_parts2.append("n.memory_scope = $mscope")
                            params2["mscope"] = memory_scope
                        where_clause2 = (" WHERE " + " AND ".join(where_parts2)) if where_parts2 else ""
                        q2 = (
                            f"MATCH (s {{id:$sid}})-[r1]->(m)-[r2]->(n){where_clause2} "
                            "RETURN n.id AS nid, type(r2) AS rel, coalesce(r2.weight,0.0) AS w"
                        )
                        recs2 = sess.run(q2, **params2)
                        for rec in recs2:
                            nid = rec["nid"]
                            w = float(rec["w"] or 0.0)
                            if nid == sid:
                                continue
                            if nid not in nbrs_map or w > float(nbrs_map[nid].get("weight", 0.0)):
                                nbrs_map[nid] = {"to": nid, "rel": rec["rel"], "weight": w, "hop": 2}
                            else:
                                try:
                                    prev_hop = int(nbrs_map[nid].get("hop", 2))
                                    nbrs_map[nid]["hop"] = min(prev_hop, 2)
                                except Exception:
                                    nbrs_map[nid]["hop"] = 2
                    # finalize per sid
                    lst = sorted(nbrs_map.values(), key=lambda x: x["weight"], reverse=True)[:neighbor_cap_per_seed]
                    result["neighbors"][sid] = lst
            self._cb_fail_count = 0
            return result
        except Exception:
            self._cb_fail_count += 1
            if self._cb_fail_count >= max(1, self._cb_failure_threshold):
                self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
            return {"neighbors": {}, "edges": []}

    # Optional placeholders to align with MemoryService usage
    async def delete_node(self, node_id: str) -> None:
        return None

    # Decay relation weights by factor (0..1)
    async def decay_edges(self, *, factor: float = 0.9, rel_whitelist: Optional[List[str]] = None, min_weight: float = 0.0) -> None:
        if not self._driver:
            return None
        try:
            with self._driver.session(database=self._database) as sess:
                if rel_whitelist:
                    sess.run(
                        "MATCH ()-[r]->() WHERE type(r) IN $rels AND coalesce(r.weight,0.0) > $min SET r.weight = r.weight * $f",
                        rels=rel_whitelist, f=float(factor), min=float(min_weight),
                    )
                else:
                    sess.run(
                        "MATCH ()-[r]->() WHERE coalesce(r.weight,0.0) > $min SET r.weight = r.weight * $f",
                        f=float(factor), min=float(min_weight),
                    )
        except Exception:
            pass

    # ---- Enhanced: transactional helpers, batch MERGE and path queries ----

    def _write_with_retry(self, fn: Callable[[Any], Any]) -> None:
        import time as _t
        from modules.memory.application.metrics import inc as _inc, add_tx_latency_ms as _addtx  # type: ignore
        if not self._driver:
            return None
        attempts = 3
        backoff = 0.1
        for i in range(attempts):
            try:
                t0 = _t.perf_counter()
                with self._driver.session(database=self._database) as sess:
                    res = sess.execute_write(lambda tx: fn(tx))
                _addtx(int((_t.perf_counter() - t0) * 1000))
                if i > 0:
                    try:
                        _inc("neo4j_tx_retries_total", i)
                    except Exception:
                        pass
                return res
            except Exception:
                self._cb_fail_count += 1
                if i == attempts - 1:
                    # open breaker
                    self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
                    try:
                        _inc("neo4j_tx_failures_total", 1)
                    except Exception:
                        pass
                    raise
                _t.sleep(backoff)
                backoff = min(backoff * 2, 1.6)

    def _read_with_retry(self, fn: Callable[[Any], Any]):
        import time as _t
        from modules.memory.application.metrics import add_tx_latency_ms as _addtx, inc as _inc  # type: ignore
        if not self._driver:
            return None
        attempts = 3
        backoff = 0.1
        for i in range(attempts):
            try:
                t0 = _t.perf_counter()
                with self._driver.session(database=self._database) as sess:
                    res = sess.execute_read(lambda tx: fn(tx))
                _addtx(int((_t.perf_counter() - t0) * 1000))
                if i > 0:
                    try:
                        _inc("neo4j_tx_retries_total", i)
                    except Exception:
                        pass
                return res
            except Exception:
                if i == attempts - 1:
                    # open breaker
                    self._cb_open_until = _t.time() + max(1, self._cb_cooldown_s)
                    try:
                        _inc("neo4j_tx_failures_total", 1)
                    except Exception:
                        pass
                    raise
                _t.sleep(backoff)
                backoff = min(backoff * 2, 1.6)

    async def merge_nodes_edges_batch(
        self,
        entries: List[MemoryEntry],
        edges: Optional[List[Edge]] = None,
        *,
        chunk_size: int = 500,
        tenant_id: Optional[str] = None,
    ) -> None:
        if not self._driver or not entries and not edges:
            return None
        self._require_legacy_memory_node_enabled("merge_nodes_edges_batch")
        effective_tenant = str(tenant_id or "").strip() or str(self._infer_entry_tenant(entries) or "").strip()
        if self._strict_tenant_mode:
            effective_tenant = self._validate_tenant_context(effective_tenant, "merge_nodes_edges_batch")
        # batch nodes
        nodes_payload: List[Dict[str, Any]] = []
        for e in entries or []:
            if not e.id:
                continue
            md = dict(e.metadata)
            md_tenant = str(md.get("tenant_id") or "").strip()
            if effective_tenant:
                if md_tenant and md_tenant != effective_tenant:
                    raise ValueError(
                        f"merge_nodes_edges_batch entry tenant mismatch: expected={effective_tenant} got={md_tenant}"
                    )
                if not md_tenant:
                    md["tenant_id"] = effective_tenant
                    e.metadata = md
                    md_tenant = effective_tenant
            if self._strict_tenant_mode and not md_tenant:
                raise ValueError("merge_nodes_edges_batch strict_tenant_mode requires entry.metadata.tenant_id")
            nodes_payload.append({
                "id": e.id,
                "kind": e.kind,
                "modality": e.modality,
                "text": (e.contents[0] if e.contents else None),
                "source": md.get("source"),
                "ts": md.get("timestamp"),
                "clip": md.get("clip_id"),
                "tenant": md.get("tenant_id"),
                "domain": md.get("memory_domain"),
                "run": md.get("run_id"),
                "uids": md.get("user_id"),
                "scope": md.get("memory_scope"),
            })
        def _merge_nodes(tx, chunk: List[Dict[str, Any]]):
            if effective_tenant:
                cypher = (
                    "UNWIND $nodes AS n "
                    f"MERGE (e:{MEMORY_NODE_LABEL} {{id:n.id, tenant_id:$tenant}}) "
                    "SET e.kind=n.kind, e.modality=n.modality, e.text=n.text, e.source=n.source, "
                    "e.timestamp=n.ts, e.clip_id=n.clip, e.tenant_id=$tenant, "
                    "e.memory_domain=n.domain, e.run_id=n.run, e.user_id=n.uids, e.memory_scope=n.scope "
                    # label backfill for each node
                    "FOREACH (_ IN CASE WHEN n.kind = 'episodic' THEN [1] ELSE [] END | SET e:Episodic) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'image' THEN [1] ELSE [] END | SET e:Image) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'audio' THEN [1] ELSE [] END | SET e:Voice) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'structured' THEN [1] ELSE [] END | SET e:Structured) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND NOT n.modality IN ['image','audio','structured'] THEN [1] ELSE [] END | SET e:Semantic) "
                )
                tx.run(cypher, nodes=chunk, tenant=effective_tenant)
            else:
                cypher = (
                    "UNWIND $nodes AS n "
                    f"MERGE (e:{MEMORY_NODE_LABEL} {{id:n.id}}) "
                    "SET e.kind=n.kind, e.modality=n.modality, e.text=n.text, e.source=n.source, "
                    "e.timestamp=n.ts, e.clip_id=n.clip, e.tenant_id=CASE WHEN e.tenant_id IS NULL THEN n.tenant ELSE e.tenant_id END, "
                    "e.memory_domain=n.domain, e.run_id=n.run, e.user_id=n.uids, e.memory_scope=n.scope "
                    # label backfill for each node
                    "FOREACH (_ IN CASE WHEN n.kind = 'episodic' THEN [1] ELSE [] END | SET e:Episodic) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'image' THEN [1] ELSE [] END | SET e:Image) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'audio' THEN [1] ELSE [] END | SET e:Voice) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND n.modality = 'structured' THEN [1] ELSE [] END | SET e:Structured) "
                    "FOREACH (_ IN CASE WHEN n.kind = 'semantic' AND NOT n.modality IN ['image','audio','structured'] THEN [1] ELSE [] END | SET e:Semantic) "
                )
                tx.run(cypher, nodes=chunk)
        for i in range(0, len(nodes_payload), max(1, int(chunk_size))):
            chunk = nodes_payload[i:i+chunk_size]
            if not chunk:
                continue
            self._write_with_retry(lambda tx, c=chunk: _merge_nodes(tx, c))
            try:
                from modules.memory.application.metrics import inc as _inc  # type: ignore
                _inc("neo4j_batch_nodes_total", len(chunk))
            except Exception:
                pass

        # batch relations
        rels_payload: List[Dict[str, Any]] = []
        for r in edges or []:
            rels_payload.append({"src": r.src_id, "dst": r.dst_id, "rel": r.rel_type, "weight": float(r.weight if r.weight is not None else 1.0)})
        def _merge_rels(tx, chunk: List[Dict[str, Any]]):
            # NOTE: Neo4j 不支持参数化关系类型，使用 CASE/FOREACH 或 APOC 更灵活；这里用简单分支处理常见关系
            # 为简化，这里统一用字符串 rel 存储类型，MERGE 通过 apoc（若不可用，退化为单类型 prefer），实际生产建议按类型拆批
            # 简化实现：动态构造语句（小批次内安全）
            # 安全加固：添加关系类型白名单验证，防止Cypher注入
            ALLOWED_RELS = {
                # 基础关系
                "APPEARS_IN", "SAID_BY", "DESCRIBES", "REFERENCES", "PART_OF",
                "RELATED_TO", "SIMILAR_TO", "BEFORE", "AFTER", "EQUIVALENCE",
                # 项目扩展关系
                "TEMPORAL_NEXT",  # 时序链
                "CO_OCCURS",      # 共现
                "LOCATED_IN",     # 位置
                "EXECUTED",       # 设备执行
                "PREFER",         # 偏好/权重偏置
                "OCCURS_AT",      # 事件发生于时间节点
            }
            for r in chunk:
                rel = str(r.get("rel", "REL")).upper()
                # 安全检查：仅允许白名单中的关系类型
                if rel not in ALLOWED_RELS:
                    raise ValueError(f"Invalid relation type: {rel}. Must be one of {ALLOWED_RELS}")

                if effective_tenant:
                    cypher = (
                        f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$src, tenant_id:$tenant}}) "
                        f"MATCH (d:{MEMORY_NODE_LABEL} {{id:$dst, tenant_id:$tenant}}) "
                        f"MERGE (s)-[e:{rel}]->(d) "
                        "SET e.tenant_id = $tenant "
                        "SET e.weight = coalesce(e.weight,0.0) + $w "
                        "RETURN count(e) AS applied"
                    )
                    _res = tx.run(cypher, src=r["src"], dst=r["dst"], tenant=effective_tenant, w=r["weight"])
                    applied = self._extract_applied_count(_res)
                    if applied is not None and applied != 1:
                        raise RuntimeError(
                            f"merge_nodes_edges_batch edge tenant validation failed: src={r['src']} dst={r['dst']} tenant={effective_tenant}"
                        )
                else:
                    cypher = (
                        f"MERGE (s:{MEMORY_NODE_LABEL} {{id:$src}}) MERGE (d:{MEMORY_NODE_LABEL} {{id:$dst}}) "
                        f"MERGE (s)-[e:{rel}]->(d) SET e.weight = coalesce(e.weight,0.0) + $w"
                    )
                    tx.run(cypher, src=r["src"], dst=r["dst"], w=r["weight"])
        for i in range(0, len(rels_payload), max(1, int(chunk_size))):
            chunk = rels_payload[i:i+chunk_size]
            if not chunk:
                continue
            self._write_with_retry(lambda tx, c=chunk: _merge_rels(tx, c))
            try:
                from modules.memory.application.metrics import inc as _inc  # type: ignore
                _inc("neo4j_batch_rels_total", len(chunk))
            except Exception:
                pass

    async def merge_rel_batch(
        self,
        links: List[Tuple[str, str, str, float]],
        *,
        chunk_size: int = 1000,
        tenant_id: Optional[str] = None,
    ) -> None:
        # links: list of (src, dst, rel_type, weight)
        payload = [{"src": a, "dst": b, "rel": r, "weight": float(w)} for (a, b, r, w) in links]
        await self.merge_nodes_edges_batch(
            [],
            [Edge(src_id=p["src"], dst_id=p["dst"], rel_type=p["rel"], weight=p["weight"]) for p in payload],
            chunk_size=chunk_size,
            tenant_id=tenant_id,
        )

    async def find_paths(self, seed_ids: List[str], *, rel_whitelist: Optional[List[str]] = None, max_hops: int = 1, min_weight: float = 0.0, user_ids: Optional[List[str]] = None, memory_domain: Optional[str] = None, cap: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """Find neighbors up to max_hops with attribute filters.

        Returns {seed_id: [{to, rel, weight, hop}...]}
        """
        if not self._driver:
            return {"neighbors": {}}
        rels = list(rel_whitelist or [])
        def _read(tx):
            out: Dict[str, List[Dict[str, Any]]] = {}
            for sid in seed_ids:
                items: List[Dict[str, Any]] = []
                # hop1
                where_parts = ["coalesce(r.weight,0.0) >= $minw", "(n.published IS NULL OR n.published = true)"]
                params = {"sid": sid, "minw": float(min_weight)}
                if rels:
                    where_parts.append("type(r) IN $rels")
                    params["rels"] = rels
                if user_ids:
                    where_parts.append("($uids IS NULL OR ANY(u IN n.user_id WHERE u IN $uids))")
                    params["uids"] = user_ids
                if memory_domain is not None:
                    where_parts.append("($domain IS NULL OR n.memory_domain = $domain)")
                    params["domain"] = memory_domain
                wc = " AND ".join(where_parts)
                q1 = f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$sid}})-[r]->(n) WHERE {wc} RETURN n.id AS nid, type(r) AS rel, r.weight AS w LIMIT $cap"
                params["cap"] = int(cap)
                for rec in tx.run(q1, **params):
                    items.append({"to": rec["nid"], "rel": rec["rel"], "weight": float(rec["w"] or 0.0), "hop": 1})
                # hop2 optional
                if max_hops >= 2:
                    q2 = f"MATCH (s:{MEMORY_NODE_LABEL} {{id:$sid}})-[r1]->(m)-[r2]->(n) WHERE {' AND '.join(where_parts).replace('r','r2')} RETURN n.id AS nid, type(r2) AS rel, r2.weight AS w LIMIT $cap"
                    for rec in tx.run(q2, **params):
                        items.append({"to": rec["nid"], "rel": rec["rel"], "weight": float(rec["w"] or 0.0), "hop": 2})
                out[sid] = items[:cap]
            return out
        res = self._read_with_retry(_read)
        return {"neighbors": res or {}}
