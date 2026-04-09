from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import uuid
import os
import math
import re

from modules.memory.contracts.memory_models import (
    MemoryEntry,
    Edge,
    SearchFilters,
    SearchResult,
    Version,
)
from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.domain.governance import (
    compute_importance,
    compute_stability,
    default_ttl_seconds,
)
from modules.memory.domain.dedup import should_merge, merge_entries
from modules.memory.application.config import (
    load_memory_config,
    get_search_weights,
    get_graph_settings,
    resolve_lexical_hybrid_settings,
    SEARCH_TOPK_HARD_LIMIT,
)
from modules.memory.application import runtime_config as rtconf
from modules.memory.application.metrics import inc, add_latency_ms
from datetime import datetime, timezone
import time
import asyncio
import threading
try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # fallback when not installed
    BM25Okapi = None  # type: ignore

class _SimpleBM25:
    """Minimal BM25 implementation used when rank_bm25 is unavailable.

    Keeps MemoryService behavior predictable for experiments without adding a hard dependency.
    """

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        import math

        self._k1 = float(k1)
        self._b = float(b)
        self._N = len(corpus)
        self._avgdl = float(sum(len(doc) for doc in corpus)) / self._N if self._N else 0.0

        self._doc_freqs: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        df: dict[str, int] = {}

        for doc in corpus:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self._doc_freqs.append(freq)
            doc_len = len(doc)
            self._doc_len.append(doc_len)
            for token in freq:
                df[token] = df.get(token, 0) + 1

        self._idf: dict[str, float] = {}
        for token, freq in df.items():
            # standard BM25 idf with +1 smoothing
            self._idf[token] = math.log(1.0 + ((self._N - freq + 0.5) / (freq + 0.5)))

        # guard against division by zero when corpus is empty
        self._avgdl = self._avgdl or 1.0

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freq, doc_len in zip(self._doc_freqs, self._doc_len):
            score = 0.0
            denom_base = self._k1 * (1 - self._b + self._b * (doc_len / self._avgdl))
            for token in query_tokens:
                if token not in freq:
                    continue
                idf = self._idf.get(token)
                if idf is None:
                    continue
                tf = freq[token]
                score += idf * (tf * (self._k1 + 1)) / (tf + denom_base)
            scores.append(score)
        return scores


_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", flags=re.UNICODE)


def _bm25_tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for lexical retrieval.

    - English/digits/underscores are kept as whole words (lower-cased).
    - Chinese spans are split into character bigrams, with single-char fallback.
    """
    s = str(text or "").strip()
    if not s:
        return []
    out: list[str] = []
    for match in _BM25_TOKEN_RE.finditer(s):
        seg = match.group(0)
        if not seg:
            continue
        if seg.isascii():
            out.append(seg.lower())
            continue
        if len(seg) == 1:
            out.append(seg)
            continue
        out.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return out


def _minmax_normalize_score_map(scores: Dict[str, float], *, log1p: bool = False) -> Dict[str, float]:
    if not scores:
        return {}
    transformed: Dict[str, float] = {}
    for key, value in scores.items():
        val = max(0.0, float(value or 0.0))
        transformed[str(key)] = math.log1p(val) if log1p else val
    vals = list(transformed.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= 0.0:
        return {key: 0.0 for key in transformed}
    if hi == lo:
        return {key: (1.0 if value > 0.0 else 0.0) for key, value in transformed.items()}
    return {key: (value - lo) / (hi - lo) for key, value in transformed.items()}



class MemoryService:
    """Use-case orchestration for the unified memory store.

    This class implements Normalize & Enrich Pipeline and delegates to injected infra stores.
    Infra stores must implement minimal methods: upsert_vectors(), search_vectors(), merge_nodes_edges(), audit().
    """

    def __init__(self, vector_store, graph_store, audit_store, usage_wal: Optional[Any] = None):
        self.vectors = vector_store
        self.graph = graph_store
        self.audit = audit_store
        self.usage_wal = usage_wal
        # Optional LLM-style update decision hook (mem0 alignment)
        # Signature: decider(existing_neighbors: list[MemoryEntry], new_entry: MemoryEntry) -> tuple[str, str|None]
        # where action in {"ADD","UPDATE","DELETE","NONE"} and target_id for UPDATE/DELETE
        self.update_decider = None
        self._bm25_warned: bool = False
        # load search weights from config
        cfg = load_memory_config()
        w = get_search_weights(cfg)
        self._w_alpha = w.get("alpha_vector", 0.6)
        self._w_beta = w.get("beta_bm25", 0.2)
        self._w_gamma = w.get("gamma_graph", 0.15)
        self._w_delta = w.get("delta_recency", 0.05)
        # Load persisted runtime overrides once per instance (ensures test isolation)
        try:
            rtconf.load_overrides()
        except Exception:
            pass
        # simple in-memory cache for last written entries
        self._cache: dict[str, MemoryEntry] = {}
        # Configuration cache to avoid repeated disk I/O
        self._cfg_cache: Optional[Dict[str, Any]] = None
        self._cfg_cache_time: float = 0.0
        self._cfg_cache_ttl: float = 30.0  # 30秒缓存
        # search cache (hot queries)
        self._search_cache_enabled: bool = True
        self._search_cache_ttl_s: int = 60
        self._search_cache_max: int = 256
        from collections import OrderedDict
        self._search_cache: "OrderedDict[str, tuple[float, SearchResult]]" = OrderedDict()
        # Thread-safe lock for search cache operations
        self._search_cache_lock = threading.RLock()
        # Touch/refresh control for graph nodes
        self._touch_min_interval_s: float = float(os.getenv("GRAPH_TOUCH_MIN_INTERVAL_S", 30.0))
        self._touch_max_batch: int = int(os.getenv("GRAPH_TOUCH_MAX_BATCH", 64))
        self._touch_extend_seconds: float = float(os.getenv("GRAPH_TOUCH_EXTEND_SECONDS", 0) or 0.0)
        self._touch_last: dict[str, float] = {}
        self._touch_last_max: int = 2048
        self._touch_lock = threading.RLock()
        self._graph_touch_tenant: Optional[str] = os.getenv("GRAPH_TOUCH_TENANT_ID")
        # Relation base weights for graph contribution (lowercase rel names)
        self._rel_base_weights: dict[str, float] = {
            "temporal_next": 0.8,
            "occurs_at": 0.6,
            "describes": 0.6,
            "said_by": 0.6,
            "appears_in": 0.4,
            "co_occurs": 0.4,
            "equivalence": 0.5,
            "executed": 0.5,
            "located_in": 0.5,
            "references": 0.5,
            "prefer": 0.5,
            "default": 1.0,
        }
        # write batching
        self._batch_enabled: bool = False
        self._batch_max_items: int = 50
        self._batch_max_bytes: int = 8_000_000  # 8MB default guardrail (can be overridden)
        self._batch_flush_chunk_items: int = 512  # chunk writes to reduce pressure
        self._batch_flush_chunk_bytes: int = 2_000_000  # 2MB per-chunk budget
        self._batch_max_pending: int = 10_000
        self._batch_entries: list[MemoryEntry] = []
        self._batch_links: list[Edge] = []
        self._batch_bytes: int = 0
        # safety / editing policy
        self._safety_sensitive_rels: list[str] = ["equivalence"]
        self._safety_require_confirm_hard_delete: bool = False
        self._safety_require_confirm_sensitive_link: bool = True
        self._safety_require_reason_delete: bool = False
        self._safety_confirmer: Optional[Callable[[Dict[str, Any]], bool]] = None
        # load cache/batch defaults from config when available
        try:
            cfg2 = load_memory_config()
            cache_cfg = ((cfg2.get("memory", {}) or {}).get("search", {}) or {}).get("cache", {}) or {}
            self._search_cache_enabled = bool(cache_cfg.get("enabled", True))
            self._search_cache_ttl_s = int(cache_cfg.get("ttl_seconds", 60))
            self._search_cache_max = int(cache_cfg.get("max_entries", 256))
            batch_cfg = ((cfg2.get("memory", {}) or {}).get("write", {}) or {}).get("batch", {}) or {}
            self._batch_enabled = bool(batch_cfg.get("enabled", False))
            self._batch_max_items = int(batch_cfg.get("max_items", 50))
            # optional guardrails (with safe defaults)
            try:
                self._batch_max_bytes = int(batch_cfg.get("max_bytes", self._batch_max_bytes))
            except Exception:
                pass
            try:
                self._batch_flush_chunk_items = int(batch_cfg.get("flush_chunk_items", self._batch_flush_chunk_items))
            except Exception:
                pass
            try:
                self._batch_flush_chunk_bytes = int(batch_cfg.get("flush_chunk_bytes", self._batch_flush_chunk_bytes))
            except Exception:
                pass
            try:
                self._batch_max_pending = int(batch_cfg.get("max_pending", self._batch_max_pending))
            except Exception:
                pass
            # sampling config
            samp_cfg = ((cfg2.get("memory", {}) or {}).get("search", {}) or {}).get("sampling", {}) or {}
            self._sampling_enabled = bool(samp_cfg.get("enabled", False))
            self._sampling_rate = float(samp_cfg.get("rate", 0.05))
        except Exception:
            pass
        # search sampling
        self._search_sampler: Optional[Callable[[Dict[str, Any]], None]] = None
        # optional event publisher injection: Callable[[str, Dict[str, Any]], None]
        self._event_publisher: Optional[Callable[[str, Dict[str, Any]], None]] = None
        # relation whitelist（P1 最小安全）：仅允许以下关系类型
        self._allowed_rel_types: set[str] = {
            "appears_in",
            "said_by",
            "located_in",
            "equivalence",
            "prefer",
            "executed",
            "describes",
            "temporal_next",
            "co_occurs",
        }

    async def embed_query(self, query: str, *, tenant_id: Optional[str] = None) -> Optional[List[float]]:
        embed_query = getattr(self.vectors, "embed_query", None)
        if callable(embed_query):
            try:
                vec = await embed_query(str(query or ""), tenant_id=tenant_id)
                if isinstance(vec, list) and vec:
                    return list(vec)
            except Exception:
                return None
        embed_text = getattr(self.vectors, "embed_text", None)
        if not callable(embed_text):
            return None
        try:
            vec = await asyncio.to_thread(embed_text, str(query or ""))
        except Exception:
            return None
        if isinstance(vec, list) and vec:
            return list(vec)
        return None

    async def graph_upsert_v0(self, body: "GraphUpsertRequest") -> None:
        """Best-effort local Graph upsert entry (same surface as HTTP /graph/v0/upsert).

        This keeps the in-process client pipeline aligned with the server surface:
        - If underlying graph store supports `upsert_graph_v0`, delegate via GraphService (enforces invariants).
        - Otherwise, raise NotImplementedError so caller can degrade cleanly.
        """
        graph_upsert_fn = getattr(self.graph, "upsert_graph_v0", None)
        if not callable(graph_upsert_fn):
            raise NotImplementedError("graph_store.upsert_graph_v0 is not available")
        # Avoid import cycles at module import time.
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        # Pass vector_store to enable TKG vector writes (Event/Entity -> Qdrant)
        await GraphService(self.graph, gating=gating, vector_store=self.vectors).upsert(body)

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
        if not callable(getattr(self.graph, "query_events", None)):
            raise NotImplementedError("graph_list_events requires a graph store with query_events()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).list_events(
            tenant_id=str(tenant_id),
            segment_id=segment_id,
            entity_id=entity_id,
            place_id=place_id,
            source_id=source_id,
            limit=int(limit),
        )

    async def graph_list_places(
        self,
        *,
        tenant_id: str,
        name: Optional[str] = None,
        segment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        if not callable(getattr(self.graph, "query_places", None)):
            raise NotImplementedError("graph_list_places requires a graph store with query_places()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).list_places(
            tenant_id=str(tenant_id),
            name=name,
            segment_id=segment_id,
            limit=int(limit),
        )

    async def graph_event_detail(self, *, tenant_id: str, event_id: str) -> dict:
        if not callable(getattr(self.graph, "query_event_detail", None)):
            raise NotImplementedError("graph_event_detail requires a graph store with query_event_detail()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).event_detail(
            tenant_id=str(tenant_id),
            event_id=str(event_id),
        )

    async def graph_place_detail(self, *, tenant_id: str, place_id: str) -> dict:
        if not callable(getattr(self.graph, "query_place_detail", None)):
            raise NotImplementedError("graph_place_detail requires a graph store with query_place_detail()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).place_detail(
            tenant_id=str(tenant_id),
            place_id=str(place_id),
        )

    async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:
        if not callable(getattr(self.graph, "query_event_evidence", None)):
            raise NotImplementedError("graph_explain_event_evidence requires a graph store with query_event_evidence()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).explain_event_evidence(
            tenant_id=str(tenant_id),
            event_id=str(event_id),
        )

    async def graph_search_v1(
        self,
        *,
        tenant_id: str,
        query: str,
        topk: int = 10,
        source_id: Optional[str] = None,
        include_evidence: bool = True,
    ) -> dict:
        if not callable(getattr(self.graph, "search_event_candidates", None)):
            raise NotImplementedError("graph_search_v1 requires a graph store with search_event_candidates()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).search_events_v1(
            tenant_id=str(tenant_id),
            query=str(query),
            topk=int(topk),
            source_id=source_id,
            include_evidence=bool(include_evidence),
        )

    async def graph_resolve_entities(
        self,
        *,
        tenant_id: str,
        name: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        if not callable(getattr(self.graph, "query_entities_by_name", None)):
            raise NotImplementedError("graph_resolve_entities requires a graph store with query_entities_by_name()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).resolve_entities(
            tenant_id=str(tenant_id),
            name=str(name),
            entity_type=(str(entity_type) if entity_type is not None else None),
            limit=int(limit),
        )

    async def graph_list_timeslices_range(
        self,
        *,
        tenant_id: str,
        start_iso: Optional[str],
        end_iso: Optional[str],
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        if not callable(getattr(self.graph, "query_time_slices_by_range", None)):
            raise NotImplementedError("graph_list_timeslices_range requires a graph store with query_time_slices_by_range()")
        from modules.memory.application.graph_service import GraphService

        cfg = load_memory_config()
        gating = get_graph_settings(cfg)
        return await GraphService(self.graph, gating=gating).list_time_slices_by_range(
            tenant_id=str(tenant_id),
            start_iso=start_iso,
            end_iso=end_iso,
            kind=kind,
            limit=int(limit),
        )

    # ---- Lifecycle management ----
    def close(self) -> None:
        """Close underlying stores if they expose close(); idempotent."""
        for obj in (getattr(self, "vectors", None), getattr(self, "graph", None), getattr(self, "audit", None)):
            if obj is None:
                continue
            try:
                fn = getattr(obj, "close", None)
                if callable(fn):
                    fn()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---- Safety policy ----
    def set_safety_policy(
        self,
        *,
        sensitive_rel_types: Optional[list[str]] = None,
        require_confirm_hard_delete: Optional[bool] = None,
        require_confirm_sensitive_link: Optional[bool] = None,
        require_reason_delete: Optional[bool] = None,
    ) -> None:
        if sensitive_rel_types is not None:
            self._safety_sensitive_rels = list(sensitive_rel_types)
        if require_confirm_hard_delete is not None:
            self._safety_require_confirm_hard_delete = bool(require_confirm_hard_delete)
        if require_confirm_sensitive_link is not None:
            self._safety_require_confirm_sensitive_link = bool(require_confirm_sensitive_link)
        if require_reason_delete is not None:
            self._safety_require_reason_delete = bool(require_reason_delete)

    def set_safety_confirmer(self, confirmer: Optional[Callable[[Dict[str, Any]], bool]]) -> None:
        """Inject confirmer callback. It receives a context dict and returns True/False."""
        self._safety_confirmer = confirmer

    def set_update_decider(self, decider):
        self.update_decider = decider

    def set_event_publisher(self, publisher: Callable[[str, Dict[str, Any]], None] | None) -> None:
        """Inject an event publisher.

        If None, no events will be published from this service instance.
        """
        self._event_publisher = publisher

    # ---- Search sampling ----
    def set_search_sampler(self, sampler: Callable[[Dict[str, Any]], None] | None, *, enabled: Optional[bool] = None, rate: Optional[float] = None) -> None:
        if enabled is not None:
            self._sampling_enabled = bool(enabled)
        if rate is not None:
            try:
                self._sampling_rate = max(0.0, min(1.0, float(rate)))
            except Exception:
                pass
        self._search_sampler = sampler

    # ---- Search cache controls ----
    def set_search_cache(self, *, enabled: bool | None = None, ttl_seconds: int | None = None, max_entries: int | None = None) -> None:
        if enabled is not None:
            self._search_cache_enabled = bool(enabled)
        if ttl_seconds is not None:
            self._search_cache_ttl_s = int(ttl_seconds)
        if max_entries is not None:
            self._search_cache_max = int(max_entries)

    def _search_cache_key(
        self,
        query: str,
        topk: int,
        filters: Optional[SearchFilters | Dict[str, Any]],
        expand_graph: bool,
        graph_backend: Optional[str],
        threshold: Optional[float],
        scope: Optional[str] = None,
        graph_sig: Optional[Dict[str, Any]] = None,
        weight_sig: Optional[str] = None,
    ) -> str:
        import json
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        else:
            fdict = dict(filters or {})
        try:
            ftxt = json.dumps(fdict, ensure_ascii=False, sort_keys=True)
        except Exception:
            ftxt = str(fdict)
        gs_txt = ""
        if graph_sig is not None:
            try:
                gs_txt = json.dumps(graph_sig, ensure_ascii=False, sort_keys=True)
            except Exception:
                gs_txt = str(graph_sig)
        gb = str(graph_backend or "memory").strip().lower()
        return f"q={query}|k={topk}|exp={int(bool(expand_graph))}|gb={gb}|th={threshold}|scope={scope}|g={gs_txt}|w={weight_sig}|f={ftxt}"

    def _search_cache_get(self, key: str) -> SearchResult | None:
        import time as _t
        with self._search_cache_lock:
            item = self._search_cache.get(key)
            if not item:
                return None
            exp, res = item
            if _t.time() > exp:
                # expired
                self._search_cache.pop(key, None)
                return None
            try:
                inc("search_cache_hits_total", 1)
            except Exception:
                pass
            # LRU: move to end on access
            try:
                self._search_cache.move_to_end(key)
            except Exception:
                pass
            return res

    def _search_cache_put(self, key: str, res: SearchResult) -> None:
        import time as _t
        with self._search_cache_lock:
            # evict if overflow (LRU: popitem(last=False))
            while len(self._search_cache) >= max(1, self._search_cache_max):
                try:
                    self._search_cache.popitem(last=False)
                    inc("search_cache_evictions_total", 1)
                except Exception:
                    break
            self._search_cache[key] = (_t.time() + max(1, self._search_cache_ttl_s), res)

    def _get_cached_config(self) -> Dict[str, Any]:
        """获取缓存的配置对象，避免重复磁盘I/O。"""
        import time as _t
        if self._cfg_cache is None or (_t.time() - self._cfg_cache_time) > self._cfg_cache_ttl:
            self._cfg_cache = load_memory_config()
            self._cfg_cache_time = _t.time()
        return self._cfg_cache

    def _invalidate_cfg_cache(self) -> None:
        """手动使配置缓存失效（用于测试或配置热更新）。"""
        self._cfg_cache = None

    @staticmethod
    def _content_hash(contents: list[str]) -> str:
        import hashlib
        text = "\n".join([str(c) for c in contents])
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _tkg_event_id_from_entry_meta(meta: Dict[str, Any]) -> Optional[str]:
        """Best-effort mapping from MemoryEntry metadata -> TKG Event ID.

        Prefer explicit `tkg_event_id` (e.g. utterance index); otherwise derive from dialog pipeline fields.
        """
        ev = str(meta.get("tkg_event_id") or "").strip()
        if ev:
            return ev

        tenant_id = str(meta.get("tenant_id") or "").strip()
        if not tenant_id:
            return None

        sample_id = str(meta.get("source_sample_id") or meta.get("sample_id") or meta.get("run_id") or "").strip()
        if not sample_id:
            return None

        candidates: list[object] = []
        if meta.get("turn_index") is not None:
            candidates.append(meta.get("turn_index"))
        if meta.get("turn") is not None:
            candidates.append(meta.get("turn"))
        stids = meta.get("source_turn_ids")
        if isinstance(stids, list):
            candidates.extend(list(stids))

        def _parse_turn(v: object) -> Optional[int]:
            if v is None:
                return None
            if isinstance(v, int):
                return int(v) if v > 0 else None
            s = str(v).strip()
            if not s:
                return None
            for sep in (":", "_"):
                if sep in s:
                    try:
                        n = int(s.split(sep)[-1])
                        return n if n > 0 else None
                    except Exception:
                        return None
            try:
                n = int(s)
                return n if n > 0 else None
            except Exception:
                return None

        for c in candidates:
            n = _parse_turn(c)
            if n is None:
                continue
            try:
                from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid
            except Exception:
                return None
            return generate_uuid("tkg.dialog.event", f"{tenant_id}|{sample_id}|{n}")
        return None

    async def _expand_neighbors_tkg_via_explain(
        self,
        *,
        seed_ids: List[str],
        id_to_entry: Dict[str, MemoryEntry],
        tenant_id: str,
        neighbor_cap_per_seed: int,
        topn_seeds: int,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build a neighbors structure from TKG explain payload (best-effort).

        Output shape matches graph_store.expand_neighbors:
          {"neighbors": {seed_id: [{"to": id, "rel": str, "weight": float, "hop": int}, ...]}, "edges": []}
        """
        started = time.perf_counter()
        out: Dict[str, Any] = {"neighbors": {}, "edges": []}
        trace: Dict[str, Any] = {"tkg_explain_calls": 0, "tkg_explain_mapped_seeds": 0}

        seeds = [sid for sid in seed_ids if sid in id_to_entry]
        seeds = seeds[: max(0, int(topn_seeds))]
        for sid in seeds:
            entry = id_to_entry.get(sid)
            if entry is None:
                continue
            meta = dict(entry.metadata or {})
            event_id = self._tkg_event_id_from_entry_meta(meta)
            if not event_id:
                continue
            trace["tkg_explain_mapped_seeds"] = int(trace.get("tkg_explain_mapped_seeds", 0)) + 1
            try:
                ex = await self.graph_explain_event_evidence(tenant_id=str(tenant_id), event_id=str(event_id))
                trace["tkg_explain_calls"] = int(trace.get("tkg_explain_calls", 0)) + 1
            except Exception:
                continue

            nbrs_raw: List[Dict[str, Any]] = []
            for it in list(ex.get("utterances") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "SUPPORTED_BY", "weight": 1.0, "hop": 1})
            for it in list(ex.get("evidences") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "SUPPORTED_BY", "weight": 1.0, "hop": 1})
            for it in list(ex.get("entities") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "INVOLVES", "weight": 1.0, "hop": 1})
            for it in list(ex.get("places") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "OCCURS_AT", "weight": 1.0, "hop": 1})
            for it in list(ex.get("timeslices") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "COVERS_EVENT", "weight": 1.0, "hop": 1})
            for it in list(ex.get("knowledge") or []):
                tid = str((it or {}).get("id") or "").strip()
                if tid:
                    nbrs_raw.append({"to": tid, "rel": "DERIVED_FROM", "weight": 1.0, "hop": 1})

            merged: Dict[str, Dict[str, Any]] = {}
            for n in nbrs_raw:
                to_id = str(n.get("to") or "").strip()
                if not to_id:
                    continue
                w = float(n.get("weight") or 0.0)
                hop = int(n.get("hop") or 1)
                prev = merged.get(to_id)
                if prev is None or w > float(prev.get("weight") or 0.0):
                    merged[to_id] = dict(n)
                else:
                    try:
                        prev_hop = int(prev.get("hop") or hop)
                        prev["hop"] = min(prev_hop, hop)
                    except Exception:
                        prev["hop"] = hop

            nbrs = list(merged.values())
            nbrs.sort(key=lambda x: float(x.get("weight") or 0.0), reverse=True)
            out["neighbors"][sid] = nbrs[: max(0, int(neighbor_cap_per_seed))]

        trace["tkg_explain_latency_ms"] = (time.perf_counter() - started) * 1000
        return out, trace

    async def search(
        self,
        query: str,
        *,
        topk: int = 10,
        filters: Optional[SearchFilters] = None,
        expand_graph: bool = True,
        graph_backend: str = "memory",
        threshold: Optional[float] = None,
        scope: Optional[str] = None,
        graph_params: Optional[Dict[str, Any]] = None,
        query_vector: Optional[List[float]] = None,
    ) -> SearchResult:
        """ANN → graph neighborhood expansion → re-rank → hints.

        - Vector recall via vector_store
        - Optional neighbor expansion via graph_store
        - Re-rank: vector + BM25 + graph + recency
        - Hints: include evidence and brief explanation
        """
        t0 = time.perf_counter()
        # clamp topk 到安全上限，防止单次请求拉取过多结果
        try:
            topk = int(topk or 10)
        except Exception:
            topk = 10
        if topk < 1:
            topk = 1
        if topk > SEARCH_TOPK_HARD_LIMIT:
            topk = SEARCH_TOPK_HARD_LIMIT
        
        # ---- Graph Touch (Move here) ----
        # Move graph touch to BEFORE search core logic to avoid potential unhandled errors during touch
        # causing search to be perceived as broken if exceptions bubble up.
        # Actually, touching AFTER search is better for freshness, but let's wrap it carefully.
        # The issue might be related to how `asyncio.create_task` interacts with server shutdown/errors.
        # However, looking at the logs, the 500 happens during IMPORT (/write), not search.
        # So this change is just minor cleanup.
        
        # hot reload weights from config on each search (使用缓存避免重复磁盘I/O)
        try:
            cfg = self._get_cached_config()  # 使用缓存的配置
            w = get_search_weights(cfg)
            # runtime overrides (if any) take precedence
            w_override = rtconf.get_rerank_weights_override()
            if w_override:
                # 直接覆盖，允许配置里没有但override想加的键（例如delta_recency）
                w.update(w_override)
            self._w_alpha = w.get("alpha_vector", self._w_alpha)
            self._w_beta = w.get("beta_bm25", self._w_beta)
            self._w_gamma = w.get("gamma_graph", self._w_gamma)
            self._w_delta = w.get("delta_recency", self._w_delta)
            self._w_user_boost = w.get("user_boost", getattr(self, "_w_user_boost", 0.0))
            self._w_domain_boost = w.get("domain_boost", getattr(self, "_w_domain_boost", 0.0))
            self._w_session_boost = w.get("session_boost", getattr(self, "_w_session_boost", 0.0))
            lex = resolve_lexical_hybrid_settings(
                cfg,
                rtconf.get_lexical_hybrid_override(),
            )
            lexical_sig = (
                f"{int(bool(lex.get('enabled', False)))}_"
                f"{int(lex.get('corpus_limit', 500) or 500)}_"
                f"{int(lex.get('lexical_topn', 50) or 50)}_"
                f"{int(bool(lex.get('normalize_scores', True)))}"
            )
        except Exception:
            lexical_sig = "na"
        # 先使用 filters.threshold（如果提供），否则回落到显式参数/配置默认值
        try:
            if threshold is None and isinstance(filters, SearchFilters):
                fthr = filters.threshold
                if fthr is not None and str(fthr).strip() != "" and not str(fthr).strip().startswith("${"):
                    threshold = float(fthr)
        except Exception:
            # ignore and continue to config lookup
            pass
        # Priority: explicit arg > filters.threshold > config default
        if threshold is None and filters is not None and filters.threshold is not None:
            threshold = filters.threshold

        # Default threshold from config if not provided by caller
        threshold_from_config = False
        relax_threshold_on_empty = False
        try:
            if threshold is None:
                cfg = self._get_cached_config()  # 使用缓存的配置
                ann_cfg = (((cfg.get("memory", {}) or {}).get("search", {}) or {}).get("ann", {}) or {})
                # Optional guardrail: only relax threshold if explicitly enabled.
                relax_threshold_on_empty = bool(ann_cfg.get("relax_threshold_on_empty", False))
                thr = ann_cfg.get("threshold")
                if thr is not None and str(thr).strip() != "" and not str(thr).strip().startswith("${"):
                    threshold = float(thr)
                    threshold_from_config = True
        except Exception:
            # keep threshold as None on any error
            pass
        # Build a stable weight signature for cache key differentiation
        weight_sig = f"{self._w_alpha:.4f}_{self._w_beta:.4f}_{self._w_gamma:.4f}_{self._w_delta:.4f}|lex={lexical_sig}"

        # Build scoping attempts and fallback (session -> domain -> user by default)
        base_filters: Dict[str, Any] = filters.model_dump(exclude_none=True) if isinstance(filters, SearchFilters) else {}
        if "published" not in base_filters:
            base_filters["published"] = True
        # P2: character expansion from query → filters.character_id; widen modalities and strip markers
        q = str(query or "")
        try:
            # read toggle from config (default: enabled)
            ch_cfg = (((cfg.get("memory", {}) or {}).get("search", {}) or {}).get("character_expansion", {}) or {})
            ch_enabled = bool(ch_cfg.get("enabled", True))
        except Exception:
            ch_enabled = True
        if ch_enabled:
            try:
                import re as _re
                chars: list[str] = []
                for m in _re.finditer(r"(?:^|\s)(?:character:|<character_)([A-Za-z0-9_\-]+)(?:>|\b)", q):
                    name = m.group(1)
                    if name:
                        chars.append(str(name))
                if chars:
                    prev = base_filters.get("character_id") or []
                    if isinstance(prev, list):
                        base_filters["character_id"] = list(sorted(set([str(x) for x in (prev + chars)])))
                    else:
                        base_filters["character_id"] = list(sorted(set(chars)))
                    if not base_filters.get("modality"):
                        base_filters["modality"] = ["text", "image", "audio"]
                    q = _re.sub(r"(?:^|\s)(?:character:|<character_)([A-Za-z0-9_\-]+)(?:>|\b)", " ", q).strip()
            except Exception:
                q = str(query or "")
        else:
            q = str(query or "")
        req_user = base_filters.get("user_id")
        req_dom = base_filters.get("memory_domain")
        req_run = base_filters.get("run_id")
        user_match = (base_filters.get("user_match") or "any").lower()

        # scoping config defaults (will be formalized in config step)
        scfg = {}
        try:
            scfg = (cfg.get("memory", {}) or {}).get("search", {}).get("scoping", {})  # type: ignore[name-defined]
        except Exception:
            try:
                scfg = (load_memory_config().get("memory", {}) or {}).get("search", {}).get("scoping", {})
            except Exception:
                scfg = {}
        # overlay runtime scoping override
        try:
            s_override = rtconf.get_scoping_override()
            if s_override:
                scfg = {**scfg, **s_override}
        except Exception:
            pass
        default_scope = str(scfg.get("default_scope", "domain")).lower()
        fallback_order = list(scfg.get("fallback_order", ["session", "domain", "user"]))
        require_user = bool(scfg.get("require_user", False))
        user_match_mode_default = str(scfg.get("user_match_mode", "any")).lower()
        if "user_match" not in base_filters:
            user_match = user_match_mode_default

        # ANN default modalities handling (if modality not specified)
        if "modality" not in base_filters or not base_filters.get("modality"):
            try:
                from modules.memory.application.config import get_ann_settings  # type: ignore
                ann_cfg = get_ann_settings(cfg)
            except Exception:
                ann_cfg = {"default_modalities": None, "default_all_modalities": False}
            # runtime override has priority
            try:
                ann_override = rtconf.get_ann_override()
            except Exception:
                ann_override = {}
            dm = ann_override.get("default_modalities", ann_cfg.get("default_modalities"))
            all_flag = bool(ann_override.get("default_all_modalities", ann_cfg.get("default_all_modalities", False)))
            if dm is not None and isinstance(dm, list) and dm:
                base_filters["modality"] = list(dm)
            elif all_flag:
                base_filters["modality"] = ["text", "image", "audio"]
            else:
                base_filters["modality"] = ["text"]

        def viable(sc: str) -> bool:
            if sc == "session":
                return bool(req_run)
            if sc == "domain":
                return bool(req_dom)
            if sc == "user":
                return bool(req_user)
            if sc == "global":
                return True
            return False

        chosen = (scope or default_scope or "domain").lower()
        order = [chosen] + [s for s in fallback_order if s != chosen]

        # enforce require_user unless using 'global'
        if require_user and not req_user and chosen != "global":
            # return empty result early
            empty = SearchResult(hits=[], neighbors={}, hints="", trace={"scope_used": None, "attempts": []})
            return empty

        residue = {k: v for k, v in base_filters.items() if k not in {"user_id", "memory_domain", "run_id", "user_match"}}

        attempts: list[tuple[str, Dict[str, Any]]] = []
        for sc in order:
            if not viable(sc):
                continue
            f: Dict[str, Any] = dict(residue)
            if sc in {"session", "domain", "user"}:
                if req_user:
                    f["user_id"] = list(req_user)
                    f["user_match"] = user_match
            if sc in {"session", "domain"} and req_dom is not None:
                f["memory_domain"] = req_dom
            if sc == "session" and req_run is not None:
                f["run_id"] = req_run
            attempts.append((sc, f))
        if not attempts and not require_user:
            # final open fallback (backward compatibility): no scoping filters
            attempts.append(("open", dict(residue)))

        vec_results = []
        applied_scope = None
        applied_filters: Dict[str, Any] = {}
        trace_attempts = []

        # Prepare a graph signature for cache key to avoid stale hits across graph-toggles
        try:
            cfg_g = get_graph_settings(cfg if 'cfg' in locals() else load_memory_config())
        except Exception:
            cfg_g = {
                "expand": True,
                "max_hops": 1,
                "rel_whitelist": [],
                "neighbor_cap_per_seed": 5,
                "restrict_to_user": True,
                "restrict_to_domain": True,
                "restrict_to_scope": True,
                "allow_cross_user": False,
                "allow_cross_domain": False,
                "allow_cross_scope": False,
            }
        # merge runtime override + explicit graph_params
        try:
            govr = rtconf.get_graph_params_override()
            if govr.get("rel_whitelist") is not None:
                cfg_g["rel_whitelist"] = list(govr.get("rel_whitelist") or [])
            if govr.get("max_hops") is not None:
                cfg_g["max_hops"] = int(govr.get("max_hops"))
            if govr.get("neighbor_cap_per_seed") is not None:
                cfg_g["neighbor_cap_per_seed"] = int(govr.get("neighbor_cap_per_seed"))
            if govr.get("restrict_to_user") is not None:
                cfg_g["restrict_to_user"] = bool(govr.get("restrict_to_user"))
            if govr.get("restrict_to_domain") is not None:
                cfg_g["restrict_to_domain"] = bool(govr.get("restrict_to_domain"))
            if govr.get("allow_cross_user") is not None:
                cfg_g["allow_cross_user"] = bool(govr.get("allow_cross_user"))
            if govr.get("allow_cross_domain") is not None:
                cfg_g["allow_cross_domain"] = bool(govr.get("allow_cross_domain"))
            if govr.get("allow_cross_scope") is not None:
                cfg_g["allow_cross_scope"] = bool(govr.get("allow_cross_scope"))
        except Exception:
            pass
        if graph_params and isinstance(graph_params, dict):
            if graph_params.get("rel_whitelist") is not None:
                cfg_g["rel_whitelist"] = list(graph_params.get("rel_whitelist") or [])
            if graph_params.get("max_hops") is not None:
                cfg_g["max_hops"] = int(graph_params.get("max_hops"))
            if graph_params.get("neighbor_cap_per_seed") is not None:
                cfg_g["neighbor_cap_per_seed"] = int(graph_params.get("neighbor_cap_per_seed"))
            if graph_params.get("restrict_to_user") is not None:
                cfg_g["restrict_to_user"] = bool(graph_params.get("restrict_to_user"))
            if graph_params.get("restrict_to_domain") is not None:
                cfg_g["restrict_to_domain"] = bool(graph_params.get("restrict_to_domain"))
            if graph_params.get("allow_cross_user") is not None:
                cfg_g["allow_cross_user"] = bool(graph_params.get("allow_cross_user"))
            if graph_params.get("allow_cross_domain") is not None:
                cfg_g["allow_cross_domain"] = bool(graph_params.get("allow_cross_domain"))
            if graph_params.get("allow_cross_scope") is not None:
                cfg_g["allow_cross_scope"] = bool(graph_params.get("allow_cross_scope"))
        # effective restrictions reflect allow_cross_* toggles
        gsig = {
            "expand": bool(expand_graph and cfg_g.get("expand", True)),
            "max_hops": int(cfg_g.get("max_hops", 1)),
            "cap": int(cfg_g.get("neighbor_cap_per_seed", 5)),
            "rel_whitelist": list(cfg_g.get("rel_whitelist", []) or []),
            "restrict_to_user": bool(cfg_g.get("restrict_to_user", True) and not cfg_g.get("allow_cross_user", False)),
            "restrict_to_domain": bool(cfg_g.get("restrict_to_domain", True) and not cfg_g.get("allow_cross_domain", False)),
        }

        # try cache per attempt, then query backend
        async def _search_vectors_with_optional_query_vector(
            q_text: str,
            payload_filters: Dict[str, Any],
            payload_topk: int,
            payload_threshold: Optional[float],
        ) -> List[Dict[str, Any]]:
            if query_vector is not None:
                try:
                    return await self.vectors.search_vectors(
                        q_text,
                        payload_filters,
                        payload_topk,
                        payload_threshold,
                        query_vector=query_vector,
                    )
                except TypeError as exc:
                    # Backward compatibility for custom/fake vector stores used in tests.
                    if "query_vector" not in str(exc):
                        raise
                    inc("query_vector_fallback_total", 1)
                    inc("query_vector_fallback_service_total", 1)
                    logging.getLogger(__name__).warning(
                        "memory.search query_vector unsupported by vector store=%s; falling back to search without precomputed vector: %s",
                        type(self.vectors).__name__,
                        str(exc)[:200],
                    )
            return await self.vectors.search_vectors(q_text, payload_filters, payload_topk, payload_threshold)

        for sc, f in attempts:
            if self._search_cache_enabled:
                key_try = self._search_cache_key(q, topk, f, expand_graph, graph_backend, threshold, scope=sc, graph_sig=gsig, weight_sig=weight_sig)
                res_cached = self._search_cache_get(key_try)
                if res_cached is not None:
                    try:
                        inc(f"search_cache_hits_scope_{sc}_total", 1)
                    except Exception:
                        pass
                    # enrich trace and return
                    cached_trace = dict(res_cached.trace)
                    cached_trace["from_cache"] = True
                    cached_trace["scope_used"] = sc
                    # keep previous attempts if present, append current
                    prev_attempts = list(cached_trace.get("attempts") or [])
                    prev_attempts.append({"scope": sc, "filters": f, "vec_hits": len(getattr(res_cached, 'hits', []) or [])})
                    cached_trace["attempts"] = prev_attempts
                    res_cached.trace = cached_trace
                    return res_cached
            try:
                inc("search_cache_misses_total", 1)
            except Exception:
                pass
            vec_results = await _search_vectors_with_optional_query_vector(q, f, topk, threshold)
            # If config-default threshold filters everything out, retry without score_threshold.
            if (not vec_results) and threshold is not None and threshold_from_config and relax_threshold_on_empty:
                try:
                    vec_results = await _search_vectors_with_optional_query_vector(q, f, topk, None)
                    trace_attempts.append(
                        {
                            "scope": sc,
                            "filters": f,
                            "vec_hits": 0,
                            "relaxed_threshold": True,
                            "vec_hits_relaxed": len(vec_results) if isinstance(vec_results, list) else 0,
                        }
                    )
                except Exception:
                    pass
            # record attempt with hit count
            try:
                trace_attempts.append({"scope": sc, "filters": f, "vec_hits": len(vec_results) if isinstance(vec_results, list) else 0})
            except Exception:
                trace_attempts.append({"scope": sc, "filters": f})
            if vec_results:
                applied_scope = sc
                applied_filters = f
                break
        # fallback: if session scope with run_id had no hits, retry without run_id (domain/user)
        if (not vec_results) and req_run is not None and attempts:
            f_fb = dict(residue)
            if req_user:
                f_fb["user_id"] = list(req_user)
                f_fb["user_match"] = user_match
            if req_dom is not None:
                f_fb["memory_domain"] = req_dom
            vec_results = await _search_vectors_with_optional_query_vector(q, f_fb, topk, threshold)
            if (not vec_results) and threshold is not None and threshold_from_config and relax_threshold_on_empty:
                try:
                    vec_results = await _search_vectors_with_optional_query_vector(q, f_fb, topk, None)
                    trace_attempts.append(
                        {
                            "scope": "session_fallback",
                            "filters": f_fb,
                            "vec_hits": 0,
                            "relaxed_threshold": True,
                            "vec_hits_relaxed": len(vec_results) if isinstance(vec_results, list) else 0,
                        }
                    )
                except Exception:
                    pass
            trace_attempts.append({"scope": "session_fallback", "filters": f_fb, "vec_hits": len(vec_results) if isinstance(vec_results, list) else 0})
            if vec_results:
                applied_scope = "domain"
                applied_filters = f_fb
        # if still empty, preserve last attempted filters (if any)
        if applied_scope is None and attempts:
            applied_scope = attempts[-1][0]
            applied_filters = attempts[-1][1]

        # downstream use fdict as applied filters
        fdict: Dict[str, Any] = dict(applied_filters)
        # metrics: scope + filter use
        try:
            if applied_scope:
                inc(f"search_scope_used_{applied_scope}_total", 1)
            if fdict.get("user_id"):
                inc("search_filter_applied_user_total", 1)
            if fdict.get("memory_domain") is not None:
                inc("search_filter_applied_domain_total", 1)
            if fdict.get("run_id") is not None:
                inc("search_filter_applied_session_total", 1)
        except Exception:
            pass

        lexical_cfg = resolve_lexical_hybrid_settings(
            cfg,
            rtconf.get_lexical_hybrid_override(),
        )

        lexical_enabled = bool(lexical_cfg.get("enabled", False))
        lexical_corpus_limit = max(1, int(lexical_cfg.get("corpus_limit", 500) or 500))
        lexical_topn = max(1, int(lexical_cfg.get("lexical_topn", 50) or 50))
        normalize_scores = bool(lexical_cfg.get("normalize_scores", True))

        id_to_entry: Dict[str, MemoryEntry] = {}
        raw_vec_scores: Dict[str, float] = {}
        for r in vec_results:
            me = r.get("payload")
            rid = str(r.get("id"))
            if not me or not rid:
                continue
            id_to_entry[rid] = me
            raw_vec_scores[rid] = float(r.get("score", 0.0))

        query_tokens = _bm25_tokenize(q)
        bm25_source = "vector_hits"
        bm25_raw_scores: Dict[str, float] = {}
        lexical_hits_added = 0
        lexical_pool_size = 0
        lexical_positive_hits = 0
        lexical_pool_map: Dict[str, MemoryEntry] = dict(id_to_entry)

        if lexical_enabled:
            fetch_text_corpus = getattr(self.vectors, "fetch_text_corpus", None)
            if callable(fetch_text_corpus):
                try:
                    lexical_pool = await fetch_text_corpus(fdict, limit=lexical_corpus_limit)
                except Exception:
                    lexical_pool = []
                lexical_pool_size = len(lexical_pool)
                if lexical_pool:
                    bm25_source = "hybrid_corpus"
                    for item in lexical_pool:
                        rid = str(item.get("id"))
                        me = item.get("payload")
                        if not rid or not me:
                            continue
                        lexical_pool_map.setdefault(rid, me)

        corpus_ids = list(lexical_pool_map.keys())
        corpus = []
        for rid in corpus_ids:
            me = lexical_pool_map[rid]
            txt = " ".join(me.contents) if me.contents else ""
            corpus.append(_bm25_tokenize(txt))

        if corpus:
            if BM25Okapi is not None:
                try:
                    bm25 = BM25Okapi(corpus)
                except Exception:
                    if not self._bm25_warned:
                        try:
                            import logging

                            logging.getLogger(__name__).warning(
                                "rank_bm25 initialisation failed – using internal BM25 fallback",
                                exc_info=True,
                            )
                        except Exception:
                            pass
                        self._bm25_warned = True
                    bm25 = _SimpleBM25(corpus)
            else:
                if not self._bm25_warned:
                    try:
                        import logging

                        logging.getLogger(__name__).warning(
                            "rank_bm25 not available – falling back to internal BM25 implementation"
                        )
                    except Exception:
                        pass
                    self._bm25_warned = True
                bm25 = _SimpleBM25(corpus)
            try:
                score_list = bm25.get_scores(query_tokens)
            except Exception:
                score_list = [0.0 for _ in corpus]
            for rid, score in zip(corpus_ids, score_list):
                bm25_raw_scores[rid] = float(score or 0.0)
        if lexical_enabled and bm25_source == "hybrid_corpus":
            ranked_lexical = sorted(
                (
                    (rid, score)
                    for rid, score in bm25_raw_scores.items()
                    if abs(float(score)) > 1e-12
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            lexical_positive_hits = len(ranked_lexical)
            for rid, _score in ranked_lexical[:lexical_topn]:
                me = lexical_pool_map.get(rid)
                if me is None:
                    continue
                if rid not in id_to_entry:
                    lexical_hits_added += 1
                id_to_entry[rid] = me

        vec_scores_for_rank = dict(raw_vec_scores)
        bm25_scores_for_rank = dict(bm25_raw_scores)
        if lexical_enabled and normalize_scores:
            vec_scores_for_rank = _minmax_normalize_score_map(raw_vec_scores)
            bm25_scores_for_rank = _minmax_normalize_score_map(bm25_raw_scores, log1p=True)
        lexical_trace = {
            "enabled": lexical_enabled,
            "bm25_source": bm25_source,
            "query_tokens": query_tokens[:12],
            "corpus_size": len(corpus_ids),
            "lexical_corpus_limit": lexical_corpus_limit,
            "lexical_topn": lexical_topn,
            "lexical_positive_hits": lexical_positive_hits,
            "lexical_candidates_added": lexical_hits_added,
            "normalize_scores": normalize_scores,
        }

        # Graph expansion (neighbors summary)
        neighbors: Dict[str, Any] = {}
        gb_requested = str(graph_backend or "memory").strip().lower()
        if gb_requested not in ("memory", "tkg"):
            gb_requested = "memory"
        gb_used = gb_requested
        tkg_expand_trace: Dict[str, Any] = {}
        gcfg: Dict[str, Any] = {}
        if id_to_entry:
            # resolve graph expansion settings: config -> runtime overrides -> request flag
            gcfg = get_graph_settings(cfg)
            govr = rtconf.get_graph_params_override()
            if govr.get("rel_whitelist") is not None:
                gcfg["rel_whitelist"] = list(govr.get("rel_whitelist") or [])
            if govr.get("max_hops") is not None:
                try:
                    gcfg["max_hops"] = int(govr.get("max_hops"))
                except Exception:
                    pass
            if govr.get("neighbor_cap_per_seed") is not None:
                try:
                    gcfg["neighbor_cap_per_seed"] = int(govr.get("neighbor_cap_per_seed"))
                except Exception:
                    pass
            # allow runtime override for scope toggles
            if govr.get("restrict_to_user") is not None:
                gcfg["restrict_to_user"] = bool(govr.get("restrict_to_user"))
            if govr.get("restrict_to_domain") is not None:
                gcfg["restrict_to_domain"] = bool(govr.get("restrict_to_domain"))
            if govr.get("allow_cross_user") is not None:
                gcfg["allow_cross_user"] = bool(govr.get("allow_cross_user"))
            if govr.get("allow_cross_domain") is not None:
                gcfg["allow_cross_domain"] = bool(govr.get("allow_cross_domain"))
            if govr.get("restrict_to_scope") is not None:
                gcfg["restrict_to_scope"] = bool(govr.get("restrict_to_scope"))
            if govr.get("allow_cross_scope") is not None:
                gcfg["allow_cross_scope"] = bool(govr.get("allow_cross_scope"))
            # request-level graph_params override (highest priority)
            if graph_params and isinstance(graph_params, dict):
                if graph_params.get("rel_whitelist") is not None:
                    gcfg["rel_whitelist"] = list(graph_params.get("rel_whitelist") or [])
                if graph_params.get("max_hops") is not None:
                    try:
                        gcfg["max_hops"] = int(graph_params.get("max_hops"))
                    except Exception:
                        pass
                if graph_params.get("neighbor_cap_per_seed") is not None:
                    try:
                        gcfg["neighbor_cap_per_seed"] = int(graph_params.get("neighbor_cap_per_seed"))
                    except Exception:
                        pass
                if graph_params.get("restrict_to_user") is not None:
                    gcfg["restrict_to_user"] = bool(graph_params.get("restrict_to_user"))
                if graph_params.get("restrict_to_domain") is not None:
                    gcfg["restrict_to_domain"] = bool(graph_params.get("restrict_to_domain"))
                if graph_params.get("allow_cross_user") is not None:
                    gcfg["allow_cross_user"] = bool(graph_params.get("allow_cross_user"))
                if graph_params.get("allow_cross_domain") is not None:
                    gcfg["allow_cross_domain"] = bool(graph_params.get("allow_cross_domain"))
                if graph_params.get("allow_cross_scope") is not None:
                    gcfg["allow_cross_scope"] = bool(graph_params.get("allow_cross_scope"))
            # optional: relation base weights override from config
            try:
                rbw = gcfg.get("rel_base_weights") or {}
                if isinstance(rbw, dict):
                    for k, v in rbw.items():
                        try:
                            self._rel_base_weights[str(k).lower()] = float(v)
                        except Exception:
                            continue
            except Exception:
                pass
            do_expand = bool(expand_graph and gcfg.get("expand", True))
            if do_expand:
                seed_ids = list(id_to_entry.keys())
                # TKG backend: build neighbors from explain payload (best-effort, requires tenant_id).
                if gb_used == "tkg":
                    tenant_for_tkg = str(fdict.get("tenant_id") or "").strip()
                    if not tenant_for_tkg:
                        gb_used = "memory"
                        tkg_expand_trace = {"tkg_expand_skipped_reason": "missing_tenant_id"}
                    else:
                        try:
                            cap = int(gcfg.get("neighbor_cap_per_seed", 5))
                        except Exception:
                            cap = 5
                        topn = 5
                        try:
                            if graph_params and isinstance(graph_params, dict) and graph_params.get("tkg_explain_topn") is not None:
                                topn = int(graph_params.get("tkg_explain_topn") or 5)
                        except Exception:
                            topn = 5
                        topn = max(0, min(int(topn), len(seed_ids)))
                        try:
                            neighbors, tkg_expand_trace = await self._expand_neighbors_tkg_via_explain(
                                seed_ids=seed_ids,
                                id_to_entry=id_to_entry,
                                tenant_id=tenant_for_tkg,
                                neighbor_cap_per_seed=cap,
                                topn_seeds=topn,
                            )
                        except NotImplementedError:
                            gb_used = "memory"
                            tkg_expand_trace = {"tkg_expand_skipped_reason": "unsupported_graph_store"}
                        except Exception:
                            neighbors = {"neighbors": {}}
                            tkg_expand_trace = {"tkg_expand_error": "unknown"}

                # Memory backend (default): use MemoryEntry graph expansion
                if gb_used == "memory":
                    try:
                        # request-level filters.rel_types 优先级最高
                        rel_whitelist = fdict.get("rel_types") or gcfg.get("rel_whitelist")
                        # graph scope restrictions
                        user_ids_ctx = None
                        if gcfg.get("restrict_to_user", True):
                            if fdict.get("user_id"):
                                # 使用请求中指定的user_id
                                try:
                                    user_ids_ctx = list(fdict.get("user_id") or [])
                                except Exception:
                                    user_ids_ctx = None
                            else:
                                # 未指定user_id时，使用种子节点中提取的所有user_id
                                try:
                                    user_ids_ctx = []
                                    for entry in id_to_entry.values():
                                        if entry.metadata.get("user_id"):
                                            user_ids_ctx.extend(entry.metadata.get("user_id"))
                                    user_ids_ctx = list(set(user_ids_ctx)) if user_ids_ctx else None
                                except Exception:
                                    user_ids_ctx = None

                        domain_ctx = None
                        if gcfg.get("restrict_to_domain", True) and fdict.get("memory_domain") is not None:
                            domain_ctx = fdict.get("memory_domain")

                        neighbors = await self.graph.expand_neighbors(
                            seed_ids,
                            rel_whitelist=rel_whitelist,
                            max_hops=int(gcfg.get("max_hops", 1)),
                            neighbor_cap_per_seed=int(gcfg.get("neighbor_cap_per_seed", 5)),
                            user_ids=user_ids_ctx,
                            memory_domain=domain_ctx,
                            memory_scope=(fdict.get("memory_scope") if fdict.get("memory_scope") is not None else None),
                            restrict_to_user=bool(gcfg.get("restrict_to_user", True) and not gcfg.get("allow_cross_user", False)),
                            restrict_to_domain=bool(gcfg.get("restrict_to_domain", True) and not gcfg.get("allow_cross_domain", False)),
                            restrict_to_scope=bool(gcfg.get("restrict_to_scope", True) and not gcfg.get("allow_cross_scope", False)),
                        )
                    except Exception:
                        neighbors = {"neighbors": {}}

        # Combine scores over the merged dense + lexical candidate pool.
        hits_combined = []
        for rid, me in id_to_entry.items():
            vscore = float(vec_scores_for_rank.get(rid, 0.0))
            bscore = float(bm25_scores_for_rank.get(rid, 0.0))
            vraw = float(raw_vec_scores.get(rid, 0.0))
            braw = float(bm25_raw_scores.get(rid, 0.0))
            # graph contribution: sum of neighbor weights with path-aware boosts
            gscore = 0.0
            try:
                # path-aware weighting: hop1_boost, hop2_boost from config
                hop1_boost = float(gcfg.get("hop1_boost", 1.0)) if gcfg else 1.0
                hop2_boost = float(gcfg.get("hop2_boost", 0.5)) if gcfg else 0.5
                for nbr in (neighbors.get("neighbors", {}).get(rid, []) or []):
                    w = float(nbr.get("weight", 0.0))
                    # apply relation-type base weight
                    try:
                        rel_name = str(nbr.get("rel") or "").lower()
                        base = float(self._rel_base_weights.get(rel_name, self._rel_base_weights.get("default", 1.0)))
                        w *= base
                    except Exception:
                        pass
                    hop = int(nbr.get("hop", 1))
                    if hop <= 1:
                        gscore += hop1_boost * w
                    elif hop == 2:
                        gscore += hop2_boost * w
                    else:
                        # generic decay for deeper hops
                        gscore += (hop2_boost / max(2, hop)) * w
            except Exception:
                pass
            # recency: convert timestamp to freshness in [0,1], then weight
            rec_score = 0.0
            ts_raw = me.metadata.get("timestamp") or me.metadata.get("created_at")
            try:
                from datetime import datetime, timezone
                def _parse_ts(val):
                    if isinstance(val, (int, float)):
                        return datetime.fromtimestamp(float(val), tz=timezone.utc)
                    if isinstance(val, str):
                        try:
                            return datetime.fromisoformat(val)
                        except Exception:
                            return None
                    return None
                dt = _parse_ts(ts_raw)
                if dt is not None:
                    now = datetime.now(timezone.utc)
                    age_s = max(0.0, (now - dt).total_seconds())
                    # half-life 1 day: recent items ~1, week-old ~0.09
                    half_life = 24 * 3600.0
                    rec_score = 0.5 ** (age_s / half_life)
            except Exception:
                rec_score = 0.0

            score = (
                float(self._w_alpha) * vscore
                + float(self._w_beta) * bscore
                + float(self._w_gamma) * gscore
                + float(self._w_delta) * float(rec_score)
            )
            try:
                md = getattr(me, "metadata", {}) or {}
                if fdict.get("user_id"):
                    try:
                        uids = set(str(x) for x in (md.get("user_id") or [])) if md.get("user_id") is not None else set()
                        if uids.intersection(set(str(x) for x in fdict.get("user_id") or [])):
                            score += float(getattr(self, "_w_user_boost", 0.0))
                    except Exception:
                        pass
                if fdict.get("memory_domain") is not None and md.get("memory_domain") == fdict.get("memory_domain"):
                    score += float(getattr(self, "_w_domain_boost", 0.0))
                if fdict.get("run_id") is not None and md.get("run_id") == fdict.get("run_id"):
                    score += float(getattr(self, "_w_session_boost", 0.0))
            except Exception:
                pass
            hits_combined.append(
                {
                    "id": rid,
                    "score": score,
                    "entry": me,
                    "v": vscore,
                    "b": bscore,
                    "g": gscore,
                    "rec": rec_score,
                    "v_raw": vraw,
                    "b_raw": braw,
                }
            )

        hits_combined.sort(key=lambda x: x["score"], reverse=True)
        hits = hits_combined[:topk]

        # Build hints: include top contents and brief relation summary
        hint_parts = []
        for h in hits[: min(3, len(hits))]:
            entry: MemoryEntry = h["entry"]
            c = entry.contents[0] if entry.contents else ""
            nbrs = neighbors.get("neighbors", {}).get(h["id"], []) if neighbors else []
            rels = ",".join(sorted({n.get("rel") for n in nbrs})) if nbrs else ""
            hint_parts.append(f"命中:{c} 关系:{rels}")
        hints = "\n".join(hint_parts)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        inc("searches_total", 1)
        add_latency_ms(latency_ms)
        # domain distribution (topK sample)
        try:
            for h in hits:
                dom = h["entry"].metadata.get("memory_domain") if hasattr(h["entry"], "metadata") else None
                if dom is not None:
                    inc(f"domain_distribution_{dom}_total", 1)
        except Exception:
            pass
        trace = {
            "query": query,
            "topk": topk,
            "returned": len(hits),
            "threshold": threshold,
            "filters": fdict,
            "final_filters": fdict,
            "scope_used": applied_scope,
            "latency_ms": latency_ms,
            "modalities": list(fdict.get("modality") or []),
            "graph_sig": gsig,
            "graph_backend_requested": gb_requested,
            "graph_backend_used": gb_used,
            **({"tkg_expand": tkg_expand_trace} if tkg_expand_trace else {}),
            "weights_used": {
                "alpha_vector": self._w_alpha,
                "beta_bm25": self._w_beta,
                "gamma_graph": self._w_gamma,
                "delta_recency": self._w_delta,
            },
            "lexical_hybrid": lexical_trace,
            "attempts": trace_attempts,
        }
        result = SearchResult(hits=hits, neighbors=neighbors or {}, hints=hints, trace=trace)
        # Touch nodes to extend freshness (best-effort, non-blocking)
        try:
            node_ids: list[str] = []
            for h in hits:
                hid = h.get("id")
                if hid:
                    node_ids.append(hid)
            tenant = self._graph_touch_tenant
            if node_ids and tenant:
                asyncio.create_task(self._touch_nodes(node_ids, tenant_id=tenant))
        except Exception:
            pass
        # sampling log (structured)
        try:
            if getattr(self, "_sampling_enabled", False) and self._search_sampler is not None:
                import random
                if random.random() < float(getattr(self, "_sampling_rate", 0.05)):
                    sample = {
                        "query": query,
                        "filters": fdict,
                        "latency_ms": latency_ms,
                        "scope": applied_scope,
                        "user_ids": fdict.get("user_id"),
                        "memory_domain": fdict.get("memory_domain"),
                        "run_id": fdict.get("run_id"),
                        "weights": {"alpha": self._w_alpha, "beta": self._w_beta, "gamma": self._w_gamma, "delta": self._w_delta},
                        "top_hits": [
                            {
                                "id": h["id"],
                                "score": h["score"],
                                "v": h.get("v", 0.0),
                                "b": h.get("b", 0.0),
                                "g": h.get("g", 0.0),
                                "rec": h.get("rec", 0.0),
                                "text": (h["entry"].contents[0] if getattr(h["entry"], "contents", None) else ""),
                            }
                            for h in hits[: min(3, len(hits))]
                        ],
                    }
                    self._search_sampler(sample)
        except Exception:
            pass
        if self._search_cache_enabled:
            try:
                cache_key_final = self._search_cache_key(q, topk, fdict, expand_graph, graph_backend, threshold, scope=applied_scope, graph_sig=gsig, weight_sig=weight_sig)
                self._search_cache_put(cache_key_final, result)
            except Exception:
                pass
        return result

    def set_graph_tenant(self, tenant_id: Optional[str]) -> None:
        """Configure tenant id for graph touch/reinforce operations."""
        self._graph_touch_tenant = tenant_id

    def _filter_touch_ids(self, node_ids: list[str]) -> list[str]:
        if not node_ids:
            return []
        now = time.monotonic()
        allowed: list[str] = []
        with self._touch_lock:
            for nid in node_ids:
                last = self._touch_last.get(nid)
                if last is not None and now - last < self._touch_min_interval_s:
                    continue
                allowed.append(nid)
                self._touch_last[nid] = now
                if len(allowed) >= self._touch_max_batch:
                    break
            # avoid unbounded cache growth
            if len(self._touch_last) > self._touch_last_max:
                # drop oldest entries beyond capacity
                to_drop = len(self._touch_last) - self._touch_last_max
                for nid, _ in sorted(self._touch_last.items(), key=lambda kv: kv[1])[:to_drop]:
                    self._touch_last.pop(nid, None)
        return allowed

    async def _touch_nodes(self, node_ids: list[str], *, tenant_id: str) -> None:
        ids = self._filter_touch_ids(node_ids)
        if not ids or not hasattr(self.graph, "touch"):
            return
        extend = self._touch_extend_seconds if self._touch_extend_seconds > 0 else None
        try:
            await self.graph.touch(tenant_id=tenant_id, node_ids=ids, extend_seconds=extend)  # type: ignore[attr-defined]
        except Exception:
            # Best-effort: swallow errors to keep search responsive
            pass

    # ---- v0.6: Retrieval helpers focused on L1/L2 场景 ----

    async def list_places_by_time_range(
        self,
        *,
        query: str = "",
        filters: Optional[SearchFilters] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        topk_search: int = 200,
    ) -> Dict[str, Any]:
        """Return distinct 'places' within a time range.

        对标清单 L1：「某个时间范围内我去了哪些地方？」的最小实现版本：
        - 使用 search() + time_range 过滤，限定为 episodic 文本记忆；
        - 优先从 graph 的 occurs_at 边聚合 Place 节点（若图可用）；
        - 退化路径：从 metadata.entities 中提取地点标签，去重并统计频次；
        - 返回 {places: [{id, count}], trace:{...}} 结构，供上层 API 或 LLM 使用。
        """
        fdict: Dict[str, Any] = {}
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        # 限定为文本 / episodic 记忆（对“去过哪里”的近似）
        if "modality" not in fdict or not fdict.get("modality"):
            fdict["modality"] = ["text"]
        if "memory_type" not in fdict or not fdict.get("memory_type"):
            fdict["memory_type"] = ["episodic"]
        # 合并时间范围
        if start_time is not None or end_time is not None:
            rng: Dict[str, Any] = {}
            if start_time is not None:
                rng["gte"] = float(start_time)
            if end_time is not None:
                rng["lte"] = float(end_time)
            if rng:
                fdict["time_range"] = rng
        try:
            sfilters = SearchFilters.model_validate(fdict)
        except Exception:
            sfilters = None
        res = await self.search(
            query or "",
            topk=topk_search,
            filters=sfilters,
            expand_graph=False,
            threshold=0.0,
        )

        counts: Dict[str, int] = {}
        # Prefer graph OCCURS_AT aggregation when available.
        try:
            seed_ids = [h.id for h in (res.hits or []) if h and h.id]
            if seed_ids and hasattr(self.graph, "expand_neighbors"):
                g = await self.graph.expand_neighbors(
                    seed_ids,
                    rel_whitelist=["occurs_at"],
                    max_hops=1,
                    neighbor_cap_per_seed=5,
                    user_ids=(list(filters.user_id) if isinstance(filters, SearchFilters) and filters.user_id else None),
                    memory_domain=(str(filters.memory_domain) if isinstance(filters, SearchFilters) and filters.memory_domain else None),
                    restrict_to_user=False,
                    restrict_to_domain=False,
                )
                neighbors = (g or {}).get("neighbors") or {}
                for sid in seed_ids:
                    for nb in neighbors.get(sid, []) or []:
                        pid = str(nb.get("to") or "").strip()
                        if pid:
                            counts[pid] = counts.get(pid, 0) + 1
        except Exception:
            counts = {}

        # Fallback: aggregate from metadata.entities
        if not counts:
            for h in res.hits or []:
                e = h.entry
                ents = e.metadata.get("entities") or []
                try:
                    for ent in ents:
                        sval = str(ent)
                        if sval:
                            counts[sval] = counts.get(sval, 0) + 1
                except Exception:
                    continue

        places: List[Dict[str, Any]] = []
        for pid, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            name = ""
            try:
                get_fn = getattr(self.vectors, "get", None)
                if callable(get_fn):
                    ent = await get_fn(str(pid))
                    if ent is not None:
                        name = str(getattr(ent, "get_primary_content", lambda *_: "")() or "").strip()
            except Exception:
                name = ""
            places.append({"id": pid, "name": name or pid, "count": cnt})
        return {
            "places": places,
            "trace": {
                "query": query,
                "filters": fdict,
                "start_time": start_time,
                "end_time": end_time,
                "hits": len(res.hits or []),
            },
        }

    async def object_search(
        self,
        *,
        objects: List[str],
        scene: Optional[str] = None,
        query: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        modalities: Optional[List[str]] = None,
        graph_params: Optional[Dict[str, Any]] = None,
        topk: int = 20,
    ) -> Dict[str, Any]:
        """Multi-modal object/scene search with optional graph enrichment.

        - For each object o in objects, build a query: [scene?] + o + [query?]
        - Use modalities (default ["text","clip_image"]) for vector recall
        - Optionally expand graph with rel whitelist for object-scene relations
        """
        objects = [str(o).strip() for o in (objects or []) if str(o).strip()]
        if not objects:
            return {"items": [], "trace": {"warning": "no_objects"}}
        # build base filters
        fdict: Dict[str, Any] = {}
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        if modalities and isinstance(modalities, list) and modalities:
            fdict["modality"] = list(modalities)
        else:
            fdict["modality"] = ["text", "clip_image"]
        items: List[Dict[str, Any]] = []
        # graph defaults
        rel_whitelist = ["APPEARS_IN", "LOCATED_IN", "CO_OCCURS", "RELATED_TO"]
        if graph_params and isinstance(graph_params.get("rel_whitelist"), list):
            rel_whitelist = list(graph_params.get("rel_whitelist"))
        for obj in objects:
            parts = []
            if scene:
                parts.append(str(scene))
            parts.append(str(obj))
            if query:
                parts.append(str(query))
            q = " ".join(parts)
            # run search
            sf = SearchFilters.model_validate(fdict) if fdict else None
            res = await self.search(q, topk=topk, filters=sf, expand_graph=True)
            # assemble hits dicts for client
            hits: List[Dict[str, Any]] = []
            idlist: List[str] = []
            for h in (res.hits or []):
                e = h.entry
                md = dict(e.metadata or {})
                hits.append({
                    "id": e.id,
                    "score": h.score,
                    "kind": e.kind,
                    "modality": e.modality,
                    "content": e.get_primary_content(""),
                    "metadata": md,
                })
                if e.id:
                    idlist.append(e.id)
            # graph relations filtered by whitelist
            rels: List[Dict[str, Any]] = []
            nbrs = (res.neighbors or {}).get("neighbors", {}) if isinstance(res.neighbors, dict) else {}
            for sid in idlist:
                for n in nbrs.get(sid, []) or []:
                    r = str(n.get("rel") or "")
                    if r and (r.upper() in rel_whitelist):
                        rels.append({"from": sid, **n})
            items.append({"object": obj, "query": q, "hits": hits, "relations": rels})
        return {"items": items, "trace": {"objects": objects, "scene": scene, "base_filters": fdict}}

    async def speech_search(
        self,
        *,
        keywords: List[str],
        speaker: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        time_range: Optional[Dict[str, float]] = None,
        topk: int = 20,
    ) -> Dict[str, Any]:
        """Search utterances by keywords (ASR text expected) and return time-anchored results.

        v1 简化：
        - 使用 text 模态，查询为关键词拼接
        - 结果中优先筛选 metadata.source == 'asr' 的命中；否则回退 episodic 文本
        - 生成 utterances: {text, start, end, clip, entry_id}
        - 返回 anchors（首段命中的 id/clip/time），neighbors（若有）
        """
        kws = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        if not kws:
            return {"utterances": [], "trace": {"warning": "no_keywords"}}
        # base filters
        fdict: Dict[str, Any] = {}
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        # enforce text modality
        prev_mod = list(fdict.get("modality") or [])
        if "text" not in prev_mod:
            fdict["modality"] = list(sorted(set(prev_mod + ["text"])))
        if time_range and isinstance(time_range, dict):
            rng: Dict[str, Any] = {}
            if time_range.get("gte") is not None:
                rng["gte"] = float(time_range.get("gte"))
            if time_range.get("lte") is not None:
                rng["lte"] = float(time_range.get("lte"))
            if rng:
                fdict["time_range"] = rng
        q = " ".join(kws)
        sf = SearchFilters.model_validate(fdict) if fdict else None
        res = await self.search(q, topk=topk, filters=sf, expand_graph=True)
        utterances: List[Dict[str, Any]] = []
        anchors: List[Dict[str, Any]] = []
        for h in (res.hits or []):
            e = h.entry
            md = dict(e.metadata or {})
            text = e.get_primary_content("")
            start = md.get("start") or md.get("timestamp")
            end = md.get("end") or md.get("timestamp")
            clip = md.get("clip_id")
            if text:
                utterances.append({
                    "entry_id": e.id,
                    "text": text,
                    "start": float(start) if start is not None else None,
                    "end": float(end) if end is not None else None,
                    "clip": int(clip) if clip is not None else None,
                    "score": h.score,
                    "source": md.get("source"),
                })
        # anchors: pick top few utterances with time
        for u in utterances[: min(3, len(utterances))]:
            anchors.append({
                "entry_id": u.get("entry_id"),
                "clip": u.get("clip"),
                "time": u.get("start") if u.get("start") is not None else u.get("end"),
                "text": u.get("text"),
            })
        return {
            "utterances": utterances,
            "anchors": anchors,
            "neighbors": res.neighbors if isinstance(res.neighbors, dict) else {},
            "trace": {"query": q, "filters": fdict, "keywords": kws, "speaker": speaker},
        }

    async def entity_event_anchor(
        self,
        *,
        entity: str,
        action: Optional[str] = None,
        time_hint: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        graph_params: Optional[Dict[str, Any]] = None,
        topk: int = 20,
    ) -> Dict[str, Any]:
        """Anchor a (entity, action) to time via semantic→graph→episodic chain.

        v1 简化：
        - 使用 text 检索 entity(+action)；
        - 命中中若存在 episodic，直接抽取时间；否则使用 neighbors 辅助（若有）；
        - 返回 triples: {entity, action, time_range, evidence[{entry_id,score}]}
        """
        ent = (entity or "").strip()
        if not ent:
            return {"triples": [], "trace": {"warning": "no_entity"}}
        parts = [ent]
        if action:
            parts.append(str(action))
        if time_hint:
            parts.append(str(time_hint))
        q = " ".join(parts)
        fdict: Dict[str, Any] = {}
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        prev_mod = list(fdict.get("modality") or [])
        if "text" not in prev_mod:
            fdict["modality"] = list(sorted(set(prev_mod + ["text"])))
        sf = SearchFilters.model_validate(fdict) if fdict else None
        res = await self.search(q, topk=topk, filters=sf, expand_graph=True)
        triples: List[Dict[str, Any]] = []
        for h in (res.hits or []):
            e = h.entry
            md = dict(e.metadata or {})
            # prefer episodic with explicit time
            if str(e.kind).lower() == "episodic":
                st = md.get("start") or md.get("timestamp")
                ed = md.get("end") or md.get("timestamp")
                triples.append({
                    "entity": ent,
                    "action": action,
                    "time_range": [float(st) if st is not None else None, float(ed) if ed is not None else None],
                    "evidence": [{"entry_id": e.id, "score": h.score}],
                })
        # if nothing episodic found, fallback to first few hits as evidence without time
        if not triples and (res.hits or []):
            ev = [
                {"entry_id": h.entry.id, "score": h.score}
                for h in res.hits[: min(3, len(res.hits))]
            ]
            triples.append({"entity": ent, "action": action, "time_range": [None, None], "evidence": ev})
        return {"triples": triples, "neighbors": res.neighbors if isinstance(res.neighbors, dict) else {}, "trace": {"query": q, "filters": fdict}}

    async def write(
        self,
        entries: List[MemoryEntry],
        links: Optional[List[Edge]] = None,
        *,
        upsert: bool = True,
        return_id_map: bool = False,
    ) -> Version | tuple[Version, dict[str, str]]:
        """Normalize → enrich governance → dedup/merge → upsert (vectors + graph) → audit.

        .. deprecated::
            This method writes to the legacy MemoryEntry/:MemoryNode system.
            **For new code, use TKG (Typed Knowledge Graph) instead:**
            
                await svc.graph_upsert(GraphUpsertRequest(
                    events=[Event(...)],
                    entities=[Entity(...)],
                    edges=[GraphEdge(...)],
                ))
            
            TKG provides richer semantics (evidence chains, entity resolution) and
            automatic vector indexing with node_type/node_id for reverse lookup.
            See: modules/memory/contracts/graph_models.py
        """
        normalized: List[MemoryEntry] = []

        # Enrich governance metadata and normalize IDs (replace temporary ids)
        id_map: dict[str, str] = {}
        # Enrich governance metadata
        for e in entries:
            # ensure id
            # Replace placeholder ids like tmp-*, dev-*, loc-*, char-* with UUIDs for persistence
            def _is_placeholder(s: str | None) -> bool:
                if not s:
                    return True
                s2 = str(s)
                return s2.startswith("tmp-") or s2.startswith("dev-") or s2.startswith("loc-") or s2.startswith("char-")

            if _is_placeholder(e.id):
                old = e.id or ""
                new_id = str(uuid.uuid4())
                e.id = new_id
                if old:
                    id_map[old] = new_id
            md = dict(e.metadata)
            if "created_at" not in md:
                md["created_at"] = datetime.now(timezone.utc).isoformat()
            # initial write: record content hash
            md["hash"] = self._content_hash(e.contents)
            if "importance" not in md:
                md["importance"] = compute_importance({
                    "modality": e.modality,
                    "source": md.get("source"),
                    "kind": e.kind,
                })
            if "stability" not in md:
                md["stability"] = compute_stability({"kind": e.kind})
            if "ttl" not in md:
                ttl = default_ttl_seconds(md["importance"])  # 0 means keep long
                md["ttl"] = ttl
            # normalize scoping metadata (user_id list, memory_domain/run_id strings)
            try:
                if "user_id" in md:
                    uid = md.get("user_id")
                    if isinstance(uid, list):
                        md["user_id"] = [str(x) for x in uid if x is not None]
                    elif uid is not None:
                        md["user_id"] = [str(uid)]
                if "memory_domain" in md and md.get("memory_domain") is not None:
                    md["memory_domain"] = str(md.get("memory_domain"))
                if "run_id" in md and md.get("run_id") is not None:
                    md["run_id"] = str(md.get("run_id"))
            except Exception:
                pass
            # per-domain governance overrides: ttl / importance (unless pinned)
            try:
                cfg_gov = (load_memory_config().get("memory", {}) or {}).get("governance", {})
                domain = md.get("memory_domain")
                if domain is not None:
                    # TTL override
                    if not bool(md.get("ttl_pinned", False)):
                        dom_ttl_cfg = (cfg_gov.get("per_domain_ttl") or {}) if isinstance(cfg_gov.get("per_domain_ttl"), dict) else {}
                        if str(domain) in dom_ttl_cfg:
                            raw = dom_ttl_cfg.get(str(domain))
                            def _parse_dur(v: object) -> int:
                                try:
                                    s = str(v).strip()
                                    if s.isdigit():
                                        return int(s)
                                    mult = 1
                                    s2 = s
                                    if s.endswith("s"):
                                        mult = 1
                                        s2 = s[:-1]
                                    elif s.endswith("m"):
                                        mult = 60
                                        s2 = s[:-1]
                                    elif s.endswith("h"):
                                        mult = 3600
                                        s2 = s[:-1]
                                    elif s.endswith("d"):
                                        mult = 86400
                                        s2 = s[:-1]
                                    return int(float(s2) * mult)
                                except Exception:
                                    return md.get("ttl", 0)
                            md["ttl"] = max(0, _parse_dur(raw))
                    # importance override (additive)
                    if not bool(md.get("importance_pinned", False)):
                        dom_imp_cfg = (cfg_gov.get("importance_overrides") or {}) if isinstance(cfg_gov.get("importance_overrides"), dict) else {}
                        if str(domain) in dom_imp_cfg:
                            try:
                                delta = float(dom_imp_cfg.get(str(domain)) or 0.0)
                                md["importance"] = max(0.0, min(1.0, float(md.get("importance", 0.5)) + delta))
                            except Exception:
                                pass
            except Exception:
                pass
            e.metadata = md
            normalized.append(e)

        write_override = rtconf.get_write_override()
        dedup_enabled = bool(write_override.get("dedup_enabled", True))

        # Naive dedup/merge: if a near-duplicate exists, merge contents/metadata
        if not dedup_enabled:
            to_write: List[MemoryEntry] = list(normalized)
        else:
            to_write = []
            for e in normalized:
                # Per-entry escape hatch: some pipelines (e.g., raw dialogue turns) must never be merged.
                # If set, skip neighbor search and write as-is (upsert by id if provided).
                try:
                    md0 = dict(e.metadata or {})
                    if md0.get("dedup_skip") is True:
                        md0.pop("dedup_skip", None)
                        e.metadata = md0
                        to_write.append(e)
                        continue
                except Exception:
                    pass
                # find neighbors for decision
                query = e.contents[0] if e.contents else ""
                try:
                    # IMPORTANT: dedup candidates must be isolated by tenant/user/domain to avoid cross-subject merges.
                    # Only add isolation keys when present to preserve legacy behavior for older payloads.
                    dedup_filters: Dict[str, Any] = {
                        "modality": [e.modality],
                        "memory_type": [e.kind],
                    }
                    md = dict(e.metadata or {})
                    tenant_id = md.get("tenant_id")
                    if tenant_id is not None and str(tenant_id).strip() != "":
                        dedup_filters["tenant_id"] = str(tenant_id)
                    user_ids = md.get("user_id")
                    if user_ids:
                        if not isinstance(user_ids, list):
                            user_ids = [user_ids]
                        dedup_filters["user_id"] = [str(x) for x in user_ids if x is not None and str(x).strip() != ""]
                        dedup_filters["user_match"] = "all"
                    dom = md.get("memory_domain")
                    if dom is not None and str(dom).strip() != "":
                        dedup_filters["memory_domain"] = str(dom)
                    scope = md.get("memory_scope")
                    if scope is not None and str(scope).strip() != "":
                        dedup_filters["memory_scope"] = str(scope)
                    # Prefer session-local dedup for episodic entries to protect timeline integrity.
                    if e.kind == "episodic":
                        rid = md.get("run_id")
                        if rid is not None and str(rid).strip() != "":
                            dedup_filters["run_id"] = str(rid)
                    neighbors = await self.vectors.search_vectors(query, dedup_filters, topk=5, threshold=None)
                except Exception:
                    neighbors = []
                existing_list: list[MemoryEntry] = []
                for n in neighbors:
                    p = n.get("payload")
                    if isinstance(p, MemoryEntry) and p.id:
                        existing_list.append(p)

                action = "ADD"
                target_id = None
                if self.update_decider is not None:
                    try:
                        decision = self.update_decider(existing_list, e)
                        if isinstance(decision, tuple) and len(decision) == 2:
                            action, target_id = decision
                    except Exception:
                        action, target_id = "ADD", None
                else:
                    # default heuristic
                    if existing_list and should_merge(existing_list[0], e):
                        action, target_id = "UPDATE", existing_list[0].id

                if action == "ADD":
                    to_write.append(e)
                elif action == "UPDATE" and target_id:
                    base = next((x for x in existing_list if x.id == target_id), existing_list[0] if existing_list else None)
                    if base is not None:
                        merged = merge_entries(base, e)
                        merged.id = base.id
                        # refresh hash
                        mdm = dict(merged.metadata)
                        mdm["hash"] = self._content_hash(merged.contents)
                        merged.metadata = mdm
                        to_write.append(merged)
                    else:
                        to_write.append(e)
                elif action == "DELETE" and target_id:
                    # prefer soft delete for safety
                    await self.delete(target_id, soft=True, reason="LLM_DECIDER")
                    # not adding the new one
                elif action == "NONE":
                    # skip writing
                    continue
                else:
                    to_write.append(e)

        # Vector validation & observation (P0.3): enforce dimension, prefer precomputed vectors, truncate if oversized
        def _expected_dim(mod: str) -> int:
            try:
                cfg = self._get_cached_config()  # 使用缓存的配置
                emb = (cfg.get("memory", {}) or {}).get("vector_store", {}).get("embedding", {}) or {}
                if mod == "text":
                    return int(emb.get("dim") or 768)
                if mod == "image":
                    return int(((emb.get("image") or {}).get("dim") or 512))
                if mod == "audio":
                    return int(((emb.get("audio") or {}).get("dim") or 192))
            except Exception:
                pass
            return 0

        # mutate a deep copy for write safety
        safe_to_write: List[MemoryEntry] = []
        for e in to_write:
            mod = e.modality
            # validate/truncate
            try:
                key = "text" if mod == "text" else ("image" if mod == "image" else ("audio" if mod == "audio" else None))
                if key and e.vectors and isinstance(e.vectors.get(key), list):
                    vec = list(e.vectors.get(key) or [])
                    need = _expected_dim(mod)
                    if need:
                        if len(vec) > need:
                            vec = vec[:need]
                            # write back truncated vector
                            ev = dict(e.vectors)
                            ev[key] = vec
                            e.vectors = ev
                            try:
                                inc("vector_truncations_total", 1)
                            except Exception:
                                pass
                        elif 0 < len(vec) < need:
                            # dimension too small: fail fast
                            raise RuntimeError(f"vector_dim_too_small: modality={mod} expected={need} got={len(vec)} id={e.id}")
            except Exception:
                raise
            # observe vector size (after validation/truncation)
            try:
                from modules.memory.application.metrics import observe_vector_size
                key = "text" if mod == "text" else ("image" if mod == "image" else ("audio" if mod == "audio" else None))
                if key and e.vectors and isinstance(e.vectors.get(key), list):
                    observe_vector_size(mod, len(e.vectors.get(key) or []))
            except Exception:
                pass
            safe_to_write.append(e)

        # P2: append character index entries (structured) for any character_id present
        try:
            present_chars = set()
            tenant_values: set[str] = set()
            default_user_ids: Optional[List[str]] = None
            default_domain: Optional[str] = None
            for e in safe_to_write:
                md = e.metadata if isinstance(e.metadata, dict) else {}
                cid = md.get("character_id")
                if cid:
                    present_chars.add(str(cid))
                tid = str(md.get("tenant_id") or "").strip()
                if tid:
                    tenant_values.add(tid)
                if default_user_ids is None and isinstance(md.get("user_id"), list):
                    default_user_ids = [str(x) for x in md.get("user_id") if x is not None and str(x).strip()]
                if default_domain is None and str(md.get("memory_domain") or "").strip():
                    default_domain = str(md.get("memory_domain")).strip()
            if len(tenant_values) > 1:
                raise RuntimeError(f"cross_tenant_write_forbidden: write payload mixed tenants {sorted(tenant_values)}")
            default_tenant = next(iter(tenant_values)) if tenant_values else None
            for ch in sorted(present_chars):
                ce_md: Dict[str, Any] = {"entity_type": "character"}
                if default_tenant:
                    ce_md["tenant_id"] = default_tenant
                if default_user_ids:
                    ce_md["user_id"] = list(default_user_ids)
                if default_domain:
                    ce_md["memory_domain"] = default_domain
                ce = MemoryEntry(kind="semantic", modality="structured", contents=[ch], metadata=ce_md)
                if not ce.id:
                    ce.id = f"char-{len(safe_to_write)}"
                safe_to_write.append(ce)
        except RuntimeError:
            raise
        except Exception:
            pass

        # ---- Chunked writes to reduce memory/back-end pressure ----
        chunk_items = max(1, int(getattr(self, "_batch_flush_chunk_items", 0) or len(safe_to_write) or 1))
        # 1) vectors in chunks
        for i in range(0, len(safe_to_write), chunk_items):
            part = safe_to_write[i : i + chunk_items]
            await self.vectors.upsert_vectors(part)
        # rewrite edge endpoints if ids were normalized
        links2 = links
        try:
            if links and id_map:
                links2 = []
                for ed in links:
                    src = id_map.get(ed.src_id, ed.src_id)
                    dst = id_map.get(ed.dst_id, ed.dst_id)
                    links2.append(Edge(src_id=src, dst_id=dst, rel_type=ed.rel_type, weight=ed.weight))
        except Exception:
            links2 = links
        tenant_values: set[str] = set()
        for e in safe_to_write:
            md = e.metadata if isinstance(e.metadata, dict) else {}
            tid = str(md.get("tenant_id") or "").strip()
            if tid:
                tenant_values.add(tid)
        if len(tenant_values) > 1:
            raise RuntimeError(f"cross_tenant_write_forbidden: graph write mixed tenants {sorted(tenant_values)}")
        graph_tenant = next(iter(tenant_values)) if tenant_values else None

        async def _merge_nodes_part(part: List[MemoryEntry]) -> None:
            if hasattr(self.graph, "merge_nodes_edges_batch"):
                try:
                    await self.graph.merge_nodes_edges_batch(part, None, tenant_id=graph_tenant)
                    return
                except TypeError as exc:
                    if "tenant_id" not in str(exc):
                        raise
                    await self.graph.merge_nodes_edges_batch(part, None)
                    return
            try:
                await self.graph.merge_nodes_edges(part, None, tenant_id=graph_tenant)
            except TypeError as exc:
                if "tenant_id" not in str(exc):
                    raise
                await self.graph.merge_nodes_edges(part, None)

        async def _merge_edges_only(part_edges: List[Edge]) -> None:
            if hasattr(self.graph, "merge_nodes_edges_batch"):
                try:
                    await self.graph.merge_nodes_edges_batch([], part_edges, tenant_id=graph_tenant)
                    return
                except TypeError as exc:
                    if "tenant_id" not in str(exc):
                        raise
                    await self.graph.merge_nodes_edges_batch([], part_edges)
                    return
            if hasattr(self.graph, "merge_rel_batch"):
                lst = [(e.src_id, e.dst_id, e.rel_type, (e.weight if e.weight is not None else 1.0)) for e in part_edges]
                try:
                    await self.graph.merge_rel_batch(lst, tenant_id=graph_tenant)
                except TypeError as exc:
                    if "tenant_id" not in str(exc):
                        raise
                    await self.graph.merge_rel_batch(lst)
                return
            for ed in part_edges:
                try:
                    await self.graph.merge_rel(ed.src_id, ed.dst_id, ed.rel_type, weight=ed.weight, tenant_id=graph_tenant)
                except TypeError as exc:
                    if "tenant_id" not in str(exc):
                        raise
                    await self.graph.merge_rel(ed.src_id, ed.dst_id, ed.rel_type, weight=ed.weight)

        # 2) graph nodes first (chunked), then edges (batched)
        for i in range(0, len(safe_to_write), chunk_items):
            part = safe_to_write[i : i + chunk_items]
            await _merge_nodes_part(part)
        # edges second
        if links2:
            await _merge_edges_only(links2)
        if links:
            try:
                inc("graph_rel_merges_total", len(links))
            except Exception:
                pass
        version = await self.audit.add_batch("ADD", safe_to_write)
        try:
            inc("writes_total", len(to_write))
        except Exception:
            pass
        for e in safe_to_write:
            if e.id:
                self._cache[e.id] = e
        # Optional event publish (memory_ready)
        try:
            cfg = self._get_cached_config()  # 使用缓存的配置
            ev_enabled = bool(cfg.get("memory", {}).get("events", {}).get("publish_memory_ready", True))
        except Exception:
            ev_enabled = True
        if ev_enabled and self._event_publisher is not None:
            try:
                clip_ids = sorted({(e.metadata.get("clip_id") or e.metadata.get("timestamp")) for e in to_write if isinstance(e.metadata, dict)})
                payload = {
                    "version": version,
                    "count": len(to_write),
                    "clip_ids": [c for c in clip_ids if c is not None],
                    "ids": [e.id for e in to_write if e.id],
                    "source_stats": sorted(list({e.metadata.get("source", "?") for e in to_write})),
                }
                self._event_publisher("memory_ready", payload)
            except Exception:
                pass
        ver_obj = Version(value=version)
        if return_id_map:
            return ver_obj, dict(id_map)
        return ver_obj

    async def update(self, memory_id: str, patch: dict, *, reason: Optional[str] = None, confirm: Optional[bool] = None) -> Version:
        existing: MemoryEntry | None = None
        if hasattr(self.vectors, "get"):
            existing = await self.vectors.get(memory_id)
        if existing is None:
            existing = self._cache.get(memory_id)
        if existing is None:
            # nothing to update
            v = await self.audit.add_one("UPDATE", memory_id, {"skipped": True}, reason=reason)
            return Version(value=v)
        # snapshot before modification for rollback
        prev_snapshot = existing.model_dump()
        # apply patch (shallow for contents/metadata)
        if "contents" in patch and isinstance(patch["contents"], list):
            existing.contents = list(patch["contents"])  # type: ignore[assignment]
        if "metadata" in patch and isinstance(patch["metadata"], dict):
            md = dict(existing.metadata)
            md.update(patch["metadata"])
            existing.metadata = md
        if "published" in patch:
            existing.published = patch.get("published")
        # re-enrich governance minimal
        md = dict(existing.metadata)
        # update metadata.updated_at and hash
        md["updated_at"] = datetime.now(timezone.utc).isoformat()
        md["hash"] = self._content_hash(existing.contents)
        if "importance" not in md:
            md["importance"] = compute_importance({
                "modality": existing.modality,
                "source": md.get("source"),
                "kind": existing.kind,
            })
        if "stability" not in md:
            md["stability"] = compute_stability({"kind": existing.kind})
        existing.metadata = md
        await self.vectors.upsert_vectors([existing])
        await self.graph.merge_nodes_edges([existing], None)
        # include previous snapshot for potential rollback
        v = await self.audit.add_one("UPDATE", memory_id, {"patch": patch, "prev": prev_snapshot}, reason=reason)
        return Version(value=v)

    async def publish_entries(
        self,
        *,
        tenant_id: str,
        entry_ids: List[str],
        graph_node_ids: List[str],
        published: bool = True,
    ) -> Dict[str, int]:
        vec_updated = 0
        graph_updated = 0
        if entry_ids:
            set_pub = getattr(self.vectors, "set_published", None)
            if callable(set_pub):
                try:
                    vec_updated = int(await set_pub(list(entry_ids), bool(published)))
                except Exception:
                    vec_updated = 0
            else:
                # Fallback: per-entry update (may re-embed)
                for eid in entry_ids:
                    try:
                        await self.update(str(eid), {"published": published}, reason="publish_entries")
                        vec_updated += 1
                    except Exception:
                        continue
            for eid in entry_ids:
                cached = self._cache.get(str(eid))
                if cached is not None:
                    try:
                        cached.published = published
                        self._cache[str(eid)] = cached
                    except Exception:
                        continue
        if graph_node_ids:
            set_nodes = getattr(self.graph, "set_nodes_published", None)
            if callable(set_nodes):
                try:
                    graph_updated = int(await set_nodes(tenant_id=str(tenant_id), node_ids=list(graph_node_ids), published=bool(published)))
                except Exception:
                    graph_updated = 0
        return {"vectors": vec_updated, "graph": graph_updated}

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a MemoryEntry by id (vector side), best-effort.

        This is primarily used for strict idempotency checks (e.g., session markers) where
        ANN search + scoping fallback would be incorrect.
        """
        try:
            if hasattr(self.vectors, "get"):
                entry = await self.vectors.get(memory_id)
                if entry is not None:
                    return entry
        except Exception:
            pass
        try:
            cached = self._cache.get(str(memory_id))
            return cached.model_copy(deep=True) if cached is not None else None
        except Exception:
            return None

    async def delete(self, memory_id: str, *, soft: bool = True, reason: Optional[str] = None, confirm: Optional[bool] = None) -> Version:
        prev_snapshot = None
        # safety checks
        if self._safety_require_reason_delete and not soft and not reason:
            raise SafetyError("hard delete requires reason")
        if not soft and self._safety_require_confirm_hard_delete:
            ok = bool(confirm)
            if not ok and self._safety_confirmer is not None:
                ctx = {"op": "delete", "id": memory_id, "soft": soft, "reason": reason}
                try:
                    ok = bool(self._safety_confirmer(ctx))
                except Exception:
                    ok = False
            if not ok:
                raise SafetyError("hard delete requires confirmation")
        if soft:
            # mark soft deleted
            existing: MemoryEntry | None = None
            if hasattr(self.vectors, "get"):
                existing = await self.vectors.get(memory_id)
            if existing is None:
                existing = self._cache.get(memory_id)
            if existing is not None:
                prev_snapshot = existing.model_dump()
                md = dict(existing.metadata)
                md["is_deleted"] = True
                md["deleted_at"] = datetime.now(timezone.utc).isoformat()
                existing.metadata = md
                await self.vectors.upsert_vectors([existing])
                await self.graph.merge_nodes_edges([existing], None)
        else:
            if hasattr(self.vectors, "delete_ids"):
                await self.vectors.delete_ids([memory_id])
            if hasattr(self.graph, "delete_node"):
                await self.graph.delete_node(memory_id)
        v = await self.audit.add_one("DELETE", memory_id, {"soft": soft, "prev": prev_snapshot}, reason=reason)
        return Version(value=v)

    # Rollback minimal support for UPDATE/DELETE soft/hard using audit snapshots
    async def rollback_version(self, version: str) -> bool:
        if not hasattr(self.audit, "get_event"):
            return False
        evt = self.audit.get_event(version)  # type: ignore[attr-defined]
        if not evt:
            return False
        event = evt.get("event")
        obj_id = evt.get("obj_id")
        payload = evt.get("payload") or {}
        if event == "UPDATE":
            prev = payload.get("prev")
            if not prev or not obj_id:
                return False
            entry = MemoryEntry(**prev)
            await self.vectors.upsert_vectors([entry])
            await self.graph.merge_nodes_edges([entry], None)
            await self.audit.add_one("ROLLBACK", obj_id, {"of": version})
            try:
                inc("rollbacks_total", 1)
            except Exception:
                pass
            return True
        if event == "DELETE":
            prev = payload.get("prev")
            if not obj_id:
                return False
            if prev:
                entry = MemoryEntry(**prev)
                md = dict(entry.metadata)
                md.pop("is_deleted", None)
                md.pop("deleted_at", None)
                entry.metadata = md
                await self.vectors.upsert_vectors([entry])
                await self.graph.merge_nodes_edges([entry], None)
                await self.audit.add_one("ROLLBACK", obj_id, {"of": version})
                try:
                    inc("rollbacks_total", 1)
                except Exception:
                    pass
                return True
        return False

    # Backward-compatible synchronous helpers for legacy notebooks
    @staticmethod
    def _run_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Dict[str, Any] = {"value": None, "error": None}

        def _worker():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result["value"] = new_loop.run_until_complete(coro)
            except Exception as exc:  # pragma: no cover
                result["error"] = exc
            finally:
                new_loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if result["error"] is not None:
            raise result["error"]
        return result["value"]

    def search_memories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or "")
        limit = int(payload.get("limit", 10))
        expand_graph = bool(payload.get("expand_graph", False))
        scope = payload.get("scope")
        threshold = payload.get("threshold")
        filters = payload.get("filters") or {}
        if isinstance(filters, SearchFilters):
            search_filters = filters
        else:
            search_filters = SearchFilters.model_validate(filters)

        async def _do():
            res = await self.search(
                query,
                topk=limit,
                filters=search_filters,
                expand_graph=expand_graph,
                scope=scope,
                threshold=threshold,
            )
            hits = [
                {
                    "id": h.id,
                    "score": h.score,
                    "entry": h.entry.model_dump() if hasattr(h.entry, "model_dump") else h.entry,
                    "neighbors": getattr(h, "neighbors", None),
                }
                for h in res.hits
            ]
            return {
                "results": hits,
                "trace": getattr(res, "trace", None),
                "neighbors": getattr(res, "neighbors", None),
            }

        return self._run_sync(_do())

    def search_graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Graph-first search: if `seeds` provided, expand neighbors directly without vector seed.

        Payload:
          - seeds: List[str] optional; if present, bypass vector search and call graph.expand_neighbors
          - filters: SearchFilters or dict (user_id, memory_domain, etc.)
          - rel_whitelist, max_hops, neighbor_cap_per_seed (optional tuning)
        """
        seeds = payload.get("seeds") or []
        filters = payload.get("filters") or {}
        if isinstance(filters, SearchFilters):
            search_filters = filters
        else:
            search_filters = SearchFilters.model_validate(filters)
        rel_whitelist = payload.get("rel_whitelist") or [
            "APPEARS_IN",
            "SAID_BY",
            "DESCRIBES",
            "TEMPORAL_NEXT",
            "OCCURS_AT",
            "LOCATED_IN",
            "EQUIVALENCE",
            "EXECUTED",
        ]
        max_hops = int(payload.get("max_hops", 1))
        cap = int(payload.get("neighbor_cap_per_seed", 5))

        async def _do_graph():
            # Expand directly in graph store if seeds provided
            return await self.graph.expand_neighbors(
                list(seeds),
                rel_whitelist=rel_whitelist,
                max_hops=max_hops,
                neighbor_cap_per_seed=cap,
                user_ids=search_filters.user_id,
                memory_domain=search_filters.memory_domain,
                restrict_to_user=bool(search_filters.user_id is not None),
                restrict_to_domain=bool(search_filters.memory_domain is not None),
            )

        if seeds:
            return self._run_sync(_do_graph())

        # Fallback: no seeds provided → use vector search with expand_graph
        query = str(payload.get("query") or "")
        limit = int(payload.get("limit", 20))
        scope = payload.get("scope")
        threshold = payload.get("threshold")

        async def _do():
            res = await self.search(
                query,
                topk=limit,
                filters=search_filters,
                expand_graph=True,
                scope=scope,
                threshold=threshold,
            )
            return {
                "neighbors": getattr(res, "neighbors", {}),
                "trace": getattr(res, "trace", None),
            }

        return self._run_sync(_do())

    async def link(
        self,
        src_id: str,
        dst_id: str,
        rel_type: str,
        *,
        weight: Optional[float] = None,
        confirm: Optional[bool] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        # whitelist check
        if rel_type not in self._allowed_rel_types:
            raise SafetyError(f"relation '{rel_type}' not allowed")
        # safety on sensitive relations
        if rel_type in set(self._safety_sensitive_rels) and self._safety_require_confirm_sensitive_link:
            ok = bool(confirm)
            if not ok and self._safety_confirmer is not None:
                ctx = {"op": "link", "src": src_id, "dst": dst_id, "rel": rel_type, "weight": weight}
                try:
                    ok = bool(self._safety_confirmer(ctx))
                except Exception:
                    ok = False
            if not ok:
                raise SafetyError(f"link '{rel_type}' requires confirmation")
        # TODO: validate nodes exist, then merge relation
        try:
            await self.graph.merge_rel(src_id, dst_id, rel_type, weight=weight, tenant_id=tenant_id)
        except TypeError as exc:
            if "tenant_id" not in str(exc):
                raise
            await self.graph.merge_rel(src_id, dst_id, rel_type, weight=weight)

        audit_payload: Dict[str, Any] = {"weight": weight}
        if tenant_id:
            audit_payload["tenant_id"] = tenant_id
        await self.audit.add_one("LINK", f"{src_id}->{dst_id}:{rel_type}", audit_payload)
        return True

    # ---- Deep Health Check ----
    # Configuration defaults (can be overridden via environment variables)
    _HEALTH_LLM_CACHE_TTL_S: float = 60.0
    _health_llm_cache: Optional[Dict[str, Any]] = None
    _health_llm_cache_time: float = 0.0

    async def _check_llm_provider_health(self) -> Dict[str, Any]:
        """Check LLM provider (OpenRouter) connectivity, auth, and balance.
        
        Returns a structured status dict with auth/balance details and specific error codes.
        Uses caching to avoid hitting OpenRouter API too frequently.
        """
        import httpx
        import time as _time
        
        provider = "openrouter"
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        min_usd = float(os.getenv("MEMORY_HEALTH_OPENROUTER_MIN_USD", "1.0"))
        cache_ttl = float(os.getenv("MEMORY_HEALTH_LLM_CACHE_TTL_S", "60"))
        
        result: Dict[str, Any] = {
            "status": "fail",
            "provider": provider,
            "auth": {"status": "fail"},
            "balance": None,
            "latency_ms": None,
        }
        
        # Check if API key is configured
        if not api_key:
            result["auth"]["error"] = "API_KEY_MISSING"
            return result
        
        # Check cache
        now = _time.time()
        if (
            self._health_llm_cache is not None
            and (now - self._health_llm_cache_time) < cache_ttl
        ):
            return dict(self._health_llm_cache)
        
        # Make actual API calls
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(10.0, connect=5.0)
        
        try:
            start = _time.perf_counter()
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Step 1: Auth check via /auth/key
                auth_resp = await client.get(f"{base_url}/auth/key", headers=headers)
                latency_ms = int((_time.perf_counter() - start) * 1000)
                result["latency_ms"] = latency_ms
                
                if auth_resp.status_code == 401 or auth_resp.status_code == 403:
                    result["auth"]["error"] = "AUTH_FAILED"
                    result["auth"]["detail"] = f"HTTP {auth_resp.status_code}"
                    self._health_llm_cache = dict(result)
                    self._health_llm_cache_time = now
                    return result
                
                if auth_resp.status_code != 200:
                    result["auth"]["error"] = "AUTH_FAILED"
                    result["auth"]["detail"] = f"HTTP {auth_resp.status_code}: {auth_resp.text[:200]}"
                    self._health_llm_cache = dict(result)
                    self._health_llm_cache_time = now
                    return result
                
                auth_data = auth_resp.json().get("data", {})
                result["auth"] = {
                    "status": "ok",
                    "label": auth_data.get("label", "unknown"),
                }
                
                # Step 2: Balance check via /credits
                credits_resp = await client.get(f"{base_url}/credits", headers=headers)
                if credits_resp.status_code == 200:
                    credits_data = credits_resp.json().get("data", {})
                    total_credits = float(credits_data.get("total_credits", 0.0))
                    total_usage = float(credits_data.get("total_usage", 0.0))
                    remaining = total_credits - total_usage
                    
                    balance_ok = remaining >= min_usd
                    result["balance"] = {
                        "status": "ok" if balance_ok else "fail",
                        "total_credits_usd": round(total_credits, 2),
                        "used_usd": round(total_usage, 2),
                        "remaining_usd": round(remaining, 2),
                        "threshold_usd": min_usd,
                    }
                    if not balance_ok:
                        result["balance"]["error"] = "BALANCE_BELOW_THRESHOLD"
                        result["status"] = "fail"
                    else:
                        result["status"] = "ok"
                else:
                    result["balance"] = {
                        "status": "fail",
                        "error": "CREDITS_API_FAILED",
                        "detail": f"HTTP {credits_resp.status_code}",
                    }
                    
        except httpx.ConnectTimeout as e:
            result["auth"]["error"] = "CONNECTION_FAILED"
            result["auth"]["detail"] = f"ConnectTimeout: {str(e)}"
        except httpx.TimeoutException as e:
            result["auth"]["error"] = "CONNECTION_FAILED"
            result["auth"]["detail"] = f"Timeout: {str(e)}"
        except httpx.ConnectError as e:
            result["auth"]["error"] = "CONNECTION_FAILED"
            result["auth"]["detail"] = f"ConnectError: {str(e)}"
        except Exception as e:
            result["auth"]["error"] = "CONNECTION_FAILED"
            result["auth"]["detail"] = f"{type(e).__name__}: {str(e)}"
        
        # Cache result
        self._health_llm_cache = dict(result)
        self._health_llm_cache_time = now
        return result

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk space availability for the data storage path.
        
        Returns structured status with actual free space and threshold.
        """
        import shutil
        
        # Determine path to check
        disk_path = os.getenv("MEMORY_HEALTH_DISK_PATH", "").strip()
        if not disk_path:
            # Fallback to ingest job store directory or current directory
            disk_path = os.getenv("MEMORY_INGEST_JOB_DB_PATH", "")
            if disk_path:
                disk_path = os.path.dirname(os.path.abspath(disk_path))
            if not disk_path:
                disk_path = os.getcwd()
        
        min_free_mb = float(os.getenv("MEMORY_HEALTH_DISK_MIN_FREE_MB", "512"))
        
        result: Dict[str, Any] = {
            "status": "fail",
            "path": disk_path,
            "threshold_mb": min_free_mb,
        }
        
        # Check path accessibility
        if not os.path.exists(disk_path):
            result["error"] = "PATH_NOT_ACCESSIBLE"
            result["detail"] = f"Path does not exist: {disk_path}"
            return result
        
        if not os.access(disk_path, os.R_OK | os.W_OK):
            result["error"] = "PATH_NOT_ACCESSIBLE"
            result["detail"] = f"Path not readable/writable: {disk_path}"
            return result
        
        try:
            usage = shutil.disk_usage(disk_path)
            total_mb = usage.total / (1024 * 1024)
            used_mb = usage.used / (1024 * 1024)
            free_mb = usage.free / (1024 * 1024)
            
            result["total_mb"] = round(total_mb, 2)
            result["used_mb"] = round(used_mb, 2)
            result["free_mb"] = round(free_mb, 2)
            
            if free_mb >= min_free_mb:
                result["status"] = "ok"
            else:
                result["status"] = "fail"
                result["error"] = "SPACE_BELOW_THRESHOLD"
                
        except PermissionError as e:
            result["error"] = "PATH_NOT_ACCESSIBLE"
            result["detail"] = f"PermissionError: {str(e)}"
        except Exception as e:
            result["error"] = "PATH_NOT_ACCESSIBLE"
            result["detail"] = f"{type(e).__name__}: {str(e)}"
        
        return result

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive deep health check including LLM provider and disk space.
        
        Returns:
            Dict with 'status' ('ok'/'fail'), 'timestamp', and detailed 'dependencies' dict.
            Individual dependency checks include actual values and specific error codes.
        """
        from datetime import datetime, timezone
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Base infrastructure checks
        try:
            vec_status = await self.vectors.health()
        except Exception as e:
            vec_status = {"status": "fail", "error": str(e)}
        
        try:
            graph_status = await self.graph.health()
        except Exception as e:
            graph_status = {"status": "fail", "error": str(e)}
        
        # Deep checks
        llm_status = await self._check_llm_provider_health()
        disk_status = self._check_disk_health()
        
        # Aggregate status
        all_ok = (
            vec_status.get("status") == "ok"
            and graph_status.get("status") == "ok"
            and llm_status.get("status") == "ok"
            and disk_status.get("status") == "ok"
        )
        
        return {
            "status": "ok" if all_ok else "fail",
            "timestamp": timestamp,
            "dependencies": {
                "vectors": vec_status,
                "graph": graph_status,
                "llm_provider": llm_status,
                "disk": disk_status,
            },
        }

    # ---- Write batching ----
    def enable_write_batching(self, *, enabled: bool = True, max_items: int | None = None) -> None:
        self._batch_enabled = bool(enabled)
        if max_items is not None:
            self._batch_max_items = int(max_items)

    async def enqueue_write(self, entries: List[MemoryEntry], links: Optional[List[Edge]] = None, *, upsert: bool = True) -> None:
        if not entries and not links:
            return
        if not self._batch_enabled:
            # fallback to immediate write
            await self.write(entries, links, upsert=upsert)
            return
        # track approximate size to avoid OOM
        def _est_bytes(e: MemoryEntry) -> int:
            # lightweight size estimate: contents text + small overhead for metadata/vectors
            b = 16
            try:
                if e.contents:
                    for c in e.contents:
                        b += len(str(c))
            except Exception:
                pass
            try:
                if e.vectors:
                    for v in e.vectors.values():
                        if isinstance(v, list):
                            b += 8 * len(v)
            except Exception:
                pass
            try:
                for k, v in (e.metadata or {}).items():
                    b += len(str(k)) + len(str(v))
            except Exception:
                pass
            return b

        add_bytes = 0
        for e in entries or []:
            self._batch_entries.append(e)
            add_bytes += _est_bytes(e)
        self._batch_bytes += add_bytes
        if links:
            self._batch_links.extend(links)
        # flush conditions: items, bytes, pending
        if (
            len(self._batch_entries) >= max(1, self._batch_max_items)
            or self._batch_bytes >= max(1, self._batch_max_bytes)
            or len(self._batch_entries) >= max(1, self._batch_max_pending)
        ):
            await self.flush_write_batch(upsert=upsert)

    async def flush_write_batch(self, *, upsert: bool = True) -> None:
        if not self._batch_entries and not self._batch_links:
            return
        entries = list(self._batch_entries)
        links = list(self._batch_links)
        self._batch_entries.clear()
        self._batch_links.clear()
        self._batch_bytes = 0
        # single batched write
        await self.write(entries, links, upsert=upsert)
        try:
            inc("write_batch_flush_total", 1)
        except Exception:
            pass

    async def timeline_summary(
        self,
        *,
        query: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_segments: int = 100,
        granularity: str = "clip",
        include_semantic: bool = True,
        graph_params: Optional[Dict[str, Any]] = None,
        topk_per_segment: int = 3,
        topk_search: int = 200,
    ) -> Dict[str, Any]:
        """Build a semantic timeline summary from episodic events.

        v1 实现：
        - 调用 search()（可选 query）获取较大 topK 命中，expand_graph=True
        - 仅使用 kind=episodic 的命中，按 timestamp 排序；按 clip_id 分组（granularity=clip）
        - 每段生成描述与 anchors；可附邻居（若 include_semantic=True）
        """
        # Merge time range into filters
        fdict: Dict[str, Any] = {}
        if isinstance(filters, SearchFilters):
            fdict = filters.model_dump(exclude_none=True)
        if start_time is not None or end_time is not None:
            rng: Dict[str, Any] = {}
            if start_time is not None:
                rng["gte"] = float(start_time)
            if end_time is not None:
                rng["lte"] = float(end_time)
            if rng:
                fdict["time_range"] = rng
        try:
            search_filters = SearchFilters.model_validate(fdict) if fdict else None
        except Exception:
            search_filters = None
        res = await self.search(
            query or "",
            topk=topk_search,
            filters=search_filters,
            expand_graph=True,
            # timeline 摘要允许得分为0的事件参与，避免默认阈值截断
            threshold=0.0,
        )
        # Collect episodic hits
        episodic: list[dict[str, Any]] = []
        for h in (res.hits or []):
            e = h.entry
            try:
                if str(e.kind).lower() != "episodic":
                    continue
                md = e.metadata or {}
                ts = md.get("timestamp")
                clip = md.get("clip_id")
                if ts is None and clip is None:
                    continue
                episodic.append({
                    "id": e.id,
                    "score": h.score,
                    "timestamp": float(ts) if ts is not None else None,
                    "clip_id": int(clip) if clip is not None else None,
                    "text": e.get_primary_content("") or "",
                    "metadata": md,
                })
            except Exception:
                continue
        episodic.sort(key=lambda x: (float(x.get("timestamp") or 0.0), -float(x.get("score") or 0.0)))
        # Group by clip (v1)
        by_clip: dict[int, list[dict[str, Any]]] = {}
        for it in episodic:
            cid = it.get("clip_id")
            if isinstance(cid, int):
                by_clip.setdefault(cid, []).append(it)
        events: list[dict[str, Any]] = []
        clip_ids = sorted(by_clip.keys())[: max_segments]
        for idx, cid in enumerate(clip_ids, 1):
            lst = by_clip.get(cid, [])
            if not lst:
                continue
            ts_vals = [x.get("timestamp") for x in lst if x.get("timestamp") is not None]
            start_ts = min(ts_vals) if ts_vals else None
            end_ts = max(ts_vals) if ts_vals else None
            texts = [x.get("text") for x in lst if x.get("text")] [: topk_per_segment]
            desc = "；".join(texts) if texts else f"clip={cid}"
            ev: dict[str, Any] = {
                "id": f"{idx:02d}",
                "clip": cid,
                "time_range": f"[{(start_ts or 0.0):.2f}–{(end_ts or (start_ts or 0.0)):.2f}s]",
                "description": desc,
                "anchors": [{"entry_id": x.get("id"), "score": x.get("score")} for x in lst[: topk_per_segment]],
            }
            if include_semantic:
                try:
                    best = lst[0].get("id") if lst else None
                    if best and isinstance(res.neighbors, dict):
                        nbrs = (res.neighbors.get("neighbors") or {}).get(best, [])
                        ev["neighbors"] = nbrs
                except Exception:
                    pass
            events.append(ev)
        all_ts = [x.get("timestamp") for x in episodic if x.get("timestamp") is not None]
        tmin = min(all_ts) if all_ts else 0.0
        tmax = max(all_ts) if all_ts else 0.0
        timeline = {
            "provider": "MemoryService",
            "origin": "timeline_summary",
            "adapter": "search+group",
            "segments": len(events),
            "duration": f"{(tmax - tmin):.2f}s",
            "time_range": f"{tmin:.2f}s → {tmax:.2f}s",
            "events": events,
            "total_clips": len(clip_ids),
            "statistics": {
                "node_count": len(episodic),
                "edge_count": sum(len(v) for v in ((res.neighbors or {}).get("neighbors", {}) or {}).values()) if isinstance(res.neighbors, dict) else 0,
                "segment_count": len(events),
                "avg_clip_duration": ((tmax - tmin) / max(1, len(clip_ids))) if clip_ids else 0.0,
            },
            "trace": {
                "query": query,
                "filters": fdict,
                "start_time": start_time,
                "end_time": end_time,
                "hits": len(res.hits or []),
            },
        }
        return timeline

    # ---- Maintenance jobs ----
    async def run_ttl_cleanup_now(self) -> int:
        """Trigger TTL cleanup if vector store supports in-memory dump.

        Returns number of entries marked deleted (if available), else 0.
        """
        try:
            # optional import to avoid circular
            from modules.memory.application.ttl_jobs import run_ttl_cleanup
            return await run_ttl_cleanup(self.vectors)
        except Exception:
            return 0

    async def decay_graph_edges(self, *, factor: float = 0.9, rel_whitelist: Optional[list[str]] = None, min_weight: float = 0.0) -> bool:
        try:
            if hasattr(self.graph, "decay_edges"):
                await self.graph.decay_edges(factor=factor, rel_whitelist=rel_whitelist, min_weight=min_weight)
                return True
        except Exception:
            return False
        return False

    # ---- Equivalence pending workflow ----
    async def add_pending_equivalence(self, pairs: list[tuple[str, str]], *, scores: Optional[list[float]] = None, reasons: Optional[list[str]] = None) -> int:
        """Add or mark equivalence relations as pending.

        Returns number of pairs accepted.
        """
        if not pairs:
            return 0
        n = 0
        if hasattr(self.graph, "add_pending_equivalence"):
            for idx, (src, dst) in enumerate(pairs):
                try:
                    sc = scores[idx] if scores and idx < len(scores) else None  # type: ignore[index]
                    rs = reasons[idx] if reasons and idx < len(reasons) else None  # type: ignore[index]
                    await self.graph.add_pending_equivalence(src, dst, score=sc, reason=rs)
                    n += 1
                except Exception:
                    continue
        return n

    async def list_pending_equivalence(self, *, limit: int = 50) -> dict:
        if hasattr(self.graph, "list_pending_equivalence"):
            return await self.graph.list_pending_equivalence(limit=limit)
        return {"pending": []}

    async def confirm_pending_equivalence(self, pairs: list[tuple[str, str]], *, weight: float | None = None) -> int:
        if hasattr(self.graph, "confirm_equivalence"):
            try:
                cnt = await self.graph.confirm_equivalence(pairs, weight=weight)
            except TypeError:
                # older signature without weight
                cnt = await self.graph.confirm_equivalence(pairs)  # type: ignore[misc]
            try:
                from modules.memory.application.metrics import inc as _inc
                _inc("equivalence_confirmed_total", int(cnt))
            except Exception:
                pass
            return int(cnt)
        return 0

    async def remove_pending_equivalence(self, pairs: list[tuple[str, str]]) -> int:
        if hasattr(self.graph, "delete_equivalence"):
            try:
                cnt = await self.graph.delete_equivalence(pairs)
                return int(cnt)
            except Exception:
                return 0
        return 0

class SafetyError(Exception):
    pass
