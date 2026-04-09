from __future__ import annotations

from typing import List, Any, Dict, Tuple, Optional
import asyncio
import concurrent.futures
import httpx
import requests
import hashlib
import logging
import os
import threading

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.metrics import inc, observe_payload_items, observe_vector_size
from modules.memory.application.embedding_adapter import (
    build_embedding_from_settings,
    build_image_embedding_from_settings,
    build_audio_embedding_from_settings,
)


class QdrantStore:
    """Minimal REST-based Qdrant facade（文本集合 MVP）。

    预期 settings：
    {
      "host": "127.0.0.1", "port": 6333, "api_key": "...",
      "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}
    }
    """

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        host = self.settings.get("host", "127.0.0.1")
        port = int(self.settings.get("port", 6333))
        self.base = f"http://{host}:{port}"
        self.api_key = self.settings.get("api_key")
        # Allow arbitrary modality→collection mapping; keep legacy defaults
        self.collections = self.settings.get(
            "collections",
            {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"},
        )
        self.session = requests.Session()
        self._session_lock = threading.Lock()
        self._async_client: httpx.AsyncClient | None = None
        # Never use environment proxies for local Qdrant by default. Proxying localhost is a common source
        # of "writes succeeded but points_count is 0" confusion (writes go to a proxy/other endpoint).
        try:
            self.session.trust_env = False
        except Exception:
            pass
        self._closed: bool = False
        if self.api_key:
            self.session.headers.update({"api-key": self.api_key})
        transport_cfg = self.settings.get("transport") or {}
        if not isinstance(transport_cfg, dict):
            transport_cfg = {}
        self._use_async_http = bool(transport_cfg.get("use_async_http", False))
        self._http_timeout_s = float(transport_cfg.get("timeout_seconds") or 15.0)
        self._http_max_connections = int(transport_cfg.get("max_connections") or 128)
        self._http_max_keepalive_connections = int(transport_cfg.get("max_keepalive_connections") or 32)
        sharding_cfg = self.settings.get("sharding") or {}
        if not isinstance(sharding_cfg, dict):
            sharding_cfg = {}
        self._tenant_sharding_enabled = bool(sharding_cfg.get("enabled", False))
        self._tenant_sharding_key_field = str(sharding_cfg.get("key_field") or "tenant_id").strip() or "tenant_id"
        self._tenant_sharding_method = str(sharding_cfg.get("method") or "custom").strip().lower() or "custom"
        try:
            self._tenant_sharding_shard_number = int(sharding_cfg.get("shard_number") or 1)
        except Exception:
            self._tenant_sharding_shard_number = 1
        try:
            self._tenant_sharding_replication_factor = int(sharding_cfg.get("replication_factor") or 1)
        except Exception:
            self._tenant_sharding_replication_factor = 1
        try:
            self._tenant_sharding_write_consistency_factor = int(
                sharding_cfg.get("write_consistency_factor") or 1
            )
        except Exception:
            self._tenant_sharding_write_consistency_factor = 1
        self._tenant_sharding_namespace_ids_by_tenant = bool(sharding_cfg.get("namespace_ids_by_tenant", False))
        self._known_shard_keys: set[Tuple[str, str]] = set()
        self._known_shard_keys_lock = threading.Lock()
        # Build embedding functions (text + multi-modal slots)
        emb_cfg = self.settings.get("embedding", {}) or {}
        # Validate embedding/collections coherence (fail-fast for explicit dims)
        self._validate_embedding_config(emb_cfg)
        # text
        self.embed_text = build_embedding_from_settings(emb_cfg if isinstance(emb_cfg, dict) else {})
        # image/audio/clip_image 子配置可用 embedding.image / embedding.audio / embedding.clip_image
        img_cfg = emb_cfg.get("image") if isinstance(emb_cfg, dict) else None
        aud_cfg = emb_cfg.get("audio") if isinstance(emb_cfg, dict) else None
        clip_img_cfg = emb_cfg.get("clip_image") if isinstance(emb_cfg, dict) else None
        self.embed_image = build_image_embedding_from_settings(img_cfg)
        # clip_image 复用 OpenCLIP 文本对齐编码；单独配置维度
        self.embed_clip_image = build_image_embedding_from_settings(clip_img_cfg)
        self.embed_audio = build_audio_embedding_from_settings(aud_cfg)
        # Reliability settings
        rel = (self.settings.get("reliability") or {}) if isinstance(self.settings.get("reliability"), dict) else {}
        self._retry_attempts = int(rel.get("retries", {}).get("max_attempts", 2))
        self._backoff_base_ms = int(rel.get("retries", {}).get("backoff_base_ms", 100))
        self._backoff_max_ms = int(rel.get("retries", {}).get("backoff_max_ms", 1000))
        self._cb_failure_threshold = int(rel.get("circuit_breaker", {}).get("failure_threshold", 5))
        self._cb_cooldown_s = int(rel.get("circuit_breaker", {}).get("cooldown_seconds", 30))
        self._cb_fail_count = 0
        self._cb_open_until = 0.0
        self._logger = logging.getLogger(__name__)
        # Qdrant upsert is async by default (wait=false). When service reports "COMPLETED", callers expect
        # points to be queryable immediately; default to wait=true for correctness.
        try:
            self._upsert_wait = str(os.getenv("MEMORY_QDRANT_UPSERT_WAIT", "true")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        except Exception:
            self._upsert_wait = True
        # modality score weights (optional), e.g., {"text":1.0,"clip_image":0.8,"image":0.5}
        try:
            self._mod_weights = dict(((self.settings.get("search") or {}).get("modality_weights") or {}))
        except Exception:
            self._mod_weights = {}

    def _request_sync(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Legacy sync HTTP request guarded by a session lock."""
        with self._session_lock:
            fn = getattr(self.session, method.lower(), None)
            if not callable(fn):
                req = getattr(self.session, "request", None)
                if callable(req):
                    return req(method, url, **kwargs)
                raise AttributeError(f"qdrant_session_missing_method: {method.lower()}")
            return fn(url, **kwargs)

    def _session_has_runtime_overrides(self) -> bool:
        """Detect monkeypatched requests.Session methods and honor those overrides."""
        try:
            if not isinstance(self.session, requests.Session):
                return True
            session_dict = getattr(self.session, "__dict__", {})
            for name in ("request", "get", "post", "put", "delete", "patch"):
                if name in session_dict:
                    return True
        except Exception:
            return False
        return False

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            headers: Dict[str, str] = {}
            if self.api_key:
                headers["api-key"] = str(self.api_key)
            self._async_client = httpx.AsyncClient(
                trust_env=False,
                headers=headers,
                timeout=httpx.Timeout(self._http_timeout_s),
                limits=httpx.Limits(
                    max_connections=max(1, self._http_max_connections),
                    max_keepalive_connections=max(1, self._http_max_keepalive_connections),
                ),
            )
        return self._async_client

    async def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        # True async concurrent path (default). Fall back to sync+lock when tests/runtime override session methods.
        if self._use_async_http and not self._session_has_runtime_overrides():
            client = self._get_async_client()
            return await client.request(method.upper(), url, **kwargs)
        return await asyncio.to_thread(self._request_sync, method, url, **kwargs)

    # ---- Lifecycle management ----
    def close(self) -> None:
        """Close underlying HTTP session; idempotent."""
        if getattr(self, "_closed", False):
            return
        try:
            aclient = self._async_client
            self._async_client = None
            if aclient is not None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    try:
                        asyncio.run(aclient.aclose())
                    except Exception:
                        pass
                else:
                    try:
                        loop.create_task(aclient.aclose())
                    except Exception:
                        pass
            try:
                self.session.close()
            except Exception:
                pass
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # allow runtime override of modality weights (e.g., via API)
    def set_modality_weights(self, weights: Dict[str, float] | None) -> None:
        if not isinstance(weights, dict):
            self._mod_weights = {}
            return
        neww: Dict[str, float] = {}
        for k, v in weights.items():
            try:
                neww[str(k)] = float(v)
            except Exception:
                continue
        self._mod_weights = neww

    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any] | None:
        """构造 Qdrant 兼容的过滤条件。

        设计约束：
        - 仅使用 Qdrant Filter 支持的顶层字段：must / should / must_not；
        - 不再使用 minimum_should_match（Qdrant 不支持，会触发 400）；
        - Soft TTL 由上层治理（ttl/created_at）负责，这里不再对 expires_at 做硬过滤，
          以避免因为缺失字段导致所有文档被意外过滤掉。
        """

        must: list[Dict[str, Any]] = []
        should: list[Dict[str, Any]] = []
        must_not: list[Dict[str, Any]] = []

        # modality (AND across dimensions, OR within dimension)
        mods = filters.get("modality")
        if isinstance(mods, list) and mods:
            if len(mods) == 1:
                must.append({"key": "modality", "match": {"value": mods[0]}})
            else:
                # flatten OR modalities into top-level should
                should.extend([{"key": "modality", "match": {"value": m}} for m in mods])

        # memory_type (kind)
        mtypes = filters.get("memory_type")
        if isinstance(mtypes, list) and mtypes:
            if len(mtypes) == 1:
                must.append({"key": "kind", "match": {"value": mtypes[0]}})
            else:
                should.extend([{"key": "kind", "match": {"value": t}} for t in mtypes])

        # source (metadata.source)
        srcs = filters.get("source")
        if isinstance(srcs, list) and srcs:
            if len(srcs) == 1:
                must.append({"key": "metadata.source", "match": {"value": srcs[0]}})
            else:
                should.extend([{"key": "metadata.source", "match": {"value": s}} for s in srcs])
        elif isinstance(srcs, str) and srcs.strip():
            # be tolerant to single string and coerce to list semantics
            must.append({"key": "metadata.source", "match": {"value": srcs}})

        # internal: exclude sources by default (must_not)
        ex_srcs = filters.get("__exclude_sources")
        if isinstance(ex_srcs, list) and ex_srcs:
            for s in ex_srcs:
                if s is None:
                    continue
                ss = str(s).strip()
                if not ss:
                    continue
                must_not.append({"key": "metadata.source", "match": {"value": ss}})

        # tenant_id: 硬租户边界，直接映射到 metadata.tenant_id（若提供）
        tenant_id = filters.get("tenant_id")
        if tenant_id is not None:
            must.append({"key": "metadata.tenant_id", "match": {"value": tenant_id}})

        # user_id (array) with match mode any/all; 我们在 metadata.user_id 中存储。
        # - mode=all: 所有 id 作为 MUST；
        # - mode=any: OR 语义，使用 should（Qdrant 原生支持，无需 minimum_should_match）。
        user_ids = filters.get("user_id")
        if isinstance(user_ids, list) and user_ids:
            mode = str(filters.get("user_match") or "any").lower()
            if mode == "all":
                # require all user ids to be present
                for uid in user_ids:
                    must.append({"key": "metadata.user_id", "match": {"value": uid}})
            else:
                # any: OR 语义，保留全部 user_id
                should.extend([{"key": "metadata.user_id", "match": {"value": uid}} for uid in user_ids])
        # memory_domain exact
        dom = filters.get("memory_domain")
        if dom is not None:
            must.append({"key": "metadata.memory_domain", "match": {"value": dom}})
        # run_id exact
        rid = filters.get("run_id")
        if rid is not None:
            must.append({"key": "metadata.run_id", "match": {"value": rid}})
        # memory_scope exact
        mscope = filters.get("memory_scope")
        if mscope is not None:
            must.append({"key": "metadata.memory_scope", "match": {"value": mscope}})
        # published (top-level)
        pub = filters.get("published")
        if pub is True:
            # include published or missing; exclude explicit false
            must_not.append({"key": "published", "match": {"value": False}})
        elif pub is False:
            must.append({"key": "published", "match": {"value": False}})
        # character_id (OR across provided ids)
        chars = filters.get("character_id")
        if isinstance(chars, list) and chars:
            if len(chars) == 1:
                must.append({"key": "metadata.character_id", "match": {"value": chars[0]}})
            else:
                should.extend([{"key": "metadata.character_id", "match": {"value": c}} for c in chars])
        elif isinstance(chars, str):
            must.append({"key": "metadata.character_id", "match": {"value": chars}})

        # clip_id
        clip_id = filters.get("clip_id")
        if clip_id is not None:
            must.append({"key": "metadata.clip_id", "match": {"value": clip_id}})

        # time_range (only if numeric timestamps are provided; strings are skipped)
        tr = filters.get("time_range")
        if isinstance(tr, dict) and ("gte" in tr or "lte" in tr):
            rng: Dict[str, Any] = {}
            if isinstance(tr.get("gte"), (int, float)):
                rng["gte"] = float(tr.get("gte"))
            if isinstance(tr.get("lte"), (int, float)):
                rng["lte"] = float(tr.get("lte"))
            if rng:
                must.append({"key": "metadata.timestamp", "range": rng})

        # ---- New context keys ----
        # agent_persona
        persona = filters.get("agent_persona")
        if persona and isinstance(persona, str):
            must.append({"key": "metadata.agent_persona", "match": {"value": persona}})

        # session_id
        session = filters.get("session_id")
        if session and isinstance(session, str):
            must.append({"key": "metadata.session_id", "match": {"value": session}})

        # invoked_by
        invoker = filters.get("invoked_by")
        if invoker and isinstance(invoker, str):
            must.append({"key": "metadata.invoked_by", "match": {"value": invoker}})

        # memory_about (list, OR logic)
        about = filters.get("memory_about")
        if about and isinstance(about, list):
            should.extend([{"key": "metadata.memory_about", "match": {"value": a}} for a in about])

        # topic_path (list or str)
        topic_paths = filters.get("topic_path")
        if isinstance(topic_paths, list) and topic_paths:
            must.append({"key": "metadata.topic_path", "match": {"any": topic_paths}})
        elif isinstance(topic_paths, str) and topic_paths.strip():
            must.append({"key": "metadata.topic_path", "match": {"value": topic_paths}})

        # tags (list or str, OR logic)
        tags = filters.get("tags")
        if isinstance(tags, list) and tags:
            must.append({"key": "metadata.tags", "match": {"any": tags}})
        elif isinstance(tags, str) and tags.strip():
            must.append({"key": "metadata.tags", "match": {"value": tags}})

        # keywords (list or str, OR logic)
        keywords = filters.get("keywords")
        if isinstance(keywords, list) and keywords:
            must.append({"key": "metadata.keywords", "match": {"any": keywords}})
        elif isinstance(keywords, str) and keywords.strip():
            must.append({"key": "metadata.keywords", "match": {"value": keywords}})

        # time_bucket (list or str, OR logic)
        time_bucket = filters.get("time_bucket")
        if isinstance(time_bucket, list) and time_bucket:
            must.append({"key": "metadata.time_bucket", "match": {"any": time_bucket}})
        elif isinstance(time_bucket, str) and time_bucket.strip():
            must.append({"key": "metadata.time_bucket", "match": {"value": time_bucket}})

        # tags_vocab_version (str)
        tags_vocab_version = filters.get("tags_vocab_version")
        if isinstance(tags_vocab_version, str) and tags_vocab_version.strip():
            must.append({"key": "metadata.tags_vocab_version", "match": {"value": tags_vocab_version}})

        # fold into filter（Qdrant Filter 仅支持 must/should/must_not）
        out: Dict[str, Any] = {}
        if must:
            out["must"] = must
        if should:
            out["should"] = should
        if must_not:
            out["must_not"] = must_not
        return out or None

    def tenant_sharding_enabled(self) -> bool:
        return bool(self._tenant_sharding_enabled)

    def tenant_scoped_ids_enabled(self) -> bool:
        return bool(self._tenant_sharding_enabled and self._tenant_sharding_namespace_ids_by_tenant)

    def _normalize_shard_key(self, value: Any) -> str | None:
        raw = str(value or "").strip()
        return raw or None

    def _resolve_shard_key_from_filters(self, filters: Dict[str, Any] | None) -> str | None:
        if not self._tenant_sharding_enabled:
            return None
        if not isinstance(filters, dict):
            return None
        return self._normalize_shard_key(filters.get(self._tenant_sharding_key_field))

    def _resolve_shard_key_from_entry(self, entry: MemoryEntry) -> str | None:
        if not self._tenant_sharding_enabled:
            return None
        try:
            meta = entry.metadata or {}
        except Exception:
            meta = {}
        return self._normalize_shard_key(meta.get(self._tenant_sharding_key_field))

    def _with_shard_key(self, payload: Dict[str, Any], shard_key: str | None) -> Dict[str, Any]:
        if shard_key is None:
            return payload
        body = dict(payload)
        body["shard_key"] = shard_key
        return body

    def _tenant_from_qdrant_filter(self, filters: Dict[str, Any] | None) -> str | None:
        if not self._tenant_sharding_enabled:
            return None
        if not isinstance(filters, dict):
            return None
        must = filters.get("must")
        if not isinstance(must, list):
            return None
        key_name = f"metadata.{self._tenant_sharding_key_field}"
        for clause in must:
            if not isinstance(clause, dict):
                continue
            if str(clause.get("key") or "").strip() != key_name:
                continue
            match = clause.get("match")
            if not isinstance(match, dict):
                continue
            shard_key = self._normalize_shard_key(match.get("value"))
            if shard_key:
                return shard_key
        return None

    async def _ensure_custom_shard(self, collection: str, shard_key: str | None) -> None:
        if not self._tenant_sharding_enabled or self._tenant_sharding_method != "custom":
            return
        shard = self._normalize_shard_key(shard_key)
        coll = str(collection or "").strip()
        if not coll or not shard:
            return
        cache_key = (coll, shard)
        with self._known_shard_keys_lock:
            if cache_key in self._known_shard_keys:
                return
        url = f"{self.base}/collections/{coll}/shards"
        resp = await self._request("PUT", url, json={"shard_key": shard}, timeout=10)
        status = int(getattr(resp, "status_code", 500))
        if status >= 400 and status != 409:
            body = str(getattr(resp, "text", "") or "")[:500]
            lowered = body.lower()
            if "already exists" not in lowered and "exists" not in lowered:
                raise RuntimeError(
                    f"qdrant_create_shard_failed: collection={coll} shard_key={shard} status={status} body={body}"
                )
        with self._known_shard_keys_lock:
            self._known_shard_keys.add(cache_key)


    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:
        # Buckets accept multiple modality-target collections, including extended ones.
        # When tenant sharding is enabled, split each modality bucket by shard key.
        buckets: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
        # helper: expected dims per modality
        emb_cfg = self.settings.get("embedding", {}) or {}
        exp_dims = {
            "text": int(emb_cfg.get("dim") or 768),
            "image": int(((emb_cfg.get("image") or {}).get("dim") or 512)),
            "clip_image": int(((emb_cfg.get("clip_image") or {}).get("dim") or ((emb_cfg.get("image") or {}).get("dim") or 512))),
            "audio": int(((emb_cfg.get("audio") or {}).get("dim") or 192)),
            "face": int(((emb_cfg.get("face") or {}).get("dim") or 512)),
        }

        def _assert_dim(mod: str, vec: List[float], entry: MemoryEntry) -> None:
            try:
                need = int(exp_dims.get(mod) or 0)
                got = int(len(vec or []))
                if need and got != need:
                    # surface a clear error early to aid debugging instead of opaque 400 from backend
                    raise RuntimeError(
                        f"vector_dim_mismatch: modality={mod} expected={need} got={got} id={entry.id} clip_id={entry.metadata.get('clip_id')} source={entry.metadata.get('source')}"
                    )
            except Exception as _e:
                # re-raise to fail fast
                raise

        # Pre-batch embedding for text modality (throughput boost with local models)
        text_need_idx: list[int] = []
        text_inputs: list[str] = []
        for idx, e in enumerate(entries):
            if e.modality == "text":
                vec = None
                if e.vectors and isinstance(e.vectors.get("text"), list):
                    vec = e.vectors.get("text")  # type: ignore[assignment]
                if vec is None and (e.contents and isinstance(e.contents[0], str)):
                    text_need_idx.append(idx)
                    text_inputs.append(e.contents[0])
        text_vecs: list[list[float]] = []
        if text_need_idx:
            # try batch encode if available
            batch_fn = getattr(self.embed_text, "encode_batch", None)
            if callable(batch_fn):
                try:
                    # optional batch size from settings
                    bsz = None
                    emb_cfg = self.settings.get("embedding", {}) or {}
                    try:
                        bsz = int(emb_cfg.get("batch_size", 0) or 0) or None
                    except Exception:
                        bsz = None
                    text_vecs = list(await asyncio.to_thread(batch_fn, text_inputs, bsz=bsz))  # type: ignore[misc]
                except Exception:
                    text_vecs = []
        # Index for mapped vectors
        text_idx_to_vec: dict[int, List[float]] = {}
        if text_need_idx and text_vecs and len(text_vecs) == len(text_need_idx):
            for i, idx in enumerate(text_need_idx):
                text_idx_to_vec[idx] = list(text_vecs[i])

        for i, e in enumerate(entries):
            if not e.contents:
                continue
            shard_key = self._resolve_shard_key_from_entry(e)
            if self._tenant_sharding_enabled and shard_key is None:
                raise RuntimeError(
                    f"qdrant_shard_key_missing: field={self._tenant_sharding_key_field} id={e.id} modality={e.modality}"
                )
            # Observe payload size per entry (helps monitor large contents like base64 images)
            try:
                observe_payload_items(e.modality, len(e.contents or []))
            except Exception:
                pass
            payload = e.model_dump()
            # For easier inspection and compatibility with older tooling,
            # expose a single-string "content" alongside the canonical "contents" list.
            try:
                if isinstance(payload, dict):
                    contents = payload.get("contents") or []
                    if contents and not payload.get("content"):
                        first = contents[0]
                        # Avoid surprising non-str types; MemoryEntry already stringifies,
                        # but be defensive here as well.
                        payload["content"] = str(first)
            except Exception:
                # Best-effort only; never block writes because of debug fields.
                pass
            # Qdrant requires point IDs to be either uint64 or UUID strings.
            # Convert readable IDs (like 'tkg_event_evt_001') to valid UUIDs.
            raw_id = e.id or hashlib.md5(str(payload).encode("utf-8")).hexdigest()
            try:
                # Check if already a valid UUID
                import uuid as _uuid
                _uuid.UUID(raw_id)
                pid = raw_id
            except (ValueError, AttributeError):
                # Not a valid UUID, hash it to create a deterministic UUID
                pid = str(_uuid.UUID(hashlib.md5(raw_id.encode("utf-8")).hexdigest()))
            if e.modality == "text":
                vec = None
                if e.vectors and isinstance(e.vectors.get("text"), list):
                    vec = e.vectors.get("text")  # type: ignore[assignment]
                if vec is None:
                    if i in text_idx_to_vec:
                        vec = text_idx_to_vec[i]
                    else:
                        vec = await asyncio.to_thread(self.embed_text, e.contents[0])
                try:
                    observe_vector_size("text", len(vec or []))
                except Exception:
                    pass
                _assert_dim("text", vec, e)
                buckets.setdefault(("text", shard_key), []).append({"id": pid, "vector": vec, "payload": payload})
            elif e.modality == "image":
                # Support multiple vector spaces from a single image entry:
                # - face: identity vector (insightface)
                # - clip_image: OpenCLIP-aligned vector for cross-modal text->image
                # - image: legacy fallback
                # 1) face vector
                if e.vectors and isinstance(e.vectors.get("face"), list):
                    vec_face = e.vectors.get("face")  # type: ignore[assignment]
                    try:
                        observe_vector_size("face", len(vec_face or []))
                    except Exception:
                        pass
                    _assert_dim("face", vec_face, e)
                    buckets.setdefault(("face", shard_key), []).append({"id": pid, "vector": vec_face, "payload": payload})
                # 2) clip_image vector
                vec_ci = None
                if e.vectors and isinstance(e.vectors.get("clip_image"), list):
                    vec_ci = e.vectors.get("clip_image")  # type: ignore[assignment]
                if vec_ci is None:
                    # embed from contents if available
                    try:
                        vec_ci = await asyncio.to_thread(self.embed_clip_image, e.contents[0])
                    except Exception:
                        vec_ci = None
                if vec_ci is not None:
                    try:
                        observe_vector_size("clip_image", len(vec_ci or []))
                    except Exception:
                        pass
                    _assert_dim("clip_image", vec_ci, e)
                    buckets.setdefault(("clip_image", shard_key), []).append({"id": pid, "vector": vec_ci, "payload": payload})
                # 3) legacy image vector
                if e.vectors and isinstance(e.vectors.get("image"), list):
                    vec_img = e.vectors.get("image")  # type: ignore[assignment]
                    try:
                        observe_vector_size("image", len(vec_img or []))
                    except Exception:
                        pass
                    _assert_dim("image", vec_img, e)
                    buckets.setdefault(("image", shard_key), []).append({"id": pid, "vector": vec_img, "payload": payload})
            elif e.modality == "audio":
                vec = None
                if e.vectors and isinstance(e.vectors.get("audio"), list):
                    vec = e.vectors.get("audio")  # type: ignore[assignment]
                if vec is None:
                    vec = await asyncio.to_thread(self.embed_audio, e.contents[0])
                try:
                    observe_vector_size("audio", len(vec or []))
                except Exception:
                    pass
                _assert_dim("audio", vec, e)
                buckets.setdefault(("audio", shard_key), []).append({"id": pid, "vector": vec, "payload": payload})
            else:
                # structured/no-vector; skip vector upsert
                continue

        async def _flush_bucket(kind: str, shard_key: str | None, pts: List[Dict[str, Any]]) -> None:
            if not pts:
                return
            coll = self.collections.get(kind)
            if not coll:
                return
            await self._ensure_custom_shard(coll, shard_key)
            wait_qs = "?wait=true" if getattr(self, "_upsert_wait", True) else ""
            url = f"{self.base}/collections/{coll}/points{wait_qs}"
            chunk_size = 512
            for i in range(0, len(pts), chunk_size):
                chunk = pts[i : i + chunk_size]
                body = self._with_shard_key({"points": chunk}, shard_key)
                resp = await self._request("PUT", url, json=body, timeout=60)
                if resp.status_code >= 400:
                    try:
                        self._logger.error(
                            "qdrant.upsert.error",
                            extra={
                                "event": "qdrant.upsert.error",
                                "module": "memory",
                                "entity": "vector",
                                "verb": "upsert",
                                "status": "error",
                                "code": resp.status_code,
                                "collection": coll,
                                "count": len(chunk),
                                "shard_key": shard_key,
                            },
                        )
                        import logging as _rootlog
                        _rootlog.error(
                            "qdrant.upsert.error",
                            extra={
                                "event": "qdrant.upsert.error",
                                "module": "memory",
                                "entity": "vector",
                                "verb": "upsert",
                                "status": "error",
                                "code": resp.status_code,
                                "collection": coll,
                                "count": len(chunk),
                                "shard_key": shard_key,
                            },
                        )
                    except Exception:
                        pass
                    import logging as _rootlog
                    body_text = resp.text[:500] if resp.text else ""
                    _rootlog.error(
                        f"qdrant.upsert.error: collection={coll} status={resp.status_code} shard_key={shard_key} body={body_text}"
                    )
                    print(
                        f"[QDRANT UPSERT ERROR] collection={coll} status={resp.status_code} shard_key={shard_key} body={body_text}"
                    )
                    raise RuntimeError(
                        f"qdrant_upsert_failed: modality={kind} collection={coll} shard_key={shard_key} status={resp.status_code} body={body_text}"
                    )
                try:
                    if hasattr(resp, "json"):
                        data = resp.json()
                        st = str((data or {}).get("status") or "").strip().lower() if isinstance(data, dict) else ""
                        if st and st != "ok":
                            body_text = resp.text[:500] if getattr(resp, "text", None) else str(data)[:500]
                            raise RuntimeError(
                                f"qdrant_upsert_unexpected_status: modality={kind} collection={coll} shard_key={shard_key} status={st} body={body_text}"
                            )
                except Exception as e:
                    body_text = resp.text[:500] if getattr(resp, "text", None) else str(e)[:500]
                    raise RuntimeError(
                        f"qdrant_upsert_unexpected_response: modality={kind} collection={coll} shard_key={shard_key} body={body_text}"
                    ) from e

        await asyncio.gather(
            *[
                _flush_bucket(kind, shard_key, pts)
                for (kind, shard_key), pts in buckets.items()
                if pts
            ]
        )
        return None

    async def search_vectors(
        self,
        query: str,
        filters: Dict[str, Any],
        topk: int,
        threshold: float | None = None,
        query_vector: List[float] | None = None,
    ) -> List[Dict[str, Any]]:
        """Search across modality-specific collections and merge results.

        - If filters.modality is provided, only those modalities are queried; otherwise defaults to ['text'].
        - Each modality uses its corresponding embedder (text/image/audio) on the textual query.
        - Results are merged and sorted by score; topk returned overall.
        """
        # Determine modalities to search
        mods_req = []
        try:
            mods_req = list(filters.get("modality") or [])
        except Exception:
            mods_req = []
        if not mods_req:
            mods_req = ["text"]

        # Prepare per-modality search payloads
        async def _vec_for(mod: str) -> List[float]:
            try:
                if mod == "text":
                    if isinstance(query_vector, list) and query_vector:
                        return list(query_vector)
                    return await asyncio.to_thread(self.embed_text, query)
                if mod == "image":
                    return await asyncio.to_thread(self.embed_image, query)
                if mod == "clip_image":
                    return await asyncio.to_thread(self.embed_clip_image, query)
                if mod == "audio":
                    return await asyncio.to_thread(self.embed_audio, query)
            except Exception:
                pass
            return []

        # Common filter building (safe for per-collection).
        # Default: hide internal index sources unless caller explicitly requests them via filters.source.
        INTERNAL_EXCLUDED_SOURCES = ["tkg_dialog_utterance_index_v1", "tkg_dialog_event_index_v1"]
        eff_filters = dict(filters or {})
        try:
            src = eff_filters.get("source")
            src_list: list[str] = []
            if isinstance(src, str) and src.strip():
                src_list = [src.strip()]
            elif isinstance(src, list):
                src_list = [str(x).strip() for x in src if x is not None and str(x).strip()]
            if not set(src_list).intersection(set(INTERNAL_EXCLUDED_SOURCES)):
                eff_filters["__exclude_sources"] = list(INTERNAL_EXCLUDED_SOURCES)
        except Exception:
            eff_filters["__exclude_sources"] = list(INTERNAL_EXCLUDED_SOURCES)

        built_filter = self._build_filter(eff_filters)
        shard_key = self._resolve_shard_key_from_filters(eff_filters)

        # Circuit breaker short-circuit
        import time as _t
        now = _t.time()
        if self._cb_open_until and now < self._cb_open_until:
            try:
                inc("circuit_breaker_short_total", 1)
            except Exception:
                pass
            return []

        merged: List[Dict[str, Any]] = []
        last_exc: Exception | None = None

        # Query each requested modality independently
        debug = str((self.settings.get("debug") or "")).lower() in ("1","true","yes") or str(__import__('os').getenv('MEMORY_SEARCH_DEBUG','')).lower() in ("1","true","yes")
        for mod in mods_req:
            coll = self.collections.get(mod)
            if not coll:
                continue
            try:
                # count ANN calls per modality
                inc(f"ann_calls_total_{mod}", 1)
            except Exception:
                pass
            import time as _t
            t0_mod = _t.perf_counter()
            vec = await _vec_for(mod)
            payload: Dict[str, Any] = {
                "vector": vec,
                "limit": max(1, topk),
                "with_payload": True,
            }
            if threshold is not None:
                payload["score_threshold"] = float(threshold)
            if built_filter:
                payload["filter"] = built_filter
            if shard_key is not None:
                payload["shard_key"] = shard_key
            url = f"{self.base}/collections/{coll}/points/search"

            # retry with exponential backoff per modality
            attempts = max(1, self._retry_attempts)
            backoff = max(1, self._backoff_base_ms) / 1000.0
            last_exc = None
            resp = None
            for i in range(attempts):
                try:
                    resp = await self._request("POST", url, json=payload, timeout=10)
                    if resp.status_code < 500:
                        self._cb_fail_count = 0
                        break
                    last_exc = RuntimeError(f"qdrant status {resp.status_code}")
                except Exception as e:
                    last_exc = e
                # failure path
                self._cb_fail_count += 1
                try:
                    inc("errors_total", 1)
                    inc("backend_retries_total", 1)
                except Exception:
                    pass
                if i < attempts - 1:
                    await asyncio.sleep(min(backoff, self._backoff_max_ms / 1000.0))
                    backoff = min(backoff * 2, self._backoff_max_ms / 1000.0)
            else:
                if self._cb_fail_count >= max(1, self._cb_failure_threshold):
                    self._cb_open_until = now + max(1, self._cb_cooldown_s)
                    try:
                        inc("circuit_breaker_open_total", 1)
                    except Exception:
                        pass
                try:
                    self._logger.error(
                        "qdrant.search.error",
                        extra={
                            "event": "qdrant.search.error",
                            "module": "memory",
                            "entity": "vector",
                            "verb": "search",
                            "status": "error",
                            "collection": coll,
                            "reason": repr(last_exc),
                        },
                    )
                    import logging as _rootlog
                    _rootlog.error(
                        "qdrant.search.error",
                        extra={
                            "event": "qdrant.search.error",
                            "module": "memory",
                            "entity": "vector",
                            "verb": "search",
                            "status": "error",
                            "collection": coll,
                            "reason": repr(last_exc),
                        },
                    )
                except Exception:
                    pass
                import logging as _rootlog
                _rootlog.error("qdrant.search.error")
                try:
                    inc("errors_total", 1)
                    inc("errors_5xx_total", 1)
                except Exception:
                    pass
                continue

            if debug:
                try:
                    st = (resp.status_code if resp is not None else 'NONE')
                    err = (repr(last_exc) if last_exc is not None else '')
                    print(f"[QdrantSearch] mod={mod} vec_len={len(vec)} status={st} has_filter={bool(built_filter)} error={err}")
                except Exception:
                    pass
            if resp is None or resp.status_code >= 400:
                try:
                    inc("errors_total", 1)
                    if resp is not None and resp.status_code >= 500:
                        inc("errors_5xx_total", 1)
                    elif resp is not None:
                        inc("errors_4xx_total", 1)
                except Exception:
                    pass
                try:
                    self._logger.warning(
                        "qdrant.search.http_error",
                        extra={
                            "event": "qdrant.search.http_error",
                            "module": "memory",
                            "entity": "vector",
                            "verb": "search",
                            "status": "warn",
                            "code": (resp.status_code if resp is not None else None),
                            "collection": coll,
                            "body": (resp.text[:500] if resp is not None else ""),  # 增加响应体
                        },
                    )
                    import logging as _rootlog
                    _rootlog.error(
                        "qdrant.search.http_error",
                        extra={
                            "event": "qdrant.search.http_error",
                            "module": "memory",
                            "entity": "vector",
                            "verb": "search",
                            "status": "warn",
                            "code": (resp.status_code if resp is not None else None),
                            "collection": coll,
                        },
                    )
                except Exception:
                    pass
                import logging as _rootlog
                _rootlog.error("qdrant.search.http_error")
                continue
            # record ANN latency per modality
            try:
                from modules.memory.application.metrics import add_ann_latency_ms  # type: ignore
                add_ann_latency_ms(mod, int((_t.perf_counter() - t0_mod) * 1000))
            except Exception:
                pass

            data = {}
            try:
                data = resp.json()
            except Exception:
                continue

            res_list = data.get("result", [])
            if debug:
                try:
                    print(f"[QdrantSearch] mod={mod} returned={len(res_list)}")
                except Exception:
                    pass
            for pt in res_list:
                pid = pt.get("id")
                score = float(pt.get("score", 0.0))
                try:
                    w = float(self._mod_weights.get(mod, 1.0))
                    score = score * w
                except Exception:
                    pass
                pl = pt.get("payload") or {}
                try:
                    entry = MemoryEntry.model_validate(pl)
                    merged.append({"id": pid, "score": score, "payload": entry})
                except Exception:
                    continue

        # Merge duplicates by id (take max score among modalities)
        best_by_id: Dict[str, Dict[str, Any]] = {}
        for item in merged:
            pid = str(item.get("id"))
            cur = best_by_id.get(pid)
            if cur is None or float(item.get("score", 0.0)) > float(cur.get("score", 0.0)):
                best_by_id[pid] = item
        deduped = list(best_by_id.values())
        deduped.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return deduped[: max(1, topk)]

    async def fetch_text_corpus(self, filters: Dict[str, Any], *, limit: int = 500) -> List[Dict[str, Any]]:
        """Best-effort scroll to fetch a text payload corpus for BM25 fallback.

        Returns list of {"id": str, "payload": MemoryEntry} up to `limit`.
        Safe on absence or backend errors (returns empty list).
        """
        try:
            coll = self.collections.get("text")
            if not coll:
                return []
            INTERNAL_EXCLUDED_SOURCES = ["tkg_dialog_utterance_index_v1", "tkg_dialog_event_index_v1"]
            eff_filters = dict(filters or {})
            try:
                src = eff_filters.get("source")
                src_list: list[str] = []
                if isinstance(src, str) and src.strip():
                    src_list = [src.strip()]
                elif isinstance(src, list):
                    src_list = [str(x).strip() for x in src if x is not None and str(x).strip()]
                if not set(src_list).intersection(set(INTERNAL_EXCLUDED_SOURCES)):
                    eff_filters["__exclude_sources"] = list(INTERNAL_EXCLUDED_SOURCES)
            except Exception:
                eff_filters["__exclude_sources"] = list(INTERNAL_EXCLUDED_SOURCES)

            built_filter = self._build_filter(eff_filters)
            shard_key = self._resolve_shard_key_from_filters(eff_filters)
            url = f"{self.base}/collections/{coll}/points/scroll"
            page_limit = 128
            next_page = None
            out: List[Dict[str, Any]] = []
            while len(out) < max(1, int(limit)):
                payload: Dict[str, Any] = {
                    "with_payload": True,
                    "limit": min(page_limit, max(1, int(limit) - len(out))),
                }
                if built_filter:
                    payload["filter"] = built_filter
                if shard_key is not None:
                    payload["shard_key"] = shard_key
                if next_page is not None:
                    payload["offset"] = next_page
                r = await self._request("POST", url, json=payload, timeout=10)
                if r.status_code >= 400:
                    try:
                        self._logger.warning(
                            "qdrant.scroll.http_error",
                            extra={
                                "event": "qdrant.scroll.http_error",
                                "entity": "vector",
                                "verb": "scroll",
                                "status": "warn",
                                "code": r.status_code,
                                "collection": coll,
                            },
                        )
                    except Exception:
                        pass
                    try:
                        import logging as _rootlog
                        _rootlog.error("qdrant.scroll.http_error")
                    except Exception:
                        pass
                    break
                data = r.json() or {}
                pts = data.get("result", {}).get("points", [])
                for pt in pts:
                    try:
                        pl = pt.get("payload") or {}
                        entry = MemoryEntry.model_validate(pl)
                        out.append({"id": pt.get("id"), "payload": entry})
                    except Exception:
                        continue
                next_page = (data.get("result", {}) or {}).get("next_page_offset")
                if not pts or next_page is None:
                    break
            return out
        except Exception:
            try:
                self._logger.error(
                    "qdrant.scroll.error",
                    extra={
                        "event": "qdrant.scroll.error",
                        "entity": "vector",
                        "verb": "scroll",
                        "status": "error",
                        "collection": self.collections.get("text"),
                    },
                )
                import logging as _rootlog
                _rootlog.error("qdrant.scroll.error")
            except Exception:
                pass
            return []

    async def tenant_exists(self, tenant_id: str) -> bool | None:
        """Return True if tenant has any points in Qdrant, False if none, None on error."""
        try:
            tid = str(tenant_id or "").strip()
            if not tid:
                return None
            built_filter = self._build_filter({"tenant_id": tid})
            if not built_filter:
                return None
            cols = self._collection_names()
            for col in cols:
                url = f"{self.base}/collections/{col}/points/count"
                payload = self._with_shard_key({"filter": built_filter, "exact": True}, tid)
                resp = await self._request("POST", url, json=payload, timeout=5)
                if resp.status_code >= 400:
                    return None
                data = resp.json() or {}
                count = int(((data.get("result") or {}).get("count") or 0))
                if count > 0:
                    return True
            return False
        except Exception:
            return None

    def _collection_names(self) -> List[str]:
        cols = list(self.collections.values()) if isinstance(self.collections, dict) else []
        if not cols:
            cols = ["memory_text", "memory_image", "memory_audio"]
        # Deduplicate while preserving order to avoid double counting when aliases point to the same collection.
        return list(dict.fromkeys(str(col) for col in cols if str(col or "").strip()))

    async def count_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        built_filter = self._build_filter({"tenant_id": tenant_id})
        if not built_filter:
            return 0
        if api_key_id is not None:
            must = list(built_filter.get("must") or [])
            must.append({"key": "metadata.api_key_id", "match": {"value": str(api_key_id)}})
            built_filter = dict(built_filter)
            built_filter["must"] = must
        total = 0
        shard_key = self._resolve_shard_key_from_filters({"tenant_id": tenant_id})
        for col in self._collection_names():
            url = f"{self.base}/collections/{col}/points/count"
            payload = self._with_shard_key({"filter": built_filter, "exact": True}, shard_key)
            resp = await self._request("POST", url, json=payload, timeout=10)
            if resp.status_code == 404:
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"qdrant_count_failed:{col}:{resp.status_code}:{resp.text[:200]}")
            data = resp.json() or {}
            total += int(((data.get("result") or {}).get("count") or 0))
        return total

    async def list_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        built_filter = self._build_filter({"tenant_id": tenant_id})
        if not built_filter:
            return []
        if api_key_id is not None:
            must = list(built_filter.get("must") or [])
            must.append({"key": "metadata.api_key_id", "match": {"value": str(api_key_id)}})
            built_filter = dict(built_filter)
            built_filter["must"] = must
        out: List[str] = []
        seen: set[str] = set()
        shard_key = self._resolve_shard_key_from_filters({"tenant_id": tenant_id})
        for col in self._collection_names():
            url = f"{self.base}/collections/{col}/points/scroll"
            next_page = None
            while True:
                payload: Dict[str, Any] = self._with_shard_key({
                    "filter": built_filter,
                    "with_payload": False,
                    "with_vector": False,
                    "limit": 256,
                }, shard_key)
                if next_page is not None:
                    payload["offset"] = next_page
                resp = await self._request("POST", url, json=payload, timeout=10)
                if resp.status_code == 404:
                    break
                if resp.status_code >= 400:
                    raise RuntimeError(f"qdrant_scroll_failed:{col}:{resp.status_code}:{resp.text[:200]}")
                data = resp.json() or {}
                points = list(((data.get("result") or {}).get("points") or []))
                for point in points:
                    pid = str(point.get("id") or "").strip()
                    if pid and pid not in seen:
                        seen.add(pid)
                        out.append(pid)
                next_page = (data.get("result") or {}).get("next_page_offset")
                if not points or next_page is None:
                    break
        return out

    async def list_entry_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        built_filter = self._build_filter({"tenant_id": tenant_id})
        if not built_filter:
            return []
        if api_key_id is not None:
            must = list(built_filter.get("must") or [])
            must.append({"key": "metadata.api_key_id", "match": {"value": str(api_key_id)}})
            built_filter = dict(built_filter)
            built_filter["must"] = must
        out: List[str] = []
        seen: set[str] = set()
        shard_key = self._resolve_shard_key_from_filters({"tenant_id": tenant_id})
        for col in self._collection_names():
            url = f"{self.base}/collections/{col}/points/scroll"
            next_page = None
            while True:
                payload: Dict[str, Any] = self._with_shard_key({
                    "filter": built_filter,
                    "with_payload": True,
                    "with_vector": False,
                    "limit": 256,
                }, shard_key)
                if next_page is not None:
                    payload["offset"] = next_page
                resp = await self._request("POST", url, json=payload, timeout=10)
                if resp.status_code == 404:
                    break
                if resp.status_code >= 400:
                    raise RuntimeError(f"qdrant_scroll_failed:{col}:{resp.status_code}:{resp.text[:200]}")
                data = resp.json() or {}
                points = list(((data.get("result") or {}).get("points") or []))
                for point in points:
                    payload_data = point.get("payload") or {}
                    entry_id = str(payload_data.get("id") or "").strip()
                    if not entry_id:
                        entry_id = str(point.get("id") or "").strip()
                    if entry_id and entry_id not in seen:
                        seen.add(entry_id)
                        out.append(entry_id)
                next_page = (data.get("result") or {}).get("next_page_offset")
                if not points or next_page is None:
                    break
        return out

    async def delete_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        built_filter = self._build_filter({"tenant_id": tenant_id})
        if not built_filter:
            return 0
        if api_key_id is not None:
            must = list(built_filter.get("must") or [])
            must.append({"key": "metadata.api_key_id", "match": {"value": str(api_key_id)}})
            built_filter = dict(built_filter)
            built_filter["must"] = must
        total = 0
        shard_key = self._resolve_shard_key_from_filters({"tenant_id": tenant_id})
        for col in self._collection_names():
            count_url = f"{self.base}/collections/{col}/points/count"
            count_payload = self._with_shard_key({"filter": built_filter, "exact": True}, shard_key)
            count_resp = await self._request("POST", count_url, json=count_payload, timeout=10)
            if count_resp.status_code == 404:
                continue
            if count_resp.status_code >= 400:
                raise RuntimeError(f"qdrant_count_failed:{col}:{count_resp.status_code}:{count_resp.text[:200]}")
            count_data = count_resp.json() or {}
            count = int(((count_data.get("result") or {}).get("count") or 0))
            if count <= 0:
                continue
            delete_url = f"{self.base}/collections/{col}/points/delete"
            delete_payload = self._with_shard_key({"filter": built_filter}, shard_key)
            delete_resp = await self._request("POST", delete_url, json=delete_payload, timeout=20)
            if delete_resp.status_code >= 400:
                raise RuntimeError(f"qdrant_delete_failed:{col}:{delete_resp.status_code}:{delete_resp.text[:200]}")
            total += count
        return total

    async def set_published(self, ids: List[str], published: bool) -> int:
        if not ids:
            return 0
        updated = 0
        payload = {"published": bool(published)}
        cols = list(self.collections.values()) if isinstance(self.collections, dict) else []
        if not cols:
            cols = ["memory_text", "memory_image", "memory_audio"]

        def _chunks(lst: List[str], size: int = 128) -> List[List[str]]:
            return [lst[i:i + size] for i in range(0, len(lst), size)]

        for col in cols:
            url = f"{self.base}/collections/{col}/points/payload"
            for chunk in _chunks([str(x) for x in ids if str(x).strip()], 128):
                if not chunk:
                    continue
                body = {"payload": payload, "points": chunk}
                try:
                    resp = await self._request("POST", url, json=body, timeout=10)
                    if int(getattr(resp, "status_code", 500)) < 400:
                        updated += len(chunk)
                    else:
                        self._logger.warning("qdrant.set_published failed: %s %s", resp.status_code, resp.text[:200])
                except Exception as e:
                    self._logger.warning("qdrant.set_published error: %s", str(e)[:200])
        return updated

    async def set_payload_by_filter(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> int:
        if not payload:
            return 0
        if not filters:
            return 0
        updated = 0
        cols = list(self.collections.values()) if isinstance(self.collections, dict) else []
        if not cols:
            cols = ["memory_text", "memory_image", "memory_audio"]
        shard_key = self._tenant_from_qdrant_filter(filters)
        for col in cols:
            url = f"{self.base}/collections/{col}/points/payload"
            body = self._with_shard_key({"payload": payload, "filter": filters}, shard_key)
            try:
                resp = await self._request("POST", url, json=body, timeout=10)
                if int(getattr(resp, "status_code", 500)) < 400:
                    updated += 1
                else:
                    self._logger.warning("qdrant.set_payload_by_filter failed: %s %s", resp.status_code, resp.text[:200])
            except Exception as e:
                self._logger.warning("qdrant.set_payload_by_filter error: %s", str(e)[:200])
        return updated

    async def set_payload_by_node(self, *, tenant_id: str, node_id: str, payload: Dict[str, Any]) -> int:
        tid = str(tenant_id or "").strip()
        nid = str(node_id or "").strip()
        if not tid or not nid:
            return 0
        filters = {
            "must": [
                {"key": "metadata.tenant_id", "match": {"value": tid}},
                {"key": "metadata.node_id", "match": {"value": nid}},
            ]
        }
        return await self.set_payload_by_filter(payload, filters)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        try:
            url = f"{self.base}/collections/{self.collections['text']}/points"
            r = await self._request("GET", url, params={"ids": [entry_id]}, timeout=5)
            if r.status_code >= 400:
                return None
            data = r.json()
            res = data.get("result", [])
            if not res:
                return None
            pl = res[0].get("payload")
            if not pl:
                return None
            return MemoryEntry.model_validate(pl)
        except Exception:
            return None

    async def delete_ids(self, ids: List[str]) -> None:
        url = f"{self.base}/collections/{self.collections['text']}/points/delete"
        await self._request("POST", url, json={"points": ids}, timeout=10)
        return None

    async def health(self) -> Dict[str, Any]:
        try:
            url = f"{self.base}/collections/{self.collections['text']}"
            r = await self._request("GET", url, timeout=5)
            return {"status": "ok" if r.status_code == 200 else "unknown", "endpoint": self.base}
        except Exception:
            return {"status": "unknown", "endpoint": self.base}

    def _payload_index_field_schemas(self) -> Dict[str, str]:
        # Keep this list in sync with _build_filter usage.
        return {
            "metadata.tenant_id": "keyword",
            "metadata.user_id": "keyword",
            "metadata.memory_domain": "keyword",
            "metadata.memory_scope": "keyword",
            "metadata.run_id": "keyword",
            "metadata.source": "keyword",
            "metadata.character_id": "keyword",
            "metadata.clip_id": "keyword",
            "metadata.agent_persona": "keyword",
            "metadata.session_id": "keyword",
            "metadata.invoked_by": "keyword",
            "metadata.memory_about": "keyword",
            "metadata.topic_path": "keyword",
            "metadata.tags": "keyword",
            "metadata.keywords": "keyword",
            "metadata.time_bucket": "keyword",
            "metadata.tags_vocab_version": "keyword",
            "metadata.api_key_id": "keyword",
            "metadata.node_id": "keyword",
            "metadata.timestamp": "float",
            "published": "bool",
            "modality": "keyword",
            "kind": "keyword",
        }

    def _get_collection_payload_schema_sync(self, coll: str) -> Dict[str, Any]:
        info = self._get_collection_info_sync(coll)
        result = info.get("result") if isinstance(info, dict) else None
        schema = (result or {}).get("payload_schema") if isinstance(result, dict) else None
        return schema if isinstance(schema, dict) else {}

    def _get_collection_info_sync(self, coll: str) -> Dict[str, Any]:
        url = f"{self.base}/collections/{coll}"
        timeout_s = max(10.0, float(self._http_timeout_s or 15.0))
        try:
            resp = self._request_sync("GET", url, timeout=timeout_s)
        except Exception as exc:
            self._logger.warning(
                "qdrant.ensure_payload_indexes inspect error: collection=%s error=%s",
                coll,
                str(exc)[:200],
            )
            return {}
        if int(getattr(resp, "status_code", 500)) >= 400:
            self._logger.warning(
                "qdrant.ensure_payload_indexes inspect failed: collection=%s status=%s",
                coll,
                getattr(resp, "status_code", "unknown"),
            )
            return {}
        try:
            data = resp.json()
        except Exception as exc:
            self._logger.warning(
                "qdrant.ensure_payload_indexes inspect decode error: collection=%s error=%s",
                coll,
                str(exc)[:200],
            )
            return {}
        return data if isinstance(data, dict) else {}

    def _payload_index_exists(self, payload_schema: Dict[str, Any], field_name: str, field_schema: str) -> bool:
        if not isinstance(payload_schema, dict):
            return False
        existing = payload_schema.get(field_name)
        if not isinstance(existing, dict):
            return False
        existing_type = str(existing.get("data_type") or existing.get("type") or "").strip().lower()
        if not existing_type:
            return False
        aliases = {
            "boolean": "bool",
        }
        normalized_existing = aliases.get(existing_type, existing_type)
        normalized_expected = aliases.get(str(field_schema).strip().lower(), str(field_schema).strip().lower())
        return normalized_existing == normalized_expected

    def _ensure_collections_sync(self) -> None:
        """Ensure required collections (text/image/audio/clip_image/face) exist with configured dims."""
        def _create(coll: str, size: int, distance: str) -> None:
            url = f"{self.base}/collections/{coll}"
            # create if missing
            spec = {
                "vectors": {"size": int(size), "distance": (distance or "Cosine").capitalize()},
            }
            if self._tenant_sharding_enabled and self._tenant_sharding_method == "custom":
                spec["sharding_method"] = "custom"
                spec["shard_number"] = max(1, int(self._tenant_sharding_shard_number or 1))
                spec["replication_factor"] = max(1, int(self._tenant_sharding_replication_factor or 1))
                spec["write_consistency_factor"] = max(
                    1, int(self._tenant_sharding_write_consistency_factor or 1)
                )
            r = self._request_sync("PUT", url, json=spec, timeout=10)
            # 200 ok or 409 already exists (qdrant returns 200 for idempotent create)
            if r.status_code >= 400 and r.status_code != 409:
                raise RuntimeError(f"qdrant create collection {coll} failed: {r.status_code}")
            if self._tenant_sharding_enabled and self._tenant_sharding_method == "custom":
                info = self._get_collection_info_sync(coll)
                params = ((info.get("result") or {}).get("config") or {}).get("params") or {}
                current_method = str(params.get("sharding_method") or "auto").strip().lower()
                if current_method != "custom":
                    raise RuntimeError(
                        f"qdrant_collection_sharding_mismatch: collection={coll} expected=custom actual={current_method}"
                    )

        def _ensure_payload_indexes(coll: str) -> None:
            field_schemas = self._payload_index_field_schemas()
            index_url = f"{self.base}/collections/{coll}/index"
            payload_schema = self._get_collection_payload_schema_sync(coll)
            index_timeout_s = max(30.0, float(self._http_timeout_s or 15.0))
            skipped = 0
            missing_fields: List[tuple[str, str]] = []
            for field_name, field_schema in field_schemas.items():
                if self._payload_index_exists(payload_schema, field_name, field_schema):
                    skipped += 1
                else:
                    missing_fields.append((field_name, field_schema))

            def _create_index(field_name: str, field_schema: str) -> tuple[str, str, requests.Response]:
                body = {"field_name": field_name, "field_schema": field_schema}
                if self._session_has_runtime_overrides():
                    response = self._request_sync("PUT", index_url, json=body, timeout=index_timeout_s)
                    return field_name, field_schema, response
                with requests.Session() as session:
                    try:
                        session.trust_env = False
                    except Exception:
                        pass
                    if self.api_key:
                        session.headers.update({"api-key": self.api_key})
                    response = session.put(index_url, json=body, timeout=index_timeout_s)
                    return field_name, field_schema, response

            created = 0
            if self._session_has_runtime_overrides() or len(missing_fields) <= 1:
                results = [_create_index(field_name, field_schema) for field_name, field_schema in missing_fields]
            else:
                try:
                    max_workers = int(os.getenv("MEMORY_QDRANT_PAYLOAD_INDEX_CONCURRENCY", "4") or 4)
                except Exception:
                    max_workers = 4
                max_workers = max(1, min(max_workers, len(missing_fields)))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_create_index, field_name, field_schema)
                        for field_name, field_schema in missing_fields
                    ]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]

            for field_name, field_schema, r in results:
                if r.status_code >= 400 and r.status_code != 409:
                    raise RuntimeError(
                        f"qdrant create payload index failed: collection={coll} field={field_name} status={r.status_code}"
                    )
                created += 1
                payload_schema[field_name] = {"data_type": field_schema}
            self._logger.info(
                "qdrant.ensure_payload_indexes: collection=%s created=%d skipped=%d",
                coll,
                created,
                skipped,
            )

        emb = self.settings.get("embedding", {}) or {}
        txt_dim = int(emb.get("dim") or 768)
        dist = (emb.get("distance") or "cosine").lower()
        if self.collections.get("text"):
            _create(self.collections["text"], txt_dim, dist)
        # image/audio optional dims
        img_dim = int(((emb.get("image") or {}).get("dim") or 512))
        if self.collections.get("image"):
            _create(self.collections["image"], img_dim, dist)
        # clip_image (OpenCLIP) optional
        clip_dim = int(((emb.get("clip_image") or {}).get("dim") or img_dim))
        if self.collections.get("clip_image"):
            _create(self.collections["clip_image"], clip_dim, dist)
        aud_dim = int(((emb.get("audio") or {}).get("dim") or 256))
        if self.collections.get("audio"):
            _create(self.collections["audio"], aud_dim, dist)
        # face (identity space) optional
        face_dim = int(((emb.get("face") or {}).get("dim") or 512))
        if self.collections.get("face"):
            _create(self.collections["face"], face_dim, dist)

        seen: set[str] = set()
        for collection_name in self.collections.values():
            coll = str(collection_name or "").strip()
            if not coll or coll in seen:
                continue
            seen.add(coll)
            _ensure_payload_indexes(coll)

    async def ensure_collections(self) -> None:
        """Ensure required collections (text/image/audio/clip_image/face) exist with configured dims."""
        await asyncio.to_thread(self._ensure_collections_sync)

    # ---- Config validation ----
    def _validate_embedding_config(self, emb: Dict[str, Any]) -> None:
        """Ensure required dims exist for enabled collections to avoid silent mismatches.

        Rules:
        - text.dim must exist (defaults to 768 if missing)
        - if 'clip_image' collection is configured, embedding.clip_image.dim must be provided explicitly
        - if 'face' collection is configured, embedding.face.dim must be provided explicitly
        """
        cols = self.settings.get("collections", {}) or {}
        # clip_image
        if cols.get("clip_image") and not ((emb.get("clip_image") or {}).get("dim")):
            raise RuntimeError("embedding_config_missing: clip_image.dim is required when collections.clip_image is set")
        # face
        if cols.get("face") and not ((emb.get("face") or {}).get("dim")):
            raise RuntimeError("embedding_config_missing: face.dim is required when collections.face is set")
