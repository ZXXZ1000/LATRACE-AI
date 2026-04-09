from __future__ import annotations

from typing import Any, Dict, List
import asyncio
import json
import logging
import threading

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.metrics import inc, observe_payload_items, observe_vector_size
from modules.memory.application.embedding_adapter import (
    build_embedding_from_settings,
    build_image_embedding_from_settings,
    build_audio_embedding_from_settings,
)


class MilvusStore:
    """Minimal Milvus facade using pymilvus.

    Settings:
    {
      "host": "127.0.0.1", "port": 19530,
      "collections": {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"},
      "embedding": {...}
    }
    """

    _PAYLOAD_FIELD = "payload"
    _VECTOR_FIELD = "vector"
    _ID_FIELD = "id"
    _USER_BUCKET_FIELD = "user_id_bucket"
    _ABOUT_BUCKET_FIELD = "memory_about_bucket"

    def __init__(self, settings: Dict[str, Any] | None = None) -> None:
        self.settings = settings or {}
        host = self.settings.get("host", "127.0.0.1")
        port = int(self.settings.get("port", 19530))
        self.host = str(host)
        self.port = int(port)
        self.collections = self.settings.get(
            "collections",
            {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"},
        )
        self._alias = f"memory_milvus_{id(self)}"
        self._conn_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._closed = False

        emb_cfg = self.settings.get("embedding", {}) or {}
        self._validate_embedding_config(emb_cfg)
        self.embed_text = build_embedding_from_settings(emb_cfg if isinstance(emb_cfg, dict) else {})
        img_cfg = emb_cfg.get("image") if isinstance(emb_cfg, dict) else None
        aud_cfg = emb_cfg.get("audio") if isinstance(emb_cfg, dict) else None
        clip_img_cfg = emb_cfg.get("clip_image") if isinstance(emb_cfg, dict) else None
        self.embed_image = build_image_embedding_from_settings(img_cfg)
        self.embed_clip_image = build_image_embedding_from_settings(clip_img_cfg)
        self.embed_audio = build_audio_embedding_from_settings(aud_cfg)

        rel = (self.settings.get("reliability") or {}) if isinstance(self.settings.get("reliability"), dict) else {}
        self._retry_attempts = int(rel.get("retries", {}).get("max_attempts", 2))
        self._backoff_base_ms = int(rel.get("retries", {}).get("backoff_base_ms", 100))
        self._backoff_max_ms = int(rel.get("retries", {}).get("backoff_max_ms", 1000))

        try:
            self._mod_weights = dict(((self.settings.get("search") or {}).get("modality_weights") or {}))
        except Exception:
            self._mod_weights = {}

        self._payload_json = False
        self._data_type = None
        self._connect()

    def _connect(self) -> None:
        with self._conn_lock:
            if self._closed:
                return
            from pymilvus import connections, DataType

            connections.connect(alias=self._alias, host=self.host, port=self.port)
            self._data_type = DataType
            self._payload_json = hasattr(DataType, "JSON")

    def close(self) -> None:
        if self._closed:
            return
        try:
            from pymilvus import connections

            connections.disconnect(alias=self._alias)
        except Exception:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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

    def _metric_type(self) -> str:
        emb = self.settings.get("embedding", {}) or {}
        dist = str(emb.get("distance") or "cosine").lower()
        if dist in ("dot", "ip", "inner"):
            return "IP"
        if dist in ("l2", "euclid", "euclidean"):
            return "L2"
        return "COSINE"

    def _pack_list(self, values: Any) -> str:
        if isinstance(values, str):
            items = [values.strip()]
        elif isinstance(values, list):
            items = [str(v).strip() for v in values if v is not None and str(v).strip()]
        else:
            items = []
        items = [v for v in items if v]
        if not items:
            return ""
        return "|" + "|".join(items) + "|"

    def _scalar_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = payload.get("metadata") or {}
        out: Dict[str, Any] = {
            "tenant_id": str(meta.get("tenant_id") or ""),
            "user_id": "",
            "memory_domain": str(meta.get("memory_domain") or ""),
            "run_id": str(meta.get("run_id") or ""),
            "memory_scope": str(meta.get("memory_scope") or ""),
            "modality": str(payload.get("modality") or ""),
            "kind": str(payload.get("kind") or ""),
            "source": str(meta.get("source") or ""),
            "clip_id": str(meta.get("clip_id") or ""),
            "character_id": str(meta.get("character_id") or ""),
            "agent_persona": str(meta.get("agent_persona") or ""),
            "session_id": str(meta.get("session_id") or ""),
            "invoked_by": str(meta.get("invoked_by") or ""),
            self._USER_BUCKET_FIELD: self._pack_list(meta.get("user_id")),
            self._ABOUT_BUCKET_FIELD: self._pack_list(meta.get("memory_about")),
            "published": bool(payload.get("published", True)),
        }
        ts = meta.get("timestamp")
        if isinstance(ts, (int, float)):
            out["timestamp"] = float(ts)
        else:
            out["timestamp"] = 0.0
        return out

    def _build_schema(self, dim: int):
        dt = self._data_type
        if dt is None:
            raise RuntimeError("milvus_not_ready")
        from pymilvus import FieldSchema, CollectionSchema

        payload_field = (
            FieldSchema(name=self._PAYLOAD_FIELD, dtype=dt.JSON, is_nullable=True)
            if self._payload_json
            else FieldSchema(name=self._PAYLOAD_FIELD, dtype=dt.VARCHAR, max_length=65535, is_nullable=True)
        )
        fields = [
            FieldSchema(
                name=self._ID_FIELD,
                dtype=dt.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=256,
            ),
            FieldSchema(
                name=self._VECTOR_FIELD,
                dtype=dt.FLOAT_VECTOR,
                dim=int(dim),
            ),
            payload_field,
            FieldSchema(name="tenant_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="user_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name=self._USER_BUCKET_FIELD, dtype=dt.VARCHAR, max_length=2048, is_nullable=True),
            FieldSchema(name="memory_domain", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="run_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="memory_scope", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="modality", dtype=dt.VARCHAR, max_length=64, is_nullable=True),
            FieldSchema(name="kind", dtype=dt.VARCHAR, max_length=64, is_nullable=True),
            FieldSchema(name="source", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="clip_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="character_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="agent_persona", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="session_id", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name="invoked_by", dtype=dt.VARCHAR, max_length=256, is_nullable=True),
            FieldSchema(name=self._ABOUT_BUCKET_FIELD, dtype=dt.VARCHAR, max_length=2048, is_nullable=True),
            FieldSchema(name="timestamp", dtype=dt.DOUBLE, is_nullable=True),
            FieldSchema(name="published", dtype=dt.BOOL, is_nullable=True),
        ]
        return CollectionSchema(fields=fields, description="memory vectors")
    def _ensure_collection_sync(self, name: str, dim: int) -> None:
        from pymilvus import Collection, utility

        if not utility.has_collection(name, using=self._alias):
            schema = self._build_schema(dim)
            Collection(name, schema=schema, using=self._alias)
        collection = Collection(name, using=self._alias)
        metric = self._metric_type()
        index_params = {"metric_type": metric, "index_type": "FLAT", "params": {}}
        try:
            collection.create_index(self._VECTOR_FIELD, index_params=index_params)
        except Exception:
            pass

    async def ensure_collections(self) -> None:
        emb = self.settings.get("embedding", {}) or {}
        txt_dim = int(emb.get("dim") or 768)
        img_dim = int(((emb.get("image") or {}).get("dim") or 512))
        clip_dim = int(((emb.get("clip_image") or {}).get("dim") or img_dim))
        aud_dim = int(((emb.get("audio") or {}).get("dim") or 256))
        face_dim = int(((emb.get("face") or {}).get("dim") or 512))
        tasks = []
        if self.collections.get("text"):
            tasks.append((self.collections["text"], txt_dim))
        if self.collections.get("image"):
            tasks.append((self.collections["image"], img_dim))
        if self.collections.get("clip_image"):
            tasks.append((self.collections["clip_image"], clip_dim))
        if self.collections.get("audio"):
            tasks.append((self.collections["audio"], aud_dim))
        if self.collections.get("face"):
            tasks.append((self.collections["face"], face_dim))
        for name, dim in tasks:
            await asyncio.to_thread(self._ensure_collection_sync, name, dim)

    def _build_expr(self, filters: Dict[str, Any]) -> str:
        terms: List[str] = []

        def _q(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (int, float)):
                return str(v)
            return json.dumps(str(v))

        def _expr_in(field: str, values: List[Any]) -> str:
            items = ", ".join(_q(v) for v in values if v is not None and str(v) != "")
            if not items:
                return ""
            return f"{field} in [{items}]"

        def _expr_like(field: str, pat: str) -> str:
            return f"{field} like {_q(pat)}"

        mods = filters.get("modality")
        if isinstance(mods, list) and mods:
            expr = _expr_in("modality", mods)
            if expr:
                terms.append(expr)

        mtypes = filters.get("memory_type")
        if isinstance(mtypes, list) and mtypes:
            expr = _expr_in("kind", mtypes)
            if expr:
                terms.append(expr)

        srcs = filters.get("source")
        if isinstance(srcs, list) and srcs:
            expr = _expr_in("source", srcs)
            if expr:
                terms.append(expr)
        elif isinstance(srcs, str) and srcs.strip():
            terms.append(f"source == {_q(srcs)}")

        ex_srcs = filters.get("__exclude_sources")
        if isinstance(ex_srcs, list) and ex_srcs:
            expr = _expr_in("source", ex_srcs)
            if expr:
                terms.append(f"not ({expr})")

        tenant_id = filters.get("tenant_id")
        if tenant_id is not None:
            terms.append(f"tenant_id == {_q(tenant_id)}")

        user_ids = filters.get("user_id")
        if isinstance(user_ids, list) and user_ids:
            mode = str(filters.get("user_match") or "any").lower()
            exprs = [_expr_like(self._USER_BUCKET_FIELD, f"%|{str(uid)}|%") for uid in user_ids if uid is not None]
            exprs = [e for e in exprs if e]
            if exprs:
                joiner = " and " if mode == "all" else " or "
                terms.append("(" + joiner.join(exprs) + ")")

        dom = filters.get("memory_domain")
        if dom is not None:
            terms.append(f"memory_domain == {_q(dom)}")

        rid = filters.get("run_id")
        if rid is not None:
            terms.append(f"run_id == {_q(rid)}")

        mscope = filters.get("memory_scope")
        if mscope is not None:
            terms.append(f"memory_scope == {_q(mscope)}")

        pub = filters.get("published")
        if pub is True:
            terms.append("published == true")
        elif pub is False:
            terms.append("published == false")

        chars = filters.get("character_id")
        if isinstance(chars, list) and chars:
            expr = _expr_in("character_id", chars)
            if expr:
                terms.append(expr)
        elif isinstance(chars, str):
            terms.append(f"character_id == {_q(chars)}")

        clip_id = filters.get("clip_id")
        if clip_id is not None:
            terms.append(f"clip_id == {_q(clip_id)}")

        tr = filters.get("time_range")
        if isinstance(tr, dict):
            if isinstance(tr.get("gte"), (int, float)):
                terms.append(f"timestamp >= {float(tr.get('gte'))}")
            if isinstance(tr.get("lte"), (int, float)):
                terms.append(f"timestamp <= {float(tr.get('lte'))}")

        persona = filters.get("agent_persona")
        if persona and isinstance(persona, str):
            terms.append(f"agent_persona == {_q(persona)}")

        session = filters.get("session_id")
        if session and isinstance(session, str):
            terms.append(f"session_id == {_q(session)}")

        invoker = filters.get("invoked_by")
        if invoker and isinstance(invoker, str):
            terms.append(f"invoked_by == {_q(invoker)}")

        about = filters.get("memory_about")
        if about and isinstance(about, list):
            exprs = [_expr_like(self._ABOUT_BUCKET_FIELD, f"%|{str(a)}|%") for a in about if a is not None]
            exprs = [e for e in exprs if e]
            if exprs:
                terms.append("(" + " or ".join(exprs) + ")")

        return " and ".join(terms)

    def _collection_names(self) -> List[str]:
        cols = list(self.collections.values()) if isinstance(self.collections, dict) else []
        if not cols:
            cols = ["memory_text", "memory_image", "memory_audio"]
        return list(dict.fromkeys(str(col) for col in cols if str(col or "").strip()))

    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:
        buckets: Dict[str, List[Dict[str, Any]]] = {"text": [], "image": [], "audio": [], "clip_image": [], "face": []}
        emb_cfg = self.settings.get("embedding", {}) or {}
        exp_dims = {
            "text": int(emb_cfg.get("dim") or 768),
            "image": int(((emb_cfg.get("image") or {}).get("dim") or 512)),
            "clip_image": int(((emb_cfg.get("clip_image") or {}).get("dim") or ((emb_cfg.get("image") or {}).get("dim") or 512))),
            "audio": int(((emb_cfg.get("audio") or {}).get("dim") or 192)),
            "face": int(((emb_cfg.get("face") or {}).get("dim") or 512)),
        }

        def _assert_dim(mod: str, vec: List[float], entry: MemoryEntry) -> None:
            need = int(exp_dims.get(mod) or 0)
            got = int(len(vec or []))
            if need and got != need:
                raise RuntimeError(
                    f"vector_dim_mismatch: modality={mod} expected={need} got={got} id={entry.id} clip_id={entry.metadata.get('clip_id')} source={entry.metadata.get('source')}"
                )

        text_need_idx: list[int] = []
        text_inputs: list[str] = []
        for idx, e in enumerate(entries):
            if e.modality == "text":
                vec = None
                if e.vectors and isinstance(e.vectors.get("text"), list):
                    vec = e.vectors.get("text")
                if vec is None and (e.contents and isinstance(e.contents[0], str)):
                    text_need_idx.append(idx)
                    text_inputs.append(e.contents[0])
        text_vecs: list[list[float]] = []
        if text_need_idx:
            batch_fn = getattr(self.embed_text, "encode_batch", None)
            if callable(batch_fn):
                try:
                    bsz = None
                    try:
                        bsz = int(emb_cfg.get("batch_size", 0) or 0) or None
                    except Exception:
                        bsz = None
                    text_vecs = list(await asyncio.to_thread(batch_fn, text_inputs, bsz=bsz))
                except Exception:
                    text_vecs = []
        text_idx_to_vec: dict[int, List[float]] = {}
        if text_need_idx and text_vecs and len(text_vecs) == len(text_need_idx):
            for i, idx in enumerate(text_need_idx):
                text_idx_to_vec[idx] = list(text_vecs[i])

        for i, e in enumerate(entries):
            if not e.contents:
                continue
            try:
                observe_payload_items(e.modality, len(e.contents or []))
            except Exception:
                pass
            payload = e.model_dump()
            try:
                if isinstance(payload, dict):
                    contents = payload.get("contents") or []
                    if contents and not payload.get("content"):
                        payload["content"] = str(contents[0])
            except Exception:
                pass
            pid = str(e.id or "")
            if not pid:
                import hashlib as _hashlib
                pid = _hashlib.md5(json.dumps(payload, ensure_ascii=True).encode("utf-8")).hexdigest()
            base_fields = self._scalar_fields(payload)
            base_fields[self._ID_FIELD] = pid
            base_fields[self._PAYLOAD_FIELD] = payload if self._payload_json else json.dumps(payload, ensure_ascii=True)

            if e.modality == "text":
                vec = None
                if e.vectors and isinstance(e.vectors.get("text"), list):
                    vec = e.vectors.get("text")
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
                row = dict(base_fields)
                row[self._VECTOR_FIELD] = list(vec or [])
                buckets["text"].append(row)
            elif e.modality == "image":
                if e.vectors and isinstance(e.vectors.get("face"), list):
                    vec_face = e.vectors.get("face")
                    try:
                        observe_vector_size("face", len(vec_face or []))
                    except Exception:
                        pass
                    _assert_dim("face", vec_face, e)
                    row = dict(base_fields)
                    row[self._VECTOR_FIELD] = list(vec_face or [])
                    buckets["face"].append(row)
                vec_ci = None
                if e.vectors and isinstance(e.vectors.get("clip_image"), list):
                    vec_ci = e.vectors.get("clip_image")
                if vec_ci is None:
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
                    row = dict(base_fields)
                    row[self._VECTOR_FIELD] = list(vec_ci or [])
                    buckets["clip_image"].append(row)
                vec_img = None
                if e.vectors and isinstance(e.vectors.get("image"), list):
                    vec_img = e.vectors.get("image")
                if vec_img is None:
                    try:
                        vec_img = await asyncio.to_thread(self.embed_image, e.contents[0])
                    except Exception:
                        vec_img = None
                if vec_img is not None:
                    try:
                        observe_vector_size("image", len(vec_img or []))
                    except Exception:
                        pass
                    _assert_dim("image", vec_img, e)
                    row = dict(base_fields)
                    row[self._VECTOR_FIELD] = list(vec_img or [])
                    buckets["image"].append(row)
            elif e.modality == "audio":
                vec = None
                if e.vectors and isinstance(e.vectors.get("audio"), list):
                    vec = e.vectors.get("audio")
                if vec is None:
                    vec = await asyncio.to_thread(self.embed_audio, e.contents[0])
                try:
                    observe_vector_size("audio", len(vec or []))
                except Exception:
                    pass
                _assert_dim("audio", vec, e)
                row = dict(base_fields)
                row[self._VECTOR_FIELD] = list(vec or [])
                buckets["audio"].append(row)

        for mod, rows in buckets.items():
            if not rows:
                continue
            coll = self.collections.get(mod)
            if not coll:
                continue
            await asyncio.to_thread(self._upsert_rows_sync, coll, rows)
    def _upsert_rows_sync(self, collection_name: str, rows: List[Dict[str, Any]]) -> None:
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        ids = [row[self._ID_FIELD] for row in rows if row.get(self._ID_FIELD)]
        if not ids:
            return
        expr_ids = ", ".join(json.dumps(str(x)) for x in ids)
        try:
            collection.delete(expr=f"{self._ID_FIELD} in [{expr_ids}]")
        except Exception:
            pass
        collection.insert(rows)

    async def search_vectors(
        self,
        query: str,
        filters: Dict[str, Any],
        topk: int,
        threshold: float | None = None,
        query_vector: List[float] | None = None,
    ) -> List[Dict[str, Any]]:
        mods_req = []
        try:
            mods_req = list(filters.get("modality") or [])
        except Exception:
            mods_req = []
        if not mods_req:
            mods_req = ["text"]

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

        expr = self._build_expr(eff_filters)

        merged: List[Dict[str, Any]] = []
        last_exc: Exception | None = None
        metric = self._metric_type()

        for mod in mods_req:
            coll = self.collections.get(mod)
            if not coll:
                continue
            try:
                inc(f"ann_calls_total_{mod}", 1)
            except Exception:
                pass
            vec = await _vec_for(mod)
            if not vec:
                continue
            params = {"metric_type": metric, "params": {}}
            attempts = max(1, self._retry_attempts)
            backoff = max(1, self._backoff_base_ms) / 1000.0
            last_exc = None
            hits = None
            for i in range(attempts):
                try:
                    hits = await asyncio.to_thread(
                        self._search_sync,
                        coll,
                        vec,
                        expr,
                        max(1, topk),
                        params,
                    )
                    break
                except Exception as e:
                    last_exc = e
                    try:
                        inc("errors_total", 1)
                        inc("backend_retries_total", 1)
                    except Exception:
                        pass
                    if i < attempts - 1:
                        await asyncio.sleep(min(backoff, self._backoff_max_ms / 1000.0))
                        backoff = min(backoff * 2, self._backoff_max_ms / 1000.0)
            if hits is None:
                if last_exc is not None:
                    try:
                        self._logger.error("milvus.search.error: %s", repr(last_exc))
                    except Exception:
                        pass
                continue
            for hit in hits:
                try:
                    score = float(hit.get("score", 0.0))
                except Exception:
                    score = 0.0
                if threshold is not None and score < float(threshold):
                    continue
                try:
                    w = float(self._mod_weights.get(mod, 1.0))
                    score = score * w
                except Exception:
                    pass
                try:
                    entry = MemoryEntry.model_validate(hit.get("payload") or {})
                except Exception:
                    continue
                merged.append({"id": hit.get("id"), "score": score, "payload": entry})

        best_by_id: Dict[str, Dict[str, Any]] = {}
        for item in merged:
            pid = str(item.get("id"))
            cur = best_by_id.get(pid)
            if cur is None or float(item.get("score", 0.0)) > float(cur.get("score", 0.0)):
                best_by_id[pid] = item
        deduped = list(best_by_id.values())
        deduped.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return deduped[: max(1, topk)]

    def _search_sync(self, collection_name: str, vec: List[float], expr: str, topk: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()
        results = collection.search(
            data=[vec],
            anns_field=self._VECTOR_FIELD,
            param=params,
            limit=topk,
            expr=(expr or None),
            output_fields=[self._PAYLOAD_FIELD],
        )
        hits = []
        for hit in results[0]:
            payload = None
            try:
                payload = hit.entity.get(self._PAYLOAD_FIELD)
                if not self._payload_json and isinstance(payload, str):
                    payload = json.loads(payload)
            except Exception:
                payload = None
            hits.append({"id": hit.id, "score": float(hit.score), "payload": payload})
        return hits
    async def fetch_text_corpus(self, filters: Dict[str, Any], *, limit: int = 500) -> List[Dict[str, Any]]:
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
            expr = self._build_expr(eff_filters)
            return await asyncio.to_thread(self._fetch_corpus_sync, coll, expr, limit)
        except Exception:
            try:
                self._logger.error("milvus.scroll.error")
            except Exception:
                pass
            return []

    def _fetch_corpus_sync(self, collection_name: str, expr: str, limit: int) -> List[Dict[str, Any]]:
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()
        out: List[Dict[str, Any]] = []
        offset = 0
        page = 128
        while len(out) < max(1, int(limit)):
            rows = collection.query(
                expr=(expr or None),
                offset=offset,
                limit=min(page, max(1, int(limit) - len(out))),
                output_fields=[self._ID_FIELD, self._PAYLOAD_FIELD],
            )
            if not rows:
                break
            for row in rows:
                pl = row.get(self._PAYLOAD_FIELD)
                if pl is None:
                    continue
                if not self._payload_json and isinstance(pl, str):
                    try:
                        pl = json.loads(pl)
                    except Exception:
                        continue
                try:
                    entry = MemoryEntry.model_validate(pl)
                    out.append({"id": row.get(self._ID_FIELD), "payload": entry})
                except Exception:
                    continue
            offset += len(rows)
            if len(rows) < page:
                break
        return out

    async def set_published(self, ids: List[str], published: bool) -> int:
        if not ids:
            return 0
        updated = 0
        for entry_id in ids:
            entry = await self.get(entry_id)
            if entry is None:
                continue
            entry.published = bool(published)
            await self.upsert_vectors([entry])
            updated += 1
        return updated

    async def get(self, entry_id: str) -> MemoryEntry | None:
        try:
            collection_names: List[str] = []
            seen: set[str] = set()
            for name in self.collections.values():
                cname = str(name or "").strip()
                if not cname or cname in seen:
                    continue
                seen.add(cname)
                collection_names.append(cname)
            for coll in collection_names:
                row = await asyncio.to_thread(self._get_sync, coll, entry_id)
                if not row:
                    continue
                pl = row.get(self._PAYLOAD_FIELD)
                if pl is None:
                    continue
                if not self._payload_json and isinstance(pl, str):
                    pl = json.loads(pl)
                return MemoryEntry.model_validate(pl)
            return None
        except Exception:
            return None

    def _get_sync(self, collection_name: str, entry_id: str) -> Dict[str, Any] | None:
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()
        expr = f"{self._ID_FIELD} == {json.dumps(str(entry_id))}"
        rows = collection.query(expr=expr, limit=1, output_fields=[self._PAYLOAD_FIELD])
        if not rows:
            return None
        return rows[0]

    async def delete_ids(self, ids: List[str]) -> None:
        coll = self.collections.get("text")
        if not coll:
            return None
        await asyncio.to_thread(self._delete_ids_sync, coll, ids)
        return None

    def _delete_ids_sync(self, collection_name: str, ids: List[str]) -> None:
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        expr_ids = ", ".join(json.dumps(str(x)) for x in ids if str(x).strip())
        if not expr_ids:
            return
        collection.delete(expr=f"{self._ID_FIELD} in [{expr_ids}]")

    async def count_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        if api_key_id is not None:
            raise NotImplementedError("milvus_api_key_clear_not_supported")
        expr = self._build_expr({"tenant_id": tenant_id})
        total = 0
        for coll in self._collection_names():
            total += await asyncio.to_thread(self._count_by_expr_sync, coll, expr)
        return total

    async def list_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        if api_key_id is not None:
            raise NotImplementedError("milvus_api_key_clear_not_supported")
        expr = self._build_expr({"tenant_id": tenant_id})
        out: List[str] = []
        seen: set[str] = set()
        for coll in self._collection_names():
            ids = await asyncio.to_thread(self._list_ids_by_expr_sync, coll, expr)
            for item in ids:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
        return out

    async def list_entry_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        return await self.list_ids_by_filter(tenant_id=tenant_id, api_key_id=api_key_id)

    def _count_by_expr_sync(self, collection_name: str, expr: str) -> int:
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using=self._alias):
            return 0
        collection = Collection(collection_name, using=self._alias)
        collection.load()
        total = 0
        offset = 0
        page = 512
        while True:
            rows = collection.query(
                expr=(expr or None),
                offset=offset,
                limit=page,
                output_fields=[self._ID_FIELD],
            )
            batch = len(rows or [])
            total += batch
            if batch < page:
                break
            offset += batch
        return total

    def _list_ids_by_expr_sync(self, collection_name: str, expr: str) -> List[str]:
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using=self._alias):
            return []
        collection = Collection(collection_name, using=self._alias)
        collection.load()
        out: List[str] = []
        offset = 0
        page = 512
        while True:
            rows = collection.query(
                expr=(expr or None),
                offset=offset,
                limit=page,
                output_fields=[self._ID_FIELD],
            )
            batch = [str(row.get(self._ID_FIELD) or "").strip() for row in (rows or []) if str(row.get(self._ID_FIELD) or "").strip()]
            if not batch:
                break
            out.extend(batch)
            if len(batch) < page:
                break
            offset += len(batch)
        return out

    async def delete_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        if api_key_id is not None:
            raise NotImplementedError("milvus_api_key_clear_not_supported")
        expr = self._build_expr({"tenant_id": tenant_id})
        total = 0
        for coll in self._collection_names():
            total += await asyncio.to_thread(self._delete_by_expr_sync, coll, expr)
        return total

    def _delete_by_expr_sync(self, collection_name: str, expr: str) -> int:
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using=self._alias):
            return 0
        collection = Collection(collection_name, using=self._alias)
        collection.load()
        ids: List[str] = []
        offset = 0
        page = 512
        while True:
            rows = collection.query(
                expr=(expr or None),
                offset=offset,
                limit=page,
                output_fields=[self._ID_FIELD],
            )
            batch = [str(row.get(self._ID_FIELD) or "").strip() for row in (rows or []) if str(row.get(self._ID_FIELD) or "").strip()]
            if not batch:
                break
            ids.extend(batch)
            if len(batch) < page:
                break
            offset += len(batch)
        if not ids:
            return 0
        chunk = 256
        for start in range(0, len(ids), chunk):
            expr_ids = ", ".join(json.dumps(item) for item in ids[start : start + chunk])
            if expr_ids:
                collection.delete(expr=f"{self._ID_FIELD} in [{expr_ids}]")
        return len(ids)

    async def health(self) -> Dict[str, Any]:
        try:
            from pymilvus import utility

            coll = self.collections.get("text")
            ok = bool(coll and utility.has_collection(coll, using=self._alias))
            return {"status": "ok" if ok else "unknown", "endpoint": f"{self.host}:{self.port}"}
        except Exception:
            return {"status": "unknown", "endpoint": f"{self.host}:{self.port}"}

    def _validate_embedding_config(self, emb: Dict[str, Any]) -> None:
        cols = self.settings.get("collections", {}) or {}
        if cols.get("clip_image") and not ((emb.get("clip_image") or {}).get("dim")):
            raise RuntimeError("embedding_config_missing: clip_image.dim is required when collections.clip_image is set")
        if cols.get("face") and not ((emb.get("face") or {}).get("dim")):
            raise RuntimeError("embedding_config_missing: face.dim is required when collections.face is set")
