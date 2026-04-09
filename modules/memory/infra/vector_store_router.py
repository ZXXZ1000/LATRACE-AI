from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import contextvars
import logging
import threading

from modules.memory.application.metrics import inc
from modules.memory.contracts.memory_models import MemoryEntry

_REQUEST_CTX: contextvars.ContextVar[Dict[str, Any] | None] = contextvars.ContextVar(
    "memory_request_ctx",
    default=None,
)


def set_request_context(ctx: Dict[str, Any]) -> contextvars.Token:
    return _REQUEST_CTX.set(ctx)


def clear_request_context(token: contextvars.Token) -> None:
    _REQUEST_CTX.reset(token)


def update_request_context(**kwargs: Any) -> None:
    ctx = _REQUEST_CTX.get()
    if not isinstance(ctx, dict):
        return
    ctx.update(kwargs)


def _get_request_context() -> Optional[Dict[str, Any]]:
    ctx = _REQUEST_CTX.get()
    return ctx if isinstance(ctx, dict) else None


class VectorStoreRouter:
    """Route vector store calls based on tenant existence in Qdrant."""

    def __init__(self, qdrant_store: Any, milvus_store: Any | None) -> None:
        self._qdrant = qdrant_store
        self._milvus = milvus_store
        self._route_table: Dict[str, str] = {}
        self._route_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def _resolve_tenant(self, tenant_id: Optional[str]) -> Optional[str]:
        if tenant_id is None:
            return None
        tid = str(tenant_id).strip()
        return tid or None

    def _tenant_from_entries(self, entries: List[MemoryEntry]) -> Optional[str]:
        for entry in entries or []:
            try:
                meta = entry.metadata or {}
                tid = self._resolve_tenant(meta.get("tenant_id"))
                if tid:
                    return tid
            except Exception:
                continue
        return None

    def _tenant_from_filters(self, filters: Dict[str, Any]) -> Optional[str]:
        try:
            return self._resolve_tenant(filters.get("tenant_id"))
        except Exception:
            return None

    async def _select_backend_async(self, tenant_id: Optional[str]) -> str:
        if self._milvus is None:
            return "qdrant"
        tid = self._resolve_tenant(tenant_id)
        if not tid:
            return "qdrant"
        with self._route_lock:
            cached = self._route_table.get(tid)
        if cached:
            return cached
        try:
            exists = await self._qdrant.tenant_exists(tid)
        except Exception:
            # Probe failures are non-authoritative; fall back to Milvus for this
            # request, but do not persist a sticky route choice.
            return "milvus"
        route = "milvus"
        if exists is True:
            route = "qdrant"
        elif exists is False:
            route = "milvus"
        else:
            route = "milvus"
        with self._route_lock:
            self._route_table[tid] = route
        return route

    def _log_route(self, tenant_id: Optional[str], route: str) -> None:
        ctx = _get_request_context()
        if not ctx:
            try:
                print(
                    "[vector.route]"
                    " request_id=unknown"
                    " time=unknown"
                    " method=unknown"
                    " path=unknown"
                    " query="
                    " content_length=unknown"
                    f" tenant_id={tenant_id}"
                    f" route={route}"
                )
            except Exception:
                pass
            return
        if ctx.get("_route_logged"):
            return
        ctx["_route_logged"] = True
        try:
            print(
                "[vector.route]"
                f" request_id={ctx.get('request_id')}"
                f" time={ctx.get('request_time')}"
                f" method={ctx.get('method')}"
                f" path={ctx.get('path')}"
                f" query={ctx.get('query')}"
                f" content_length={ctx.get('content_length')}"
                f" tenant_id={tenant_id}"
                f" route={route}"
            )
        except Exception:
            pass
        try:
            self._logger.info(
                "vector.route",
                extra={
                    "event": "vector.route",
                    "module": "memory",
                    "entity": "vector",
                    "verb": "route",
                    "status": "ok",
                    "request_id": ctx.get("request_id"),
                    "request_time": ctx.get("request_time"),
                    "method": ctx.get("method"),
                    "path": ctx.get("path"),
                    "query": ctx.get("query"),
                    "content_length": ctx.get("content_length"),
                    "tenant_id": tenant_id,
                    "route": route,
                },
            )
        except Exception:
            pass

    async def upsert_vectors(self, entries: List[MemoryEntry]) -> None:
        tenant_id = self._tenant_from_entries(entries)
        if tenant_id is None:
            ctx = _get_request_context() or {}
            tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            await self._qdrant.upsert_vectors(entries)
            return None
        await self._milvus.upsert_vectors(entries)
        return None

    async def search_vectors(
        self,
        query: str,
        filters: Dict[str, Any],
        topk: int,
        threshold: float | None = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        async def _search_with_optional_query_vector(store: Any) -> List[Dict[str, Any]]:
            if query_vector is not None:
                try:
                    return await store.search_vectors(query, filters, topk, threshold, query_vector=query_vector)
                except TypeError as exc:
                    # Backward compatibility for legacy fakes/custom stores that still use the old signature.
                    if "query_vector" not in str(exc):
                        raise
                    inc("query_vector_fallback_total", 1)
                    inc("query_vector_fallback_router_total", 1)
                    self._logger.warning(
                        "vector_store_router query_vector unsupported by backend=%s; falling back to plain search: %s",
                        type(store).__name__,
                        str(exc)[:200],
                    )
            return await store.search_vectors(query, filters, topk, threshold)

        tenant_id = self._tenant_from_filters(filters or {})
        if tenant_id is None:
            ctx = _get_request_context() or {}
            tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            return await _search_with_optional_query_vector(self._qdrant)
        return await _search_with_optional_query_vector(self._milvus)

    async def embed_query(self, query: str, *, tenant_id: Optional[str] = None) -> Optional[List[float]]:
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        backend = self._qdrant if route == "qdrant" else self._milvus
        if backend is None:
            return None
        embed_fn = getattr(backend, "embed_text", None)
        if not callable(embed_fn):
            return None
        try:
            vec = await asyncio.to_thread(embed_fn, str(query or ""))
        except Exception:
            return None
        if isinstance(vec, list) and vec:
            return list(vec)
        return None

    async def fetch_text_corpus(self, filters: Dict[str, Any], *, limit: int = 500) -> List[Dict[str, Any]]:
        tenant_id = self._tenant_from_filters(filters or {})
        if tenant_id is None:
            ctx = _get_request_context() or {}
            tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            return await self._qdrant.fetch_text_corpus(filters, limit=limit)
        return await self._milvus.fetch_text_corpus(filters, limit=limit)

    async def set_published(self, ids: List[str], published: bool) -> int:
        ctx = _get_request_context() or {}
        tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            return await self._qdrant.set_published(ids, published)
        return await self._milvus.set_published(ids, published)

    async def get(self, entry_id: str) -> MemoryEntry | None:
        ctx = _get_request_context() or {}
        tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            return await self._qdrant.get(entry_id)
        return await self._milvus.get(entry_id)

    async def delete_ids(self, ids: List[str]) -> None:
        ctx = _get_request_context() or {}
        tenant_id = self._resolve_tenant(ctx.get("tenant_id"))
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        if route == "qdrant":
            await self._qdrant.delete_ids(ids)
            return None
        await self._milvus.delete_ids(ids)
        return None

    async def count_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        backend = self._qdrant if route == "qdrant" else self._milvus
        fn = getattr(backend, "count_by_filter", None)
        if not callable(fn):
            raise NotImplementedError(f"{route}_count_by_filter_not_supported")
        return await fn(tenant_id=tenant_id, api_key_id=api_key_id)

    async def list_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        backend = self._qdrant if route == "qdrant" else self._milvus
        fn = getattr(backend, "list_ids_by_filter", None)
        if not callable(fn):
            raise NotImplementedError(f"{route}_list_ids_by_filter_not_supported")
        return await fn(tenant_id=tenant_id, api_key_id=api_key_id)

    async def list_entry_ids_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> List[str]:
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        backend = self._qdrant if route == "qdrant" else self._milvus
        fn = getattr(backend, "list_entry_ids_by_filter", None)
        if callable(fn):
            return await fn(tenant_id=tenant_id, api_key_id=api_key_id)
        fn = getattr(backend, "list_ids_by_filter", None)
        if not callable(fn):
            raise NotImplementedError(f"{route}_list_entry_ids_by_filter_not_supported")
        return await fn(tenant_id=tenant_id, api_key_id=api_key_id)

    async def delete_by_filter(self, *, tenant_id: str, api_key_id: str | None = None) -> int:
        route = await self._select_backend_async(tenant_id)
        self._log_route(tenant_id, route)
        backend = self._qdrant if route == "qdrant" else self._milvus
        fn = getattr(backend, "delete_by_filter", None)
        if not callable(fn):
            raise NotImplementedError(f"{route}_delete_by_filter_not_supported")
        return await fn(tenant_id=tenant_id, api_key_id=api_key_id)

    async def health(self) -> Dict[str, Any]:
        status = {"qdrant": await self._qdrant.health()}
        if self._milvus is not None:
            status["milvus"] = await self._milvus.health()
        return status

    async def ensure_collections(self) -> None:
        await self._qdrant.ensure_collections()
        if self._milvus is not None:
            await self._milvus.ensure_collections()
