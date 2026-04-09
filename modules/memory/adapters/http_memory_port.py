from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.contracts.memory_models import Edge, MemoryEntry, SearchFilters, SearchResult, Hit, Version


class HttpMemoryPort:
    """HTTP adapter that implements the minimal MemoryPort surface used by client pipelines.

    This is intentionally small:
    - search -> POST /search
    - write  -> POST /write
    - delete -> POST /delete

    Auth/signing headers are provided by caller (we do not generate signatures here).
    """

    def __init__(
        self,
        *,
        base_url: str,
        tenant_id: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 30.0,
        session: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not str(base_url or "").strip():
            raise ValueError("base_url is required")
        if not str(tenant_id or "").strip():
            raise ValueError("tenant_id is required")
        self.base_url = str(base_url).rstrip("/")
        self.tenant_id = str(tenant_id)
        self._headers = dict(headers or {})
        self._timeout_s = float(timeout_s)
        self._session = session or httpx.AsyncClient()

    def _merged_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", **self._headers}
        if self.tenant_id:
            h.setdefault("X-Tenant-ID", self.tenant_id)
        return h

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._merged_headers()
        body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload is not None else None
        kwargs: Dict[str, Any] = {"headers": headers, "timeout": self._timeout_s}
        if params:
            kwargs["params"] = params
        if body is not None and method.upper() != "GET":
            kwargs["content"] = body
        resp = await self._session.request(method.upper(), url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._session.aclose()

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
        payload: Dict[str, Any] = {
            "query": str(query or ""),
            "topk": int(topk),
            "expand_graph": bool(expand_graph),
        }
        if filters is not None:
            payload["filters"] = filters.model_dump(exclude_none=True)
        if threshold is not None:
            payload["threshold"] = float(threshold)
        if scope is not None:
            payload["scope"] = str(scope)

        data = await self._request_json("POST", "/search", payload=payload)
        hits: List[Hit] = []
        for item in data.get("hits", []) or []:
            try:
                entry = MemoryEntry.model_validate(item.get("entry") or {})
                hits.append(Hit(id=str(item.get("id") or ""), score=float(item.get("score") or 0.0), entry=entry))
            except Exception:
                continue
        return SearchResult(
            hits=hits,
            neighbors=data.get("neighbors") or {},
            hints=str(data.get("hints") or ""),
            trace=data.get("trace") or {},
        )

    async def write(
        self,
        entries: List[MemoryEntry],
        links: Optional[List[Edge]] = None,
        *,
        upsert: bool = True,
        return_id_map: bool = False,
    ) -> Version | tuple[Version, dict[str, str]]:
        payload: Dict[str, Any] = {
            "entries": [e.model_dump(exclude_none=True) for e in (entries or [])],
            "upsert": bool(upsert),
            "return_id_map": bool(return_id_map),
        }
        if links:
            payload["links"] = [link.model_dump(exclude_none=True) for link in links]

        data = await self._request_json("POST", "/write", payload=payload)
        ver = Version.model_validate(data)
        if return_id_map:
            id_map = data.get("id_map") or {}
            if isinstance(id_map, dict):
                return ver, {str(k): str(v) for k, v in id_map.items()}
            return ver, {}
        return ver

    async def delete(self, memory_id: str, *, soft: bool = True, reason: Optional[str] = None) -> Version:
        payload: Dict[str, Any] = {"id": str(memory_id), "soft": bool(soft)}
        if reason:
            payload["reason"] = str(reason)
        data = await self._request_json("POST", "/delete", payload=payload)
        return Version.model_validate(data)

    async def graph_upsert_v0(self, body: GraphUpsertRequest) -> None:
        payload = body.model_dump(mode="json")
        data = await self._request_json("POST", "/graph/v0/upsert", payload=payload)
        # server returns {"ok": true}
        if not bool((data or {}).get("ok", False)):
            raise RuntimeError("graph_upsert_failed")

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
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        params: Dict[str, Any] = {"limit": int(limit)}
        if segment_id is not None:
            params["segment_id"] = str(segment_id)
        if entity_id is not None:
            params["entity_id"] = str(entity_id)
        if place_id is not None:
            params["place_id"] = str(place_id)
        if source_id is not None:
            params["source_id"] = str(source_id)
        data = await self._request_json("GET", "/graph/v0/events", payload=None, params=params)
        items = data.get("items") or []
        return list(items) if isinstance(items, list) else []

    async def graph_list_places(
        self,
        *,
        tenant_id: str,
        name: Optional[str] = None,
        segment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        params: Dict[str, Any] = {"limit": int(limit)}
        if name is not None:
            params["name"] = str(name)
        if segment_id is not None:
            params["segment_id"] = str(segment_id)
        data = await self._request_json("GET", "/graph/v0/places", payload=None, params=params)
        items = data.get("items") or []
        return list(items) if isinstance(items, list) else []

    async def graph_event_detail(self, *, tenant_id: str, event_id: str) -> dict:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        data = await self._request_json("GET", f"/graph/v0/events/{str(event_id)}", payload=None, params=None)
        item = data.get("item") or {}
        return dict(item) if isinstance(item, dict) else {}

    async def graph_place_detail(self, *, tenant_id: str, place_id: str) -> dict:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        data = await self._request_json("GET", f"/graph/v0/places/{str(place_id)}", payload=None, params=None)
        item = data.get("item") or {}
        return dict(item) if isinstance(item, dict) else {}

    async def graph_explain_event_evidence(self, *, tenant_id: str, event_id: str) -> dict:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        data = await self._request_json("GET", f"/graph/v0/explain/event/{str(event_id)}", payload=None, params=None)
        item = data.get("item") or {}
        return dict(item) if isinstance(item, dict) else {}

    async def graph_search_v1(
        self,
        *,
        tenant_id: str,
        query: str,
        topk: int = 10,
        source_id: Optional[str] = None,
        include_evidence: bool = True,
    ) -> dict:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        payload: Dict[str, Any] = {
            "query": str(query or ""),
            "topk": int(topk),
            "include_evidence": bool(include_evidence),
        }
        if source_id is not None:
            payload["source_id"] = str(source_id)
        data = await self._request_json("POST", "/graph/v1/search", payload=payload)
        return dict(data) if isinstance(data, dict) else {}

    async def graph_resolve_entities(
        self,
        *,
        tenant_id: str,
        name: str,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        params: Dict[str, Any] = {"name": str(name), "limit": int(limit)}
        if entity_type is not None:
            params["type"] = str(entity_type)
        data = await self._request_json("GET", "/graph/v0/entities/resolve", payload=None, params=params)
        items = data.get("items") or []
        return list(items) if isinstance(items, list) else []

    async def graph_list_timeslices_range(
        self,
        *,
        tenant_id: str,
        start_iso: Optional[str],
        end_iso: Optional[str],
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        if str(tenant_id or "").strip() and str(tenant_id) != str(self.tenant_id):
            raise ValueError("tenant_id mismatch for this HttpMemoryPort")
        params: Dict[str, Any] = {"limit": int(limit)}
        if start_iso is not None:
            params["start_iso"] = str(start_iso)
        if end_iso is not None:
            params["end_iso"] = str(end_iso)
        if kind is not None:
            params["kind"] = str(kind)
        data = await self._request_json("GET", "/graph/v0/timeslices/range", payload=None, params=params)
        items = data.get("items") or []
        return list(items) if isinstance(items, list) else []
