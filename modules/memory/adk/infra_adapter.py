from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class HttpMemoryInfraAdapter:
    """Layer 0 HTTP adapter used by ADK runtime.

    Responsibilities:
    - Build HTTP requests with tenant/auth headers.
    - Filter request body params (`None` removed).
    - Return uniform in-band HTTP error payload:
      `{"status_code": int, "body": Any}`.

    Non-responsibilities:
    - ToolResult mapping.
    - Semantic resolution / orchestration logic.
    """

    def __init__(
        self,
        *,
        base_url: str,
        tenant_id: str,
        auth_token: str | None = None,
        auth_header: str = "Authorization",
        timeout_s: float = 30.0,
        verify_tls: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = str(base_url or "").rstrip("/")
        self._tenant_id = str(tenant_id or "").strip()
        self._auth_token = str(auth_token or "").strip() or None
        self._auth_header = str(auth_header or "Authorization").strip() or "Authorization"
        self._timeout_s = float(timeout_s)
        self._verify_tls = bool(verify_tls)
        self._client: Optional[httpx.AsyncClient] = client
        self._owns_client = client is None

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HttpMemoryInfraAdapter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def _headers(self) -> Dict[str, str]:
        out: Dict[str, str] = {"X-Tenant-ID": self._tenant_id}
        if self._auth_token:
            if self._auth_token.lower().startswith("bearer "):
                out[self._auth_header] = self._auth_token
            else:
                out[self._auth_header] = f"Bearer {self._auth_token}"
        return out

    async def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_s,
            verify=self._verify_tls,
            trust_env=False,
        )
        return self._client

    @staticmethod
    def _build_body(**kwargs: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "tenant_id":
                continue
            if value is None:
                continue
            out[key] = value
        return out

    @staticmethod
    def _build_params(**kwargs: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "tenant_id":
                continue
            if value is None:
                continue
            out[key] = value
        return out

    @staticmethod
    def _parse_error_body(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except Exception:
            return resp.text

    async def _request(
        self,
        *,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        client = await self._client_or_create()
        try:
            resp = await client.request(
                method=method.upper(),
                url=path,
                json=body,
                params=params,
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            return {"status_code": 504, "body": str(exc)}
        except httpx.HTTPError as exc:
            return {"status_code": 503, "body": str(exc)}

        if resp.status_code >= 400:
            return {
                "status_code": int(resp.status_code),
                "body": self._parse_error_body(resp),
            }

        try:
            data = resp.json()
        except Exception:
            return {
                "status_code": 502,
                "body": resp.text or "invalid_json_response",
            }

        if isinstance(data, dict):
            return data
        # Endpoints used by ADK all return object JSON.
        return {"status_code": 502, "body": "invalid_json_response_type"}

    async def resolve_entity(
        self,
        *,
        name: str,
        type: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        limit: int = 5,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(name=name, type=type, user_tokens=user_tokens, limit=limit, debug=debug, tenant_id=tenant_id)
        return await self._request(method="POST", path="/memory/v1/resolve-entity", body=body)

    async def entity_profile_api(
        self,
        *,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        facts_limit: int = 20,
        relations_limit: int = 20,
        events_limit: int = 20,
        include_quotes: bool = False,
        include_relations: bool = True,
        include_events: bool = True,
        include_states: bool = False,
        quotes_limit: int = 20,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            entity=entity,
            entity_id=entity_id,
            user_tokens=user_tokens,
            memory_domain=memory_domain,
            facts_limit=facts_limit,
            relations_limit=relations_limit,
            events_limit=events_limit,
            include_quotes=include_quotes,
            include_relations=include_relations,
            include_events=include_events,
            include_states=include_states,
            quotes_limit=quotes_limit,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/entity-profile", body=body)

    async def time_since_api(
        self,
        *,
        topic: Optional[str] = None,
        topic_id: Optional[str] = None,
        topic_path: Optional[str] = None,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 50,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            topic=topic,
            topic_id=topic_id,
            topic_path=topic_path,
            entity=entity,
            entity_id=entity_id,
            user_tokens=user_tokens,
            time_range=time_range,
            memory_domain=memory_domain,
            limit=limit,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/time-since", body=body)

    async def relations_api(
        self,
        *,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 50,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            entity=entity,
            entity_id=entity_id,
            relation_type=relation_type,
            user_tokens=user_tokens,
            time_range=time_range,
            memory_domain=memory_domain,
            limit=limit,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/relations", body=body)

    async def quotes_api(
        self,
        *,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        topic: Optional[str] = None,
        topic_id: Optional[str] = None,
        topic_path: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 50,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            entity=entity,
            entity_id=entity_id,
            topic=topic,
            topic_id=topic_id,
            topic_path=topic_path,
            user_tokens=user_tokens,
            time_range=time_range,
            memory_domain=memory_domain,
            limit=limit,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/quotes", body=body)

    async def topic_timeline_api(
        self,
        *,
        topic: Optional[str] = None,
        topic_id: Optional[str] = None,
        topic_path: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        user_tokens: Optional[List[str]] = None,
        time_range: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        memory_domain: Optional[str] = None,
        limit: int = 50,
        with_quotes: bool = False,
        with_entities: bool = False,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            topic=topic,
            topic_id=topic_id,
            topic_path=topic_path,
            keywords=keywords,
            user_tokens=user_tokens,
            time_range=time_range,
            session_id=session_id,
            memory_domain=memory_domain,
            limit=limit,
            with_quotes=with_quotes,
            with_entities=with_entities,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/topic-timeline", body=body)

    async def explain_api(
        self,
        *,
        event_id: str,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        debug: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            event_id=event_id,
            user_tokens=user_tokens,
            memory_domain=memory_domain,
            debug=debug,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/v1/explain", body=body)

    async def list_entities_api(
        self,
        *,
        user_tokens: Optional[List[str]] = None,
        type: Optional[str] = None,
        query: Optional[str] = None,
        mentioned_since: Optional[str] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        memory_domain: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = self._build_params(
            user_tokens=user_tokens,
            type=type,
            query=query,
            mentioned_since=mentioned_since,
            limit=limit,
            cursor=cursor,
            memory_domain=memory_domain,
            tenant_id=tenant_id,
        )
        return await self._request(method="GET", path="/memory/v1/entities", params=params)

    async def list_topics_api(
        self,
        *,
        user_tokens: Optional[List[str]] = None,
        query: Optional[str] = None,
        parent_path: Optional[str] = None,
        min_events: Optional[int] = None,
        limit: int = 20,
        cursor: Optional[str] = None,
        memory_domain: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = self._build_params(
            user_tokens=user_tokens,
            query=query,
            parent_path=parent_path,
            min_events=min_events,
            limit=limit,
            cursor=cursor,
            memory_domain=memory_domain,
            tenant_id=tenant_id,
        )
        return await self._request(method="GET", path="/memory/v1/topics", params=params)

    async def state_current_api(
        self,
        *,
        subject_id: str,
        property: str,
        user_tokens: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(subject_id=subject_id, property=property, user_tokens=user_tokens, tenant_id=tenant_id)
        return await self._request(method="POST", path="/memory/state/current", body=body)

    async def state_at_time_api(
        self,
        *,
        subject_id: str,
        property: str,
        t_iso: str,
        user_tokens: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(subject_id=subject_id, property=property, t_iso=t_iso, user_tokens=user_tokens, tenant_id=tenant_id)
        return await self._request(method="POST", path="/memory/state/at_time", body=body)

    async def state_what_changed_api(
        self,
        *,
        subject_id: str,
        property: str,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        limit: int = 200,
        order: str = "asc",
        user_tokens: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            subject_id=subject_id,
            property=property,
            start_iso=start_iso,
            end_iso=end_iso,
            limit=limit,
            order=order,
            user_tokens=user_tokens,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/state/what-changed", body=body)

    async def state_time_since_api(
        self,
        *,
        subject_id: str,
        property: str,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        user_tokens: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = self._build_body(
            subject_id=subject_id,
            property=property,
            start_iso=start_iso,
            end_iso=end_iso,
            user_tokens=user_tokens,
            tenant_id=tenant_id,
        )
        return await self._request(method="POST", path="/memory/state/time-since", body=body)

    async def state_properties_api(
        self,
        *,
        user_tokens: Optional[List[str]] = None,
        limit: int = 200,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = self._build_params(user_tokens=user_tokens, limit=limit, tenant_id=tenant_id)
        return await self._request(method="GET", path="/memory/v1/state/properties", params=params)
