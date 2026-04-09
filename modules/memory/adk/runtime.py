from __future__ import annotations

from typing import Dict, List, Optional

from .infra_adapter import HttpMemoryInfraAdapter
from .memory_tools import (
    entity_profile as memory_entity_profile,
    explain as memory_explain,
    list_entities as memory_list_entities,
    list_topics as memory_list_topics,
    quotes as memory_quotes,
    relations as memory_relations,
    time_since as memory_time_since,
    topic_timeline as memory_topic_timeline,
)
from .models import ToolResult
from .state_property_vocab import StatePropertyVocabManager
from .state_tools import (
    TimeRangeParserFn,
    WhenParserFn,
    entity_status as state_entity_status,
    state_time_since as state_state_time_since,
    status_changes as state_status_changes,
)
from .tool_definitions import MemoryToolDefinition, get_tool_definitions, to_mcp_tools, to_openai_tools


class MemoryAdkRuntime:
    """Layer 1 ADK runtime with semantic tool wrappers."""

    def __init__(
        self,
        *,
        tenant_id: str,
        infra: HttpMemoryInfraAdapter,
        user_tokens: Optional[List[str]] = None,
        when_parser: Optional[WhenParserFn] = None,
        time_range_parser: Optional[TimeRangeParserFn] = None,
    ) -> None:
        self.tenant_id = str(tenant_id or "").strip()
        self.infra = infra
        self.default_user_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
        self.when_parser = when_parser
        self.time_range_parser = time_range_parser
        self.vocab_manager = StatePropertyVocabManager(fetcher=self.infra.state_properties_api)

    async def aclose(self) -> None:
        await self.infra.aclose()

    async def __aenter__(self) -> "MemoryAdkRuntime":
        await self.infra.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.infra.__aexit__(exc_type, exc, tb)

    def _effective_user_tokens(self, override: Optional[List[str]]) -> List[str]:
        source = self.default_user_tokens if override is None else override
        return [str(x).strip() for x in (source or []) if str(x).strip()]

    async def entity_profile(
        self,
        *,
        entity: str | None = None,
        entity_id: str | None = None,
        include: Optional[List[str]] = None,
        limit: int = 10,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> ToolResult:
        return await memory_entity_profile(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            entity_profile_api=self.infra.entity_profile_api,
            entity=entity,
            entity_id=entity_id,
            include=include,
            limit=limit,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
        )

    async def time_since(
        self,
        *,
        entity: str | None = None,
        topic: str | None = None,
        entity_id: str | None = None,
        topic_id: str | None = None,
        topic_path: str | None = None,
        user_tokens: Optional[List[str]] = None,
        time_range: Optional[Dict[str, str]] = None,
        memory_domain: Optional[str] = None,
        limit: int = 50,
    ) -> ToolResult:
        return await memory_time_since(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            time_since_api=self.infra.time_since_api,
            entity=entity,
            topic=topic,
            entity_id=entity_id,
            topic_id=topic_id,
            topic_path=topic_path,
            user_tokens=self._effective_user_tokens(user_tokens),
            time_range=time_range,
            memory_domain=memory_domain,
            limit=limit,
        )

    async def relations(
        self,
        *,
        entity: str | None = None,
        entity_id: str | None = None,
        relation_type: str = "co_occurs_with",
        time_range: Optional[Dict[str, str]] = None,
        limit: int = 20,
        user_tokens: Optional[List[str]] = None,
    ) -> ToolResult:
        return await memory_relations(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            relations_api=self.infra.relations_api,
            entity=entity,
            entity_id=entity_id,
            relation_type=relation_type,
            time_range=time_range,
            limit=limit,
            user_tokens=self._effective_user_tokens(user_tokens),
        )

    async def quotes(
        self,
        *,
        entity: str | None = None,
        topic: str | None = None,
        entity_id: str | None = None,
        topic_id: str | None = None,
        topic_path: str | None = None,
        time_range: Optional[Dict[str, str]] = None,
        limit: int = 5,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> ToolResult:
        return await memory_quotes(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            quotes_api=self.infra.quotes_api,
            entity=entity,
            topic=topic,
            entity_id=entity_id,
            topic_id=topic_id,
            topic_path=topic_path,
            time_range=time_range,
            limit=limit,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
        )

    async def topic_timeline(
        self,
        *,
        topic: str | None = None,
        topic_id: str | None = None,
        topic_path: str | None = None,
        keywords: Optional[List[str]] = None,
        time_range: Optional[Dict[str, str]] = None,
        include: Optional[List[str]] = None,
        limit: int = 10,
        session_id: str | None = None,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> ToolResult:
        return await memory_topic_timeline(
            tenant_id=self.tenant_id,
            topic_timeline_api=self.infra.topic_timeline_api,
            topic=topic,
            topic_id=topic_id,
            topic_path=topic_path,
            keywords=keywords,
            time_range=time_range,
            include=include,
            limit=limit,
            session_id=session_id,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
        )

    async def explain(
        self,
        *,
        event_id: str,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
    ) -> ToolResult:
        return await memory_explain(
            tenant_id=self.tenant_id,
            explain_api=self.infra.explain_api,
            event_id=event_id,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
        )

    async def list_entities(
        self,
        *,
        query: str | None = None,
        entity_type: str | None = None,
        mentioned_since: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        auto_page: bool = False,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        max_pages: int = 3,
    ) -> ToolResult:
        return await memory_list_entities(
            tenant_id=self.tenant_id,
            list_entities_api=self.infra.list_entities_api,
            query=query,
            entity_type=entity_type,
            mentioned_since=mentioned_since,
            limit=limit,
            cursor=cursor,
            auto_page=auto_page,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
            max_pages=max_pages,
        )

    async def list_topics(
        self,
        *,
        query: str | None = None,
        parent_path: str | None = None,
        min_events: int | None = None,
        limit: int = 20,
        cursor: str | None = None,
        auto_page: bool = False,
        user_tokens: Optional[List[str]] = None,
        memory_domain: Optional[str] = None,
        max_pages: int = 3,
    ) -> ToolResult:
        return await memory_list_topics(
            tenant_id=self.tenant_id,
            list_topics_api=self.infra.list_topics_api,
            query=query,
            parent_path=parent_path,
            min_events=min_events,
            limit=limit,
            cursor=cursor,
            auto_page=auto_page,
            user_tokens=self._effective_user_tokens(user_tokens),
            memory_domain=memory_domain,
            max_pages=max_pages,
        )

    async def entity_status(
        self,
        *,
        entity: str | None = None,
        property: str | None = None,
        when: str | None = None,
        entity_id: str | None = None,
        property_canonical: str | None = None,
        user_tokens: Optional[List[str]] = None,
        when_parser: Optional[WhenParserFn] = None,
        force_vocab_refresh: bool = False,
    ) -> ToolResult:
        return await state_entity_status(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            vocab_manager=self.vocab_manager,
            state_current=self.infra.state_current_api,
            state_at_time=self.infra.state_at_time_api,
            entity=entity,
            property=property,
            when=when,
            entity_id=entity_id,
            property_canonical=property_canonical,
            user_tokens=self._effective_user_tokens(user_tokens),
            when_parser=when_parser or self.when_parser,
            force_vocab_refresh=force_vocab_refresh,
        )

    async def status_changes(
        self,
        *,
        entity: str | None = None,
        property: str | None = None,
        when: str | None = None,
        time_range: Optional[Dict[str, str]] = None,
        entity_id: str | None = None,
        property_canonical: str | None = None,
        user_tokens: Optional[List[str]] = None,
        time_range_parser: Optional[TimeRangeParserFn] = None,
        order: str = "desc",
        limit: int = 20,
        force_vocab_refresh: bool = False,
    ) -> ToolResult:
        return await state_status_changes(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            vocab_manager=self.vocab_manager,
            state_what_changed=self.infra.state_what_changed_api,
            entity=entity,
            property=property,
            when=when,
            time_range=time_range,
            entity_id=entity_id,
            property_canonical=property_canonical,
            user_tokens=self._effective_user_tokens(user_tokens),
            time_range_parser=time_range_parser or self.time_range_parser,
            order=order,
            limit=limit,
            force_vocab_refresh=force_vocab_refresh,
        )

    async def state_time_since(
        self,
        *,
        entity: str | None = None,
        property: str | None = None,
        when: str | None = None,
        time_range: Optional[Dict[str, str]] = None,
        entity_id: str | None = None,
        property_canonical: str | None = None,
        user_tokens: Optional[List[str]] = None,
        time_range_parser: Optional[TimeRangeParserFn] = None,
        force_vocab_refresh: bool = False,
    ) -> ToolResult:
        return await state_state_time_since(
            tenant_id=self.tenant_id,
            resolver=self.infra.resolve_entity,
            vocab_manager=self.vocab_manager,
            state_time_since_api=self.infra.state_time_since_api,
            entity=entity,
            property=property,
            when=when,
            time_range=time_range,
            entity_id=entity_id,
            property_canonical=property_canonical,
            user_tokens=self._effective_user_tokens(user_tokens),
            time_range_parser=time_range_parser or self.time_range_parser,
            force_vocab_refresh=force_vocab_refresh,
        )

    def get_tool_definitions(self, *, enabled_only: bool = True) -> List[MemoryToolDefinition]:
        return get_tool_definitions(enabled_only=enabled_only)

    def get_openai_tools(self, *, names: Optional[List[str]] = None) -> List[Dict[str, object]]:
        return to_openai_tools(names=names)

    def get_mcp_tools(self, *, names: Optional[List[str]] = None) -> List[Dict[str, object]]:
        return to_mcp_tools(names=names)


def create_memory_runtime(
    *,
    base_url: str,
    tenant_id: str,
    user_tokens: Optional[List[str]] = None,
    auth_token: str | None = None,
    timeout_s: float = 30.0,
    when_parser: Optional[WhenParserFn] = None,
    time_range_parser: Optional[TimeRangeParserFn] = None,
    auth_header: str = "Authorization",
    verify_tls: bool = True,
) -> MemoryAdkRuntime:
    """Create a ready-to-use ADK runtime with default HTTP infra adapter."""

    tenant_norm = str(tenant_id or "").strip()
    default_tokens = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not default_tokens and tenant_norm:
        default_tokens = [f"u:{tenant_norm}"]

    infra = HttpMemoryInfraAdapter(
        base_url=base_url,
        tenant_id=tenant_norm,
        auth_token=auth_token,
        auth_header=auth_header,
        timeout_s=timeout_s,
        verify_tls=verify_tls,
    )
    return MemoryAdkRuntime(
        tenant_id=tenant_norm,
        infra=infra,
        user_tokens=default_tokens,
        when_parser=when_parser,
        time_range_parser=time_range_parser,
    )
