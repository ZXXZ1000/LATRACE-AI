from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class MemoryEntry(BaseModel):
    """Unified memory entry used across sources (m3/mem0/ctrl).

    .. deprecated::
        MemoryEntry and the associated :MemoryNode graph are being phased out.
        **Prefer TKG (Typed Knowledge Graph) for new memory writes:**
        - Use `GraphUpsertRequest` with `Event`, `Entity`, `Evidence` nodes
        - TKG nodes are written to both Neo4j AND Qdrant (with node_type/node_id payload)
        - TKG provides richer semantics: evidence chains, entity resolution, temporal reasoning
        
        See: `modules/memory/contracts/graph_models.py` for TKG models.
        See: `modules/memory/application/graph_service.py::GraphService.upsert()` for TKG writes.

    - kind: episodic | semantic
    - modality: text | image | audio | structured
    - contents: primary textual form (ASR/summary/sentence). For non-text, a textual description.
      contents MUST be List[str] - never None, never contains non-string elements.
    - vectors: optional precomputed vectors; if missing, application will embed.
    - metadata: clip_id/timestamp/entities/room/device/user/importance/ttl/pinned/stability/source/etc.
    """

    id: Optional[str] = None
    kind: Literal["episodic", "semantic"]
    modality: Literal["text", "image", "audio", "structured"]
    contents: List[str] = Field(default_factory=list)
    vectors: Optional[Dict[str, List[float]]] = None  # e.g. {"text": [...], "image": [...], "audio": [...]}
    metadata: Dict[str, Any] = Field(default_factory=dict)
    published: Optional[bool] = None

    @field_validator("contents", mode="before")
    @classmethod
    def validate_contents(cls, v: Any) -> List[str]:
        """确保contents是List[str]类型，自动转换并过滤无效值。"""
        if v is None:
            return []
        if isinstance(v, list):
            # 过滤并转换为字符串
            result = []
            for item in v:
                if item is not None:
                    result.append(str(item))
            return result
        # 单个值转换为列表
        if v is not None:
            return [str(v)]
        return []

    def get_primary_content(self, default: str = "") -> str:
        """安全获取主要（第一个）内容，默认返回空字符串而不是抛出异常。"""
        if self.contents and len(self.contents) > 0:
            return self.contents[0]
        return default

    def add_content(self, content: Union[str, List[str]]) -> None:
        """安全添加内容，自动转换为List[str]并去重。"""
        if isinstance(content, str):
            if content and content not in self.contents:
                self.contents.append(content)
        elif isinstance(content, list):
            for item in content:
                if item is not None and str(item) not in self.contents:
                    self.contents.append(str(item))

    def __repr__(self) -> str:
        """友好的字符串表示，便于调试。"""
        primary = self.get_primary_content("<empty>")
        return f"MemoryEntry(id={self.id}, kind={self.kind}, modality={self.modality}, primary='{primary[:50]}...', contents_len={len(self.contents)})"


class Edge(BaseModel):
    """Graph relation between nodes in Neo4j.

    - rel_type examples: appears_in | said_by | located_in | equivalence | prefer | executed
    - weight: optional numeric weight for reinforce/weaken semantics.
    """

    src_id: str
    dst_id: str
    rel_type: str
    weight: Optional[float] = None


class SearchFilters(BaseModel):
    """Search filters used by MemoryPort.search()."""

    modality: Optional[List[str]] = None
    time_range: Optional[Dict[str, Any]] = None  # {"gte": ts, "lte": ts}
    clip_id: Optional[Any] = None
    entities: Optional[List[str]] = None
    rel_types: Optional[List[str]] = None
    importance_range: Optional[Dict[str, float]] = None
    stability_range: Optional[Dict[str, float]] = None
    ttl_range: Optional[Dict[str, Any]] = None
    memory_type: Optional[List[str]] = None  # ["episodic","semantic"]
    source: Optional[List[str]] = None  # ["m3","mem0","ctrl"]
    threshold: Optional[float] = None
    published: Optional[bool] = None
    # New scoping filters
    tenant_id: Optional[str] = None      # optional hard tenant boundary for vector store
    user_id: Optional[List[str]] = None  # strong isolation principal (can be multiple)
    memory_domain: Optional[str] = None  # domain scoping: work/home/...
    run_id: Optional[str] = None         # session/task grouping
    memory_scope: Optional[str] = None   # per-video isolation (vh::<sha>)
    user_match: Optional[Literal['any', 'all']] = 'any'  # how to match user_id list
    # Character scoping (P2): match entries annotated with a character id/name
    character_id: Optional[List[str]] = None
    # Topic/tag filters (P1): hard filters for semantic API
    topic_path: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    time_bucket: Optional[List[str]] = None
    tags_vocab_version: Optional[str] = None


class Hit(BaseModel):
    id: str
    score: float
    entry: MemoryEntry


class SearchResult(BaseModel):
    hits: List[Hit]
    neighbors: Dict[str, Any] = Field(default_factory=dict)  # graph expansion summary
    hints: str = ""  # compact text for LLM context
    trace: Dict[str, Any] = Field(default_factory=dict)


class Version(BaseModel):
    value: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
