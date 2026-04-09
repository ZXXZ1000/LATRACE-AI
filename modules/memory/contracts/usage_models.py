from __future__ import annotations

from typing import Literal, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TokenUsageDetail(BaseModel):
    """Atomic token usage breakdown (provider/model independent internal schema)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None  # Estimated cost, can be None


class EmbeddingUsage(BaseModel):
    """Usage detail for a single embedding operation."""
    provider: str
    model: str
    tokens: int  # usually prompt_tokens
    cost_usd: Optional[float] = None  # cost in USD
    currency: Optional[str] = "USD"
    source: Literal["openrouter_generation", "response", "missing", "estimation"]


class LLMUsage(BaseModel):
    """Usage detail for a single LLM call."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class UsageSummary(BaseModel):
    """Aggregated usage for an API response (Receipt)."""
    total: TokenUsageDetail
    llm: TokenUsageDetail
    embedding: TokenUsageDetail
    billable: bool = True  # False if BYOK
    details: Optional[Any] = None  # Optional detailed list for debug


class UsageEvent(BaseModel):
    """Atomic usage fact for WAL (Trust Root)."""
    event_id: Optional[str] = Field(default=None, description="UUIDv4 or stable hash")
    timestamp: str = Field(default_factory=_now_iso_z, description="ISO8601 UTC")
    
    # Trace context
    tenant_id: str
    api_key_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    job_id: Optional[str] = None
    stage: Optional[str] = None
    source: Optional[str] = None
    call_index: Optional[int] = None
    
    # Event meta
    event_type: Literal["embedding", "llm", "write", "request"]
    provider: str
    model: str
    billable: bool = True
    byok: bool = False
    
    # Usage metrics
    usage: Optional[TokenUsageDetail] = None
    
    # Status
    status: Literal["ok", "fail"] = "ok"
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    
    # Additional raw meta
    meta: Dict[str, Any] = Field(default_factory=dict)
