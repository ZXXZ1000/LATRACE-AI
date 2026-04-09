from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from datetime import timezone, datetime
from dataclasses import dataclass
import asyncio
import logging
import time
import hashlib
import re

from modules.memory.contracts.memory_models import SearchFilters
from modules.memory.application.metrics import inc
from modules.memory.domain.dialog_tkg_vector_index_v1 import TKG_DIALOG_EVENT_INDEX_SOURCE_V1
from modules.memory.ports.memory_port import MemoryPort
try:
    from modules.memory.contracts.usage_models import (
        UsageEvent, UsageSummary, TokenUsageDetail, EmbeddingUsage
    )
    from modules.memory.application.llm_adapter import (
        set_llm_usage_hook, reset_llm_usage_hook
    )
    from modules.memory.application.embedding_adapter import (
        set_embedding_usage_hook, reset_embedding_usage_hook
    )
    _USAGE_AVAILABLE = True
except ImportError:
    _USAGE_AVAILABLE = False


_logger = logging.getLogger(__name__)


def _hash_usage_event_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _safe_int(value: Optional[int]) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _build_usage_summary(
    *,
    llm_events: List[Dict[str, Any]],
    embedding_events: List["EmbeddingUsage"],
    billable: bool,
    include_details: bool,
) -> "UsageSummary":
    def _llm_total_tokens(e: Dict[str, Any]) -> int:
        total = _safe_int(e.get("total_tokens"))
        if total:
            return total
        return _safe_int(e.get("prompt_tokens")) + _safe_int(e.get("completion_tokens"))

    llm_prompt = sum(_safe_int(e.get("prompt_tokens")) for e in llm_events)
    llm_completion = sum(_safe_int(e.get("completion_tokens")) for e in llm_events)
    llm_total = sum(_llm_total_tokens(e) for e in llm_events)
    llm_cost = 0.0
    llm_cost_seen = False
    for e in llm_events:
        try:
            if e.get("cost_usd") is not None:
                llm_cost += float(e.get("cost_usd") or 0.0)
                llm_cost_seen = True
        except Exception:
            pass
    emb_total = sum(_safe_int(e.tokens) for e in embedding_events)
    emb_cost = 0.0
    emb_cost_seen = False
    for e in embedding_events:
        try:
            if e.cost_usd is not None:
                emb_cost += float(e.cost_usd or 0.0)
                emb_cost_seen = True
        except Exception:
            pass
    total_prompt = llm_prompt + emb_total
    total_completion = llm_completion
    total_tokens = llm_total + emb_total
    total_cost = (llm_cost if llm_cost_seen else 0.0) + (emb_cost if emb_cost_seen else 0.0)
    total_cost_seen = llm_cost_seen or emb_cost_seen
    details = None
    if include_details:
        details = {
            "llm": [dict(e) for e in llm_events],
            "embedding": [e.model_dump() for e in embedding_events],
        }
    return UsageSummary(
        total=TokenUsageDetail(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_tokens,
            cost_usd=(total_cost if total_cost_seen else None),
        ),
        llm=TokenUsageDetail(
            prompt_tokens=llm_prompt,
            completion_tokens=llm_completion,
            total_tokens=llm_total,
            cost_usd=(llm_cost if llm_cost_seen else None),
        ),
        embedding=TokenUsageDetail(
            prompt_tokens=emb_total,
            completion_tokens=0,
            total_tokens=emb_total,
            cost_usd=(emb_cost if emb_cost_seen else None),
        ),
        billable=bool(billable),
        details=details,
    )


def _normalize_user_tokens(user_tokens: Sequence[str], tenant_id: str = None) -> List[str]:
    """
    Normalize and validate user_tokens.

    In SaaS mode, isolation is guaranteed by tenant_id derived from the API key.
    We therefore treat user_tokens as optional and derive a stable value from
    tenant_id when the caller does not provide explicit tokens.

    Self-hosted deployments may still choose to supply explicit user_tokens to
    implement per-user isolation within a tenant; in that case we simply
    normalize the provided values.
    """
    out = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not out and tenant_id:
        out = [f"u:{tenant_id}"]
    if not out:
        raise ValueError("user_tokens must be non-empty")
    return list(sorted(dict.fromkeys(out)))


async def _compute_shared_query_vector(
    *,
    store: MemoryPort,
    query: str,
    tenant_id: Optional[str] = None,
) -> Optional[List[float]]:
    """Compute one text query embedding for dialog_v2 and reuse across vector routes."""
    embed_query = getattr(store, "embed_query", None)
    if not callable(embed_query):
        return None
    try:
        vec = await embed_query(str(query or ""), tenant_id=tenant_id)
        if isinstance(vec, list) and vec:
            return list(vec)
    except Exception:
        return None
    return None


async def _store_search_with_optional_query_vector(
    store: MemoryPort,
    query: str,
    *,
    topk: int,
    filters: SearchFilters,
    expand_graph: bool,
    query_vector: Optional[List[float]] = None,
):
    """Call store.search with query_vector when supported; degrade gracefully for stubs/fakes."""
    if query_vector is not None:
        try:
            return await store.search(
                query,
                topk=int(topk),
                filters=filters,
                expand_graph=expand_graph,
                query_vector=query_vector,
            )
        except TypeError as exc:
            if "query_vector" not in str(exc):
                raise
            inc("query_vector_fallback_total", 1)
            inc("query_vector_fallback_retrieval_total", 1)
            _logger.warning(
                "retrieval.query_vector_fallback store=%s error=%s",
                type(store).__name__,
                str(exc)[:200],
            )
    return await store.search(query, topk=int(topk), filters=filters, expand_graph=expand_graph)


def _extract_fact_hits_v2(hits) -> List[Dict[str, Any]]:
    """Match benchmark/shared/adapters/moyan_memory_qa_adapter.py::_extract_fact_hits_v2 behavior."""
    results: List[Dict[str, Any]] = []
    for hit in hits or []:
        entry = hit.entry
        meta = dict(entry.metadata or {})
        contents = list(entry.contents or [])

        turn_ids = meta.get("source_turn_ids", []) or []
        sample_id = str(meta.get("source_sample_id") or "").strip()
        event_ids: List[str] = []
        for tid in turn_ids:
            normalized = str(tid).replace(":", "_")
            event_ids.append(f"{sample_id}_{normalized}" if sample_id else normalized)

        results.append(
            {
                "fact_id": hit.id,
                "event_ids": event_ids,
                "event_id": event_ids[0] if event_ids else None,
                "text": contents[0] if contents else "",
                "score": float(hit.score or 0.0),
                "source": "fact_search",
                "fact_type": meta.get("fact_type"),
                "sample_id": sample_id,
                "source_turn_ids": list(turn_ids),
            }
        )
    return results


def _extract_event_hits_v2(hits) -> List[Dict[str, Any]]:
    """Match benchmark/shared/adapters/moyan_memory_qa_adapter.py::_extract_event_hits_v2 behavior."""
    results: List[Dict[str, Any]] = []
    for hit in hits or []:
        entry = hit.entry
        meta = dict(entry.metadata or {})
        contents = list(entry.contents or [])

        eid = meta.get("event_id") or meta.get("dia_id") or hit.id
        results.append(
            {
                "event_id": eid,
                "event_ids": [eid] if eid else [],
                "text": contents[0] if contents else "",
                "score": float(hit.score or 0.0),
                "source": "event_search",
            }
        )
    return results


_DIALOG_V2_DEFAULT_WEIGHTS: Dict[str, float] = {
    "event_vec": 0.6,
    "vec": 0.6,
    "knowledge": 0.9,
    "entity": 0.15,
    "time": 0.15,
    "match": 1.0,
    "recency": 0.0,
    "signal": 0.0,
    "score_blend_alpha": 0.7,
    "multi": 0.03,
}


def _dialog_v2_rerank_evidence_type(source: str):
    from modules.memory.application.rerank_dialog_v1 import EvidenceType

    raw = str(source or "").strip()
    if raw in {"K_vec", "fact_search"}:
        return EvidenceType.FACT
    if raw in {"reference_trace"}:
        return EvidenceType.REFERENCE
    if raw in {"E_event_vec", "E_vec", "event_search"}:
        return EvidenceType.EVENT
    return EvidenceType.UNKNOWN


_MATCH_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_MATCH_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
_TIME_TEXT_RE = re.compile(
    r"\b("
    r"\d{4}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
    r"today|yesterday|tomorrow|last|next|ago|year|years|month|months|week|weeks|day|days|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r")\b",
    flags=re.I,
)
_LIST_TEXT_RE = re.compile(r",|(?:\band\b)|(?:\bor\b)", flags=re.I)
_STATUS_TERMS = {
    "single",
    "married",
    "dating",
    "divorced",
    "engaged",
    "partner",
    "husband",
    "wife",
    "boyfriend",
    "girlfriend",
    "single parent",
}
_MATCH_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "been",
    "bookshelf",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "hers",
    "him",
    "his",
    "in",
    "is",
    "it",
    "its",
    "likely",
    "many",
    "of",
    "on",
    "she",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
}
_QUESTION_WORDS = {"what", "when", "where", "who", "why", "would", "how"}
_MATCH_CONFLICT_PAIRS = {
    "sunrise": "sunset",
    "sunset": "sunrise",
    "husband": "wife",
    "wife": "husband",
    "boyfriend": "girlfriend",
    "girlfriend": "boyfriend",
}


def _tokenize_match_text(text: str) -> Tuple[str, ...]:
    return tuple(tok.lower() for tok in _MATCH_TOKEN_RE.findall(str(text or "")))


def _extract_match_names(text: str) -> Tuple[str, ...]:
    names: List[str] = []
    for item in _MATCH_NAME_RE.findall(str(text or "")):
        val = str(item or "").strip()
        if not val:
            continue
        lower = val.lower()
        if lower in _QUESTION_WORDS:
            continue
        names.append(val)
    return tuple(dict.fromkeys(names))


def _classify_question_slot(lowered_query: str) -> str:
    q = str(lowered_query or "").strip()
    if q.startswith("when ") or "what date" in q or "how long" in q or "how many years" in q:
        return "time"
    if "relationship status" in q or ("status" in q and "relationship" in q):
        return "status"
    if q.startswith("what ") and any(tok in q for tok in ("books", "activities", "events", "types", "kinds")):
        return "list"
    return "general"


@dataclass(frozen=True)
class _QueryMatchProfile:
    raw_query: str
    lowered_query: str
    tokens: Tuple[str, ...]
    token_set: frozenset[str]
    entity_names: Tuple[str, ...]
    entity_name_set: frozenset[str]
    anchor_tokens: Tuple[str, ...]
    rare_anchor_tokens: Tuple[str, ...]
    anchor_phrases: Tuple[str, ...]
    slot_type: str


@dataclass(frozen=True)
class _CandidateMatchProfile:
    text: str
    lowered_text: str
    tokens: Tuple[str, ...]
    token_set: frozenset[str]
    entity_names: Tuple[str, ...]
    entity_name_set: frozenset[str]
    has_time_marker: bool
    has_status_marker: bool
    has_list_marker: bool


class _DialogV2MatchEvaluator:
    """Lightweight, testable query-candidate fidelity matcher for dialog_v2.

    Design goals:
    - Keep query parsing and candidate scoring separate.
    - Expose stable sub-scores for debug/ablation.
    - Reward exact-answer fidelity without replacing route-support entirely.
    """

    def __init__(self, query: str) -> None:
        self.query_profile = self._build_query_profile(query)

    def _build_query_profile(self, query: str) -> _QueryMatchProfile:
        raw = str(query or "").strip()
        lowered = raw.lower()
        tokens = _tokenize_match_text(raw)
        token_set = frozenset(tokens)
        entity_names = _extract_match_names(raw)
        entity_name_set = frozenset(name.lower() for name in entity_names)

        anchor_tokens: List[str] = []
        for tok in tokens:
            if tok in _MATCH_STOPWORDS or tok in entity_name_set:
                continue
            if tok.isdigit():
                continue
            anchor_tokens.append(tok)
        anchor_tokens = list(dict.fromkeys(anchor_tokens))

        rare_anchor_tokens = [tok for tok in anchor_tokens if len(tok) >= 5]
        anchor_phrases: List[str] = []
        for left, right in zip(tokens, tokens[1:]):
            if left in _MATCH_STOPWORDS or right in _MATCH_STOPWORDS:
                continue
            if left in entity_name_set or right in entity_name_set:
                continue
            anchor_phrases.append(f"{left} {right}")
        anchor_phrases = list(dict.fromkeys(anchor_phrases))

        return _QueryMatchProfile(
            raw_query=raw,
            lowered_query=lowered,
            tokens=tokens,
            token_set=token_set,
            entity_names=entity_names,
            entity_name_set=entity_name_set,
            anchor_tokens=tuple(anchor_tokens),
            rare_anchor_tokens=tuple(rare_anchor_tokens),
            anchor_phrases=tuple(anchor_phrases),
            slot_type=_classify_question_slot(lowered),
        )

    def build_candidate_profile(self, text: str) -> _CandidateMatchProfile:
        raw = str(text or "").strip()
        lowered = raw.lower()
        tokens = _tokenize_match_text(raw)
        token_set = frozenset(tokens)
        entity_names = _extract_match_names(raw)
        entity_name_set = frozenset(name.lower() for name in entity_names)
        return _CandidateMatchProfile(
            text=raw,
            lowered_text=lowered,
            tokens=tokens,
            token_set=token_set,
            entity_names=entity_names,
            entity_name_set=entity_name_set,
            has_time_marker=bool(_TIME_TEXT_RE.search(raw)),
            has_status_marker=any(term in lowered for term in _STATUS_TERMS),
            has_list_marker=bool(_LIST_TEXT_RE.search(raw)),
        )

    def score_candidate(self, *, text: str) -> Tuple[float, Dict[str, float]]:
        candidate = self.build_candidate_profile(text)
        qp = self.query_profile

        entity_overlap = qp.entity_name_set & candidate.entity_name_set
        entity_match = 0.0
        if entity_overlap:
            entity_match = min(len(entity_overlap), 2) * 0.004

        anchor_overlap = set(qp.anchor_tokens) & candidate.token_set
        phrase_overlap = [phrase for phrase in qp.anchor_phrases if phrase and phrase in candidate.lowered_text]
        attribute_match = min(len(anchor_overlap), 3) * 0.0015 + min(len(phrase_overlap), 2) * 0.002

        rare_overlap = set(qp.rare_anchor_tokens) & candidate.token_set
        rare_anchor_match = min(len(rare_overlap), 2) * 0.0015

        slot_match = 0.0
        if qp.slot_type == "time" and candidate.has_time_marker:
            slot_match += 0.004
        elif qp.slot_type == "status" and candidate.has_status_marker:
            slot_match += 0.005
        elif qp.slot_type == "list" and candidate.has_list_marker:
            slot_match += 0.003

        conflict_penalty = 0.0
        if qp.entity_name_set and candidate.entity_name_set and not entity_overlap:
            conflict_penalty += 0.004
        for left, right in _MATCH_CONFLICT_PAIRS.items():
            if left in qp.token_set and right in candidate.token_set:
                conflict_penalty += 0.002

        total = entity_match + attribute_match + rare_anchor_match + slot_match - conflict_penalty
        details = {
            "entity_match": float(entity_match),
            "attribute_match": float(attribute_match),
            "rare_anchor_match": float(rare_anchor_match),
            "slot_match": float(slot_match),
            "conflict_penalty": float(conflict_penalty),
        }
        return float(total), details


def _parse_time_window(
    query: str,
    *,
    time_hints: Optional[Dict[str, Any]] = None,
    tz: Optional[timezone] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort time window parsing. Returns (start_iso, end_iso, reason)."""
    if tz is None:
        tz = timezone.utc
    if isinstance(time_hints, dict):
        start_iso = time_hints.get("start_iso")
        end_iso = time_hints.get("end_iso")
        if start_iso or end_iso:
            return (str(start_iso) if start_iso else None, str(end_iso) if end_iso else None, "time_hints")

    q = str(query or "")
    if not q:
        return (None, None, "empty_query")

    # Simple date patterns: YYYY-MM-DD or YYYY/MM/DD
    import re
    from datetime import datetime, timedelta

    dates = re.findall(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", q)
    if dates:
        def _to_dt(parts: tuple[str, str, str]) -> datetime:
            y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
            return datetime(y, m, d, tzinfo=tz)

        dts = [_to_dt(p) for p in dates]
        dts.sort()
        if len(dts) >= 2:
            start = dts[0]
            end = dts[-1] + timedelta(hours=23, minutes=59, seconds=59)
        else:
            start = dts[0]
            end = dts[0] + timedelta(hours=23, minutes=59, seconds=59)
        return (start.isoformat(), end.isoformat(), "date_literal")

    now = datetime.now(tz=tz)
    q_lower = q.lower()
    if "today" in q_lower or "今天" in q:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=23, minutes=59, seconds=59)
        return (start.isoformat(), end.isoformat(), "today")
    if "yesterday" in q_lower or "昨天" in q:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=23, minutes=59, seconds=59)
        return (start.isoformat(), end.isoformat(), "yesterday")
    if "前天" in q:
        start = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=23, minutes=59, seconds=59)
        return (start.isoformat(), end.isoformat(), "day_before_yesterday")
    if "last week" in q_lower or "上周" in q or "最近一周" in q:
        start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        return (start.isoformat(), end.isoformat(), "last_week")
    if "last month" in q_lower or "上个月" in q or "最近一月" in q:
        start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        return (start.isoformat(), end.isoformat(), "last_month")

    return (None, None, "no_time_hint")


def _recency_score_from_iso(ts_iso: Optional[str]) -> float:
    if not ts_iso:
        return 0.0
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(str(ts_iso))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_s = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
        half_life = 24 * 3600.0
        return 0.5 ** (age_s / half_life)
    except Exception:
        return 0.0


def _same_dialog_key_from_event_id(event_id: Optional[str]) -> Optional[str]:
    raw = str(event_id or "").strip()
    if not raw:
        return None
    raw = raw.replace(":", "_")
    for part in raw.split("_"):
        if len(part) >= 2 and part[0] == "D" and part[1:].isdigit():
            return part
    return None


def _dialog_v2_candidate_kind(source: Optional[str]) -> str:
    raw = str(source or "").strip()
    if raw == "E_vec":
        return "Original dialogue"
    if raw == "E_event_vec":
        return "Event summary"
    if raw in {"K_vec", "fact_search"}:
        return "Fact"
    if raw == "EN":
        return "Entity-linked event"
    if raw == "T":
        return "Time-linked event"
    return "Candidate evidence"


def _extract_speaker_and_body(text: Optional[str]) -> Tuple[Optional[str], str]:
    raw = str(text or "").strip()
    if not raw:
        return None, ""
    match = re.match(r"^([A-Za-z][A-Za-z0-9' .-]{0,40}):\s+(.*)$", raw)
    if not match:
        return None, raw
    speaker = str(match.group(1) or "").strip() or None
    body = str(match.group(2) or "").strip() or raw
    return speaker, body


def _conversation_date_from_timestamp(timestamp: Optional[str]) -> Optional[str]:
    raw = str(timestamp or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return raw[:10] if len(raw) >= 10 else raw


def _rank_to_unit_interval(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    safe_rank = max(1, min(int(rank), int(total)))
    return max(0.0, 1.0 - float(safe_rank - 1) / float(total - 1))


def _normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    nums = [float(v) for v in values.values()]
    lo = min(nums)
    hi = max(nums)
    if hi - lo <= 1e-9:
        return {k: 1.0 for k in values}
    return {k: max(0.0, min(1.0, (float(v) - lo) / (hi - lo))) for k, v in values.items()}


_DIALOG_V2_TEST_ALLOWED_ROUTE_NAMES = frozenset({"event", "evidence", "knowledge", "entity", "time"})
_DIALOG_V2_TEST_ALLOWED_SIGNAL_NAMES = frozenset(
    {"event", "evidence", "knowledge", "entity", "time", "timestamp", "recency", "graph_signal"}
)
_DIALOG_V2_ROUTE_SCORE_KEY = {
    "E_event_vec": "score_event_vec",
    "E_vec": "score_vec",
    "K_vec": "score_k",
    "EN": "score_en",
    "T": "score_t",
}
_DIALOG_V2_ROUTE_PRIORITY = {"E_event_vec": 5, "E_vec": 4, "K_vec": 3, "EN": 2, "T": 1}


def _parse_dialog_v2_test_name_set(value: Any, *, allowed: frozenset[str], field_name: str) -> frozenset[str]:
    raw_items: List[str] = []
    if value is None:
        return frozenset()
    if isinstance(value, str):
        raw_items = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_items = [str(part or "").strip().lower() for part in value if str(part or "").strip()]
    else:
        raise ValueError(f"{field_name} must be a string or sequence")
    invalid = sorted({item for item in raw_items if item not in allowed})
    if invalid:
        raise ValueError(f"unsupported {field_name}: {invalid}")
    return frozenset(raw_items)


@dataclass(frozen=True)
class _DialogV2TestAblationContract:
    disabled_routes: frozenset[str] = frozenset()
    disabled_backlinks: frozenset[str] = frozenset()
    disabled_signals: frozenset[str] = frozenset()
    source_native_only: bool = False

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]]) -> "_DialogV2TestAblationContract":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            disabled_routes=_parse_dialog_v2_test_name_set(
                payload.get("disabled_routes"),
                allowed=_DIALOG_V2_TEST_ALLOWED_ROUTE_NAMES,
                field_name="disabled_routes",
            ),
            disabled_backlinks=_parse_dialog_v2_test_name_set(
                payload.get("disabled_backlinks"),
                allowed=_DIALOG_V2_TEST_ALLOWED_ROUTE_NAMES,
                field_name="disabled_backlinks",
            ),
            disabled_signals=_parse_dialog_v2_test_name_set(
                payload.get("disabled_signals"),
                allowed=_DIALOG_V2_TEST_ALLOWED_SIGNAL_NAMES,
                field_name="disabled_signals",
            ),
            source_native_only=bool(payload.get("source_native_only")),
        )

    def is_active(self) -> bool:
        return bool(self.disabled_routes or self.disabled_backlinks or self.disabled_signals or self.source_native_only)

    def route_enabled(self, name: str, enabled: bool) -> bool:
        return bool(enabled) and str(name or "").strip().lower() not in self.disabled_routes

    def backlink_disabled(self, name: str) -> bool:
        return str(name or "").strip().lower() in self.disabled_backlinks

    def signal_enabled(self, name: str) -> bool:
        return str(name or "").strip().lower() not in self.disabled_signals

    def blocks_event_backlink(self) -> bool:
        return self.backlink_disabled("event")

    def should_use_source_native_candidates(self) -> bool:
        return bool(self.source_native_only)

    def should_disable_explain(self) -> bool:
        return self.blocks_event_backlink() or not self.signal_enabled("graph_signal")

    def model_dump(self) -> Dict[str, Any]:
        return {
            "disabled_routes": sorted(self.disabled_routes),
            "disabled_backlinks": sorted(self.disabled_backlinks),
            "disabled_signals": sorted(self.disabled_signals),
            "source_native_only": bool(self.source_native_only),
        }


class _DialogV2CandidatePool:
    def __init__(self) -> None:
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self._canonical_by_tkg: Dict[str, str] = {}
        self._canonical_by_logical: Dict[str, str] = {}

    def touch(
        self,
        event_id: str,
        *,
        route: str,
        score: float,
        text: Optional[str] = None,
        timestamp: Optional[str] = None,
        reason: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
        logical_event_id: Optional[str] = None,
        tkg_event_id: Optional[str] = None,
        evaluation_event_ids: Optional[Sequence[str]] = None,
        force_raw_key: bool = False,
    ) -> None:
        raw_event_id = str(event_id or "").strip()
        logical_id = str(logical_event_id or "").strip() or None
        tkg_id = str(tkg_event_id or "").strip() or None
        if not raw_event_id and not logical_id and not tkg_id:
            return
        eval_ids = [str(item or "").strip() for item in list(evaluation_event_ids or []) if str(item or "").strip()]
        if logical_id and logical_id not in eval_ids:
            eval_ids.append(logical_id)

        if force_raw_key:
            canonical_id = raw_event_id
        elif logical_id and logical_id in self._canonical_by_logical:
            canonical_id = self._canonical_by_logical[logical_id]
        elif tkg_id and tkg_id in self._canonical_by_tkg:
            canonical_id = self._canonical_by_tkg[tkg_id]
        elif logical_id:
            canonical_id = logical_id
        elif tkg_id:
            canonical_id = tkg_id
        else:
            canonical_id = raw_event_id
        canonical_id = str(canonical_id or "").strip()
        if not canonical_id:
            return

        def _merge_into(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            try:
                dst["sources"] |= set(src.get("sources") or set())
            except Exception:
                pass
            try:
                dst["reasons"].extend(list(src.get("reasons") or []))
            except Exception:
                pass
            for key in ("score_event_vec", "score_vec", "score_k", "score_en", "score_t"):
                try:
                    dst[key] = max(float(dst.get(key) or 0.0), float(src.get(key) or 0.0))
                except Exception:
                    pass
            if not dst.get("logical_event_id") and src.get("logical_event_id"):
                dst["logical_event_id"] = src.get("logical_event_id")
            if not dst.get("tkg_event_id") and src.get("tkg_event_id"):
                dst["tkg_event_id"] = src.get("tkg_event_id")
            try:
                dst["evaluation_event_ids"] |= set(src.get("evaluation_event_ids") or set())
            except Exception:
                pass
            if bool(src.get("source_native")):
                dst["source_native"] = True
            try:
                current = dst.get("primary_source")
                incoming = src.get("primary_source")
                if current is None or _DIALOG_V2_ROUTE_PRIORITY.get(str(incoming), 0) > _DIALOG_V2_ROUTE_PRIORITY.get(str(current), 0):
                    dst["primary_source"] = incoming
            except Exception:
                pass
            if not dst.get("text") and src.get("text"):
                dst["text"] = src.get("text")
            if not dst.get("timestamp") and src.get("timestamp"):
                dst["timestamp"] = src.get("timestamp")
            try:
                dst["route_meta"].update(dict(src.get("route_meta") or {}))
            except Exception:
                pass

        candidate = self.candidates.get(canonical_id)
        if candidate is None:
            candidate = {
                "event_id": canonical_id,
                "logical_event_id": None,
                "tkg_event_id": None,
                "sources": set(),
                "reasons": [],
                "score_event_vec": 0.0,
                "score_vec": 0.0,
                "score_k": 0.0,
                "score_en": 0.0,
                "score_t": 0.0,
                "text": None,
                "timestamp": None,
                "primary_source": None,
                "route_meta": {},
                "evaluation_event_ids": set(),
                "source_native": bool(force_raw_key),
            }
            self.candidates[canonical_id] = candidate

        if not force_raw_key:
            for alt in (tkg_id, logical_id, raw_event_id):
                if not alt or alt == canonical_id:
                    continue
                existing = self.candidates.get(alt)
                if existing is None:
                    continue
                _merge_into(candidate, existing)
                self.candidates.pop(alt, None)

        candidate["sources"].add(route)
        if reason:
            candidate["reasons"].append(reason)
        score_key = _DIALOG_V2_ROUTE_SCORE_KEY.get(route)
        if score_key:
            try:
                if float(score or 0.0) > float(candidate.get(score_key) or 0.0):
                    candidate[score_key] = float(score or 0.0)
            except Exception:
                pass
        try:
            current = candidate.get("primary_source")
            if current is None or _DIALOG_V2_ROUTE_PRIORITY.get(route, 0) > _DIALOG_V2_ROUTE_PRIORITY.get(str(current), 0):
                candidate["primary_source"] = route
        except Exception:
            candidate["primary_source"] = route
        current_source = candidate.get("primary_source")
        if timestamp:
            if not candidate.get("timestamp") or _DIALOG_V2_ROUTE_PRIORITY.get(route, 0) >= _DIALOG_V2_ROUTE_PRIORITY.get(str(current_source), 0):
                candidate["timestamp"] = timestamp
        if text:
            if not candidate.get("text") or _DIALOG_V2_ROUTE_PRIORITY.get(route, 0) >= _DIALOG_V2_ROUTE_PRIORITY.get(str(current_source), 0):
                candidate["text"] = text
        if extras:
            try:
                candidate["route_meta"].update(extras)
            except Exception:
                pass
        try:
            candidate["evaluation_event_ids"] |= set(eval_ids)
        except Exception:
            pass
        if logical_id:
            candidate["logical_event_id"] = logical_id
            self._canonical_by_logical[logical_id] = canonical_id
        if tkg_id:
            candidate["tkg_event_id"] = tkg_id
            self._canonical_by_tkg[tkg_id] = canonical_id
        if force_raw_key:
            candidate["source_native"] = True


def _build_dialog_v2_rerank_text(candidate: Dict[str, Any], *, candidate_id: str) -> str:
    speaker, body = _extract_speaker_and_body(candidate.get("text"))
    logical_event_id = str(candidate.get("logical_event_id") or "").strip() or None
    same_dialog_key = _same_dialog_key_from_event_id(logical_event_id or candidate_id)
    conversation_date = _conversation_date_from_timestamp(candidate.get("timestamp"))
    lines: List[str] = [
        f"Evidence ID: {logical_event_id or candidate_id}",
        f"Type: {_dialog_v2_candidate_kind(candidate.get('primary_source'))}",
    ]
    if same_dialog_key:
        lines.append(f"Dialogue block: {same_dialog_key}")
    if speaker:
        lines.append(f"Speaker: {speaker}")
    if conversation_date:
        lines.append(f"Conversation date: {conversation_date}")
    if candidate.get("primary_source"):
        lines.append(f"Source route: {str(candidate.get('primary_source'))}")
    content = body or str(candidate.get("text") or "").strip()
    lines.append(f"Content: {content}")
    return "\n".join(lines)


async def retrieval(
    store: MemoryPort,
    *,
    tenant_id: str,
    user_tokens: Sequence[str],
    query: str,
    strategy: str = "dialog_v2",
    topk: int = 30,
    memory_domain: str = "dialog",
    user_match: str = "all",
    run_id: Optional[str] = None,
    debug: bool = False,
    with_answer: bool = False,
    task: str = "GENERAL",
    llm_policy: str = "best_effort",
    qa_generate: Optional[Callable[[str, str], str]] = None,
    rerank: Optional[Dict[str, Any]] = None,
    rerank_generate: Optional[Callable[[str, str], str]] = None,
    backend: str = "tkg",  # memory | tkg (tkg-first, with memory fallback when index is absent)
    tkg_explain: bool = True,
    tkg_explain_topn: int = 5,
    candidate_k: int = 50,
    seed_topn: int = 15,
    e_vec_oversample: int = 3,
    graph_cap: int = 5,
    rrf_k: int = 60,
    qa_evidence_cap_l2: int = 12,
    qa_evidence_cap_l4: int = 12,
    enable_event_route: bool = True,
    enable_evidence_route: bool = True,
    enable_knowledge_route: bool = True,
    enable_entity_route: bool = True,
    enable_time_route: bool = True,
    entity_hints: Optional[Sequence[str]] = None,
    time_hints: Optional[Dict[str, Any]] = None,
    dialog_v2_weights: Optional[Dict[str, float]] = None,
    dialog_v2_reranker: Optional[Dict[str, Any]] = None,
    dialog_v2_test_ablation: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    byok_route: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level retrieval orchestration.

    Strategies:
    - `dialog_v1`: benchmark-aligned fixed 3-way search + fusion
    - `dialog_v2`: Event-first parallel recall (event_vec + utterance index + entity/time routes) with dynamic fill
    - `dialog_v2_test`: Experimental dialog_v2 variant for route ablation and benchmark testing
    """
    time.perf_counter()
    strategy_norm = str(strategy or "").strip().lower()
    if strategy_norm not in ("dialog_v1", "dialog_v2", "dialog_v2_test"):
        raise ValueError(f"unsupported strategy: {strategy}")
    backend_norm = str(backend or "memory").strip().lower()
    if backend_norm not in ("memory", "tkg"):
        raise ValueError(f"unsupported backend: {backend}")
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    user_tokens_norm = _normalize_user_tokens(user_tokens, tenant_id=str(tenant_id))
    user_match_norm = "all" if str(user_match or "all").lower() != "any" else "any"
    test_ablation = (
        _DialogV2TestAblationContract.from_payload(dialog_v2_test_ablation)
        if strategy_norm == "dialog_v2_test"
        else _DialogV2TestAblationContract()
    )

    llm_events: List[Dict[str, Any]] = []
    embedding_events: List["EmbeddingUsage"] = []
    usage_wal = getattr(store, "usage_wal", None)
    llm_hook_token = None
    emb_hook_token = None
    if _USAGE_AVAILABLE:
        try:
            llm_hook_token = set_llm_usage_hook(lambda payload: llm_events.append(dict(payload)))
            emb_hook_token = set_embedding_usage_hook(lambda usage: embedding_events.append(usage))
        except Exception:
            llm_hook_token = None
            emb_hook_token = None

    result: Dict[str, Any]
    try:
        if strategy_norm == "dialog_v2":
            result = await _retrieval_dialog_v2(
                store=store,
                tenant_id=str(tenant_id),
                user_tokens_norm=list(user_tokens_norm),
                user_match_norm=str(user_match_norm),
                query=q,
                topk=int(topk),
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                debug=bool(debug),
                with_answer=bool(with_answer),
                task=str(task or "GENERAL"),
                llm_policy=str(llm_policy or "best_effort"),
                qa_generate=qa_generate,
                tkg_explain=bool(tkg_explain),
                candidate_k=int(candidate_k),
                seed_topn=int(seed_topn),
                e_vec_oversample=int(e_vec_oversample),
                graph_cap=int(graph_cap),
                rrf_k=int(rrf_k),
                qa_evidence_cap_l2=int(qa_evidence_cap_l2),
                qa_evidence_cap_l4=int(qa_evidence_cap_l4),
                enable_event_route=bool(enable_event_route),
                enable_evidence_route=bool(enable_evidence_route),
                enable_knowledge_route=bool(enable_knowledge_route),
                enable_entity_route=bool(enable_entity_route),
                enable_time_route=bool(enable_time_route),
                entity_hints=list(entity_hints) if entity_hints else None,
                time_hints=dict(time_hints) if isinstance(time_hints, dict) else None,
                dialog_v2_weights=dict(dialog_v2_weights) if isinstance(dialog_v2_weights, dict) else None,
                dialog_v2_reranker=dict(dialog_v2_reranker) if isinstance(dialog_v2_reranker, dict) else None,
                test_ablation=_DialogV2TestAblationContract(),
            )
        elif strategy_norm == "dialog_v2_test":
            result = await _retrieval_dialog_v2_test(
                store=store,
                tenant_id=str(tenant_id),
                user_tokens_norm=list(user_tokens_norm),
                user_match_norm=str(user_match_norm),
                query=q,
                topk=int(topk),
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                debug=bool(debug),
                with_answer=bool(with_answer),
                task=str(task or "GENERAL"),
                llm_policy=str(llm_policy or "best_effort"),
                qa_generate=qa_generate,
                tkg_explain=bool(tkg_explain),
                candidate_k=int(candidate_k),
                seed_topn=int(seed_topn),
                e_vec_oversample=int(e_vec_oversample),
                graph_cap=int(graph_cap),
                rrf_k=int(rrf_k),
                qa_evidence_cap_l2=int(qa_evidence_cap_l2),
                qa_evidence_cap_l4=int(qa_evidence_cap_l4),
                enable_event_route=bool(enable_event_route),
                enable_evidence_route=bool(enable_evidence_route),
                enable_knowledge_route=bool(enable_knowledge_route),
                enable_entity_route=bool(enable_entity_route),
                enable_time_route=bool(enable_time_route),
                entity_hints=list(entity_hints) if entity_hints else None,
                time_hints=dict(time_hints) if isinstance(time_hints, dict) else None,
                dialog_v2_weights=dict(dialog_v2_weights) if isinstance(dialog_v2_weights, dict) else None,
                dialog_v2_reranker=dict(dialog_v2_reranker) if isinstance(dialog_v2_reranker, dict) else None,
                test_ablation=test_ablation,
            )
        else:
            result = await _retrieval_dialog_v1(
                store=store,
                tenant_id=str(tenant_id),
                user_tokens_norm=list(user_tokens_norm),
                user_match_norm=str(user_match_norm),
                query=q,
                topk=int(topk),
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                debug=bool(debug),
                with_answer=bool(with_answer),
                task=str(task or "GENERAL"),
                llm_policy=str(llm_policy or "best_effort"),
                qa_generate=qa_generate,
                rerank=rerank,
                rerank_generate=rerank_generate,
                backend=backend,
                tkg_explain=bool(tkg_explain),
                tkg_explain_topn=int(tkg_explain_topn),
                time_hints=dict(time_hints) if isinstance(time_hints, dict) else None,
                qa_evidence_cap_l2=int(qa_evidence_cap_l2),
                qa_evidence_cap_l4=int(qa_evidence_cap_l4),
            )
    finally:
        if _USAGE_AVAILABLE:
            if llm_hook_token is not None:
                try:
                    reset_llm_usage_hook(llm_hook_token)
                except Exception:
                    pass
            if emb_hook_token is not None:
                try:
                    reset_embedding_usage_hook(emb_hook_token)
                except Exception:
                    pass

    if _USAGE_AVAILABLE:
        billable = str(byok_route or "").strip().lower() != "byok"
        summary = _build_usage_summary(
            llm_events=llm_events,
            embedding_events=embedding_events,
            billable=billable,
            include_details=bool(debug),
        )
        try:
            result["usage"] = summary.model_dump()
        except Exception:
            result["usage"] = summary.dict()

        if usage_wal is not None:
            try:
                for idx, emb in enumerate(embedding_events):
                    seed = f"{tenant_id}|{request_id or run_id or ''}|embedding|{idx}"
                    evt = UsageEvent(
                        event_id=_hash_usage_event_id("embed", seed),
                        event_type="embedding",
                        tenant_id=str(tenant_id),
                        api_key_id=None,
                        request_id=str(request_id) if request_id else None,
                        trace_id=str(trace_id) if trace_id else None,
                        session_id=str(run_id) if run_id else None,
                        stage="retrieval",
                        source="retrieval",
                        call_index=idx,
                        provider=str(emb.provider),
                        model=str(emb.model),
                        billable=billable,
                        byok=not billable,
                        usage=TokenUsageDetail(
                            prompt_tokens=int(emb.tokens),
                            completion_tokens=0,
                            total_tokens=int(emb.tokens),
                            cost_usd=emb.cost_usd,
                        ),
                        status="ok",
                        meta={
                            "currency": emb.currency,
                            "source": emb.source,
                        },
                    )
                    usage_wal.append(evt.model_dump())
                for idx, payload in enumerate(llm_events):
                    call_index = payload.get("call_index")
                    try:
                        call_index_int = int(call_index) if call_index is not None else idx
                    except Exception:
                        call_index_int = idx
                    api_key_id = str(payload.get("api_key_id") or "").strip() or None
                    byok_route_payload = str(payload.get("byok_route") or "").strip()
                    byok_payload = byok_route_payload == "byok"
                    anchor = payload.get("request_id") or request_id or run_id or ""
                    seed = f"{tenant_id}|{api_key_id or ''}|{anchor}|{payload.get('stage') or ''}|{call_index_int}"
                    prompt_tokens = _safe_int(payload.get("prompt_tokens"))
                    completion_tokens = _safe_int(payload.get("completion_tokens"))
                    total_tokens = _safe_int(payload.get("total_tokens"))
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens
                    evt = UsageEvent(
                        event_id=_hash_usage_event_id("llm", seed),
                        event_type="llm",
                        tenant_id=str(tenant_id),
                        api_key_id=api_key_id,
                        request_id=str(payload.get("request_id") or request_id or "") or None,
                        trace_id=str(trace_id) if trace_id else None,
                        session_id=str(run_id) if run_id else None,
                        stage=str(payload.get("stage") or "").strip() or None,
                        source=str(payload.get("source") or "").strip() or None,
                        call_index=call_index_int,
                        provider=str(payload.get("provider") or "unknown"),
                        model=str(payload.get("model") or "unknown"),
                        billable=not byok_payload,
                        byok=byok_payload,
                        usage=TokenUsageDetail(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cost_usd=(float(payload.get("cost_usd")) if payload.get("cost_usd") is not None else None),
                        ),
                        status=str(payload.get("status") or "ok"),
                        error_code=(str(payload.get("error_code") or "").strip() or None),
                        error_detail=(str(payload.get("error_detail") or "").strip() or None),
                        meta={
                            "tokens_missing": bool(payload.get("tokens_missing")),
                            "byok_route": byok_route_payload or None,
                            "resolver_status": payload.get("resolver_status"),
                            "generation_id": payload.get("generation_id"),
                        },
                    )
                    usage_wal.append(evt.model_dump())
            except Exception:
                pass

    return result


async def _retrieval_dialog_v1(
    *,
    store: MemoryPort,
    tenant_id: str,
    user_tokens_norm: List[str],
    user_match_norm: str,
    query: str,
    topk: int,
    memory_domain: str,
    run_id: Optional[str],
    debug: bool,
    with_answer: bool,
    task: str,
    llm_policy: str,
    qa_generate: Optional[Callable[[str, str], str]],
    rerank: Optional[Dict[str, Any]],
    rerank_generate: Optional[Callable[[str, str], str]],
    backend: str,
    tkg_explain: bool,
    tkg_explain_topn: int,
    time_hints: Optional[Dict[str, Any]],
    qa_evidence_cap_l2: int,
    qa_evidence_cap_l4: int,
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    backend_norm = str(backend or "memory").strip().lower()
    if backend_norm not in ("memory", "tkg"):
        raise ValueError(f"unsupported backend: {backend}")

    executed_calls: List[Dict[str, Any]] = []

    # 1) Fact Search
    facts: List[Dict[str, Any]] = []
    fact_error: str | None = None
    fact_start = time.perf_counter()
    try:
        f = SearchFilters(
            tenant_id=str(tenant_id),
            user_id=list(user_tokens_norm),
            user_match=user_match_norm,  # type: ignore[arg-type]
            memory_domain=str(memory_domain),
            run_id=(str(run_id) if run_id else None),
            modality=["text"],
            memory_type=["semantic"],
            source=["locomo_text_pipeline"],
        )
        res = await store.search(q, topk=int(topk), filters=f, expand_graph=False)
        facts = _extract_fact_hits_v2(res.hits)
    except Exception as exc:
        facts = []
        fact_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    fact_latency_ms = (time.perf_counter() - fact_start) * 1000
    executed_calls.append(
        {"api": "fact_search", "count": len(facts), "latency_ms": fact_latency_ms, **({"error": fact_error} if fact_error else {})}
    )

    # 2) Event Search (tkg-first, with fallback to legacy episodic search when index is absent)
    events: List[Dict[str, Any]] = []
    event_error: str | None = None
    event_start = time.perf_counter()
    effective_backend = backend_norm
    try:
        if backend_norm == "tkg":
            f_ev = SearchFilters(
                tenant_id=str(tenant_id),
                user_id=list(user_tokens_norm),
                user_match=user_match_norm,  # type: ignore[arg-type]
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                modality=["text"],
                memory_type=["semantic"],
                source=["tkg_dialog_utterance_index_v1"],
            )
            res_ev = await store.search(q, topk=int(topk), filters=f_ev, expand_graph=False)
            events = []
            for hit in res_ev.hits:
                md = dict(hit.entry.metadata or {})
                ev_id = str(md.get("tkg_event_id") or "").strip()
                if not ev_id:
                    continue
                contents = list(hit.entry.contents or [])
                events.append(
                    {
                        "event_id": ev_id,
                        "event_ids": [ev_id],
                        "text": contents[0] if contents else "",
                        "score": float(hit.score or 0.0),
                        "source": "event_search",
                        "tkg_utterance_id": md.get("tkg_utterance_id"),
                    }
                )
        else:
            f_ev = SearchFilters(
                tenant_id=str(tenant_id),
                user_id=list(user_tokens_norm),
                user_match=user_match_norm,  # type: ignore[arg-type]
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                modality=["text"],
                memory_type=["episodic"],
            )
            res_ev = await store.search(q, topk=int(topk), filters=f_ev, expand_graph=True)
            events = _extract_event_hits_v2(res_ev.hits)
    except Exception as exc:
        events = []
        event_error = f"{type(exc).__name__}: {str(exc)[:160]}"
    event_latency_ms = (time.perf_counter() - event_start) * 1000
    executed_calls.append(
        {
            "api": ("utterance_search_tkg" if backend_norm == "tkg" else "event_search"),
            "count": len(events),
            "latency_ms": event_latency_ms,
            **({"error": event_error} if event_error else {}),
        }
    )

    # If tkg backend has no usable hits, fall back to legacy episodic search to avoid breaking userspace.
    if backend_norm == "tkg" and not events:
        effective_backend = "memory"
        fb_error: str | None = None
        fb_start = time.perf_counter()
        try:
            f_fb = SearchFilters(
                tenant_id=str(tenant_id),
                user_id=list(user_tokens_norm),
                user_match=user_match_norm,  # type: ignore[arg-type]
                memory_domain=str(memory_domain),
                run_id=(str(run_id) if run_id else None),
                modality=["text"],
                memory_type=["episodic"],
            )
            res_fb = await store.search(q, topk=int(topk), filters=f_fb, expand_graph=True)
            events = _extract_event_hits_v2(res_fb.hits)
        except Exception as exc:
            events = []
            fb_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        fb_latency_ms = (time.perf_counter() - fb_start) * 1000
        executed_calls.append(
            {"api": "event_search_fallback_memory", "count": len(events), "latency_ms": fb_latency_ms, **({"error": fb_error} if fb_error else {})}
        )

    # If effective backend is TKG, remap fact event_ids to TKG event ids (turn-index based).
    if effective_backend == "tkg" and facts:
        from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid

        def _turn_index_from_source_turn_id(v: object) -> Optional[int]:
            if v is None:
                return None
            if isinstance(v, int):
                return int(v) if v > 0 else None
            s = str(v).strip()
            if not s:
                return None
            if ":" in s:
                try:
                    n = int(s.split(":")[-1])
                    return n if n > 0 else None
                except Exception:
                    return None
            if "_" in s:
                try:
                    n = int(s.split("_")[-1])
                    return n if n > 0 else None
                except Exception:
                    return None
            try:
                n = int(s)
                return n if n > 0 else None
            except Exception:
                return None

        for f in facts:
            sample_id = str(f.get("sample_id") or "").strip()
            if not sample_id:
                continue
            legacy_ids = list(f.get("event_ids") or [])
            src_turn_ids = list(f.get("source_turn_ids") or [])
            tkg_event_ids: List[str] = []
            for stid in src_turn_ids:
                turn_i = _turn_index_from_source_turn_id(stid)
                if turn_i is None:
                    continue
                tkg_event_ids.append(generate_uuid("tkg.dialog.event", f"{tenant_id}|{sample_id}|{turn_i}"))
            if tkg_event_ids:
                f["_legacy_event_ids"] = legacy_ids
                f["event_ids"] = tkg_event_ids
                f["event_id"] = tkg_event_ids[0]

    # 3) Trace References (no extra search)
    refs: List[Dict[str, Any]] = []
    trace_start = time.perf_counter()
    seen_refs: set[str] = set()
    for f in facts:
        for eid in f.get("event_ids", []) or []:
            if eid and eid not in seen_refs:
                refs.append(
                    {
                        "event_ids": [eid],
                        "event_id": eid,
                        "fact_id": f.get("fact_id", ""),
                        "fact_type": f.get("fact_type"),
                        "text": f.get("text", ""),
                        "score": float(f.get("score", 0.0)) * 0.9,
                        "source": "reference_trace",
                    }
                )
                seen_refs.add(eid)
    trace_latency_ms = (time.perf_counter() - trace_start) * 1000
    executed_calls.append({"api": "trace_references", "count": len(refs), "latency_ms": trace_latency_ms})

    # 4) Fusion (fixed weights like benchmark)
    source_boost = {"fact_search": 2.0, "reference_trace": 1.8, "event_search": 1.0}
    all_hits: List[Dict[str, Any]] = []
    for item in facts:
        item["_final_score"] = float(item.get("score") or 0.0) * source_boost["fact_search"]
        all_hits.append(item)
    for item in refs:
        item["_final_score"] = float(item.get("score") or 0.0) * source_boost["reference_trace"]
        all_hits.append(item)
    for item in events:
        item["_final_score"] = float(item.get("score") or 0.0) * source_boost["event_search"]
        all_hits.append(item)

    all_hits.sort(key=lambda x: float(x.get("_final_score", 0.0)), reverse=True)

    deduped: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for hit in all_hits:
        eid = hit.get("event_id")
        fid = hit.get("fact_id")
        key = f"event:{eid}" if eid else f"fact:{fid}"
        if key and key not in seen_ids:
            deduped.append(hit)
            seen_ids.add(key)

    retrieval_latency_ms = (time.perf_counter() - total_start) * 1000
    evidence_items: List[Dict[str, Any]] = list(deduped)

    # 4.2) Optional TKG explain enrichment (does not change benchmark prompt fields by default).
    if backend_norm == "tkg" and effective_backend == "tkg" and tkg_explain and evidence_items:
        explain_fn = getattr(store, "graph_explain_event_evidence", None)
        if callable(explain_fn):
            topn = max(0, min(int(tkg_explain_topn), len(evidence_items)))
            started = time.perf_counter()
            enriched = 0
            for item in evidence_items[:topn]:
                eid = str(item.get("event_id") or "").strip()
                if not eid:
                    continue
                try:
                    ex = await explain_fn(tenant_id=str(tenant_id), event_id=eid)
                except Exception:
                    continue
                item["tkg_explain"] = ex
                enriched += 1
            executed_calls.append(
                {
                    "api": "tkg_explain_event_evidence",
                    "count": int(enriched),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )

    # 4.5) Optional rerank (benchmark-aligned)
    rerank_latency_ms = 0.0
    rerank_cfg = rerank if isinstance(rerank, dict) else {}
    rerank_enabled = bool(rerank_cfg.get("enabled", False))
    if rerank_enabled and evidence_items:
        from modules.memory.application.rerank_dialog_v1 import (
            EvidenceType,
            RerankConfig,
            RetrievalCandidate,
            build_llm_client_from_fn,
            create_rerank_service,
        )

        original_top_n = int(rerank_cfg.get("top_n", min(int(topk), 20)) or min(int(topk), 20))
        original_top_n = max(1, min(int(topk), original_top_n))
        rerank_pool_size = int(rerank_cfg.get("rerank_pool_size", 30) or 30)
        pool_size = max(int(original_top_n), int(rerank_pool_size))

        candidates_raw = evidence_items[:pool_size]
        candidates: List[RetrievalCandidate] = []
        for e in candidates_raw:
            candidates.append(
                RetrievalCandidate(
                    query_text=q,
                    evidence_text=str(e.get("text") or ""),
                    evidence_type=EvidenceType.from_source(str(e.get("source") or "")),
                    event_id=str(e.get("event_id") or ""),
                    base_score=float(e.get("_final_score") or e.get("score") or 0.0),
                    fact_id=(str(e.get("fact_id")) if e.get("fact_id") else None),
                    timestamp=(str(e.get("timestamp")) if e.get("timestamp") else None),
                    importance=str(e.get("importance") or "medium"),
                    metadata=dict(e),
                )
            )

        llm_client = None
        if str(rerank_cfg.get("model", "noop")) == "llm":
            if rerank_generate is not None:
                llm_client = build_llm_client_from_fn(rerank_generate)
            if llm_client is None and str(llm_policy or "best_effort").lower() == "require":
                raise RuntimeError("Rerank LLM is not configured (llm_policy=require).")

        call_cfg = RerankConfig(
            enabled=True,
            model=str(rerank_cfg.get("model", "noop")),
            top_n=int(pool_size),
            weight_base=float(rerank_cfg.get("weight_base", 0.4)),
            weight_rerank=float(rerank_cfg.get("weight_rerank", 0.5)),
            weight_type=float(rerank_cfg.get("weight_type", 0.1)),
            type_bias=dict(rerank_cfg.get("type_bias") or RerankConfig.default().type_bias),
        )
        service = create_rerank_service(call_cfg, llm_client=llm_client)

        rerank_start = time.perf_counter()
        results = service.rerank(q, candidates)
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000

        reranked: List[Dict[str, Any]] = []
        for r in results:
            item = dict(r.candidate.metadata)
            item["_rerank_score"] = float(r.rerank_score)
            item["_final_score"] = float(r.final_score)
            item["_rank"] = int(r.rank)
            reranked.append(item)
        evidence_items = reranked[:original_top_n]
        executed_calls.append(
            {
                "api": "rerank",
                "before": len(candidates),
                "after": len(evidence_items),
                "latency_ms": rerank_latency_ms,
                "model": str(rerank_cfg.get("model", "noop")),
            }
        )

    evidence_ids = [h.get("event_id") for h in evidence_items if h.get("event_id")]
    out: Dict[str, Any] = {
        "strategy": "dialog_v1",
        "query": q,
        "evidence": evidence_ids[: int(topk)],
        "evidence_details": evidence_items[: int(topk)],
    }

    qa_latency_ms = 0.0
    if with_answer:
        answer_text = ""
        try:
            from modules.memory.application.qa_dialog_v1 import QA_SYSTEM_PROMPT_GENERAL, build_qa_user_prompt
            from modules.memory.application.qa_longmemeval import (
                build_longmemeval_prompts,
                extract_question_date_from_time_hints,
                should_use_longmemeval_prompt,
            )
        except Exception as exc:
            raise RuntimeError(f"qa_prompt_import_failed: {exc}") from exc

        if qa_generate is None and str(llm_policy or "best_effort").lower() == "require":
            raise RuntimeError("QA LLM is not configured (llm_policy=require).")

        qa_limit = int(topk)
        task_norm = str(task or "").upper()
        if task_norm == "L2":
            qa_limit = min(qa_limit, int(qa_evidence_cap_l2))
        elif task_norm == "L4":
            qa_limit = min(qa_limit, int(qa_evidence_cap_l4))
        evidence_for_qa = evidence_items[: int(qa_limit)]

        if not evidence_for_qa:
            answer_text = "insufficient information"
        elif qa_generate is None:
            answer_text = "Unable to answer in dummy mode."
        else:
            question_date = extract_question_date_from_time_hints(time_hints)
            system_prompt = QA_SYSTEM_PROMPT_GENERAL
            if should_use_longmemeval_prompt(memory_domain=str(memory_domain), task=str(task or "")) and question_date:
                system_prompt, user_prompt = build_longmemeval_prompts(
                    question=q,
                    question_date=str(question_date),
                    evidence=list(evidence_for_qa),
                    task=str(task or ""),
                )
            else:
                user_prompt = build_qa_user_prompt(query=q, task=str(task or "GENERAL"), evidence=list(evidence_for_qa))
            qa_start = time.perf_counter()
            try:
                answer_text = str(await asyncio.to_thread(qa_generate, system_prompt, user_prompt)).strip()
            except Exception:
                answer_text = "Unable to generate answer due to an error."
            qa_latency_ms = (time.perf_counter() - qa_start) * 1000

        out["answer"] = answer_text

    if debug:
        plan = {
            "api": "search_3way",
            "features": {},
            "thought": "No planner, fixed 3-way search.",
            "latency_ms": retrieval_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
        }
        if rerank_enabled:
            plan["rerank_latency_ms"] = rerank_latency_ms
        if with_answer:
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            plan["qa_latency_ms"] = qa_latency_ms
            plan["total_latency_ms"] = total_latency_ms

        out["debug"] = {
            "plan": plan,
            "executed_calls": executed_calls,
            "evidence_count": len(evidence_items[: int(topk)]),
        }
    return out


async def _retrieval_dialog_v2(
    *,
    store: MemoryPort,
    tenant_id: str,
    user_tokens_norm: List[str],
    user_match_norm: str,
    query: str,
    topk: int,
    memory_domain: str,
    run_id: Optional[str],
    debug: bool,
    with_answer: bool,
    task: str,
    llm_policy: str,
    qa_generate: Optional[Callable[[str, str], str]],
    tkg_explain: bool,
    candidate_k: int,
    seed_topn: int,
    e_vec_oversample: int,
    graph_cap: int,
    rrf_k: int,
    qa_evidence_cap_l2: int,
    qa_evidence_cap_l4: int,
    enable_event_route: bool,
    enable_evidence_route: bool,
    enable_knowledge_route: bool,
    enable_entity_route: bool,
    enable_time_route: bool,
    entity_hints: Optional[List[str]],
    time_hints: Optional[Dict[str, Any]],
    dialog_v2_weights: Optional[Dict[str, float]],
    dialog_v2_reranker: Optional[Dict[str, Any]],
    test_ablation: _DialogV2TestAblationContract,
    output_strategy: str = "dialog_v2",
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    q = str(query or "").strip()
    if not q:
        raise ValueError("query is required")

    K = max(1, int(candidate_k))
    seed_k = max(1, int(seed_topn))
    oversample = max(1, int(e_vec_oversample))

    weights = dict(_DIALOG_V2_DEFAULT_WEIGHTS)
    if isinstance(dialog_v2_weights, dict):
        for k, v in dialog_v2_weights.items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue
    # Backward-compatible alias: allow legacy "graph" weight to override event_vec.
    if "event_vec" not in weights and "graph" in weights:
        try:
            weights["event_vec"] = float(weights.get("graph") or 0.0)
        except Exception:
            pass
    if isinstance(dialog_v2_weights, dict) and "graph" in dialog_v2_weights and "event_vec" not in dialog_v2_weights:
        try:
            weights["event_vec"] = float(dialog_v2_weights.get("graph") or weights.get("event_vec") or 0.0)
        except Exception:
            pass
    weights.pop("graph", None)

    try:
        from modules.memory.application.config import load_memory_config, resolve_dialog_v2_reranker_settings

        _cfg = load_memory_config()
        reranker_cfg = resolve_dialog_v2_reranker_settings(
            _cfg,
            override=(dict(dialog_v2_reranker) if isinstance(dialog_v2_reranker, dict) else None),
        )
    except Exception:
        reranker_cfg = {
            "enabled": False,
            "engine": "native",
            "llm_kind": "reranker",
            "stage": "preselect",
            "score_mode": "rerank_only",
            "rerank_pool_size": 80,
            "weight_base": 0.25,
            "weight_rerank": 0.75,
            "instruct": "Given a user query, rank the passages by how directly they help answer the query.",
            "weight_type": 0.1,
            "type_bias": {"fact": 0.15, "reference": 0.10, "event": 0.0, "unknown": 0.0},
        }

    ablation = test_ablation if isinstance(test_ablation, _DialogV2TestAblationContract) else _DialogV2TestAblationContract()
    effective_enable_event_route = ablation.route_enabled("event", enable_event_route)
    effective_enable_evidence_route = ablation.route_enabled("evidence", enable_evidence_route)
    effective_enable_knowledge_route = ablation.route_enabled("knowledge", enable_knowledge_route)
    effective_enable_entity_route = ablation.route_enabled("entity", enable_entity_route)
    effective_enable_time_route = ablation.route_enabled("time", enable_time_route)
    if not ablation.signal_enabled("timestamp"):
        weights["recency"] = 0.0
    if not ablation.signal_enabled("recency"):
        weights["recency"] = 0.0
    if not ablation.signal_enabled("event"):
        weights["event_vec"] = 0.0
    if not ablation.signal_enabled("evidence"):
        weights["vec"] = 0.0
    if not ablation.signal_enabled("knowledge"):
        weights["knowledge"] = 0.0
    if not ablation.signal_enabled("entity"):
        weights["entity"] = 0.0
    if not ablation.signal_enabled("time"):
        weights["time"] = 0.0

    executed_calls: List[Dict[str, Any]] = []
    match_evaluator = _DialogV2MatchEvaluator(q)
    shared_query_vector: Optional[List[float]] = None
    if effective_enable_event_route or effective_enable_evidence_route or effective_enable_knowledge_route:
        shared_query_vector = await _compute_shared_query_vector(
            store=store,
            query=q,
            tenant_id=tenant_id,
        )

    candidate_pool = _DialogV2CandidatePool()
    candidates = candidate_pool.candidates
    unmapped_evidence: List[Dict[str, Any]] = []

    def _source_candidate_id(prefix: str, *parts: Any) -> str:
        cleaned: List[str] = [str(prefix or "").strip() or "candidate"]
        for part in parts:
            value = str(part or "").strip()
            if value:
                cleaned.append(value)
        return "::".join(cleaned)

    async def _route_event_vec() -> Dict[str, Any]:
        e_event_vec_ids: List[str] = []
        candidate_ops: List[Dict[str, Any]] = []
        e_event_vec_error: Optional[str] = None
        start_event_vec = time.perf_counter()
        if effective_enable_event_route:
            try:
                # Backward compat: include legacy event index source so historical vectors remain visible.
                f_ev = SearchFilters(
                    tenant_id=str(tenant_id),
                    user_id=list(user_tokens_norm),
                    user_match=user_match_norm,  # type: ignore[arg-type]
                    memory_domain=str(memory_domain),
                    modality=["text"],
                    memory_type=["semantic"],
                    source=[TKG_DIALOG_EVENT_INDEX_SOURCE_V1, "dialog_session_write_v1"],
                )
                res_ev = await _store_search_with_optional_query_vector(
                    store,
                    q,
                    topk=int(K),
                    filters=f_ev,
                    expand_graph=False,
                    query_vector=shared_query_vector,
                )
                for hit in res_ev.hits:
                    md = dict(hit.entry.metadata or {})
                    tkg_eid = str(md.get("tkg_event_id") or md.get("node_id") or "").strip() or None
                    logical_eid = str(md.get("event_id") or md.get("logical_event_id") or "").strip() or None
                    if not tkg_eid and not logical_eid:
                        continue
                    text = hit.entry.contents[0] if hit.entry.contents else None
                    ts = md.get("timestamp_iso") or md.get("t_start") or md.get("t_abs_start") or md.get("created_at")
                    candidate_ops.append(
                        {
                            "event_id": tkg_eid or logical_eid or "",
                            "route": "E_event_vec",
                            "score": float(hit.score or 0.0),
                            "text": str(text) if text else None,
                            "timestamp": (str(ts) if ts and ablation.signal_enabled("timestamp") else None),
                            "reason": "event_index",
                            "tkg_event_id": tkg_eid,
                            "logical_event_id": logical_eid,
                            "evaluation_event_ids": ([logical_eid] if logical_eid else []),
                        }
                    )
                    e_event_vec_ids.append(str(tkg_eid or logical_eid))
            except Exception as exc:
                e_event_vec_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            e_event_vec_error = "event_route_disabled"
        return {
            "ops": candidate_ops,
            "unmapped_evidence": [],
            "executed_call": {
                "api": "event_search_event_vec",
                "count": len(list(dict.fromkeys(e_event_vec_ids))),
                "latency_ms": (time.perf_counter() - start_event_vec) * 1000,
                **({"error": e_event_vec_error} if e_event_vec_error else {}),
            },
        }

    async def _route_utterance_vec() -> Dict[str, Any]:
        e_vec_error: Optional[str] = None
        e_vec_unique: set[str] = set()
        vec_hits = 0
        candidate_ops: List[Dict[str, Any]] = []
        local_unmapped: List[Dict[str, Any]] = []
        start_vec = time.perf_counter()
        if effective_enable_evidence_route:
            try:
                f_vec = SearchFilters(
                    tenant_id=str(tenant_id),
                    user_id=list(user_tokens_norm),
                    user_match=user_match_norm,  # type: ignore[arg-type]
                    memory_domain=str(memory_domain),
                    run_id=(str(run_id) if run_id else None),
                    modality=["text"],
                    memory_type=["semantic"],
                    source=["tkg_dialog_utterance_index_v1"],
                )
                vec_topk = max(1, K * oversample)
                res_vec = await _store_search_with_optional_query_vector(
                    store,
                    q,
                    topk=int(vec_topk),
                    filters=f_vec,
                    expand_graph=False,
                    query_vector=shared_query_vector,
                )
                for hit in res_vec.hits:
                    vec_hits += 1
                    md = dict(hit.entry.metadata or {})
                    tkg_eid = str(md.get("tkg_event_id") or "").strip() or None
                    logical_eid = str(md.get("event_id") or "").strip() or None
                    extra_ids = md.get("tkg_event_ids")
                    tkg_ids: List[str] = []
                    if tkg_eid:
                        tkg_ids.append(tkg_eid)
                    if isinstance(extra_ids, (list, tuple)):
                        for item in extra_ids:
                            val = str(item or "").strip()
                            if val:
                                tkg_ids.append(val)
                    if tkg_ids:
                        tkg_ids = list(dict.fromkeys(tkg_ids))
                    text = hit.entry.contents[0] if hit.entry.contents else None
                    ts = md.get("timestamp") or md.get("timestamp_iso") or md.get("created_at")
                    utterance_id = str(md.get("tkg_utterance_id") or getattr(hit, "id", None) or "").strip() or None
                    eval_ids = [logical_eid] if logical_eid else []
                    if ablation.should_use_source_native_candidates():
                        source_candidate_id = _source_candidate_id("utterance", utterance_id or f"vec_{vec_hits}")
                        candidate_ops.append(
                            {
                                "event_id": source_candidate_id,
                                "route": "E_vec",
                                "score": float(hit.score or 0.0),
                                "text": str(text) if text else None,
                                "timestamp": (str(ts) if ts and ablation.signal_enabled("timestamp") else None),
                                "reason": "utterance_index",
                                "extras": {"utterance_id": utterance_id or getattr(hit, "id", None)},
                                "evaluation_event_ids": eval_ids,
                                "force_raw_key": True,
                            }
                        )
                        continue
                    if not tkg_ids and not logical_eid:
                        local_unmapped.append(
                            {
                                "event_id": None,
                                "tkg_event_id": None,
                                "score": float(hit.score or 0.0),
                                "source": "utterance_search_unmapped",
                                "text": str(text) if text else "",
                                "timestamp": (str(ts) if ts and ablation.signal_enabled("timestamp") else None),
                                "unmapped_to_event": True,
                                "evidence_id": md.get("tkg_utterance_id") or getattr(hit, "id", None),
                                "utterance_id": md.get("tkg_utterance_id"),
                            }
                        )
                        continue
                    if tkg_ids:
                        for tid in tkg_ids:
                            e_vec_unique.add(tid)
                            candidate_ops.append(
                                {
                                    "event_id": tid if (len(tkg_ids) > 1 or not logical_eid) else (logical_eid or tid),
                                    "route": "E_vec",
                                    "score": float(hit.score or 0.0),
                                    "text": str(text) if text else None,
                                    "timestamp": (str(ts) if ts and ablation.signal_enabled("timestamp") else None),
                                    "reason": "utterance_index",
                                    "extras": {"utterance_id": getattr(hit, "id", None)},
                                    "logical_event_id": (logical_eid if len(tkg_ids) == 1 else None),
                                    "tkg_event_id": tid,
                                    "evaluation_event_ids": eval_ids,
                                }
                            )
                    else:
                        candidate_ops.append(
                            {
                                "event_id": logical_eid or "",
                                "route": "E_vec",
                                "score": float(hit.score or 0.0),
                                "text": str(text) if text else None,
                                "timestamp": (str(ts) if ts and ablation.signal_enabled("timestamp") else None),
                                "reason": "utterance_index",
                                "extras": {"utterance_id": getattr(hit, "id", None)},
                                "logical_event_id": logical_eid,
                                "tkg_event_id": None,
                                "evaluation_event_ids": eval_ids,
                            }
                        )
            except Exception as exc:
                e_vec_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            e_vec_error = "evidence_route_disabled"
        return {
            "ops": candidate_ops,
            "unmapped_evidence": local_unmapped,
            "executed_call": {
                "api": "event_search_utterance_vec",
                "hits": vec_hits,
                "unique_events": len(e_vec_unique),
                "latency_ms": (time.perf_counter() - start_vec) * 1000,
                **({"error": e_vec_error} if e_vec_error else {}),
            },
        }

    async def _route_knowledge() -> Dict[str, Any]:
        k_error: Optional[str] = None
        k_hits = 0
        candidate_ops: List[Dict[str, Any]] = []
        start_k = time.perf_counter()
        if effective_enable_knowledge_route:
            try:
                f_k = SearchFilters(
                    tenant_id=str(tenant_id),
                    user_id=list(user_tokens_norm),
                    user_match=user_match_norm,  # type: ignore[arg-type]
                    memory_domain=str(memory_domain),
                    run_id=(str(run_id) if run_id else None),
                    modality=["text"],
                    memory_type=["semantic"],
                    source=["locomo_text_pipeline"],
                )
                res_k = await _store_search_with_optional_query_vector(
                    store,
                    q,
                    topk=int(K),
                    filters=f_k,
                    expand_graph=False,
                    query_vector=shared_query_vector,
                )
                facts = _extract_fact_hits_v2(res_k.hits)
                for fact in facts:
                    k_hits += 1
                    fact_id = str(fact.get("fact_id") or f"fact_{k_hits}").strip()
                    fact_event_ids = [str(eid or "").strip() for eid in list(fact.get("event_ids") or []) if str(eid or "").strip()]
                    if ablation.should_use_source_native_candidates():
                        candidate_ops.append(
                            {
                                "event_id": _source_candidate_id("fact", fact_id),
                                "route": "K_vec",
                                "score": float(fact.get("score") or 0.0),
                                "text": str(fact.get("text") or "") if fact.get("text") else None,
                                "reason": "fact_search",
                                "extras": {"fact_id": fact.get("fact_id")},
                                "evaluation_event_ids": fact_event_ids,
                                "force_raw_key": True,
                            }
                        )
                        continue
                    for eid in fact.get("event_ids") or []:
                        ev_id = str(eid or "").strip()
                        if not ev_id:
                            continue
                        candidate_ops.append(
                            {
                                "event_id": ev_id,
                                "route": "K_vec",
                                "score": float(fact.get("score") or 0.0),
                                "text": str(fact.get("text") or "") if fact.get("text") else None,
                                "reason": "fact_search",
                                "extras": {"fact_id": fact.get("fact_id")},
                                "logical_event_id": ev_id,
                                "evaluation_event_ids": fact_event_ids,
                            }
                        )
            except Exception as exc:
                k_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            k_error = "knowledge_route_disabled"
        return {
            "ops": candidate_ops,
            "unmapped_evidence": [],
            "executed_call": {
                "api": "fact_search",
                "count": int(k_hits),
                "latency_ms": (time.perf_counter() - start_k) * 1000,
                **({"error": k_error} if k_error else {}),
            },
        }

    async def _route_entity() -> Dict[str, Any]:
        en_error: Optional[str] = None
        en_labels: List[str] = []
        en_events: int = 0
        candidate_ops: List[Dict[str, Any]] = []
        start_en = time.perf_counter()
        if effective_enable_entity_route:
            import re

            labels: List[str] = []
            if entity_hints:
                labels.extend([str(x).strip() for x in entity_hints if str(x).strip()])
            labels.extend([m for m in re.findall(r"\bface\d+\b", q, flags=re.I)])
            labels.extend([m for m in re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", q)])
            en_labels = list(dict.fromkeys([x for x in labels if x]))

            if en_labels:
                resolve_fn = getattr(store, "graph_resolve_entities", None)
                entity_ids: List[str] = []
                entity_scores: Dict[str, float] = {}
                if callable(resolve_fn):
                    resolve_tasks = [
                        asyncio.create_task(resolve_fn(tenant_id=tenant_id, name=name, entity_type="PERSON", limit=5))
                        for name in en_labels
                    ]
                    resolve_results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
                    for hits in resolve_results:
                        if isinstance(hits, Exception):
                            continue
                        for h in hits or []:
                            ent_id = str(h.get("entity_id") or "").strip()
                            if not ent_id:
                                continue
                            entity_ids.append(ent_id)
                            try:
                                entity_scores[ent_id] = max(float(entity_scores.get(ent_id, 0.0)), float(h.get("score") or 0.0))
                            except Exception:
                                entity_scores.setdefault(ent_id, 0.0)
                if not entity_ids:
                    try:
                        from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid

                        def _stable_entity_id(label: str) -> str:
                            key = f"tenant={tenant_id}|domain={memory_domain}|users={'|'.join([str(x) for x in user_tokens_norm])}|speaker={label}"
                            return generate_uuid("tkg.dialog.entity", key)

                        for name in en_labels:
                            entity_ids.append(_stable_entity_id(name))
                            entity_scores.setdefault(entity_ids[-1], 1.0)
                    except Exception:
                        entity_ids = []

                if entity_ids:
                    entity_ids = list(dict.fromkeys(entity_ids))
                    list_events_fn = getattr(store, "graph_list_events", None)
                    if callable(list_events_fn):
                        event_tasks = [
                            asyncio.create_task(list_events_fn(tenant_id=tenant_id, entity_id=ent_id, limit=K))
                            for ent_id in entity_ids
                        ]
                        event_results = await asyncio.gather(*event_tasks, return_exceptions=True)
                        for ent_id, evs in zip(entity_ids, event_results):
                            if isinstance(evs, Exception):
                                continue
                            for ev in evs or []:
                                eid = str(ev.get("id") or ev.get("event_id") or "").strip()
                                if not eid:
                                    continue
                                en_events += 1
                                score = float(entity_scores.get(ent_id, 0.8) or 0.8)
                                candidate_ops.append(
                                    {
                                        "event_id": eid,
                                        "route": "EN",
                                        "score": score,
                                        "reason": f"entity:{ent_id}",
                                        "tkg_event_id": eid,
                                    }
                                )
                    else:
                        en_error = "graph_list_events_unavailable"
        else:
            en_error = "entity_route_disabled"

        return {
            "ops": candidate_ops,
            "unmapped_evidence": [],
            "executed_call": {
                "api": "entity_route",
                "labels": en_labels,
                "events": int(en_events),
                "latency_ms": (time.perf_counter() - start_en) * 1000,
                **({"error": en_error} if en_error else {}),
            },
        }

    async def _route_time() -> Dict[str, Any]:
        t_error: Optional[str] = None
        t_events: int = 0
        t_reason: Optional[str] = None
        start_iso: Optional[str] = None
        end_iso: Optional[str] = None
        candidate_ops: List[Dict[str, Any]] = []
        start_t = time.perf_counter()
        if effective_enable_time_route:
            start_iso, end_iso, t_reason = _parse_time_window(q, time_hints=time_hints)
            if start_iso or end_iso:
                list_ts_fn = getattr(store, "graph_list_timeslices_range", None)
                if callable(list_ts_fn):
                    try:
                        ts_items = await list_ts_fn(
                            tenant_id=tenant_id,
                            start_iso=start_iso,
                            end_iso=end_iso,
                            kind="dialog_session",
                            limit=K,
                        )
                        for ts in ts_items or []:
                            for eid in ts.get("event_ids") or []:
                                ev_id = str(eid or "").strip()
                                if not ev_id:
                                    continue
                                t_events += 1
                                candidate_ops.append(
                                    {
                                        "event_id": ev_id,
                                        "route": "T",
                                        "score": 0.7,
                                        "reason": f"timeslice:{ts.get('id')}",
                                        "tkg_event_id": ev_id,
                                        "evaluation_event_ids": [],
                                    }
                                )
                    except Exception as exc:
                        t_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                else:
                    t_error = "timeslice_range_unavailable"
            else:
                t_error = "no_time_hint"
        else:
            t_error = "time_route_disabled"
        return {
            "ops": candidate_ops,
            "unmapped_evidence": [],
            "executed_call": {
                "api": "time_route",
                "time_window": {"start": (start_iso if enable_time_route else None), "end": (end_iso if enable_time_route else None)},
                "events": int(t_events),
                "latency_ms": (time.perf_counter() - start_t) * 1000,
                "reason": t_reason,
                **({"error": t_error} if t_error else {}),
            },
        }

    route_results = await asyncio.gather(
        _route_event_vec(),
        _route_utterance_vec(),
        _route_knowledge(),
        _route_entity(),
        _route_time(),
    )
    for route_result in route_results:
        for op in route_result.get("ops") or []:
            candidate_pool.touch(**op)
        unmapped_evidence.extend(list(route_result.get("unmapped_evidence") or []))
        executed_calls.append(dict(route_result.get("executed_call") or {}))

    # --- Candidate scoring (route support + match fidelity + recency) ---
    def _ranked(score_key: str) -> List[str]:
        items = [eid for eid, c in candidates.items() if float(c.get(score_key) or 0.0) > 0.0]
        items.sort(key=lambda eid: float(candidates[eid].get(score_key) or 0.0), reverse=True)
        return items

    route_lists = {
        "event_vec": _ranked("score_event_vec"),
        "vec": _ranked("score_vec"),
        "knowledge": _ranked("score_k"),
        "entity": _ranked("score_en"),
        "time": _ranked("score_t"),
    }
    route_score_keys = {
        "event_vec": "score_event_vec",
        "vec": "score_vec",
        "knowledge": "score_k",
        "entity": "score_en",
        "time": "score_t",
    }
    route_rank_maps = {
        route: {eid: rank for rank, eid in enumerate(items, start=1)} for route, items in route_lists.items()
    }
    route_norm_score_maps: Dict[str, Dict[str, float]] = {}
    for route, items in route_lists.items():
        score_key = route_score_keys[route]
        route_norm_score_maps[route] = _normalize_scores(
            {eid: float(candidates[eid].get(score_key) or 0.0) for eid in items}
        )
    score_blend_alpha = max(0.0, min(1.0, float(weights.get("score_blend_alpha", 1.0) or 1.0)))

    for c in candidates.values():
        c["_rrf_score"] = 0.0
        c["_route_support_score"] = 0.0

    for route, items in route_lists.items():
        weight = float(weights.get(route, 1.0))
        for rank, eid in enumerate(items, start=1):
            rank_term = 1.0 / (float(rrf_k) + rank)
            blended_term = (
                score_blend_alpha * rank_term
                + (1.0 - score_blend_alpha) * float(route_norm_score_maps.get(route, {}).get(eid, 0.0))
            )
            candidates[eid]["_rrf_score"] = float(candidates[eid].get("_rrf_score") or 0.0) + (weight * rank_term)
            candidates[eid]["_route_support_score"] = (
                float(candidates[eid].get("_route_support_score") or 0.0) + (weight * blended_term)
            )

    for c in candidates.values():
        match_score, match_details = match_evaluator.score_candidate(text=str(c.get("text") or ""))
        rec = (_recency_score_from_iso(c.get("timestamp")) if ablation.signal_enabled("recency") else 0.0)
        c["_match_fidelity_score"] = float(match_score)
        c["_match_details"] = dict(match_details)
        c["_recency_score"] = rec
        c["_preselect_score"] = (
            float(c.get("_route_support_score") or 0.0)
            + float(weights.get("match", 1.0)) * float(match_score)
            + float(weights.get("recency", 0.0)) * float(rec)
        )
        # Keep backward-compatible alias for existing debug consumers.
        c["_base_score"] = float(c.get("_preselect_score") or 0.0)
        c["_selection_score"] = float(c.get("_preselect_score") or 0.0)
    base_sorted_all = sorted(candidates.keys(), key=lambda eid: float(candidates[eid].get("_preselect_score") or 0.0), reverse=True)
    base_rank_map = {eid: rank for rank, eid in enumerate(base_sorted_all, start=1)}

    reranker_applied = False
    reranker_engine = str(reranker_cfg.get("engine") or "noop").strip().lower()
    reranker_llm_kind = str(reranker_cfg.get("llm_kind") or "reranker").strip() or "reranker"
    reranker_stage = str(reranker_cfg.get("stage") or "preselect").strip().lower() or "preselect"
    reranker_score_mode = str(reranker_cfg.get("score_mode") or "rerank_only").strip().lower() or "rerank_only"
    reranker_pool_size = max(1, min(len(base_sorted_all), int(reranker_cfg.get("rerank_pool_size") or len(base_sorted_all) or 1)))
    reranker_model_provider: Optional[str] = None
    reranker_model_name: Optional[str] = None
    reranker_error: Optional[str] = None
    rerank_latency_ms = 0.0
    reranker_request_id: Optional[str] = None
    reranker_usage_total_tokens: Optional[int] = None
    reranker_input_ids: List[str] = []
    reranker_rank_map: Dict[str, int] = {}
    reranker_input_set: set[str] = set()

    if bool(reranker_cfg.get("enabled")) and reranker_engine == "native" and base_sorted_all and reranker_stage == "preselect":
        try:
            from modules.memory.application.config import get_llm_selection, load_memory_config
            from modules.memory.application.reranker_adapter import RerankDocument, build_reranker_from_config

            cfg = load_memory_config()
            llm_sel = get_llm_selection(cfg, reranker_llm_kind)
            reranker_model_provider = str(llm_sel.get("provider") or "").strip() or None
            reranker_model_name = str(llm_sel.get("model") or "").strip() or None
            client = build_reranker_from_config(reranker_llm_kind)
            if client is None:
                reranker_error = f"native_reranker_unavailable:{reranker_llm_kind}"
            else:
                reranker_input_ids = list(base_sorted_all[: int(reranker_pool_size)])
                reranker_input_set = set(reranker_input_ids)
                rerank_documents = [
                    RerankDocument(
                        document_id=str(eid),
                        text=_build_dialog_v2_rerank_text(candidates[eid], candidate_id=str(eid)),
                        metadata={
                            "logical_event_id": candidates[eid].get("logical_event_id"),
                            "tkg_event_id": candidates[eid].get("tkg_event_id"),
                            "source": candidates[eid].get("primary_source"),
                        },
                    )
                    for eid in reranker_input_ids
                ]
                started = time.perf_counter()
                rerank_response = await asyncio.to_thread(
                    client.rerank,
                    query=q,
                    documents=rerank_documents,
                    top_n=len(rerank_documents),
                    instruct=str(reranker_cfg.get("instruct") or "").strip() or None,
                )
                rerank_latency_ms = (time.perf_counter() - started) * 1000
                reranker_request_id = rerank_response.request_id
                reranker_usage_total_tokens = rerank_response.usage_total_tokens
                raw_scores: Dict[str, float] = {}
                for result in rerank_response.results:
                    eid = str(result.document_id or "").strip()
                    if not eid or eid not in reranker_input_set:
                        continue
                    reranker_rank_map[eid] = int(result.rank)
                    raw_scores[eid] = float(result.relevance_score or 0.0)
                    candidates[eid]["_rerank_raw_score"] = float(result.relevance_score or 0.0)
                normalized_scores = _normalize_scores(raw_scores)
                unique_raw = {round(v, 8) for v in raw_scores.values()}
                base_pool_scores = {eid: float(candidates[eid].get("_preselect_score") or 0.0) for eid in reranker_input_ids}
                normalized_base_scores = _normalize_scores(base_pool_scores)
                for eid in reranker_input_ids:
                    rerank_rank = reranker_rank_map.get(eid)
                    rank_score = _rank_to_unit_interval(rerank_rank or len(reranker_input_ids), len(reranker_input_ids))
                    if len(unique_raw) <= 1:
                        rerank_score = rank_score
                    else:
                        rerank_score = float(normalized_scores.get(eid, rank_score))
                    candidates[eid]["_rerank_score"] = float(rerank_score)
                    base_score = float(normalized_base_scores.get(eid, 0.0))
                    candidates[eid]["_base_norm_score"] = base_score
                    if reranker_score_mode == "rerank_only":
                        selection_score = float(rerank_score)
                    elif reranker_score_mode == "base_only":
                        selection_score = float(base_score)
                    else:
                        selection_score = (
                            float(reranker_cfg.get("weight_base") or 0.25) * float(base_score)
                            + float(reranker_cfg.get("weight_rerank") or 0.75) * float(rerank_score)
                        )
                    candidates[eid]["_selection_score"] = float(selection_score)
                if reranker_input_ids:
                    reranker_applied = True
        except Exception as exc:
            reranker_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    selection_sorted_all = list(base_sorted_all)
    if reranker_applied and reranker_stage == "preselect" and reranker_input_ids:
        reranked_pool = sorted(
            reranker_input_ids,
            key=lambda eid: float(candidates[eid].get("_selection_score") or 0.0),
            reverse=True,
        )
        selection_sorted_all = reranked_pool + [eid for eid in base_sorted_all if eid not in reranker_input_set]

    # --- Candidate selection (score-ordered; no hard route reservation) ---
    graph_cap = max(0, min(int(graph_cap), K))
    selected = list(selection_sorted_all[:K])
    selected_set: set[str] = set(selected)
    selection_reason_map: Dict[str, str] = {
        eid: ("rerank_preselect" if reranker_applied and eid in reranker_input_set else "score_order")
        for eid in selected
    }
    selected_before_final = sorted(
        selected,
        key=lambda eid: float(candidates[eid].get("_selection_score") or candidates[eid].get("_preselect_score") or 0.0),
        reverse=True,
    )
    selected_rank_before_final_map = {eid: rank for rank, eid in enumerate(selected_before_final, start=1)}

    # --- Explain expansion (single-hop evidence) ---
    explain_fn = getattr(store, "graph_explain_event_evidence", None)
    explain_count = 0
    if tkg_explain and not ablation.should_disable_explain() and callable(explain_fn) and selected:
        seeds = selected_before_final[:seed_k]
        started = time.perf_counter()
        for eid in seeds:
            explain_event_id = str((candidates.get(eid) or {}).get("tkg_event_id") or "").strip()
            if not explain_event_id:
                continue
            try:
                ex = await explain_fn(tenant_id=str(tenant_id), event_id=explain_event_id)
            except Exception:
                continue
            candidates[eid]["tkg_explain"] = ex
            explain_count += 1
            try:
                sig = 0.0
                sig += 0.3 * min(len(ex.get("utterances") or []), 3)
                sig += 0.4 * min(len(ex.get("knowledge") or []), 3)
                sig += 0.2 * min(len(ex.get("entities") or []), 3)
                if ablation.signal_enabled("time") and ablation.signal_enabled("timestamp"):
                    sig += 0.1 * min(len(ex.get("timeslices") or []), 2)
                candidates[eid]["_graph_signal"] = sig
            except Exception:
                candidates[eid]["_graph_signal"] = 0.0
        executed_calls.append(
            {
                "api": "tkg_explain_event_evidence",
                "count": int(explain_count),
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )

    # --- Final scoring ---
    for eid in selected:
        c = candidates[eid]
        base = float(c.get("_selection_score") or c.get("_preselect_score") or 0.0)
        sig = float(c.get("_graph_signal") or 0.0)
        try:
            multi_routes = len(set(c.get("sources") or []))
        except Exception:
            multi_routes = 0
        multi_bonus = max(0, min(int(multi_routes) - 1, 4))
        c["_multi_bonus"] = float(multi_bonus)
        c["_final_score"] = base + float(weights.get("signal", 0.0)) * sig

    if bool(reranker_cfg.get("enabled")) and reranker_engine == "llm" and selected and reranker_stage == "final_only":
        try:
            from modules.memory.application.config import get_llm_selection, load_memory_config
            from modules.memory.application.llm_adapter import build_llm_from_config
            from modules.memory.application.rerank_dialog_v1 import (
                RerankConfig,
                RetrievalCandidate,
                build_llm_client_from_fn,
                create_rerank_service,
            )

            cfg = load_memory_config()
            llm_sel = get_llm_selection(cfg, reranker_llm_kind)
            reranker_model_provider = str(llm_sel.get("provider") or "").strip() or None
            reranker_model_name = str(llm_sel.get("model") or "").strip() or None
            adapter = build_llm_from_config(reranker_llm_kind)
            if adapter is None:
                reranker_error = f"llm_adapter_unavailable:{reranker_llm_kind}"
            else:
                def _rerank_generate(system_prompt: str, user_prompt: str) -> str:
                    return adapter.generate(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )

                rerank_candidates: List[RetrievalCandidate] = []
                selected_before_rerank = sorted(
                    selected,
                    key=lambda eid: float(candidates[eid].get("_final_score") or 0.0),
                    reverse=True,
                )
                reranker_pool_size = max(1, min(len(selected_before_rerank), int(reranker_cfg.get("rerank_pool_size") or len(selected_before_rerank) or 1)))
                for eid in selected_before_rerank:
                    c = candidates[eid]
                    rerank_candidates.append(
                        RetrievalCandidate(
                            query_text=q,
                            evidence_text=str(c.get("text") or ""),
                            evidence_type=_dialog_v2_rerank_evidence_type(c.get("primary_source") or ""),
                            event_id=str(eid),
                            base_score=float(c.get("_final_score") or 0.0),
                            timestamp=(str(c.get("timestamp")) if c.get("timestamp") else None),
                            metadata={"candidate_id": eid},
                        )
                    )

                rerank_service = create_rerank_service(
                    RerankConfig(
                        enabled=True,
                        model="llm",
                        top_n=int(reranker_pool_size),
                        weight_base=float(reranker_cfg.get("weight_base") or 0.4),
                        weight_rerank=float(reranker_cfg.get("weight_rerank") or 0.5),
                        weight_type=float(reranker_cfg.get("weight_type") or 0.1),
                        type_bias=dict(reranker_cfg.get("type_bias") or {}),
                    ),
                    llm_client=build_llm_client_from_fn(_rerank_generate),
                )
                started = time.perf_counter()
                rerank_results = rerank_service.rerank(q, rerank_candidates)
                rerank_latency_ms = (time.perf_counter() - started) * 1000
                reranked_selected: List[str] = []
                for result in rerank_results:
                    eid = str(result.event_id or "").strip()
                    if not eid or eid not in candidates:
                        continue
                    candidates[eid]["_rerank_score"] = float(result.rerank_score)
                    candidates[eid]["_selection_score"] = float(result.final_score)
                    candidates[eid]["_final_score"] = float(result.final_score) + float(weights.get("signal", 0.0)) * float(candidates[eid].get("_graph_signal") or 0.0)
                    reranked_selected.append(eid)
                    reranker_rank_map[eid] = int(result.rank)
                if reranked_selected:
                    selected = reranked_selected
                    selected_set = set(selected)
                    reranker_applied = True
        except Exception as exc:
            reranker_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    if bool(reranker_cfg.get("enabled")):
        executed_calls.append(
            {
                "api": "dialog_v2_rerank",
                "enabled": bool(reranker_cfg.get("enabled")),
                "applied": bool(reranker_applied),
                "engine": reranker_engine,
                "llm_kind": reranker_llm_kind,
                "provider": reranker_model_provider,
                "model": reranker_model_name,
                "stage": reranker_stage,
                "score_mode": reranker_score_mode,
                "pool_size": int(reranker_pool_size),
                "input_count": int(len(reranker_input_ids)),
                "latency_ms": rerank_latency_ms,
                **({"request_id": reranker_request_id} if reranker_request_id else {}),
                **({"usage_total_tokens": reranker_usage_total_tokens} if reranker_usage_total_tokens is not None else {}),
                **({"error": reranker_error} if reranker_error else {}),
            }
        )

    # sort selected by final score
    selected.sort(key=lambda eid: float(candidates[eid].get("_final_score") or 0.0), reverse=True)
    final_rank_map = {eid: rank for rank, eid in enumerate(selected, start=1)}

    evidence_items: List[Dict[str, Any]] = []
    for eid in selected[: int(topk)]:
        c = candidates[eid]
        out_event_id = str(eid if ablation.should_use_source_native_candidates() else (c.get("logical_event_id") or eid))
        evaluation_ids = [str(item or "").strip() for item in sorted(list(c.get("evaluation_event_ids") or [])) if str(item or "").strip()]
        item = {
            "event_id": out_event_id,
            "tkg_event_id": c.get("tkg_event_id"),
            "score": float(c.get("_final_score") or 0.0),
            "source": c.get("primary_source") or "event_search",
            "text": c.get("text") or "",
            "timestamp": (c.get("timestamp") if ablation.signal_enabled("timestamp") else None),
            "_base_score": c.get("_base_score"),
            "_preselect_score": c.get("_preselect_score"),
            "_route_support_score": c.get("_route_support_score"),
            "_match_fidelity_score": c.get("_match_fidelity_score"),
            "_recency_score": c.get("_recency_score"),
            "_rerank_score": c.get("_rerank_score"),
            "_multi_bonus": c.get("_multi_bonus"),
            "sources": sorted(list(c.get("sources") or [])),
            "route_scores": {
                "event_vec": c.get("score_event_vec", 0.0),
                "vec": c.get("score_vec", 0.0),
                "knowledge": c.get("score_k", 0.0),
                "entity": c.get("score_en", 0.0),
                "time": c.get("score_t", 0.0),
            },
            "evaluation_event_ids": evaluation_ids,
        }
        if "tkg_explain" in c:
            item["tkg_explain"] = c.get("tkg_explain")
        evidence_items.append(item)

    if unmapped_evidence and len(evidence_items) < int(topk):
        unmapped_sorted = sorted(unmapped_evidence, key=lambda x: float(x.get("score") or 0.0), reverse=True)
        remaining = max(0, int(topk) - len(evidence_items))
        evidence_items.extend(unmapped_sorted[:remaining])

    evidence_ids = [h.get("event_id") for h in evidence_items if h.get("event_id")]
    evaluation_evidence_ids: List[str] = []
    evaluation_seen: set[str] = set()
    for item in evidence_items[: int(topk)]:
        for ev_id in list(item.get("evaluation_event_ids") or []):
            normalized = str(ev_id or "").strip()
            if normalized and normalized not in evaluation_seen:
                evaluation_seen.add(normalized)
                evaluation_evidence_ids.append(normalized)
    out: Dict[str, Any] = {
        "strategy": str(output_strategy or "dialog_v2"),
        "query": q,
        "evidence": evidence_ids[: int(topk)],
        "evaluation_evidence": evaluation_evidence_ids,
        "evidence_details": evidence_items[: int(topk)],
    }

    qa_latency_ms = 0.0
    if with_answer:
        answer_text = ""
        try:
            from modules.memory.application.qa_dialog_v1 import QA_SYSTEM_PROMPT_GENERAL, build_qa_user_prompt
            from modules.memory.application.qa_longmemeval import (
                build_longmemeval_prompts,
                extract_question_date_from_time_hints,
                should_use_longmemeval_prompt,
            )
        except Exception as exc:
            raise RuntimeError(f"qa_prompt_import_failed: {exc}") from exc

        if qa_generate is None:
            try:
                from modules.memory.application.llm_adapter import build_llm_from_config, build_llm_from_env

                adapter = build_llm_from_config('qa')
                if adapter is None:
                    adapter = build_llm_from_env()
            except Exception:
                adapter = None

            if adapter is not None:
                def _gen(system_prompt: str, user_prompt: str) -> str:
                    return adapter.generate(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    )

                qa_generate = _gen

        if qa_generate is None and str(llm_policy or "best_effort").lower() == "require":
            raise RuntimeError("QA LLM is not configured (llm_policy=require).")

        evidence_for_qa = evidence_items[: int(topk)]
        if not evidence_for_qa:
            answer_text = "insufficient information"
        elif qa_generate is None:
            answer_text = "Unable to answer in dummy mode."
        else:
            question_date = extract_question_date_from_time_hints(time_hints)
            system_prompt = QA_SYSTEM_PROMPT_GENERAL
            if should_use_longmemeval_prompt(memory_domain=str(memory_domain), task=str(task or "")) and question_date:
                system_prompt, user_prompt = build_longmemeval_prompts(
                    question=q,
                    question_date=str(question_date),
                    evidence=list(evidence_for_qa),
                    task=str(task or ""),
                )
            else:
                user_prompt = build_qa_user_prompt(query=q, task=str(task or "GENERAL"), evidence=list(evidence_for_qa))
            qa_start = time.perf_counter()
            try:
                answer_text = str(await asyncio.to_thread(qa_generate, system_prompt, user_prompt)).strip()
            except Exception:
                answer_text = "Unable to generate answer due to an error."
            qa_latency_ms = (time.perf_counter() - qa_start) * 1000
        out["answer"] = answer_text

    if debug:
        retrieval_latency_ms = (time.perf_counter() - total_start) * 1000
        plan = {
            "api": "dialog_v2_parallel",
            "strategy": str(output_strategy or "dialog_v2"),
            "candidate_k": K,
            "seed_topn": seed_k,
            "graph_cap": max(0, min(int(graph_cap), K)),
            "rrf_k": int(rrf_k),
            "score_blend_alpha": float(score_blend_alpha),
            "weights": {str(k): float(v) for k, v in weights.items()},
            "route_toggles": {
                "event": bool(effective_enable_event_route),
                "evidence": bool(effective_enable_evidence_route),
                "knowledge": bool(effective_enable_knowledge_route),
                "entity": bool(effective_enable_entity_route),
                "time": bool(effective_enable_time_route),
            },
            "latency_ms": retrieval_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
        }
        if ablation.is_active():
            plan["test_ablation"] = ablation.model_dump()
        if bool(reranker_cfg.get("enabled")):
            plan["reranker"] = {
                "enabled": bool(reranker_cfg.get("enabled")),
                "applied": bool(reranker_applied),
                "engine": reranker_engine,
                "llm_kind": reranker_llm_kind,
                "provider": reranker_model_provider,
                "model": reranker_model_name,
                "stage": reranker_stage,
                "score_mode": reranker_score_mode,
                "pool_size": int(reranker_pool_size),
                "input_count": int(len(reranker_input_ids)),
                "latency_ms": rerank_latency_ms,
                "request_id": reranker_request_id,
                "usage_total_tokens": reranker_usage_total_tokens,
                "error": reranker_error,
            }
        if with_answer:
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            plan["qa_latency_ms"] = qa_latency_ms
            plan["total_latency_ms"] = total_latency_ms
        candidate_details: List[Dict[str, Any]] = []
        for eid in base_sorted_all:
            c = candidates[eid]
            logical_event_id = str(c.get("logical_event_id") or "").strip() or None
            tkg_event_id = str(c.get("tkg_event_id") or "").strip() or None
            source_native = bool(c.get("source_native"))
            evaluation_event_ids = [str(item or "").strip() for item in sorted(list(c.get("evaluation_event_ids") or [])) if str(item or "").strip()]
            candidate_details.append(
                {
                    "candidate_id": str(eid),
                    "logical_event_id": logical_event_id,
                    "tkg_event_id": tkg_event_id,
                    "evaluation_event_ids": evaluation_event_ids,
                    "utterance_id": (c.get("route_meta") or {}).get("utterance_id"),
                    "mapping_status": (
                        "source_native"
                        if source_native
                        else ("mapped_logical" if logical_event_id else ("mapped_tkg" if tkg_event_id else "unmapped"))
                    ),
                    "same_dialog_key": _same_dialog_key_from_event_id(logical_event_id or str(eid)),
                    "primary_source": c.get("primary_source") or "event_search",
                    "sources": sorted(list(c.get("sources") or [])),
                    "text_preview": str(c.get("text") or "")[:240],
                    "timestamp": (c.get("timestamp") if ablation.signal_enabled("timestamp") else None),
                    "source_native": source_native,
                    "route_scores": {
                        "event_vec": float(c.get("score_event_vec") or 0.0),
                        "vec": float(c.get("score_vec") or 0.0),
                        "knowledge": float(c.get("score_k") or 0.0),
                        "entity": float(c.get("score_en") or 0.0),
                        "time": float(c.get("score_t") or 0.0),
                    },
                    "route_norm_scores": {
                        "event_vec": float(route_norm_score_maps["event_vec"].get(eid, 0.0)),
                        "vec": float(route_norm_score_maps["vec"].get(eid, 0.0)),
                        "knowledge": float(route_norm_score_maps["knowledge"].get(eid, 0.0)),
                        "entity": float(route_norm_score_maps["entity"].get(eid, 0.0)),
                        "time": float(route_norm_score_maps["time"].get(eid, 0.0)),
                    },
                    "route_ranks": {
                        "event_vec_rank": route_rank_maps["event_vec"].get(eid),
                        "vec_rank": route_rank_maps["vec"].get(eid),
                        "knowledge_rank": route_rank_maps["knowledge"].get(eid),
                        "entity_rank": route_rank_maps["entity"].get(eid),
                        "time_rank": route_rank_maps["time"].get(eid),
                    },
                    "rrf_score": float(c.get("_rrf_score") or 0.0),
                    "route_support_score": float(c.get("_route_support_score") or 0.0),
                    "match_fidelity_score": float(c.get("_match_fidelity_score") or 0.0),
                    "match_details": dict(c.get("_match_details") or {}),
                    "recency_score": float(c.get("_recency_score") or 0.0),
                    "preselect_score": float(c.get("_preselect_score") or 0.0),
                    "selection_score": float(c.get("_selection_score") or 0.0),
                    "base_score": float(c.get("_base_score") or 0.0),
                    "graph_signal": (float(c.get("_graph_signal")) if c.get("_graph_signal") is not None else None),
                    "rerank_raw_score": (float(c.get("_rerank_raw_score")) if c.get("_rerank_raw_score") is not None else None),
                    "rerank_score": (float(c.get("_rerank_score")) if c.get("_rerank_score") is not None else None),
                    "rerank_rank": reranker_rank_map.get(eid),
                    "in_rerank_pool": eid in reranker_input_set,
                    "multi_bonus": (float(c.get("_multi_bonus")) if c.get("_multi_bonus") is not None else None),
                    "final_score": (float(c.get("_final_score")) if c.get("_final_score") is not None else None),
                    "base_rank": base_rank_map.get(eid),
                    "selected": eid in selected_set,
                    "selection_reason": selection_reason_map.get(eid, "dropped_before_selection"),
                    "selected_rank_before_final": selected_rank_before_final_map.get(eid),
                    "final_rank": final_rank_map.get(eid),
                    "in_final_topk": bool(final_rank_map.get(eid) and final_rank_map.get(eid) <= int(topk)),
                    "reasons": list(c.get("reasons") or []),
                }
            )
        unmapped_sorted = sorted(unmapped_evidence, key=lambda x: float(x.get("score") or 0.0), reverse=True)
        out["debug"] = {
            "plan": plan,
            "executed_calls": executed_calls,
            "evidence_count": len(evidence_items[: int(topk)]),
            "candidate_stats": {
                "candidate_k": K,
                "seed_topn": seed_k,
                "graph_cap": graph_cap,
                "selection_mode": ("rerank_preselect" if reranker_applied and reranker_stage == "preselect" else "score_order"),
                "reranker_enabled": bool(reranker_cfg.get("enabled")),
                "reranker_applied": bool(reranker_applied),
                "union_count": len(candidates),
                "selected_count": len(selected),
                "final_topk_count": len(evidence_items[: int(topk)]),
                "unmapped_count": len(unmapped_evidence),
                "rerank_pool_count": len(reranker_input_ids),
            },
            "candidate_details": candidate_details,
            "unmapped_evidence_details": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "utterance_id": item.get("utterance_id"),
                    "score": float(item.get("score") or 0.0),
                    "source": item.get("source"),
                    "text_preview": str(item.get("text") or "")[:240],
                    "timestamp": item.get("timestamp"),
                    "unmapped_to_event": bool(item.get("unmapped_to_event")),
                }
                for item in unmapped_sorted
            ],
        }

    return out


async def _retrieval_dialog_v2_test(
    *,
    store: MemoryPort,
    tenant_id: str,
    user_tokens_norm: List[str],
    user_match_norm: str,
    query: str,
    topk: int,
    memory_domain: str,
    run_id: Optional[str],
    debug: bool,
    with_answer: bool,
    task: str,
    llm_policy: str,
    qa_generate: Optional[Callable[[str, str], str]],
    tkg_explain: bool,
    candidate_k: int,
    seed_topn: int,
    e_vec_oversample: int,
    graph_cap: int,
    rrf_k: int,
    qa_evidence_cap_l2: int,
    qa_evidence_cap_l4: int,
    enable_event_route: bool,
    enable_evidence_route: bool,
    enable_knowledge_route: bool,
    enable_entity_route: bool,
    enable_time_route: bool,
    entity_hints: Optional[List[str]],
    time_hints: Optional[Dict[str, Any]],
    dialog_v2_weights: Optional[Dict[str, float]],
    dialog_v2_reranker: Optional[Dict[str, Any]],
    test_ablation: _DialogV2TestAblationContract,
) -> Dict[str, Any]:
    return await _retrieval_dialog_v2(
        store=store,
        tenant_id=tenant_id,
        user_tokens_norm=user_tokens_norm,
        user_match_norm=user_match_norm,
        query=query,
        topk=topk,
        memory_domain=memory_domain,
        run_id=run_id,
        debug=debug,
        with_answer=with_answer,
        task=task,
        llm_policy=llm_policy,
        qa_generate=qa_generate,
        tkg_explain=tkg_explain,
        candidate_k=candidate_k,
        seed_topn=seed_topn,
        e_vec_oversample=e_vec_oversample,
        graph_cap=graph_cap,
        rrf_k=rrf_k,
        qa_evidence_cap_l2=qa_evidence_cap_l2,
        qa_evidence_cap_l4=qa_evidence_cap_l4,
        enable_event_route=enable_event_route,
        enable_evidence_route=enable_evidence_route,
        enable_knowledge_route=enable_knowledge_route,
        enable_entity_route=enable_entity_route,
        enable_time_route=enable_time_route,
        entity_hints=entity_hints,
        time_hints=time_hints,
        dialog_v2_weights=dialog_v2_weights,
        dialog_v2_reranker=dialog_v2_reranker,
        test_ablation=test_ablation,
        output_strategy="dialog_v2_test",
    )
