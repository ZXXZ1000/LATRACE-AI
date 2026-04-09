from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .errors import AdkErrorInfo, normalize_exception, normalize_http_error


StatePropertiesFetcher = Callable[..., Awaitable[Dict[str, Any]]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    # Keep normalization intentionally conservative for deterministic behavior.
    return " ".join(text.replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True)
class StatePropertyDef:
    name: str
    description: Optional[str] = None
    value_type: Optional[str] = None
    allowed_values: Optional[Tuple[Any, ...]] = None
    allow_raw_value: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "value_type": self.value_type,
            "allowed_values": list(self.allowed_values or ()),
            "allow_raw_value": bool(self.allow_raw_value),
        }


@dataclass
class StatePropertyVocab:
    tenant_id: str
    vocab_version: Optional[str]
    properties: List[StatePropertyDef]
    by_name: Dict[str, StatePropertyDef]
    alias_index: Dict[str, str]
    normalized_canonical_index: Dict[str, List[str]]
    normalized_alias_index: Dict[str, List[str]]
    alias_map_version: str = "v1"
    fetched_at: datetime = field(default_factory=_utc_now)
    last_refresh_at: datetime = field(default_factory=_utc_now)
    cache_hit: bool = False
    cache_miss: bool = True
    vocab_refreshed: bool = False
    vocab_empty: bool = False

    def debug_meta(self) -> Dict[str, Any]:
        return {
            "vocab_version": self.vocab_version,
            "alias_map_version": self.alias_map_version,
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "vocab_refreshed": self.vocab_refreshed,
            "vocab_empty": self.vocab_empty,
            "last_refresh_at": self.last_refresh_at.isoformat(),
        }


@dataclass(frozen=True)
class PropertyResolutionResult:
    matched: bool
    needs_disambiguation: bool
    canonical: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    match_source: Optional[str] = None
    vocab_version: Optional[str] = None

    def to_debug_meta(self, *, input_text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "input": input_text,
            "vocab_version": self.vocab_version,
            "matched": bool(self.matched),
            "needs_disambiguation": bool(self.needs_disambiguation),
        }
        if self.canonical:
            out["canonical"] = self.canonical
        if self.match_source:
            out["match_source"] = self.match_source
        if self.candidates:
            out["candidates"] = list(self.candidates)
        return out


class StatePropertyVocabLoadError(RuntimeError):
    def __init__(self, info: AdkErrorInfo) -> None:
        super().__init__(info.message)
        self.info = info


def _default_aliases() -> Dict[str, str]:
    # Local ADK alias map (can be extended later / moved to config).
    return {
        "职位": "occupation",
        "岗位": "occupation",
        "工作": "occupation",
        "工作状态": "work_status",
        "状态": "status",
        "所在地": "location",
        "城市": "location",
        "住址": "location",
        "情绪": "mood",
        "心情": "mood",
        "关系状态": "relationship_status",
        "婚姻状态": "relationship_status",
        "title": "occupation",
    }


def _build_vocab(
    *,
    tenant_id: str,
    response: Dict[str, Any],
    alias_map: Optional[Dict[str, str]] = None,
    alias_map_version: str = "v1",
    previous: Optional[StatePropertyVocab] = None,
    cache_hit: bool = False,
) -> StatePropertyVocab:
    props_raw = response.get("properties") if isinstance(response, dict) else None
    props: List[StatePropertyDef] = []
    by_name: Dict[str, StatePropertyDef] = {}
    normalized_canonical: Dict[str, List[str]] = {}

    for item in (props_raw or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        prop = StatePropertyDef(
            name=name,
            description=(str(item.get("description")).strip() if item.get("description") is not None else None),
            value_type=(str(item.get("value_type")).strip() if item.get("value_type") is not None else None),
            allowed_values=tuple(item.get("allowed_values") or ()),
            allow_raw_value=bool(item.get("allow_raw_value")),
        )
        by_name[name] = prop
        props.append(prop)
        norm = _norm_text(name)
        if norm:
            normalized_canonical.setdefault(norm, [])
            if name not in normalized_canonical[norm]:
                normalized_canonical[norm].append(name)

    local_alias = dict(_default_aliases())
    if alias_map:
        local_alias.update({str(k): str(v) for k, v in alias_map.items()})
    alias_index: Dict[str, str] = {}
    normalized_alias: Dict[str, List[str]] = {}
    for alias, canonical in local_alias.items():
        a = str(alias or "").strip()
        c = str(canonical or "").strip()
        if not a or not c:
            continue
        alias_index[a] = c
        na = _norm_text(a)
        if na:
            normalized_alias.setdefault(na, [])
            if c not in normalized_alias[na]:
                normalized_alias[na].append(c)

    now = _utc_now()
    vocab_version = str(response.get("vocab_version") or "").strip() or None
    prev_version = previous.vocab_version if previous else None
    refreshed = bool(previous) and (prev_version != vocab_version)
    return StatePropertyVocab(
        tenant_id=str(tenant_id),
        vocab_version=vocab_version,
        properties=props,
        by_name=by_name,
        alias_index=alias_index,
        normalized_canonical_index=normalized_canonical,
        normalized_alias_index=normalized_alias,
        alias_map_version=alias_map_version,
        fetched_at=(previous.fetched_at if (cache_hit and previous) else now),
        last_refresh_at=(previous.last_refresh_at if (cache_hit and previous) else now),
        cache_hit=cache_hit,
        cache_miss=not cache_hit,
        vocab_refreshed=refreshed,
        vocab_empty=(len(props) == 0),
    )


class StatePropertyVocabManager:
    """In-process cache + loader for /memory/v1/state/properties."""

    def __init__(
        self,
        *,
        fetcher: StatePropertiesFetcher,
        alias_map: Optional[Dict[str, str]] = None,
        alias_map_version: str = "v1",
    ) -> None:
        self._fetcher = fetcher
        self._alias_map = dict(alias_map or {})
        self._alias_map_version = str(alias_map_version or "v1")
        self._cache: Dict[str, StatePropertyVocab] = {}

    async def load_state_property_vocab(
        self,
        *,
        tenant_id: str,
        user_tokens: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> StatePropertyVocab:
        tenant_key = str(tenant_id or "").strip()
        if not tenant_key:
            raise StatePropertyVocabLoadError(
                normalize_http_error(status_code=400, body={"detail": "missing_tenant_id"})
            )

        cached = self._cache.get(tenant_key)
        if cached is not None and not force_refresh:
            cached_copy = _build_vocab(
                tenant_id=tenant_key,
                response={
                    "vocab_version": cached.vocab_version,
                    "properties": [p.to_dict() for p in cached.properties],
                },
                alias_map=self._alias_map,
                alias_map_version=self._alias_map_version,
                previous=cached,
                cache_hit=True,
            )
            self._cache[tenant_key] = cached_copy
            return cached_copy

        try:
            response = await self._fetcher(
                tenant_id=tenant_key,
                user_tokens=list(user_tokens or []),
                limit=200,
            )
        except StatePropertyVocabLoadError:
            raise
        except Exception as exc:
            raise StatePropertyVocabLoadError(normalize_exception(exc)) from exc

        if not isinstance(response, dict):
            raise StatePropertyVocabLoadError(
                normalize_http_error(status_code=500, body={"detail": "invalid_state_properties_response"})
            )

        # Allow fetcher to pass through HTTP-like errors in-band when needed by tests/adapters.
        if "status_code" in response and int(response.get("status_code") or 200) >= 400:
            raise StatePropertyVocabLoadError(
                normalize_http_error(
                    status_code=int(response.get("status_code")),
                    body=response.get("body") or response.get("detail") or response,
                )
            )

        built = _build_vocab(
            tenant_id=tenant_key,
            response=response,
            alias_map=self._alias_map,
            alias_map_version=self._alias_map_version,
            previous=cached,
            cache_hit=False,
        )
        self._cache[tenant_key] = built
        return built

    def clear(self, *, tenant_id: Optional[str] = None) -> None:
        if tenant_id is None:
            self._cache.clear()
            return
        self._cache.pop(str(tenant_id or "").strip(), None)


def _candidate_rank_key(name: str) -> Tuple[int, str]:
    return (len(name), name)


def map_state_property(
    property_text: str,
    *,
    vocab: StatePropertyVocab,
) -> PropertyResolutionResult:
    text = str(property_text or "").strip()
    if not text:
        return PropertyResolutionResult(
            matched=False,
            needs_disambiguation=False,
            canonical=None,
            candidates=(),
            match_source=None,
            vocab_version=vocab.vocab_version,
        )

    # 1) canonical exact
    if text in vocab.by_name:
        return PropertyResolutionResult(
            matched=True,
            needs_disambiguation=False,
            canonical=text,
            candidates=(),
            match_source="canonical_exact",
            vocab_version=vocab.vocab_version,
        )

    # 2) alias exact
    if text in vocab.alias_index:
        canonical = vocab.alias_index[text]
        if canonical in vocab.by_name:
            return PropertyResolutionResult(
                matched=True,
                needs_disambiguation=False,
                canonical=canonical,
                candidates=(),
                match_source="alias_exact",
                vocab_version=vocab.vocab_version,
            )

    # 3) normalized exact (canonical + alias)
    norm = _norm_text(text)
    if norm:
        canonical_matches = tuple(sorted(set(vocab.normalized_canonical_index.get(norm) or []), key=_candidate_rank_key))
        if len(canonical_matches) == 1:
            return PropertyResolutionResult(
                matched=True,
                needs_disambiguation=False,
                canonical=canonical_matches[0],
                candidates=(),
                match_source="normalized_match",
                vocab_version=vocab.vocab_version,
            )
        if len(canonical_matches) > 1:
            return PropertyResolutionResult(
                matched=False,
                needs_disambiguation=True,
                canonical=None,
                candidates=canonical_matches,
                match_source="normalized_match",
                vocab_version=vocab.vocab_version,
            )

        alias_matches = tuple(sorted(set(vocab.normalized_alias_index.get(norm) or []), key=_candidate_rank_key))
        alias_matches = tuple(x for x in alias_matches if x in vocab.by_name)
        if len(alias_matches) == 1:
            return PropertyResolutionResult(
                matched=True,
                needs_disambiguation=False,
                canonical=alias_matches[0],
                candidates=(),
                match_source="normalized_alias",
                vocab_version=vocab.vocab_version,
            )
        if len(alias_matches) > 1:
            return PropertyResolutionResult(
                matched=False,
                needs_disambiguation=True,
                canonical=None,
                candidates=alias_matches,
                match_source="normalized_alias",
                vocab_version=vocab.vocab_version,
            )

    # 4) no match -> return bounded candidates (prefix/contains on normalized canonical names)
    candidate_hits: List[str] = []
    if norm:
        for cand_norm, names in vocab.normalized_canonical_index.items():
            if not cand_norm:
                continue
            if cand_norm.startswith(norm) or norm in cand_norm:
                candidate_hits.extend(list(names))
    candidate_hits = sorted(set(candidate_hits), key=_candidate_rank_key)
    if len(candidate_hits) > 1:
        return PropertyResolutionResult(
            matched=False,
            needs_disambiguation=True,
            canonical=None,
            candidates=tuple(candidate_hits[:5]),
            match_source=None,
            vocab_version=vocab.vocab_version,
        )
    return PropertyResolutionResult(
        matched=False,
        needs_disambiguation=False,
        canonical=None,
        candidates=tuple(candidate_hits[:3]),
        match_source=None,
        vocab_version=vocab.vocab_version,
    )

