from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager, suppress
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, TYPE_CHECKING
import logging
import time
import hashlib
import jwt
import yaml
from pathlib import Path
from jwt import InvalidTokenError, PyJWKClient
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse, JSONResponse
try:
    from dotenv import load_dotenv  # type: ignore
    import os as _os
    load_dotenv()  # root .env
    # also load module-specific .env with override to ensure keys available
    _MEM_ENV = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "config", ".env"))
    if _os.path.exists(_MEM_ENV):
        load_dotenv(_MEM_ENV, override=True)
except Exception:
    # dotenv not installed or no .env present; skip silently
    pass
from pydantic import BaseModel, ConfigDict, Field
from modules.memory.application.service import MemoryService, SafetyError
from modules.memory.application.graph_service import GraphService, GraphValidationError
from modules.memory.application.topic_normalizer import normalize_topic_text
from modules.memory.application.config import (
    SEARCH_TOPK_HARD_LIMIT,
    load_memory_config,
    get_ann_settings,
    get_api_topk_defaults,
    get_dialog_v2_ranking_settings,
    get_ingest_executor_settings,
    resolve_lexical_hybrid_settings,
    get_graph_settings,
    get_search_weights,
)
from modules.memory.application.ingest_executor import IngestExecutor, IngestExecutorConfig
from modules.memory.application.metrics import (
    get_metrics,
    as_prometheus_text,
    record_graph_request,
    add_graph_latency,
    record_ttl_cleanup,
    add,
    inc,
    observe_ingest_latency,
)
from modules.memory.application.llm_adapter import (
    LLMAdapter,
    LLMUsageContext,
    build_llm_from_byok,
    resolve_openai_compatible_chat_target,
    reset_llm_usage_context,
    reset_llm_usage_hook,
    set_llm_usage_context,
    set_llm_usage_hook,
)
from modules.memory.application import runtime_config as rtconf
from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore, IngestJobRecord
from modules.memory.infra.pg_ingest_job_store import PgIngestJobStore, PgIngestJobStoreSettings
from modules.memory.application.turn_mark_extractor_dialog_v1 import (
    apply_turn_marks,
    build_turn_mark_extractor_v1_from_env,
    default_marks_keep_all,
    generate_pin_intents,
    pin_intents_to_facts,
    validate_and_normalize_marks,
)
from modules.memory.contracts.memory_models import MemoryEntry, Edge, SearchFilters
from modules.memory.contracts.graph_models import GraphUpsertRequest, PendingEquiv
from modules.memory.contracts.usage_models import UsageEvent, TokenUsageDetail, EmbeddingUsage
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.infra.vector_store_router import (
    clear_request_context,
    set_request_context,
    update_request_context,
    VectorStoreRouter,
)
if TYPE_CHECKING:
    pass
from modules.memory.infra.equiv_store import EquivStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.usage_wal import UsageWAL, UsageWALSettings
from modules.memory.retrieval import retrieval
from modules.memory.session_write import session_write
from urllib.parse import urlparse, urlunparse

MAX_REQUEST_BYTES_FALLBACK = 10 * 1024 * 1024  # 10 MiB
RATE_LIMIT_BURST_FALLBACK = 30
RATE_LIMIT_PER_MINUTE_FALLBACK = 0
HIGH_COST_TIMEOUT_FALLBACK = 15.0
HIGH_COST_FAILURE_THRESHOLD = 4
HIGH_COST_COOLDOWN_SECONDS = 20
DEFAULT_JWKS_CACHE_SECONDS = 300
TOPIC_STATUS_ONGOING_DAYS = int(os.getenv("MEMORY_TOPIC_STATUS_ONGOING_DAYS", "14") or 14)
TOPIC_STATUS_PAUSED_DAYS = int(os.getenv("MEMORY_TOPIC_STATUS_PAUSED_DAYS", "90") or 90)
PUBLIC_PATHS = {"/health", "/metrics", "/metrics_prom"}
PATH_SCOPE_REQUIREMENTS: Dict[str, str] = {
    "/api/list": "memory.read",
    "/ingest": "memory.write",
    "/ingest/dialog/v1": "memory.write",
    "/ingest/jobs/execute": "memory.admin",
    "/ingest/jobs/": "memory.read",
    "/ingest/sessions/": "memory.read",
    "/retrieval": "memory.read",
    "/retrieval/dialog/v2": "memory.read",
    "/search": "memory.read",
    "/timeline_summary": "memory.read",
    "/object_search": "memory.read",
    "/speech_search": "memory.read",
    "/entity_event_anchor": "memory.read",
    "/write": "memory.admin",
    "/update": "memory.admin",
    "/delete": "memory.admin",
    "/link": "memory.admin",
    "/rollback": "memory.admin",
    "/batch_delete": "memory.admin",
    "/batch_link": "memory.admin",
    "/equiv/pending": "memory.admin",
    "/equiv/pending/add": "memory.admin",
    "/equiv/pending/confirm": "memory.admin",
    "/equiv/pending/remove": "memory.admin",
    "/graph/v0/": "memory.admin",
    "/graph/v1/": "memory.read",
    "/memory/state/pending/": "memory.admin",
    "/memory/state/": "memory.read",
    "/memory/agentic/": "memory.read",
    "/memory/v1/clear": "memory.clear",
    "/memory/v1/": "memory.read",
    "/config/": "memory.admin",
    "/admin/": "memory.admin",
}


class AuthContext(TypedDict, total=False):
    tenant_id: str
    subject: Optional[str]
    claims: Dict[str, Any]
    token: str
    method: str
    scopes: List[str]

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("memory.audit")

_svc: Optional[MemoryService] = None
_graph_svc: Optional[Any] = None
_equiv_store: Optional[Any] = None
_rtconf_loaded = False
_init_lock = threading.RLock()
_tenant_clear_locks: Dict[str, asyncio.Lock] = {}
_tenant_clear_locks_guard = threading.Lock()


class _LazyProxy:
    def __init__(self, getter: Callable[[], Any], label: str, ready_check: Callable[[], bool]) -> None:
        object.__setattr__(self, "_getter", getter)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_ready_check", ready_check)

    def _target(self) -> Any:
        return object.__getattribute__(self, "_getter")()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if object.__getattribute__(self, "_ready_check")():
            setattr(self._target(), name, value)
            return
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"<LazyProxy {object.__getattribute__(self, '_label')}>"


def _resolve_override(name: str) -> Optional[Any]:
    val = globals().get(name)
    if isinstance(val, _LazyProxy):
        return None
    return val


def _get_svc() -> MemoryService:
    global _svc, _rtconf_loaded
    override = _resolve_override("svc")
    if _svc is None and isinstance(override, MemoryService):
        _svc = override
    with _init_lock:
        if _svc is None:
            _svc = create_service()
        if not _rtconf_loaded:
            rtconf.load_overrides()
            _rtconf_loaded = True
    return _svc


def _get_graph_svc() -> GraphService:
    global _graph_svc
    override = _resolve_override("graph_svc")
    if _graph_svc is None and override is not None:
        _graph_svc = override
        return _graph_svc  # type: ignore[return-value]
    with _init_lock:
        if _graph_svc is None:
            svc_local = _get_svc()
            # Pass vector_store to enable TKG vector writes to Qdrant
            _graph_svc = GraphService(svc_local.graph, vector_store=svc_local.vectors)
    return _graph_svc  # type: ignore[return-value]


def _get_equiv_store() -> EquivStore:
    global _equiv_store
    override = _resolve_override("equiv_store")
    if _equiv_store is None and override is not None:
        _equiv_store = override
        return _equiv_store  # type: ignore[return-value]
    with _init_lock:
        if _equiv_store is None:
            graph_local = _get_graph_svc()
            _equiv_store = EquivStore(graph_local.store)
    return _equiv_store  # type: ignore[return-value]

def _as_int(val: Any, default: int) -> int:
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip()
        if not s or s.startswith("${"):
            return default
        return int(s)
    except Exception:
        return default


def _as_bool(val: Any, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _resolve_env_like_str(val: Any, default: str) -> str:
    try:
        s = str(val).strip()
        if not s or s.startswith("${"):
            return default
        return s
    except Exception:
        return default


def _normalize_neo4j_uri(raw_uri: str, *, default_host: str = "127.0.0.1", default_port: int = 7687) -> Tuple[str, Dict[str, str]]:
    """
    Normalize Neo4j URI for single-node deployments.

    - Force scheme to bolt://
    - Convert localhost/::1 to 127.0.0.1
    - Default or override port to 7687 (rewriting 7474)
    """
    info: Dict[str, str] = {"input": raw_uri or ""}
    try:
        uri = (raw_uri or "").strip()
        if not uri:
            final = f"bolt://{default_host}:{default_port}"
            info.update({"scheme": "bolt", "host": default_host, "port": str(default_port)})
            return final, info
        if "://" not in uri:
            uri = f"bolt://{uri}"
        parsed = urlparse(uri)
        parsed.scheme.lower() if parsed.scheme else "bolt"
        host = parsed.hostname or default_host
        port = parsed.port or default_port
        if host in ("localhost", "::1"):
            host = default_host
        if port in (None, 0, 7474):
            port = default_port
        info.update({"scheme": "bolt", "host": host, "port": str(port)})
        normalized = urlunparse(("bolt", f"{host}:{port}", "", "", "", ""))
        return normalized, info
    except Exception:
        final = f"bolt://{default_host}:{default_port}"
        info.update({"scheme": "bolt", "host": default_host, "port": str(default_port), "error": "normalize_failed"})
        return final, info


class TenantRateLimiter:
    """Simple token-bucket rate limiter keyed by tenant token/header.

    - Refills continuously based on requests_per_minute.
    - Burst controls max tokens (0 == no burst; all requests denied when rate also 0).
    """

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self.requests_per_minute = max(0, requests_per_minute)
        self.burst = max(0, burst)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.perf_counter()
        last, tokens = self._buckets.get(key, (now, float(self.burst)))
        # refill tokens based on elapsed time
        if self.requests_per_minute > 0:
            refill_rate_per_sec = self.requests_per_minute / 60.0
            tokens = min(self.burst, tokens + (now - last) * refill_rate_per_sec)
        self._buckets[key] = (now, tokens)
        if tokens < 1.0:
            return False
        self._buckets[key] = (now, tokens - 1.0)
        return True


class SimpleCircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: int) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(1, cooldown_seconds)
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> Tuple[bool, int]:
        now = time.time()
        if self.open_until and now < self.open_until:
            return False, int(self.open_until - now)
        return True, 0

    def record_success(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open_until = time.time() + self.cooldown_seconds
            self.failures = 0
            inc("circuit_breaker_open_total", 1)


def create_service() -> MemoryService:
    global _usage_wal
    cfg = load_memory_config()
    vcfg = cfg.get("memory", {}).get("vector_store", {})
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    rcfg = cfg.get("memory", {}).get("reliability", {})

    # MEMORY_BACKEND=inmem: use in-memory stores for local development without Docker
    backend = str(os.getenv("MEMORY_BACKEND", "")).strip().lower()
    if backend == "inmem":
        import logging
        from modules.memory.infra.inmem_vector_store import InMemVectorStore
        from modules.memory.infra.inmem_graph_store import InMemGraphStore
        logging.getLogger(__name__).warning(
            "[LATRACE] MEMORY_BACKEND=inmem: using in-memory stores (data will not persist across restarts)"
        )
        vectors: Any = InMemVectorStore()
        neo: Any = InMemGraphStore()
        audit = AuditStore()
        if _usage_wal is None:
            try:
                _usage_wal = _init_usage_wal()
            except Exception:
                pass
        return MemoryService(vectors, neo, audit, usage_wal=_usage_wal)

    vkind = str(vcfg.get("kind", "qdrant") or "qdrant").strip().lower()
    q_host = os.getenv("QDRANT_HOST") or vcfg.get("host", "127.0.0.1")
    q_port = os.getenv("QDRANT_PORT") or vcfg.get("port", 6333)
    q_api = os.getenv("QDRANT_API_KEY") or vcfg.get("api_key", "")
    qdr = QdrantStore({
        "host": _resolve_env_like_str(q_host, "127.0.0.1"),
        "port": _as_int(q_port, 6333),
        "api_key": _resolve_env_like_str(q_api, ""),
        "collections": vcfg.get("collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}),
        "embedding": vcfg.get("embedding", {}),
        "transport": vcfg.get("transport", {}),
        "sharding": vcfg.get("sharding", {}),
        "reliability": rcfg,
    })

    milvus = None
    m_host = os.getenv("MILVUS_HOST")
    m_port = os.getenv("MILVUS_PORT")
    if str(m_host or "").strip() and str(m_port or "").strip():
        from modules.memory.infra.milvus_store import MilvusStore
        milvus = MilvusStore({
            "host": _resolve_env_like_str(m_host, "127.0.0.1"),
            "port": _as_int(m_port, 19530),
            "collections": vcfg.get("collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}),
            "embedding": vcfg.get("embedding", {}),
            "reliability": rcfg,
        })

    vectors: Any
    if vkind == "milvus":
        vectors = milvus or qdr
    elif milvus is not None:
        vectors = VectorStoreRouter(qdr, milvus)
    else:
        vectors = qdr
    n_uri_raw = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    n_uri_norm, uri_meta = _normalize_neo4j_uri(_resolve_env_like_str(n_uri_raw, "bolt://127.0.0.1:7687"))
    n_user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    n_pass = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")
    n_db = os.getenv("NEO4J_DATABASE") or gcfg.get("database", "neo4j")
    env_strict_tenant_mode = os.getenv("MEMORY_GRAPH_STRICT_TENANT_MODE")
    strict_tenant_mode_raw = (
        env_strict_tenant_mode if env_strict_tenant_mode is not None else gcfg.get("strict_tenant_mode", False)
    )
    env_enable_legacy_memory_node = os.getenv("MEMORY_GRAPH_ENABLE_LEGACY_MEMORY_NODE")
    enable_legacy_memory_node_raw = (
        env_enable_legacy_memory_node
        if env_enable_legacy_memory_node is not None
        else gcfg.get("enable_legacy_memory_node", True)
    )
    strict_tenant_mode = _as_bool(strict_tenant_mode_raw, False)
    enable_legacy_memory_node = _as_bool(enable_legacy_memory_node_raw, True)
    try:
        import logging
        # Silence verbose neo4j driver logs
        logging.getLogger("neo4j").setLevel(logging.WARNING)
        logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
        if not enable_legacy_memory_node:
            logging.getLogger(__name__).warning(
                "graph_store.enable_legacy_memory_node=false: current write/link paths still depend on legacy MemoryNode methods; writes may fail until fully migrated."
            )
        # Optional debug line, controlled by env MEMORY_GRAPH_DEBUG
        if str(os.getenv("MEMORY_GRAPH_DEBUG", "")).lower() in ("1", "true", "yes"):  # pragma: no cover
            logging.getLogger(__name__).warning(
                "[GRAPH DEBUG] env_uri=%s final_uri=%s user=%s pwd_set=%s",
                n_uri_raw,
                n_uri_norm,
                _resolve_env_like_str(n_user, "neo4j"),
                bool(_resolve_env_like_str(n_pass, "")),
            )
    except Exception:
        pass
    neo = Neo4jStore({
        "uri": n_uri_norm,
        "user": _resolve_env_like_str(n_user, "neo4j"),
        "password": _resolve_env_like_str(n_pass, "password"),
        "database": _resolve_env_like_str(n_db, "neo4j"),
        "strict_tenant_mode": strict_tenant_mode,
        "enable_legacy_memory_node": enable_legacy_memory_node,
        "reliability": rcfg,
        "uri_debug": uri_meta,
    })
    try:
        neo.ensure_schema_v0()
    except Exception:
        pass
    audit = AuditStore()

    # Initialize usage WAL if not already active (ensure singleton usage)
    if _usage_wal is None:
        try:
            _usage_wal = _init_usage_wal()
        except Exception:
            pass

    return MemoryService(vectors, neo, audit, usage_wal=_usage_wal)


def _auth_settings() -> dict:
    cfg = load_memory_config()
    api = ((cfg.get("memory", {}) or {}).get("api", {}) or {})
    auth = (api.get("auth") or {}) if isinstance(api.get("auth"), dict) else {}
    token_map = {}
    jwt_cfg = auth.get("jwt") or {}
    signing_cfg = auth.get("signing") or {}
    try:
        tokens_cfg = auth.get("tokens") or []
        if isinstance(tokens_cfg, dict):
            # allow {token: tenant_id} shorthand
            token_map = {str(k): str(v) for k, v in tokens_cfg.items()}
        elif isinstance(tokens_cfg, list):
            for item in tokens_cfg:
                if not isinstance(item, dict):
                    continue
                tok = item.get("token")
                ten = item.get("tenant_id")
                if tok and ten:
                    token_map[str(tok)] = str(ten)
    except Exception:
        token_map = {}
    # env overrides
    try:
        env_enabled = str(os.getenv("MEMORY_API_AUTH_ENABLED", "")).lower()
        if env_enabled in ("true", "1", "yes"):
            auth["enabled"] = True
    except Exception:
        pass
    if os.getenv("MEMORY_API_TOKEN"):
        auth["token"] = os.getenv("MEMORY_API_TOKEN")
    if os.getenv("MEMORY_API_TENANT_ID"):
        auth["tenant_id"] = os.getenv("MEMORY_API_TENANT_ID")
    signing_required_env = os.getenv("MEMORY_API_SIGNING_REQUIRED")
    if signing_required_env:
        signing_cfg["required"] = str(signing_required_env).lower() in {"1", "true", "yes", "on"}
    return {
        "enabled": bool(auth.get("enabled", False)),
        "header": str(auth.get("header", "X-API-Token")),
        "token": str(auth.get("token", "")),
        "tenant_id": str(auth.get("tenant_id", "")),
        "token_map": token_map,
        "jwks_url": _resolve_env_like_str(jwt_cfg.get("jwks_url", ""), ""),
        "audience": _resolve_env_like_str(jwt_cfg.get("audience", ""), "") or None,
        "issuer": _resolve_env_like_str(jwt_cfg.get("issuer", ""), "") or None,
        "tenant_claim": str(jwt_cfg.get("tenant_claim", "tenant_id")),
        "algorithms": jwt_cfg.get("algorithms") or ["RS256"],
        "jwks_cache_seconds": _as_int(jwt_cfg.get("jwks_cache_seconds", DEFAULT_JWKS_CACHE_SECONDS), DEFAULT_JWKS_CACHE_SECONDS),
        "signing": {
            "required": bool(signing_cfg.get("required", False)),
            "header": str(signing_cfg.get("header", "X-Signature")),
            "timestamp_header": str(signing_cfg.get("timestamp_header", "X-Signature-Ts")),
            "max_skew_seconds": _as_int(signing_cfg.get("max_skew_seconds", 300), 300),
            "default_secret": str(signing_cfg.get("default_secret", "")),
            "secrets": {str(k): str(v) for k, v in (signing_cfg.get("secrets") or {}).items()},
        },
    }


def _api_limits() -> dict:
    cfg = load_memory_config()
    api = ((cfg.get("memory", {}) or {}).get("api", {}) or {})
    limits = (api.get("limits") or {}) if isinstance(api.get("limits"), dict) else {}
    env_bytes = os.getenv("MEMORY_API_MAX_REQUEST_BYTES")
    env_rate = os.getenv("MEMORY_API_RATE_LIMIT_PER_MINUTE")
    env_burst = os.getenv("MEMORY_API_RATE_LIMIT_BURST")
    env_high_cost_timeout = os.getenv("MEMORY_API_HIGH_COST_TIMEOUT_SECONDS")
    enabled_env = os.getenv("MEMORY_API_RATE_LIMIT_ENABLED")
    try:
        max_bytes = int(env_bytes) if env_bytes not in (None, "", "${MEMORY_API_MAX_REQUEST_BYTES}") else int(
            limits.get("max_request_bytes", MAX_REQUEST_BYTES_FALLBACK)
        )
    except Exception:
        max_bytes = MAX_REQUEST_BYTES_FALLBACK
    try:
        per_min = int(env_rate) if env_rate not in (None, "", "${MEMORY_API_RATE_LIMIT_PER_MINUTE}") else int(
            limits.get("requests_per_minute", RATE_LIMIT_PER_MINUTE_FALLBACK)
        )
    except Exception:
        per_min = RATE_LIMIT_PER_MINUTE_FALLBACK
    try:
        burst = int(env_burst) if env_burst not in (None, "", "${MEMORY_API_RATE_LIMIT_BURST}") else int(
            limits.get("burst", RATE_LIMIT_BURST_FALLBACK)
        )
    except Exception:
        burst = RATE_LIMIT_BURST_FALLBACK
    try:
        high_cost_timeout = float(env_high_cost_timeout) if env_high_cost_timeout not in (None, "", "${MEMORY_API_HIGH_COST_TIMEOUT_SECONDS}") else float(
            limits.get("high_cost_timeout_seconds", HIGH_COST_TIMEOUT_FALLBACK)
        )
    except Exception:
        high_cost_timeout = HIGH_COST_TIMEOUT_FALLBACK
    enabled = bool(limits.get("rate_limit_enabled", False))
    if enabled_env:
        enabled = str(enabled_env).lower() in {"1", "true", "yes", "on"}
    return {
        "max_request_bytes": max(0, max_bytes),
        "requests_per_minute": max(0, per_min),
        "burst": max(0, burst),
        "rate_limit_enabled": enabled,
        "per_tenant": limits.get("per_tenant", {}),
        "high_cost_timeout_seconds": max(1.0, high_cost_timeout),
    }


def _with_answer_settings() -> dict:
    cfg = load_memory_config()
    api = ((cfg.get("memory", {}) or {}).get("api", {}) or {})
    retrieval = api.get("retrieval") or {}
    enabled = bool(retrieval.get("with_answer_enabled", True))
    scope = str(retrieval.get("with_answer_scope") or "").strip()
    enabled_env = os.getenv("MEMORY_API_WITH_ANSWER_ENABLED")
    scope_env = os.getenv("MEMORY_API_WITH_ANSWER_SCOPE")
    if enabled_env not in (None, "", "${MEMORY_API_WITH_ANSWER_ENABLED}"):
        enabled = str(enabled_env).lower() in {"1", "true", "yes", "on"}
    if scope_env not in (None, "", "${MEMORY_API_WITH_ANSWER_SCOPE}"):
        scope = str(scope_env).strip()
    return {
        "enabled": enabled,
        "required_scope": scope,
    }


_JWK_CLIENTS: Dict[str, Tuple[PyJWKClient, float]] = {}


def _jwk_client(jwks_url: str, cache_seconds: int) -> PyJWKClient:
    now = time.time()
    cached = _JWK_CLIENTS.get(jwks_url)
    if cached and cached[1] > now:
        return cached[0]
    client = PyJWKClient(jwks_url, cache_keys=True, lifespan=cache_seconds)
    _JWK_CLIENTS[jwks_url] = (client, now + cache_seconds)
    return client


def _decode_jwt_token(token: str, settings: dict) -> Dict[str, Any]:
    jwks_url = settings.get("jwks_url") or ""
    if not jwks_url:
        raise InvalidTokenError("jwks_url_not_configured")
    client = _jwk_client(jwks_url, settings.get("jwks_cache_seconds", DEFAULT_JWKS_CACHE_SECONDS))
    signing_key = client.get_signing_key_from_jwt(token).key
    return jwt.decode(
        token,
        signing_key,
        algorithms=settings.get("algorithms") or ["RS256"],
        audience=settings.get("audience"),
        issuer=settings.get("issuer"),
    )


def _tenant_from_claims(claims: Dict[str, Any], settings: dict) -> str:
    claim_name = settings.get("tenant_claim", "tenant_id") or "tenant_id"
    tenant_val = claims.get(claim_name) or claims.get("tid")
    return str(tenant_val) if tenant_val is not None else ""


def _record_security_event(
    event: str, status: str, *, tenant: Optional[str], detail: Optional[str], request: Optional[Request]
) -> None:
    payload = {
        "event": event,
        "status": status,
        "tenant_id": tenant or "unknown",
        "path": getattr(request, "url", None).path if request else None,
    }
    if detail:
        payload["detail"] = detail
    audit_logger.info("[SECURITY] %s", payload)
    if event == "auth" and status != "ok":
        inc("auth_failures_total", 1)
    if event == "signature" and status != "ok":
        inc("signature_failures_total", 1)


def _tenant_from_headers(request: Request) -> str:
    tenant = request.headers.get("X-Tenant-ID") or request.headers.get("x-tenant-id")
    if not tenant:
        raise HTTPException(status_code=400, detail="missing X-Tenant-ID header")
    return str(tenant)


def _extract_bearer_token(raw: str) -> str:
    val = str(raw or "").strip()
    if not val:
        return ""
    parts = val.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return val


def _normalize_scopes(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        return list(sorted(dict.fromkeys(parts)))
    if isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
        return list(sorted(dict.fromkeys(parts)))
    return []


def _required_scope_for_path(path: str) -> Optional[str]:
    if path in PUBLIC_PATHS:
        return None
    scope, _ = _match_scope_mapping(path)
    if scope is not None:
        return scope
    return "memory.admin"


def _check_scope(ctx: AuthContext, required_scope: Optional[str]) -> None:
    if required_scope is None:
        return
    if ctx.get("method") == "disabled":
        return
    scopes = list(ctx.get("scopes") or [])
    if not scopes:
        # Legacy tokens without scopes: allow to preserve userspace compatibility.
        return
    if "memory.admin" in scopes:
        return
    if required_scope not in scopes:
        raise HTTPException(status_code=403, detail="insufficient_scope")


def _require_explicit_scope(ctx: AuthContext, *allowed_scopes: str) -> None:
    if ctx.get("method") == "disabled":
        return
    scopes = list(ctx.get("scopes") or [])
    if "memory.admin" in scopes:
        return
    if not scopes:
        raise HTTPException(status_code=403, detail="insufficient_scope")
    if not any(scope in scopes for scope in allowed_scopes):
        raise HTTPException(status_code=403, detail="insufficient_scope")


def _resolve_request_api_key_id(request: Request, ctx: AuthContext) -> Optional[str]:
    for header in ("X-API-Key-Id", "X-Principal-ApiKey-Id"):
        value = str(request.headers.get(header) or "").strip()
        if value:
            return value
    subject = str(ctx.get("subject") or "").strip()
    return subject or None


def _get_tenant_clear_lock(tenant_id: str) -> asyncio.Lock:
    tid = str(tenant_id or "").strip()
    with _tenant_clear_locks_guard:
        lock = _tenant_clear_locks.get(tid)
        if lock is None:
            lock = asyncio.Lock()
            _tenant_clear_locks[tid] = lock
        return lock


def _lookup_scope_mapping(path: str) -> tuple[Optional[str], bool]:
    """Return (required_scope, matched_explicit_rule)."""
    if path in PUBLIC_PATHS:
        return None, True
    scope, matched = _match_scope_mapping(path)
    if matched:
        return scope, True
    return "memory.admin", False


def _match_scope_mapping(path: str) -> tuple[Optional[str], bool]:
    # Exact matches must win over prefix rules, otherwise `/memory/v1/clear`
    # gets swallowed by the broader `/memory/v1/` read scope.
    for pattern, scope in PATH_SCOPE_REQUIREMENTS.items():
        if not pattern.endswith("/") and path == pattern:
            return scope, True
    for pattern, scope in PATH_SCOPE_REQUIREMENTS.items():
        if pattern.endswith("/") and path.startswith(pattern):
            return scope, True
    return None, False


def _category_for_path(path: str) -> tuple[str, str]:
    """Business-domain classification used by /api/list."""
    if path in PUBLIC_PATHS or path.startswith("/metrics"):
        return "health_metrics", "健康与监控"
    if path.startswith("/ingest"):
        return "ingest", "写入与任务"
    if path.startswith("/retrieval"):
        return "retrieval", "检索编排"
    if path in {"/search", "/timeline_summary", "/object_search", "/speech_search", "/entity_event_anchor"}:
        return "search", "核心检索"
    if path.startswith("/graph"):
        return "graph", "图谱接口"
    if path.startswith("/memory/v1"):
        return "memory_semantic", "语义记忆"
    if path.startswith("/memory/agentic"):
        return "memory_agentic", "Agentic 语义路由"
    if path.startswith("/memory/state"):
        return "memory_state", "状态记忆"
    if path.startswith("/config"):
        return "config", "配置管理"
    if path.startswith("/admin") or path.startswith("/equiv") or path in {
        "/write",
        "/update",
        "/delete",
        "/link",
        "/rollback",
        "/batch_delete",
        "/batch_link",
    }:
        return "admin", "管理接口"
    if path.startswith("/api/"):
        return "meta", "元信息"
    return "other", "其他"


def _iter_api_routes() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in app.routes:
        path = getattr(r, "path", None)
        if not path or not isinstance(path, str):
            continue
        if path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/redoc"):
            continue
        methods = list(getattr(r, "methods", []) or [])
        # Drop implicit HEAD/OPTIONS to reduce noise for callers.
        methods = sorted({m for m in methods if m and m.upper() not in {"HEAD", "OPTIONS"}})
        out.append({"path": path, "methods": methods})
    out.sort(key=lambda x: (x["path"], ",".join(x["methods"])))
    return out


_AGENTIC_ROUTER_SYSTEM_PROMPT = """你是 memory tools 的单步路由器。
你只能选择 1 个最合适的工具，不能生成自然语言答案。
不要在工具调用之外输出任何文本。
必须严格输出工具调用：function.name + function.arguments(JSON)。
禁止臆造 UUID；优先使用实体名/话题名/属性名文本参数。
若没有合适工具，不调用工具。
高频意图映射：entity_profile=实体画像；topic_timeline=话题时间线；time_since=上次提及时间；quotes=原话；relations=关系。"""


def _env_flag(name: str, default: bool = True) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _agentic_router_model_override() -> Optional[str]:
    raw = str(os.getenv("MEMORY_AGENTIC_ROUTER_MODEL") or "").strip()
    return raw or None


def _agentic_router_provider_override() -> Optional[str]:
    raw = str(os.getenv("MEMORY_AGENTIC_ROUTER_PROVIDER") or "").strip()
    return raw or None


def _agentic_router_timeout_s() -> float:
    try:
        return max(1.0, min(float(os.getenv("MEMORY_AGENTIC_ROUTER_TIMEOUT_S", "8") or 8), 60.0))
    except Exception:
        return 8.0


def _agentic_runtime_timeout_s() -> float:
    try:
        return max(1.0, min(float(os.getenv("MEMORY_AGENTIC_RUNTIME_TIMEOUT_S", "30") or 30), 120.0))
    except Exception:
        return 30.0


def _agentic_infra_base_url(request: Request) -> str:
    raw = str(os.getenv("MEMORY_AGENTIC_INFRA_BASE_URL") or "").strip()
    if raw:
        return raw.rstrip("/")
    return str(request.base_url).rstrip("/")


def _agentic_request_id(body_request_id: Optional[str], request: Request) -> Optional[str]:
    rid = str(body_request_id or "").strip() or _request_id_from_request(request)
    return str(rid or "").strip() or None


def _agentic_effective_tool_names(
    tool_whitelist: Optional[List[str]],
    *,
    include_disabled_default: bool = False,
) -> List[str]:
    from modules.memory.adk.tool_definitions import TOOL_DEFINITIONS, get_tool_definitions

    if tool_whitelist is None:
        defs = get_tool_definitions(enabled_only=(not include_disabled_default))
        return [d.name for d in defs]

    names: List[str] = []
    for raw in tool_whitelist:
        name = str(raw or "").strip()
        if not name:
            continue
        if name not in names:
            names.append(name)
    if not names:
        raise HTTPException(status_code=400, detail="empty_tool_whitelist")
    unknown = [n for n in names if n not in TOOL_DEFINITIONS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown_tools:{','.join(unknown)}")
    return names


def _agentic_tools_payload(format_name: str, names: List[str]) -> List[Dict[str, Any]]:
    from modules.memory.adk import to_mcp_tools, to_openai_tools

    fmt = str(format_name or "openai").strip().lower()
    if fmt == "openai":
        return to_openai_tools(names=names)
    if fmt == "mcp":
        return to_mcp_tools(names=names)
    raise HTTPException(status_code=400, detail="invalid_tools_format")


def _agentic_validate_tool_args(tool_name: str, args: Any) -> Dict[str, Any]:
    from modules.memory.adk.tool_definitions import TOOL_DEFINITIONS

    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="tool_args_must_be_object")
    definition = TOOL_DEFINITIONS.get(str(tool_name or "").strip())
    if definition is None:
        raise HTTPException(status_code=400, detail=f"unknown_tool:{tool_name}")
    schema = dict(definition.input_schema or {})
    props = schema.get("properties") or {}
    allowed = set(str(k) for k in props.keys())
    normalized: Dict[str, Any] = {}
    for key, value in args.items():
        name = str(key or "").strip()
        if not name:
            continue
        if name not in allowed:
            raise HTTPException(status_code=400, detail=f"unknown_tool_arg:{name}")
        normalized[name] = value
    required = [str(x) for x in (schema.get("required") or []) if str(x).strip()]
    missing: List[str] = []
    for key in required:
        val = normalized.get(key)
        if val is None:
            missing.append(key)
            continue
        if isinstance(val, str) and not val.strip():
            missing.append(key)
    if missing:
        raise HTTPException(status_code=400, detail=f"missing_required_args:{','.join(missing)}")
    return normalized


def _agentic_auth_token_from_request(request: Request) -> Optional[str]:
    direct = _extract_bearer_token(request.headers.get("Authorization") or request.headers.get("authorization") or "")
    if direct:
        return direct
    settings = _auth_settings()
    header = str(settings.get("header") or "").strip()
    if not header:
        return None
    raw = str(request.headers.get(header) or "").strip()
    if not raw:
        return None
    return _extract_bearer_token(raw) or raw


def _create_agentic_runtime(*, request: Request, tenant_id: str, user_tokens: List[str]):
    from modules.memory.adk import create_memory_runtime

    token = _agentic_auth_token_from_request(request)
    return create_memory_runtime(
        base_url=_agentic_infra_base_url(request),
        tenant_id=tenant_id,
        user_tokens=list(user_tokens),
        auth_token=token,
        auth_header="Authorization",
        timeout_s=_agentic_runtime_timeout_s(),
        verify_tls=_env_flag("MEMORY_AGENTIC_RUNTIME_VERIFY_TLS", True),
    )


async def _agentic_route_tool_call(
    *,
    query: str,
    tools: List[Dict[str, Any]],
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    from openai import AsyncOpenAI

    target = resolve_openai_compatible_chat_target(
        kind="agentic_router",
        provider_override=_agentic_router_provider_override(),
        model_override=str(model or _agentic_router_model_override() or "").strip() or None,
        base_url_override=str(os.getenv("MEMORY_AGENTIC_ROUTER_BASE_URL") or "").strip() or None,
    )
    if not target:
        raise HTTPException(status_code=503, detail="agentic_router_unavailable")

    model_name = str(target.get("model") or "").strip()
    if not model_name:
        raise HTTPException(status_code=503, detail="agentic_router_unavailable")

    client_kwargs: Dict[str, Any] = {
        "timeout": _agentic_router_timeout_s(),
        "api_key": str(target.get("api_key") or "EMPTY"),
    }
    base_url = str(target.get("base_url") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _AGENTIC_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
    except Exception as exc:
        logger.warning(
            "agentic.router.error",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "error": str(exc)[:200],
            },
        )
        raise HTTPException(status_code=503, detail="agentic_router_unavailable") from None

    choice = (list(resp.choices or []) or [None])[0]
    message = choice.message if choice is not None else None
    tool_calls = list(getattr(message, "tool_calls", []) or [])
    out: Dict[str, Any] = {
        "has_tool_call": bool(tool_calls),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "model": str(getattr(resp, "model", model_name) or model_name),
        "provider": str(target.get("provider") or ""),
    }
    if not tool_calls:
        return out

    call = tool_calls[0]
    fn = getattr(call, "function", None)
    tool_name = str(getattr(fn, "name", "") or "").strip()
    raw_args = str(getattr(fn, "arguments", "") or "")
    parsed_args: Dict[str, Any] = {}
    args_invalid = False
    if raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                parsed_args = parsed
            else:
                args_invalid = True
        except Exception:
            args_invalid = True

    out.update(
        {
            "tool_call_id": str(getattr(call, "id", "") or ""),
            "tool_name": tool_name,
            "tool_args": parsed_args,
            "tool_args_raw": raw_args,
            "tool_args_invalid": bool(args_invalid),
        }
    )
    return out


async def _execute_agentic_tool(*, runtime: Any, tool_name: str, args: Dict[str, Any]):
    from modules.memory.adk import ToolResult

    fn = getattr(runtime, str(tool_name or "").strip(), None)
    if not callable(fn):
        return ToolResult.no_match(message=f"unknown tool: {tool_name}")
    try:
        result = await fn(**dict(args or {}))
    except TypeError:
        return ToolResult.no_match(message="tool_args_invalid")
    except Exception:
        logger.exception("agentic.execute.error", extra={"tool_name": tool_name})
        return ToolResult.no_match(message="tool_execution_failed")

    if hasattr(result, "to_llm_dict") and hasattr(result, "to_wire_dict"):
        return result
    if isinstance(result, dict):
        return ToolResult.success(data=result)
    return ToolResult.no_match(message="tool_execution_failed")


def _authenticate_request(request: Request) -> AuthContext:
    settings = _auth_settings()
    header = settings.get("header") or "X-API-Token"
    token = request.headers.get(header, "")
    if str(header).lower() == "authorization":
        token = _extract_bearer_token(token)
    if not token:
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization") or ""
        token = _extract_bearer_token(auth_header)
    ctx: AuthContext = {"claims": {}, "token": token}
    if not settings.get("enabled"):
        tenant = _tenant_from_headers(request)
        ctx.update({"tenant_id": tenant, "method": "disabled", "scopes": []})
        return ctx
    if not token:
        _record_security_event("auth", "missing_token", tenant=None, detail=None, request=request)
        raise HTTPException(status_code=401, detail="missing_token")

    token_map = settings.get("token_map") or {}
    try:
        if settings.get("jwks_url"):
            claims = _decode_jwt_token(token, settings)
            scopes = _normalize_scopes(claims.get("scopes") or claims.get("scope"))
            tenant_id = _tenant_from_claims(claims, settings) or token_map.get(token) or settings.get("tenant_id") or ""
            if not tenant_id:
                _record_security_event("auth", "tenant_claim_missing", tenant=None, detail=None, request=request)
                raise HTTPException(status_code=401, detail="tenant_claim_missing")
            ctx.update({
                "claims": claims,
                "tenant_id": tenant_id,
                "subject": str(claims.get("sub", "")),
                "method": "jwt",
                "scopes": scopes,
            })
            _record_security_event("auth", "ok", tenant=tenant_id, detail="jwt", request=request)
            return ctx
        if token in token_map:
            tenant_id = token_map[token]
            ctx.update({"tenant_id": tenant_id, "method": "token_map", "scopes": []})
            _record_security_event("auth", "ok", tenant=tenant_id, detail="token_map", request=request)
            return ctx
        expected = settings.get("token") or ""
        if expected and token == expected:
            tenant_id = settings.get("tenant_id") or token_map.get(token)
            if not tenant_id:
                _record_security_event("auth", "tenant_missing", tenant=None, detail="token_missing_tenant", request=request)
                raise HTTPException(status_code=401, detail="tenant_not_configured")
            ctx.update({"tenant_id": tenant_id, "method": "static_token", "scopes": []})
            _record_security_event("auth", "ok", tenant=tenant_id, detail="static_token", request=request)
            return ctx
    except InvalidTokenError as exc:
        _record_security_event("auth", "invalid_token", tenant=None, detail=str(exc), request=request)
        raise HTTPException(status_code=401, detail="invalid_token") from None

    _record_security_event("auth", "unauthorized", tenant=None, detail="token_rejected", request=request)
    raise HTTPException(status_code=401, detail="unauthorized")


async def _get_request_body_bytes(request: Request) -> bytes:
    if hasattr(request, "_body") and getattr(request, "_body") is not None:
        return getattr(request, "_body")
    body = await request.body()
    request._body = body
    return body


def _parse_payload_raw(payload_raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload_raw:
        return None
    try:
        if isinstance(payload_raw, (bytes, bytearray)):
            payload_raw = payload_raw.decode("utf-8")
        data = json.loads(payload_raw)
        return dict(data) if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _parse_time_value(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except Exception:
            return None
    return _parse_iso_datetime(raw)


def _parse_time_range_dict(time_range: Optional[Dict[str, Any]]) -> tuple[Optional[datetime], Optional[datetime]]:
    if not isinstance(time_range, dict):
        return None, None
    start = time_range.get("start") or time_range.get("gte") or time_range.get("from")
    end = time_range.get("end") or time_range.get("lte") or time_range.get("to")
    return _parse_time_value(start), _parse_time_value(end)


def _decode_cursor(cursor: Optional[str]) -> int:
    raw = str(cursor or "").strip()
    if not raw:
        return 0
    if raw.startswith(("c:", "o:")):
        raw = raw[2:]
    if raw.isdigit():
        return max(0, int(raw))
    return 0


def _encode_cursor(offset: int) -> str:
    return f"c:{max(0, int(offset))}"


def _parse_user_tokens_param(value: Optional[List[str] | str]) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    raw = str(value or "").strip()
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _normalize_confidence(score: Any) -> Optional[float]:
    if score is None:
        return None
    try:
        val = float(score)
    except Exception:
        return None
    # Heuristic: scores around ~2-3 are common for fulltext; map to (0,1]
    return max(0.0, min(1.0, val / 3.0))


def _map_match_source(matched: Optional[str]) -> Optional[str]:
    raw = str(matched or "").strip().lower()
    if not raw:
        return None
    if "alias" in raw:
        return "alias"
    if raw in ("fulltext", "exact"):
        return "exact"
    return "fuzzy"


def _topic_status_from_last(last_ts: Optional[datetime]) -> str:
    if last_ts is None:
        return "paused"
    now = datetime.now(timezone.utc)
    delta = now - last_ts
    days = delta.total_seconds() / 86400.0
    if days <= max(1, int(TOPIC_STATUS_ONGOING_DAYS)):
        return "ongoing"
    # If no explicit "done" signal is available, keep paused for long gaps
    if days <= max(1, int(TOPIC_STATUS_PAUSED_DAYS)):
        return "paused"
    return "paused"


async def _resolve_entity_candidates(
    *,
    tenant_id: str,
    name: str,
    entity_type: Optional[str] = None,
    limit: int = 5,
) -> tuple[Optional[str], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    resolve_kwargs = {
        "tenant_id": tenant_id,
        "name": name,
        "limit": limit,
    }
    if entity_type:
        resolve_kwargs["entity_type"] = entity_type
    hits = await graph_svc.resolve_entities(**resolve_kwargs)
    candidates: List[Dict[str, Any]] = []
    for h in hits or []:
        ent_id = str(h.get("entity_id") or "").strip()
        if not ent_id:
            continue
        cand = {
            "id": ent_id,
            "name": h.get("name"),
            "type": h.get("type"),
            "confidence": _normalize_confidence(h.get("score")),
            "match_source": _map_match_source(h.get("matched")),
        }
        candidates.append(cand)

    resolved = candidates[0] if candidates else None
    entity_id = resolved.get("id") if resolved else None
    if not candidates:
        return None, None, []

    # ambiguity rules
    top1_conf = resolved.get("confidence") if resolved else None
    top2_conf = candidates[1].get("confidence") if len(candidates) > 1 else None
    exact_matches = [c for c in candidates if c.get("match_source") in ("exact", "alias")]
    ambiguous = False
    if top1_conf is None or top1_conf < 0.7:
        ambiguous = True
    if top2_conf is not None and top1_conf is not None and (top1_conf - top2_conf) < 0.2:
        ambiguous = True
    if len(exact_matches) > 1:
        ambiguous = True

    return entity_id, resolved, (candidates if ambiguous else [])


_STATE_PROPERTIES_CACHE: Optional[Dict[str, Any]] = None


def _load_state_properties() -> Dict[str, Any]:
    global _STATE_PROPERTIES_CACHE
    if _STATE_PROPERTIES_CACHE is not None:
        return _STATE_PROPERTIES_CACHE
    root = Path(__file__).resolve().parents[3]
    fp = root / "modules" / "memory" / "vocab" / "state_properties.yaml"
    data: Dict[str, Any] = {}
    try:
        data = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    _STATE_PROPERTIES_CACHE = data
    return data


def _entity_display_name(ent: Dict[str, Any]) -> Optional[str]:
    if not isinstance(ent, dict):
        return None
    for key in ("name", "manual_name", "cluster_label"):
        val = ent.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def _build_entity_name_map(entities: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ent in entities or []:
        eid = str(ent.get("id") or ent.get("entity_id") or "").strip()
        if not eid:
            continue
        name = _entity_display_name(ent)
        if name:
            out[eid] = name
    return out


def _quotes_from_bundle(
    bundle: Dict[str, Any],
    *,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    quotes: List[Dict[str, Any]] = []
    event_obj = bundle.get("event") if isinstance(bundle, dict) else None
    event_id = None
    event_time = None
    topic_path = None
    if isinstance(event_obj, dict):
        event_id = str(event_obj.get("id") or "").strip() or None
        event_time = event_obj.get("t_abs_start") or event_obj.get("t_abs_end")
        topic_path = event_obj.get("topic_path")

    speaker_map: Dict[str, str] = {}
    speaker_name_map: Dict[str, str] = {}
    entity_name_map = _build_entity_name_map(list(bundle.get("entities") or []))
    for item in bundle.get("utterance_speakers") or []:
        uid = str(item.get("utterance_id") or "").strip()
        sid = str(item.get("entity_id") or "").strip()
        if uid and sid:
            speaker_map[uid] = sid
            if sid in entity_name_map:
                speaker_name_map[uid] = entity_name_map[sid]

    for utt in bundle.get("utterances") or []:
        utt_id = str(utt.get("id") or utt.get("utterance_id") or "").strip()
        text = str(utt.get("raw_text") or utt.get("text") or "").strip()
        if not utt_id or not text:
            continue
        speaker_id = speaker_map.get(utt_id)
        if entity_id and speaker_id and speaker_id != entity_id:
            continue
        if entity_id and speaker_id is None:
            continue
        quotes.append(
            {
                "utterance_id": utt_id,
                "text": text,
                "event_id": event_id,
                "when": event_time,
                "t_media_start": utt.get("t_media_start"),
                "t_media_end": utt.get("t_media_end"),
                "speaker_id": speaker_id,
                "speaker_name": speaker_name_map.get(utt_id),
                "speaker_track_id": utt.get("speaker_track_id"),
                "topic_path": topic_path,
            }
        )
    return quotes


def _timeline_status(events: List[Dict[str, Any]], *, recent_days: int = 30) -> str:
    if not events:
        return "done"
    last_ts: Optional[datetime] = None
    for ev in reversed(events):
        ts = _parse_iso_datetime(ev.get("t_abs_start") or ev.get("timestamp_iso") or ev.get("timestamp"))
        if ts is not None:
            last_ts = ts
            break
    if last_ts is None:
        return "paused"
    now = datetime.now(timezone.utc)
    if now - last_ts <= dt.timedelta(days=max(1, int(recent_days))):
        return "ongoing"
    return "paused"


async def _require_signature(request: Request, ctx: AuthContext) -> None:
    settings = _auth_settings()
    signing = settings.get("signing") or {}
    if not signing.get("required"):
        return
    tenant_id = ctx.get("tenant_id") or ""
    secret_map = signing.get("secrets") or {}
    secret = secret_map.get(tenant_id) or signing.get("default_secret")
    if not secret:
        _record_security_event("signature", "secret_missing", tenant=tenant_id, detail=None, request=request)
        raise HTTPException(status_code=401, detail="signature_required")
    sig_header = signing.get("header", "X-Signature")
    ts_header = signing.get("timestamp_header", "X-Signature-Ts")
    sig = request.headers.get(sig_header)
    ts_raw = request.headers.get(ts_header)
    if not sig or not ts_raw:
        _record_security_event("signature", "missing", tenant=tenant_id, detail=None, request=request)
        raise HTTPException(status_code=401, detail="signature_required")
    try:
        ts_val = int(ts_raw)
    except Exception:
        _record_security_event("signature", "timestamp_invalid", tenant=tenant_id, detail=None, request=request)
        raise HTTPException(status_code=401, detail="signature_invalid")
    skew = abs(time.time() - ts_val)
    if skew > int(signing.get("max_skew_seconds", 300)):
        _record_security_event("signature", "expired", tenant=tenant_id, detail=str(skew), request=request)
        raise HTTPException(status_code=401, detail="signature_expired")
    body = await _get_request_body_bytes(request)
    payload = f"{ts_val}.{request.url.path}".encode() + b"." + body
    expected = hmac.new(str(secret).encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        _record_security_event("signature", "mismatch", tenant=tenant_id, detail=None, request=request)
        raise HTTPException(status_code=401, detail="signature_invalid")
    _record_security_event("signature", "ok", tenant=tenant_id, detail=None, request=request)


def _enforce_with_answer(ctx: AuthContext) -> None:
    if not WITH_ANSWER_SETTINGS.get("enabled", True):
        raise HTTPException(status_code=403, detail="with_answer_disabled")
    required_scope = str(WITH_ANSWER_SETTINGS.get("required_scope") or "").strip()
    if not required_scope:
        return
    scopes = list(ctx.get("scopes") or [])
    if not scopes:
        return
    if "memory.admin" in scopes:
        return
    if required_scope not in scopes:
        raise HTTPException(status_code=403, detail="with_answer_forbidden")


async def _enforce_security(request: Request, *, require_signature: bool = False) -> AuthContext:
    ctx = _authenticate_request(request)
    required_scope = _required_scope_for_path(request.url.path)
    _check_scope(ctx, required_scope)
    if require_signature:
        await _require_signature(request, ctx)
    try:
        update_request_context(tenant_id=str(ctx.get("tenant_id") or ""))
    except Exception:
        pass
    return ctx


def _require_auth(request: Request) -> None:
    _authenticate_request(request)


def _resolve_tenant(request: Request) -> str:
    """Resolve tenant id via authentication context."""

    ctx = _authenticate_request(request)
    tenant_id = ctx.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    return str(tenant_id)


def _inject_tenant(body: Any, tenant_id: str) -> GraphUpsertRequest:
    """Force all nodes/edges to carry the caller tenant_id to prevent cross-tenant writes.

    Accepts dict or GraphUpsertRequest for flexibility; returns validated model.
    """

    try:
        data = body.model_dump() if hasattr(body, "model_dump") else dict(body)
    except Exception:
        raise HTTPException(status_code=422, detail="invalid_graph_payload")

    for seq_name in ("segments", "evidences", "entities", "events", "places", "time_slices", "edges"):
        items = data.get(seq_name) or []
        new_items = []
        for item in items:
            if not isinstance(item, dict):
                try:
                    item = dict(item)
                except Exception:
                    continue
            item["tenant_id"] = tenant_id
            new_items.append(item)
        data[seq_name] = new_items
    # Pending equivs
    peq_items = data.get("pending_equivs") or []
    new_peq = []
    for item in peq_items:
        if not isinstance(item, dict):
            try:
                item = dict(item)
            except Exception:
                continue
        item["tenant_id"] = tenant_id
        new_peq.append(item)
    data["pending_equivs"] = new_peq
    try:
        return GraphUpsertRequest.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _content_length(request: Request) -> int:
    try:
        return int(request.headers.get("content-length", "0"))
    except Exception:
        return 0


def _ensure_request_id(request: Request) -> str:
    rid = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
    if rid:
        rid = str(rid).strip()
    if not rid:
        rid = f"req_{uuid.uuid4().hex}"
    request.state.request_id = rid
    return rid


def _request_id_from_request(request: Request) -> Optional[str]:
    rid = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
    if rid:
        return str(rid).strip()
    return getattr(request.state, "request_id", None)


def _resolve_llm_adapter_from_client_meta(
    client_meta: Optional[Dict[str, Any]],
) -> Tuple[Optional[LLMAdapter], str, str]:
    """Resolve per-request LLM adapter from client_meta.llm_* (BYOK or platform)."""
    if not isinstance(client_meta, dict):
        return None, "platform", "missing"
    llm_mode = str(client_meta.get("llm_mode") or "").strip().lower()
    provider = str(client_meta.get("llm_provider") or "").strip()
    model = str(client_meta.get("llm_model") or "").strip()
    api_key = str(client_meta.get("llm_api_key") or "").strip()
    base_url = str(
        client_meta.get("llm_base_url")
        or client_meta.get("llm_api_base")
        or ""
    ).strip()

    has_any = bool(provider or model or api_key or base_url)
    if llm_mode == "platform":
        return None, "platform", "platform"
    if not llm_mode and not has_any:
        return None, "platform", "missing"
    if provider and model and api_key:
        adapter = build_llm_from_byok(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=(base_url if base_url else None),
        )
        if adapter is None:
            return None, "platform", "adapter_missing"
        return adapter, "byok", "hit"
    if llm_mode == "byok":
        if provider and model and not api_key:
            return None, "platform", "platform_key_missing"
        return None, "platform", "config_incomplete"
    if provider or model or api_key:
        return None, "platform", "config_incomplete"
    return None, "platform", "missing"


def _normalize_ingest_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize ingest turns for job execution.

    Stage2 turn marks require `turn_id` and `text` fields. Some clients send
    only {role, content}; normalize to a stable shape.
    """
    out: List[Dict[str, Any]] = []
    idx = 0
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        idx += 1
        d = dict(t)
        tid = str(d.get("turn_id") or "").strip()
        if not tid:
            d["turn_id"] = f"t{idx:04d}"
        text = d.get("text")
        if not isinstance(text, str) or not text.strip():
            content = d.get("content")
            if isinstance(content, str) and content.strip():
                d["text"] = content
        out.append(d)
    return out


def _missing_core_requirements(missing: List[str], *, target: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "code": "missing_core_requirements",
            "message": f"Missing required data for core {target}",
            "missing": list(missing),
        },
    )


def _llm_meta_missing(client_meta: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(client_meta, dict):
        return []
    llm_mode = str(client_meta.get("llm_mode") or "").strip().lower()
    provider = str(client_meta.get("llm_provider") or "").strip()
    model = str(client_meta.get("llm_model") or "").strip()
    api_key = str(client_meta.get("llm_api_key") or "").strip()
    base_url = str(
        client_meta.get("llm_base_url")
        or client_meta.get("llm_api_base")
        or ""
    ).strip()
    has_any = bool(provider or model or api_key or base_url)
    if llm_mode == "platform" or (not llm_mode and not has_any):
        return []
    missing: List[str] = []
    if not provider:
        missing.append("client_meta.llm_provider")
    if not model:
        missing.append("client_meta.llm_model")
    if not api_key:
        missing.append("client_meta.llm_api_key")
    return missing


def _client_meta_missing(client_meta: Optional[Dict[str, Any]]) -> List[str]:
    missing: List[str] = []
    if not isinstance(client_meta, dict):
        return ["client_meta.memory_policy", "client_meta.user_id"]
    if not str(client_meta.get("memory_policy") or "").strip():
        missing.append("client_meta.memory_policy")
    if not str(client_meta.get("user_id") or "").strip():
        missing.append("client_meta.user_id")
    return missing


def _resolve_rate_limit_key(request: Request) -> str:
    """Best-effort tenant key for throttling without enforcing auth in middleware."""

    settings = _auth_settings()
    header = settings.get("header") or "X-API-Token"
    token_map = settings.get("token_map") or {}
    token = request.headers.get(header, "")
    if token and token in token_map:
        return token_map[token]
    if token and token == (settings.get("token") or "") and settings.get("tenant_id"):
        return settings["tenant_id"]
    tenant = request.headers.get("X-Tenant-ID") or request.headers.get("x-tenant-id")
    if tenant:
        return str(tenant)
    client_host = getattr(request.client, "host", "") or "anonymous"
    return f"anon:{client_host}"


API_LIMITS = _api_limits()
WITH_ANSWER_SETTINGS = _with_answer_settings()
MAX_REQUEST_BYTES = API_LIMITS.get("max_request_bytes", MAX_REQUEST_BYTES_FALLBACK)
RATE_LIMITER: Optional[TenantRateLimiter] = None
if API_LIMITS.get("rate_limit_enabled"):
    RATE_LIMITER = TenantRateLimiter(
        API_LIMITS.get("requests_per_minute", RATE_LIMIT_PER_MINUTE_FALLBACK),
        API_LIMITS.get("burst", RATE_LIMIT_BURST_FALLBACK),
    )
PER_TENANT_LIMITERS: Dict[str, TenantRateLimiter] = {}
for tenant_key, cfg in (API_LIMITS.get("per_tenant") or {}).items():
    if not isinstance(cfg, dict):
        continue
    try:
        rpm = _as_int(
            cfg.get("requests_per_minute"), API_LIMITS.get("requests_per_minute", RATE_LIMIT_PER_MINUTE_FALLBACK)
        )
        burst = _as_int(cfg.get("burst"), API_LIMITS.get("burst", RATE_LIMIT_BURST_FALLBACK))
        PER_TENANT_LIMITERS[str(tenant_key)] = TenantRateLimiter(rpm, burst)
    except Exception:
        continue
cfg = load_memory_config()
reliability_cfg = ((cfg.get("memory", {}) or {}).get("reliability", {}) or {})
cb_cfg = reliability_cfg.get("circuit_breaker", {}) if isinstance(reliability_cfg.get("circuit_breaker"), dict) else {}
_cb_failure_threshold = _as_int(cb_cfg.get("failure_threshold"), HIGH_COST_FAILURE_THRESHOLD)
_cb_cooldown_seconds = _as_int(cb_cfg.get("cooldown_seconds"), HIGH_COST_COOLDOWN_SECONDS)
HIGH_COST_TIMEOUT = float(API_LIMITS.get("high_cost_timeout_seconds", HIGH_COST_TIMEOUT_FALLBACK))
_api_topk = get_api_topk_defaults(cfg)
_API_DEFAULT_TOPK_SEARCH = int(_api_topk.get("search", 10))
_API_DEFAULT_TOPK_RETRIEVAL = int(_api_topk.get("retrieval", 15))
_API_DEFAULT_TOPK_GRAPH_SEARCH = int(_api_topk.get("graph_search", 10))
_ingest_exec = get_ingest_executor_settings(cfg)
_INGEST_EXECUTOR_CONFIG = IngestExecutorConfig(
    enabled=bool(_ingest_exec.get("enabled", True)),
    worker_count=int(_ingest_exec.get("worker_count", 2) or 2),
    queue_maxsize=int(_ingest_exec.get("queue_maxsize", 0) or 0),
    global_concurrency=int(_ingest_exec.get("global_concurrency", 2) or 2),
    per_tenant_concurrency=int(_ingest_exec.get("per_tenant_concurrency", 1) or 1),
    job_timeout_s=int(_ingest_exec.get("job_timeout_s", 300) or 300),
    shutdown_grace_s=int(_ingest_exec.get("shutdown_grace_s", 30) or 30),
    recover_stale_s=int(_ingest_exec.get("recover_stale_s", 3600) or 3600),
    retry_delay_s=int(_ingest_exec.get("retry_delay_s", 60) or 60),
    max_retries=int(_ingest_exec.get("max_retries", 3) or 3),
)
_INGEST_EXECUTOR_ENABLED = bool(_INGEST_EXECUTOR_CONFIG.enabled)
HIGH_COST_BREAKERS: Dict[str, SimpleCircuitBreaker] = {
    "search": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "timeline_summary": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "graph_expand": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "graph_search": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "retrieval": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "topic_timeline": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "entity_profile": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "quotes": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "relations": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
    "time_since": SimpleCircuitBreaker(_cb_failure_threshold, _cb_cooldown_seconds),
}

_LIMIT_SKIP_PATHS = {"/health", "/metrics", "/metrics_prom"}
_MUTATION_PATHS = {
    "/write",
    "/update",
    "/delete",
    "/link",
    "/rollback",
    "/batch_delete",
    "/batch_link",
    "/graph/v0/upsert",
    "/graph/v0/admin/equiv/pending",
    "/graph/v0/admin/equiv/approve",
    "/graph/v0/admin/equiv/reject",
    "/graph/v0/admin/purge_source",
    "/ingest",
    "/ingest/dialog/v1",
}


async def _ensure_request_size(request: Request) -> Optional[PlainTextResponse]:
    if MAX_REQUEST_BYTES <= 0:
        return None
    if request.method.upper() not in {"POST", "PUT", "PATCH"}:
        return None
    if request.url.path in _LIMIT_SKIP_PATHS:
        return None
    content_len = _content_length(request)
    if content_len > MAX_REQUEST_BYTES:
        inc("request_too_large_total", 1)
        return PlainTextResponse("request entity too large", status_code=413)
    if content_len == 0:
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            inc("request_too_large_total", 1)
            return PlainTextResponse("request entity too large", status_code=413)
        request._body = body  # allow downstream reuse
    return None


async def _maybe_rate_limit(request: Request) -> Optional[PlainTextResponse]:
    if request.url.path not in _MUTATION_PATHS:
        return None
    key = _resolve_rate_limit_key(request)
    limiter = PER_TENANT_LIMITERS.get(key) or RATE_LIMITER
    if limiter is None:
        return None
    if limiter.allow(key):
        return None
    inc("throttled_requests_total", 1)
    return PlainTextResponse("rate limit exceeded", status_code=429, headers={"Retry-After": "60"})


def _graph_status_from_code(code: int) -> str:
    if 400 <= code < 500:
        return "client_error"
    if code >= 500:
        return "server_error"
    return "ok"


def _record_graph_metrics(endpoint: str, started_at: float, status: str = "ok") -> None:
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    record_graph_request(endpoint, status)
    add_graph_latency(endpoint, duration_ms)


def _breaker_for(endpoint_key: str) -> Optional[SimpleCircuitBreaker]:
    return HIGH_COST_BREAKERS.get(endpoint_key)


def _gate_high_cost(endpoint_key: str) -> Optional[PlainTextResponse]:
    breaker = _breaker_for(endpoint_key)
    if not breaker:
        return None
    allowed, retry_after = breaker.allow()
    if allowed:
        return None
    inc("circuit_breaker_short_total", 1)
    return PlainTextResponse("temporarily unavailable", status_code=503, headers={"Retry-After": str(max(1, retry_after))})


def _record_breaker_outcome(endpoint_key: str, exc: Optional[Exception]) -> None:
    breaker = _breaker_for(endpoint_key)
    if not breaker:
        return
    if exc is None:
        breaker.record_success()
        return
    breaker.record_failure()


def _create_ingest_store():
    """Factory function to create ingest store based on backend configuration.

    Supports SQLite (default) and PostgreSQL backends.
    Set MEMORY_STORE_BACKEND=postgresql to use PostgreSQL.
    """
    backend = os.getenv("MEMORY_STORE_BACKEND", "sqlite").lower().strip()

    if backend == "postgresql":
        settings = PgIngestJobStoreSettings.from_env()
        logging.getLogger(__name__).info(
            f"Using PostgreSQL ingest store: {settings.host}:{settings.port}/{settings.database}"
        )
        return PgIngestJobStore(settings)
    else:
        db_path = os.getenv("MEMORY_INGEST_JOB_DB_PATH", "").strip() or "modules/memory/outputs/ingest_jobs.db"
        logging.getLogger(__name__).info(f"Using SQLite ingest store: {db_path}")
        return AsyncIngestJobStore({"sqlite_path": db_path})


ingest_store = _create_ingest_store()
_ingest_executor: Optional[IngestExecutor] = None
_usage_wal: Optional[UsageWAL] = None
_usage_wal_stop: Optional[asyncio.Event] = None
_usage_wal_task: Optional[asyncio.Task] = None
_llm_usage_hook_token: Optional[object] = None
_embedding_usage_hook_token: Optional[object] = None


def _init_usage_wal() -> Optional[UsageWAL]:
    settings = UsageWALSettings.from_env()
    if not settings.enabled:
        return None
    return UsageWAL(settings)


def _emit_usage_event(payload: Dict[str, Any]) -> None:
    if _usage_wal is None:
        return
    try:
        _usage_wal.append(payload)
    except Exception:
        # Usage WAL is best-effort locally; upstream delivery is guaranteed by WAL flush.
        return


def _hash_usage_event_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _handle_llm_usage_event(payload: Dict[str, Any]) -> None:
    tenant_id = str(payload.get("tenant_id") or "").strip()
    stage = str(payload.get("stage") or "").strip()
    if not tenant_id or not stage:
        return
    api_key_id = str(payload.get("api_key_id") or "").strip() or None
    request_id = str(payload.get("request_id") or "").strip() or None
    job_id = str(payload.get("job_id") or "").strip() or None
    session_id = str(payload.get("session_id") or "").strip() or None
    call_index = payload.get("call_index")
    try:
        call_index_int = int(call_index) if call_index is not None else 0
    except Exception:
        call_index_int = 0
    anchor = job_id or request_id or ""
    if not anchor:
        return
    seed = f"{tenant_id}|{api_key_id or ''}|{anchor}|{stage}|{call_index_int}"
    event_id = _hash_usage_event_id("llm", seed)
    byok_route = str(payload.get("byok_route") or "").strip()
    byok = byok_route == "byok"
    prompt_tokens = int(payload.get("prompt_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or 0) or (prompt_tokens + completion_tokens)
    usage = TokenUsageDetail(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=(float(payload.get("cost_usd")) if payload.get("cost_usd") is not None else None),
    )
    try:
        add("usage_llm_calls_total", 1)
        add("usage_llm_prompt_tokens_total", prompt_tokens)
        add("usage_llm_completion_tokens_total", completion_tokens)
        add("usage_llm_total_tokens_total", total_tokens)
        if usage.cost_usd is not None:
            add("usage_llm_cost_usd_total", float(usage.cost_usd))
    except Exception:
        pass
    evt = UsageEvent(
        event_id=event_id,
        event_type="llm",
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        request_id=request_id,
        job_id=job_id,
        session_id=session_id,
        stage=stage,
        source=str(payload.get("source") or "").strip() or None,
        call_index=call_index_int,
        provider=str(payload.get("provider") or "unknown"),
        model=str(payload.get("model") or "unknown"),
        billable=not byok,
        byok=byok,
        usage=usage,
        status=str(payload.get("status") or "ok"),
        error_code=(str(payload.get("error_code") or "").strip() or None),
        error_detail=(str(payload.get("error_detail") or "").strip() or None),
        meta={
            "tokens_missing": bool(payload.get("tokens_missing")),
            "byok_route": byok_route or None,
            "credential_fingerprint": payload.get("credential_fingerprint"),
            "resolver_status": payload.get("resolver_status"),
            "generation_id": payload.get("generation_id"),
        },
    )
    _emit_usage_event(evt.model_dump())


def _handle_embedding_usage_event(usage: EmbeddingUsage) -> None:
    """Best-effort embedding usage metrics/WAL.

    EmbeddingUsage hook does not carry tenant context; record global counters only.
    """
    try:
        tokens = int(getattr(usage, "tokens", 0) or 0)
        add("usage_embedding_calls_total", 1)
        add("usage_embedding_tokens_total", tokens)
        cost_usd = getattr(usage, "cost_usd", None)
        if cost_usd is not None:
            add("usage_embedding_cost_usd_total", float(cost_usd))
    except Exception:
        return


async def _run_ingest_job(
    *,
    job_id: str,
    tenant_id: str,
    user_tokens: List[str],
    memory_domain: str,
    llm_policy: str,
    payload: Optional[Dict[str, Any]] = None,
    claim: bool = True,
) -> None:
    stage2_start = time.perf_counter()
    # Use asyncio.to_thread to prevent SQLite blocking the event loop
    record = await ingest_store.get_job(job_id)
    if record is None:
        return
    if record.status == "COMPLETED":
        return
    # Avoid duplicate concurrent runners. For both auto-execute and the (internal) execute endpoint, only
    # a RECEIVED job may be claimed for running. This prevents status rollback / duplicate writes.
    if str(record.status) in {"STAGE2_RUNNING", "STAGE3_RUNNING"}:
        return
    if str(record.status) != "RECEIVED":
        return
    claimed = await ingest_store.try_transition_status(job_id, from_statuses=["RECEIVED"], to_status="STAGE2_RUNNING")
    if not claimed:
        return
    await ingest_store.update_status(job_id, stage="stage2", attempt_inc=True)

    payload_data = payload or _parse_payload_raw(record.payload_raw)
    turns = list(record.turns or [])
    client_meta = dict(record.client_meta or {})
    if isinstance(payload_data, dict):
        payload_turns = payload_data.get("turns")
        if isinstance(payload_turns, list) and payload_turns:
            turns = list(payload_turns)
        payload_meta = payload_data.get("client_meta")
        if isinstance(payload_meta, dict):
            client_meta = dict(payload_meta)
        payload_domain = payload_data.get("memory_domain")
        if payload_domain:
            memory_domain = str(payload_domain)
        payload_llm = payload_data.get("llm_policy")
        if payload_llm:
            llm_policy = str(payload_llm)
    turns = _normalize_ingest_turns(turns)

    # Guardrails: never block on an invalid/corrupted job payload.
    # user_tokens comes from record.user_tokens (populated by api-dev BFF in SaaS mode),
    # NOT from payload_raw (which is the original SDK request). In SaaS, if user_tokens
    # is empty, derive from tenant_id as fallback.
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]
    if not user_tokens:
        err = {"stage": "stage3", "code": "invalid_job_payload", "message": "user_tokens must be non-empty"}
        await ingest_store.update_status(job_id, status="STAGE3_FAILED", error=err)
        return
    if not turns:
        err = {"stage": "stage3", "code": "invalid_job_payload", "message": "turns must be non-empty"}
        await ingest_store.update_status(job_id, status="STAGE3_FAILED", error=err)
        return

    marks = list(record.stage2_marks or [])
    pin_intents = list(record.stage2_pin_intents or [])
    llm_adapter, llm_route, llm_status = _resolve_llm_adapter_from_client_meta(client_meta)
    stage2_strict = bool(client_meta.get("stage2_strict")) if client_meta.get("stage2_strict") is not None else False
    # Stage2 (turn marks) default is OFF: worker-driven ingest should never be blocked by LLM marking.
    # Enable explicitly via client_meta.stage2_enabled=true or env MEMORY_STAGE2_ENABLED=true.
    stage2_enabled = client_meta.get("stage2_enabled")
    if stage2_enabled is None:
        stage2_enabled_env = str(os.getenv("MEMORY_STAGE2_ENABLED", "false")).strip().lower()
        stage2_enabled = stage2_enabled_env not in ("0", "false", "no", "")
    stage2_enabled = bool(stage2_enabled)
    stage2_skip_llm = bool(client_meta.get("stage2_skip_llm")) if client_meta.get("stage2_skip_llm") is not None else False
    stage3_extract = bool(client_meta.get("stage3_extract")) if client_meta.get("stage3_extract") is not None else True
    overwrite_existing = bool(client_meta.get("overwrite_existing")) if client_meta.get("overwrite_existing") is not None else False
    if not marks:
        if not stage2_enabled or stage2_skip_llm:
            marks = default_marks_keep_all(turns)
        else:
            extractor = build_turn_mark_extractor_v1_from_env(adapter=llm_adapter)
            if extractor is None:
                if str(llm_policy or "best_effort").lower() == "require":
                    err = {"stage": "stage2", "code": "llm_missing", "message": "stage2_extractor_missing"}
                    await ingest_store.update_status(job_id, status="STAGE2_FAILED", error=err)
                    return
                marks = default_marks_keep_all(turns)
            else:
                try:
                    ctx_token = set_llm_usage_context(
                        LLMUsageContext(
                            tenant_id=str(record.tenant_id),
                            api_key_id=(str(record.api_key_id) if record.api_key_id else None),
                            request_id=(str(record.request_id) if record.request_id else None),
                            stage="stage2",
                            job_id=str(record.job_id),
                            session_id=str(record.session_id),
                            call_index=None,
                            source="turn_mark_extractor",
                            byok_route=llm_route,
                            resolver_status=llm_status,
                        )
                    )
                    try:
                        raw_marks = await asyncio.to_thread(extractor, list(turns))
                    finally:
                        reset_llm_usage_context(ctx_token)
                    marks = validate_and_normalize_marks(turns=turns, marks=raw_marks, strict=stage2_strict)
                except Exception as exc:
                    err = {"stage": "stage2", "code": "stage2_invalid", "message": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    await ingest_store.update_status(job_id, status="STAGE2_FAILED", error=err)
                    return
        pin_intents = generate_pin_intents(turns, marks, window=4)
        await ingest_store.update_stage2(job_id, marks=marks, pin_intents=pin_intents)
    elif marks and not pin_intents:
        pin_intents = generate_pin_intents(turns, marks, window=4)
        if pin_intents:
            await ingest_store.update_stage2(job_id, marks=marks, pin_intents=pin_intents)

    kept_turns_list = apply_turn_marks(turns, marks) if marks else list(turns)
    relaxed_keep_all = False
    if not kept_turns_list and turns:
        kept_turns_list = list(turns)
        relaxed_keep_all = True
    kept_turns = len(kept_turns_list)
    stage2_ms = int((time.perf_counter() - stage2_start) * 1000)
    try:
        observe_ingest_latency("stage2", stage2_ms)
    except Exception:
        pass
    await ingest_store.update_status(
        job_id,
        status="STAGE3_RUNNING", stage="stage3", attempt_inc=True,
        metrics_patch={
            "kept_turns": kept_turns,
            "kept_turns_relaxed": bool(relaxed_keep_all),
            "stage2_ms": stage2_ms,
        },
    )
    try:
        stage3_start = time.perf_counter()
        extra_facts = pin_intents_to_facts(pin_intents=pin_intents, turns=kept_turns_list)
        tkg_extractor_override = None
        if llm_adapter is not None:
            try:
                from modules.memory.application.dialog_tkg_unified_extractor_v1 import (
                    build_dialog_tkg_unified_extractor_v1_from_env,
                )

                tkg_extractor_override = build_dialog_tkg_unified_extractor_v1_from_env(
                    session_id=str(record.session_id),
                    reference_time_iso=None,
                    adapter=llm_adapter,
                )
            except Exception:
                tkg_extractor_override = None
        ctx_token = set_llm_usage_context(
            LLMUsageContext(
                tenant_id=str(record.tenant_id),
                api_key_id=(str(record.api_key_id) if record.api_key_id else None),
                request_id=(str(record.request_id) if record.request_id else None),
                stage="stage3",
                job_id=str(record.job_id),
                session_id=str(record.session_id),
                call_index=None,
                source="session_write",
                byok_route=llm_route,
                resolver_status=llm_status,
            )
        )
        route_token = set_request_context(
            {
                "request_id": str(record.request_id or record.job_id),
                "request_time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "method": "JOB",
                "path": "/ingest/worker",
                "query": "",
                "content_length": len(record.payload_raw or b""),
                "tenant_id": str(record.tenant_id),
            }
        )
        try:
            res = await session_write(
                svc,
                tenant_id=str(tenant_id),
                user_tokens=list(user_tokens),
                session_id=str(record.session_id),
                turns=list(kept_turns_list),
                memory_domain=str(memory_domain),
                llm_policy=str(llm_policy),
                extra_facts=list(extra_facts),
                turn_marks=list(marks),
                tkg_extractor=tkg_extractor_override,
                extract=bool(stage3_extract),
                write_facts=bool(stage3_extract),
                overwrite_existing=bool(overwrite_existing),
            )
        finally:
            clear_request_context(route_token)
            reset_llm_usage_context(ctx_token)
        status_val = str(res.get("status") or "") if isinstance(res, dict) else ""
        if status_val and status_val not in ("ok", "skipped_existing"):
            trace = res.get("trace") if isinstance(res, dict) else {}
            err_msg = ""
            try:
                err_msg = str((trace or {}).get("error") or "")[:200]
            except Exception:
                err_msg = ""
            err = {
                "stage": "stage3",
                "code": "session_write_failed",
                "message": (err_msg or f"session_write status={status_val}")[:200],
            }
            await ingest_store.update_status(job_id, status="STAGE3_FAILED", error=err)
            return
        trace = res.get("trace") if isinstance(res, dict) else {}
        stage3_ms = int((time.perf_counter() - stage3_start) * 1000)
        try:
            observe_ingest_latency("stage3", stage3_ms)
        except Exception:
            pass
        timing = trace.get("timing_ms") if isinstance(trace, dict) else {}
        if isinstance(timing, dict):
            try:
                if timing.get("extract_ms") is not None:
                    observe_ingest_latency("stage3_extract", float(timing.get("extract_ms")))
                if timing.get("build_ms") is not None:
                    observe_ingest_latency("stage3_build", float(timing.get("build_ms")))
                if timing.get("graph_upsert_ms") is not None:
                    observe_ingest_latency("stage3_graph", float(timing.get("graph_upsert_ms")))
                if timing.get("vector_write_ms") is not None:
                    observe_ingest_latency("stage3_vector", float(timing.get("vector_write_ms")))
                if timing.get("publish_ms") is not None:
                    observe_ingest_latency("stage3_publish", float(timing.get("publish_ms")))
                if timing.get("overwrite_delete_ms") is not None:
                    observe_ingest_latency("stage3_overwrite_delete", float(timing.get("overwrite_delete_ms")))
            except Exception:
                pass
        graph_ids = trace.get("graph_ids", {}) if isinstance(trace, dict) else {}
        event_ids = list(graph_ids.get("event_ids") or []) if isinstance(graph_ids, dict) else []
        timeslice_id = graph_ids.get("timeslice_id") if isinstance(graph_ids, dict) else None
        graph_nodes_written = len(event_ids) + (1 if timeslice_id else 0)
        vector_written = int(res.get("written_entries") or 0) if isinstance(res, dict) else 0
        await ingest_store.update_status(
            job_id,
            status="COMPLETED",
            metrics_patch={
                "graph_nodes_written": graph_nodes_written,
                "vector_points_written": vector_written,
                "stage3_ms": stage3_ms,
                "stage3_extract_ms": timing.get("extract_ms") if isinstance(timing, dict) else None,
                "stage3_build_ms": timing.get("build_ms") if isinstance(timing, dict) else None,
                "stage3_graph_ms": timing.get("graph_upsert_ms") if isinstance(timing, dict) else None,
                "stage3_vector_ms": timing.get("vector_write_ms") if isinstance(timing, dict) else None,
                "stage3_publish_ms": timing.get("publish_ms") if isinstance(timing, dict) else None,
                "stage3_overwrite_delete_ms": timing.get("overwrite_delete_ms") if isinstance(timing, dict) else None,
            },
            error=None,
            next_retry_at=None,
        )
        seed = f"{record.tenant_id}|{record.api_key_id or ''}|{record.job_id}|stage3|0"
        evt = UsageEvent(
            event_id=_hash_usage_event_id("write", seed),
            event_type="write",
            tenant_id=str(record.tenant_id),
            api_key_id=(str(record.api_key_id) if record.api_key_id else None),
            request_id=(str(record.request_id) if record.request_id else None),
            job_id=str(record.job_id),
            session_id=str(record.session_id),
            stage="stage3",
            source="session_write",
            provider="memory_service",
            model="write",
            billable=False,
            byok=False,
            usage=None,
            status="ok",
            meta={
                "archived_turns": int(record.metrics.get("archived_turns") or 0),
                "kept_turns": int(kept_turns),
                "graph_nodes_written": int(graph_nodes_written),
                "vector_points_written": int(vector_written),
            },
        )
        _emit_usage_event(evt.model_dump())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        err = {"stage": "stage3", "code": "server_error", "message": f"{type(exc).__name__}: {str(exc)[:200]}"}
        await ingest_store.update_status(job_id, status="STAGE3_FAILED", error=err)

        return


async def _run_ingest_job_from_record(record: IngestJobRecord) -> None:
    await _run_ingest_job(
        job_id=str(record.job_id),
        tenant_id=str(record.tenant_id),
        user_tokens=list(record.user_tokens),
        memory_domain=str(record.memory_domain),
        llm_policy=str(record.llm_policy),
        payload=None,
        claim=True,
    )


# Lazy services (avoid import-time side effects)
svc = _LazyProxy(_get_svc, "svc", lambda: _svc is not None)
graph_svc = _LazyProxy(_get_graph_svc, "graph_svc", lambda: _graph_svc is not None)
equiv_store = _LazyProxy(_get_equiv_store, "equiv_store", lambda: _equiv_store is not None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Startup: Best-effort initialization for development ergonomics.
    - Ensure Qdrant collections exist if vectors backend supports it.
    - Never fail startup if this step errors; keep service responsive.
    """
    try:
        enable_startup_ensure = str(os.getenv("MEMORY_STARTUP_ENSURE_COLLECTIONS", "true")).strip().lower()
        if enable_startup_ensure in ("1", "true", "yes", "on"):
            async def _ensure_in_background() -> None:
                try:
                    svc_local: Optional[MemoryService] = _svc
                    override = _resolve_override("svc")
                    if svc_local is None and isinstance(override, MemoryService):
                        svc_local = override
                    if svc_local is None:
                        svc_local = await asyncio.to_thread(_get_svc)
                    vectors = getattr(svc_local, "vectors", None)
                    if vectors is None or not hasattr(vectors, "ensure_collections"):
                        return
                    timeout_s = float(os.getenv("MEMORY_STARTUP_ENSURE_COLLECTIONS_TIMEOUT_S", "120.0") or 120.0)
                    ensure_coro = vectors.ensure_collections()  # type: ignore[attr-defined]
                    if timeout_s > 0:
                        await asyncio.wait_for(ensure_coro, timeout=timeout_s)
                    else:
                        await ensure_coro
                except Exception:
                    logger.warning("memory.startup.ensure_collections failed", exc_info=True)

            asyncio.create_task(_ensure_in_background())
    except Exception:
        # Do not block service on init failure
        pass
    try:
        ensure_graph = str(os.getenv("MEMORY_STARTUP_ENSURE_GRAPH_SCHEMA", "true")).strip().lower()
        if ensure_graph not in ("0", "false", "no"):
            graph_local = _graph_svc
            override_graph = _resolve_override("graph_svc")
            if graph_local is None and override_graph is not None:
                graph_local = override_graph
            if graph_local is not None and hasattr(graph_local, "store") and hasattr(graph_local.store, "ensure_schema_v0"):
                timeout_s = float(os.getenv("MEMORY_STARTUP_ENSURE_GRAPH_SCHEMA_TIMEOUT_S", "2.0") or 2.0)

                async def _ensure_graph_schema() -> None:
                    try:
                        await asyncio.wait_for(asyncio.to_thread(graph_local.store.ensure_schema_v0), timeout=timeout_s)
                    except Exception:
                        return

                asyncio.create_task(_ensure_graph_schema())
    except Exception:
        pass
    try:
        global _usage_wal, _usage_wal_stop, _usage_wal_task, _llm_usage_hook_token, _embedding_usage_hook_token
        if _usage_wal is None:
            _usage_wal = _init_usage_wal()
        if _usage_wal is not None and _usage_wal_task is None:
            _usage_wal_stop = asyncio.Event()
            _usage_wal_task = asyncio.create_task(_usage_wal.run_flush_loop(_usage_wal_stop))
        if _usage_wal is not None and _llm_usage_hook_token is None:
            _llm_usage_hook_token = set_llm_usage_hook(_handle_llm_usage_event)
        if _usage_wal is not None and _embedding_usage_hook_token is None:
            try:
                from modules.memory.application.embedding_adapter import set_embedding_usage_hook

                _embedding_usage_hook_token = set_embedding_usage_hook(_handle_embedding_usage_event)
            except Exception:
                _embedding_usage_hook_token = None
    except Exception:
        pass
    try:
        global _ingest_executor
        if _INGEST_EXECUTOR_ENABLED:
            if _ingest_executor is None:
                _ingest_executor = IngestExecutor(
                    store=ingest_store,
                    run_job=_run_ingest_job_from_record,
                    config=_INGEST_EXECUTOR_CONFIG,
                )
            await _ingest_executor.start()
    except Exception:
        logging.getLogger(__name__).exception("Failed to start ingest executor")
    yield
    if _llm_usage_hook_token is not None:
        reset_llm_usage_hook(_llm_usage_hook_token)
        _llm_usage_hook_token = None
    if _embedding_usage_hook_token is not None:
        try:
            from modules.memory.application.embedding_adapter import reset_embedding_usage_hook

            reset_embedding_usage_hook(_embedding_usage_hook_token)
        except Exception:
            pass
        _embedding_usage_hook_token = None
    if _usage_wal_stop is not None:
        _usage_wal_stop.set()
    if _usage_wal_task is not None:
        _usage_wal_task.cancel()
        with suppress(asyncio.CancelledError):
            await _usage_wal_task
    if _ingest_executor is not None:
        await _ingest_executor.stop()


app = FastAPI(title="MOYAN Memory Service", lifespan=lifespan)


@app.middleware("http")
async def enforce_api_limits(request: Request, call_next):
    request_id = _ensure_request_id(request)
    token = set_request_context(
        {
            "request_id": request_id,
            "request_time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query or ""),
            "content_length": _content_length(request),
        }
    )
    size_resp = await _ensure_request_size(request)
    if size_resp is not None:
        size_resp.headers["X-Request-ID"] = request_id
        clear_request_context(token)
        return size_resp
    rate_resp = await _maybe_rate_limit(request)
    if rate_resp is not None:
        rate_resp.headers["X-Request-ID"] = request_id
        clear_request_context(token)
        return rate_resp
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = request_id
    clear_request_context(token)
    return resp


class SearchBody(BaseModel):
    query: str
    topk: Optional[int] = None
    expand_graph: bool = True
    graph_backend: str = "memory"  # memory|tkg (optional; default keeps userspace stable)
    threshold: Optional[float] = None
    graph_params: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None


class TimelineSummaryBody(BaseModel):
    query: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_segments: int = 100
    granularity: str = "clip"
    include_semantic: bool = True
    graph_params: Optional[Dict[str, Any]] = None
    topk_per_segment: int = 3
    topk_search: int = 200


class ObjectSearchBody(BaseModel):
    objects: List[str]
    scene: Optional[str] = None
    query: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    graph_params: Optional[Dict[str, Any]] = None
    topk: int = 20


class SpeechSearchBody(BaseModel):
    keywords: List[str]
    speaker: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, float]] = None
    topk: int = 20


class EntityAnchorBody(BaseModel):
    entity: str
    action: Optional[str] = None
    time_hint: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    graph_params: Optional[Dict[str, Any]] = None
    topk: int = 20


class IngestCursorBody(BaseModel):
    base_turn_id: Optional[str] = None


class IngestExecuteBody(BaseModel):
    job_id: str


class IngestDialogBody(BaseModel):
    session_id: Optional[str] = None
    user_tokens: List[str] = []
    memory_domain: str = "dialog"
    turns: List[Dict[str, Any]] = []
    commit_id: Optional[str] = None
    cursor: Optional[IngestCursorBody] = None
    client_meta: Optional[Dict[str, Any]] = None
    llm_policy: str = "require"


async def _enqueue_ingest_job(record: IngestJobRecord) -> bool:
    if _ingest_executor is None:
        return False
    return await _ingest_executor.enqueue(record.job_id, tenant_id=str(record.tenant_id))


class RetrievalDialogBody(BaseModel):
    query: Optional[str] = None
    user_tokens: List[str] = []
    memory_domain: str = "dialog"
    run_id: Optional[str] = None
    strategy: str = "dialog_v2"
    topk: Optional[int] = None
    debug: bool = False
    with_answer: bool = False
    task: str = "GENERAL"
    llm_policy: str = "best_effort"
    backend: str = "tkg"
    tkg_explain: bool = True
    entity_hints: Optional[List[str]] = None
    time_hints: Optional[Dict[str, Any]] = None
    client_meta: Optional[Dict[str, Any]] = None
    candidate_k: Optional[int] = None
    seed_topn: Optional[int] = None
    e_vec_oversample: Optional[int] = None
    graph_cap: Optional[int] = None
    rrf_k: Optional[int] = None
    qa_evidence_cap_l2: Optional[int] = None
    qa_evidence_cap_l4: Optional[int] = None
    enable_event_route: Optional[bool] = None
    enable_evidence_route: Optional[bool] = None
    enable_knowledge_route: Optional[bool] = None
    enable_entity_route: Optional[bool] = None
    enable_time_route: Optional[bool] = None
    dialog_v2_weights: Optional[Dict[str, float]] = None
    dialog_v2_reranker: Optional[Dict[str, Any]] = None
    dialog_v2_test_ablation: Optional[Dict[str, Any]] = None


class TopicTimelineBody(BaseModel):
    topic: Optional[str] = None
    topic_id: Optional[str] = None
    topic_path: Optional[str] = None
    keywords: Optional[List[str]] = None
    user_tokens: Optional[List[str]] = None
    time_range: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    memory_domain: Optional[str] = None
    limit: int = 50
    with_quotes: bool = False
    with_entities: bool = False
    debug: bool = False


class EntityProfileBody(BaseModel):
    entity: Optional[str] = None
    entity_id: Optional[str] = None
    user_tokens: Optional[List[str]] = None
    memory_domain: Optional[str] = None
    facts_limit: int = 20
    relations_limit: int = 20
    events_limit: int = 20
    include_quotes: bool = False
    include_relations: bool = True
    include_events: bool = True
    include_states: bool = False
    quotes_limit: int = 20
    debug: bool = False


class QuotesBody(BaseModel):
    entity: Optional[str] = None
    entity_id: Optional[str] = None
    topic: Optional[str] = None
    topic_id: Optional[str] = None
    topic_path: Optional[str] = None
    user_tokens: Optional[List[str]] = None
    time_range: Optional[Dict[str, Any]] = None
    memory_domain: Optional[str] = None
    limit: int = 50
    debug: bool = False


class RelationsBody(BaseModel):
    entity: Optional[str] = None
    entity_id: Optional[str] = None
    relation_type: Optional[str] = None
    user_tokens: Optional[List[str]] = None
    time_range: Optional[Dict[str, Any]] = None
    memory_domain: Optional[str] = None
    limit: int = 50
    debug: bool = False


class ExplainBody(BaseModel):
    event_id: str
    user_tokens: Optional[List[str]] = None
    memory_domain: Optional[str] = None
    debug: bool = False


class ResolveEntityBody(BaseModel):
    name: str
    type: Optional[str] = None
    user_tokens: Optional[List[str]] = None
    limit: int = 5
    debug: bool = False


class TimeSinceBody(BaseModel):
    topic: Optional[str] = None
    topic_id: Optional[str] = None
    topic_path: Optional[str] = None
    entity: Optional[str] = None
    entity_id: Optional[str] = None
    user_tokens: Optional[List[str]] = None
    time_range: Optional[Dict[str, Any]] = None
    memory_domain: Optional[str] = None
    limit: int = 50
    debug: bool = False


class AgenticQueryBody(BaseModel):
    query: str
    tool_whitelist: Optional[List[str]] = None
    include_debug: bool = False
    request_id: Optional[str] = None


class AgenticExecuteBody(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}
    include_debug: bool = False
    request_id: Optional[str] = None


class WriteBody(BaseModel):
    entries: List[Dict[str, Any]]
    links: Optional[List[Dict[str, Any]]] = None
    upsert: bool = True
    return_id_map: bool = False  # 是否返回 logical_id → UUID 映射


class UpdateBody(BaseModel):
    id: str
    patch: Dict[str, Any]
    reason: Optional[str] = None
    confirm: Optional[bool] = None


class DeleteBody(BaseModel):
    id: str
    soft: bool = True
    reason: Optional[str] = None
    confirm: Optional[bool] = None


class LinkBody(BaseModel):
    src_id: str
    dst_id: str
    rel_type: str
    weight: Optional[float] = None
    confirm: Optional[bool] = None


class RollbackBody(BaseModel):
    version: str


class ClearRequest(BaseModel):
    scope: Literal["tenant", "apikey"] = "tenant"
    api_key_id: Optional[str] = None
    confirm: bool = False
    reason: str


class ClearStoreStatus(BaseModel):
    status: Literal["estimated", "cleared", "error"]
    count: int = 0
    detail: Optional[str] = None


class ClearResponse(BaseModel):
    tenant_id: str
    scope: Literal["tenant", "apikey"] = "tenant"
    api_key_id: Optional[str] = None
    dry_run: bool
    status: Literal["completed", "partial_failure"]
    estimated_vectors: int = 0
    estimated_graph_nodes: int = 0
    estimated_ingest_jobs: int = 0
    cleared_vectors: int = 0
    cleared_graph_nodes: int = 0
    cleared_ingest_jobs: int = 0
    warnings: List[str] = Field(default_factory=list)
    stores: Dict[str, ClearStoreStatus] = Field(default_factory=dict)


@app.get("/health")
async def health():
    result = await svc.health_check()
    status_code = 200 if result.get("status") == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/metrics")
async def metrics():
    return get_metrics()


@app.get("/metrics_prom")
async def metrics_prom():
    text = as_prometheus_text()
    return PlainTextResponse(text)


@app.post("/search")
async def search(body: SearchBody, request: Request):
    await _enforce_security(request)
    gate = _gate_high_cost("search")
    if gate:
        return gate
    topk = int(body.topk) if body.topk is not None else _API_DEFAULT_TOPK_SEARCH
    if topk <= 0:
        topk = _API_DEFAULT_TOPK_SEARCH
    if topk > SEARCH_TOPK_HARD_LIMIT:
        topk = SEARCH_TOPK_HARD_LIMIT
    filters = SearchFilters.model_validate(body.filters) if body.filters else None
    # If tenant_id is not provided in filters, allow X-Tenant-ID header to fill it (useful for TKG graph backend).
    try:
        if filters is not None and filters.tenant_id is None:
            tenant_hdr = request.headers.get("X-Tenant-ID")
            if tenant_hdr:
                filters.tenant_id = str(tenant_hdr)
    except Exception:
        pass
    try:
        res = await asyncio.wait_for(
            svc.search(
                body.query,
                topk=topk,
                filters=filters,
                expand_graph=body.expand_graph,
                graph_backend=body.graph_backend,
                threshold=body.threshold,
                graph_params=body.graph_params,
            ),
            timeout=HIGH_COST_TIMEOUT,
        )
        _record_breaker_outcome("search", None)
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("search", exc)
        raise HTTPException(status_code=504, detail="search_timeout") from None
    except HTTPException as exc:
        if exc.status_code >= 500:
            _record_breaker_outcome("search", exc)
        else:
            _record_breaker_outcome("search", None)
        raise
    except Exception as exc:
        _record_breaker_outcome("search", exc)
        raise
    # Pydantic models are JSON-serializable as dicts
    return res.model_dump()


@app.post("/timeline_summary")
async def timeline_summary(body: TimelineSummaryBody, request: Request):
    await _enforce_security(request)
    gate = _gate_high_cost("timeline_summary")
    if gate:
        return gate
    filters = SearchFilters.model_validate(body.filters) if body.filters else None
    try:
        res = await asyncio.wait_for(
            svc.timeline_summary(
                query=body.query,
                filters=filters,
                start_time=body.start_time,
                end_time=body.end_time,
                max_segments=body.max_segments,
                granularity=body.granularity,
                include_semantic=body.include_semantic,
                graph_params=body.graph_params,
                topk_per_segment=body.topk_per_segment,
                topk_search=body.topk_search,
            ),
            timeout=HIGH_COST_TIMEOUT,
        )
        _record_breaker_outcome("timeline_summary", None)
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("timeline_summary", exc)
        raise HTTPException(status_code=504, detail="timeline_timeout") from None
    except HTTPException as exc:
        if exc.status_code >= 500:
            _record_breaker_outcome("timeline_summary", exc)
        else:
            _record_breaker_outcome("timeline_summary", None)
        raise
    except Exception as exc:
        _record_breaker_outcome("timeline_summary", exc)
        raise
    return res


@app.post("/speech_search")
async def speech_search(body: SpeechSearchBody, request: Request):
    await _enforce_security(request)
    filters = SearchFilters.model_validate(body.filters) if body.filters else None
    res = await svc.speech_search(
        keywords=body.keywords,
        speaker=body.speaker,
        filters=filters,
        time_range=body.time_range,
        topk=body.topk,
    )
    return res


@app.post("/entity_event_anchor")
async def entity_event_anchor(body: EntityAnchorBody, request: Request):
    await _enforce_security(request)
    filters = SearchFilters.model_validate(body.filters) if body.filters else None
    res = await svc.entity_event_anchor(
        entity=body.entity,
        action=body.action,
        time_hint=body.time_hint,
        filters=filters,
        graph_params=body.graph_params,
        topk=body.topk,
    )
    return res


@app.post("/object_search")
async def object_search(body: ObjectSearchBody, request: Request):
    await _enforce_security(request)
    filters = SearchFilters.model_validate(body.filters) if body.filters else None
    res = await svc.object_search(
        objects=body.objects,
        scene=body.scene,
        query=body.query,
        filters=filters,
        modalities=body.modalities,
        graph_params=body.graph_params,
        topk=body.topk,
    )
    return res


@app.post("/retrieval")  # Short path alias for SaaS gateway compatibility
@app.post("/retrieval/dialog/v2")  # Full versioned path for direct access
async def retrieval_dialog_v2(body: RetrievalDialogBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("retrieval")
    if gate:
        return gate
    topk = int(body.topk) if body.topk is not None else _API_DEFAULT_TOPK_RETRIEVAL
    if topk <= 0:
        topk = _API_DEFAULT_TOPK_RETRIEVAL
    if topk > SEARCH_TOPK_HARD_LIMIT:
        topk = SEARCH_TOPK_HARD_LIMIT
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    missing: List[str] = []
    query = str(body.query or "").strip()
    if not query:
        missing.append("query")
    # SaaS: isolation is guaranteed by tenant_id; user_tokens are optional and
    # primarily for self-hosted / advanced deployments. If the caller does not
    # provide user_tokens, derive a stable value from tenant_id so that
    # retrieval can still scope correctly.
    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]
    if not user_tokens:
        missing.append("user_tokens")
    client_meta = dict(body.client_meta) if isinstance(body.client_meta, dict) else None
    missing.extend(_client_meta_missing(client_meta))
    missing.extend(_llm_meta_missing(client_meta))
    if missing:
        return _missing_core_requirements(missing, target="retrieval")
    llm_ctx_token: Optional[object] = None
    qa_generate: Optional[Callable[[str, str], str]] = None
    llm_adapter, llm_route, llm_status = _resolve_llm_adapter_from_client_meta(client_meta)
    if bool(body.with_answer):
        _enforce_with_answer(ctx)
        if llm_adapter is not None:
            def _gen(system_prompt: str, user_prompt: str) -> str:
                return llm_adapter.generate(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )

            qa_generate = _gen
        llm_ctx_token = set_llm_usage_context(
            LLMUsageContext(
                tenant_id=str(tenant_id),
                api_key_id=(str(ctx.get("subject")) if ctx.get("subject") else None),
                request_id=_request_id_from_request(request),
                stage="retrieval_qa",
                job_id=None,
                session_id=(str(body.run_id) if body.run_id else None),
                call_index=None,
                source="retrieval_qa",
                byok_route=llm_route,
                resolver_status=llm_status,
            )
        )
    try:
        dialog_v2_ranking = get_dialog_v2_ranking_settings(cfg)
        retrieval_kwargs: Dict[str, Any] = {
            "tenant_id": str(tenant_id),
            "user_tokens": list(user_tokens),
            "query": query,
            "strategy": str(body.strategy or "dialog_v2"),
            "topk": topk,
            "memory_domain": str(body.memory_domain or "dialog"),
            "user_match": "all",
            "run_id": (str(body.run_id) if body.run_id else None),
            "debug": bool(body.debug),
            "with_answer": bool(body.with_answer),
            "task": str(body.task or "GENERAL"),
            "llm_policy": str(body.llm_policy or "best_effort"),
            "backend": str(body.backend or "tkg"),
            "tkg_explain": bool(body.tkg_explain),
            "qa_generate": qa_generate,
            "entity_hints": list(body.entity_hints or []) if body.entity_hints else None,
            "time_hints": dict(body.time_hints) if isinstance(body.time_hints, dict) else None,
            "request_id": _request_id_from_request(request),
            "trace_id": str(request.headers.get("X-Trace-ID") or "").strip() or None,
            "byok_route": llm_route,
            "rrf_k": int(dialog_v2_ranking.get("rrf_k") or 60),
            "dialog_v2_weights": dict(dialog_v2_ranking.get("weights") or {}),
        }
        if body.candidate_k is not None:
            retrieval_kwargs["candidate_k"] = int(body.candidate_k)
        if body.seed_topn is not None:
            retrieval_kwargs["seed_topn"] = int(body.seed_topn)
        if body.e_vec_oversample is not None:
            retrieval_kwargs["e_vec_oversample"] = int(body.e_vec_oversample)
        if body.graph_cap is not None:
            retrieval_kwargs["graph_cap"] = int(body.graph_cap)
        if body.rrf_k is not None:
            retrieval_kwargs["rrf_k"] = int(body.rrf_k)
        if body.qa_evidence_cap_l2 is not None:
            retrieval_kwargs["qa_evidence_cap_l2"] = int(body.qa_evidence_cap_l2)
        if body.qa_evidence_cap_l4 is not None:
            retrieval_kwargs["qa_evidence_cap_l4"] = int(body.qa_evidence_cap_l4)
        if body.enable_event_route is not None:
            retrieval_kwargs["enable_event_route"] = bool(body.enable_event_route)
        if body.enable_evidence_route is not None:
            retrieval_kwargs["enable_evidence_route"] = bool(body.enable_evidence_route)
        if body.enable_knowledge_route is not None:
            retrieval_kwargs["enable_knowledge_route"] = bool(body.enable_knowledge_route)
        if body.enable_entity_route is not None:
            retrieval_kwargs["enable_entity_route"] = bool(body.enable_entity_route)
        if body.enable_time_route is not None:
            retrieval_kwargs["enable_time_route"] = bool(body.enable_time_route)
        if isinstance(body.dialog_v2_weights, dict):
            retrieval_kwargs["dialog_v2_weights"] = dict(body.dialog_v2_weights)
        if isinstance(body.dialog_v2_reranker, dict):
            retrieval_kwargs["dialog_v2_reranker"] = dict(body.dialog_v2_reranker)
        if isinstance(body.dialog_v2_test_ablation, dict):
            retrieval_kwargs["dialog_v2_test_ablation"] = dict(body.dialog_v2_test_ablation)

        res = await asyncio.wait_for(
            retrieval(
                svc,
                **retrieval_kwargs,
            ),
            timeout=HIGH_COST_TIMEOUT,
        )
        _record_breaker_outcome("retrieval", None)
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("retrieval", exc)
        raise HTTPException(status_code=504, detail="retrieval_timeout") from None
    except ValueError as exc:
        _record_breaker_outcome("retrieval", None)
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except HTTPException as exc:
        if exc.status_code >= 500:
            _record_breaker_outcome("retrieval", exc)
        else:
            _record_breaker_outcome("retrieval", None)
        raise
    except Exception as exc:
        _record_breaker_outcome("retrieval", exc)
        raise
    finally:
        if llm_ctx_token is not None:
            reset_llm_usage_context(llm_ctx_token)
    if isinstance(res, dict) and "usage" not in res:
        billable = str(llm_route or "").strip().lower() != "byok"
        res["usage"] = {
            "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": None},
            "llm": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": None},
            "embedding": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": None},
            "billable": bool(billable),
            "details": None,
        }
    return res


@app.post("/ingest")  # Short path alias for SaaS gateway compatibility
@app.post("/ingest/dialog/v1")  # Full versioned path for direct access
async def ingest_dialog_v1(
    body: IngestDialogBody,
    request: Request,
    wait: bool = Query(False),
    wait_timeout_ms: Optional[int] = Query(None),
):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    missing: List[str] = []
    session_id = str(body.session_id or "").strip()
    if not session_id:
        fallback = str(_request_id_from_request(request) or "").strip()
        session_id = fallback
    if not session_id:
        missing.append("session_id")
    # In SaaS mode, isolation is guaranteed by tenant_id derived from the API key.
    # user_tokens remain supported for backward compatibility / self-hosted mode,
    # but are no longer required for SaaS: we can safely derive a stable token
    # from tenant_id when none is provided.
    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]
    turns = _normalize_ingest_turns(list(body.turns or []))
    if not turns:
        missing.append("turns")
    client_meta = dict(body.client_meta) if isinstance(body.client_meta, dict) else None
    missing.extend(_client_meta_missing(client_meta))
    missing.extend(_llm_meta_missing(client_meta))
    if missing:
        return _missing_core_requirements(missing, target="ingest")
    base_turn_id = None
    if body.cursor is not None and str(body.cursor.base_turn_id or "").strip():
        base_turn_id = str(body.cursor.base_turn_id or "").strip()
    client_meta_payload = dict(client_meta or {})

    record, created = await ingest_store.create_job(
        session_id=str(session_id),
        commit_id=(str(body.commit_id).strip() if body.commit_id else None),
        tenant_id=str(tenant_id),
        api_key_id=(str(ctx.get("subject")) if ctx.get("subject") else None),
        request_id=_request_id_from_request(request),
        turns=turns,
        user_tokens=list(user_tokens),
        base_turn_id=base_turn_id,
        client_meta=client_meta_payload,
        memory_domain=str(body.memory_domain or "dialog"),
        llm_policy=str(body.llm_policy or "require"),
        payload_raw=None,
    )
    accepted = len(turns) if created else 0
    deduped = 0 if created else len(turns)
    if bool(wait):
        raise HTTPException(status_code=400, detail="wait_not_supported")

    enqueued = await _enqueue_ingest_job(record)
    if not enqueued:
        err = {"stage": "enqueue", "code": "enqueue_failed", "message": "ingest_executor_unavailable"}
        await ingest_store.update_status(record.job_id, status="ENQUEUE_FAILED", error=err)
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "session_id": record.session_id,
                "job_id": record.job_id,
                "accepted_turns": accepted,
                "deduped_turns": deduped,
                "status": "ENQUEUE_FAILED",
                "status_url": f"/ingest/jobs/{record.job_id}",
                "error": err,
            },
        )

    # Refresh before returning 202 to avoid confusing "RECEIVED" responses after wait or fast completion.
    cur = await ingest_store.get_job(record.job_id)
    if cur is not None:
        record = cur

    return JSONResponse(
        status_code=202,
        content={
            "ok": True,
            "session_id": record.session_id,
            "job_id": record.job_id,
            "accepted_turns": accepted,
            "deduped_turns": deduped,
            "status": record.status,
            "status_url": f"/ingest/jobs/{record.job_id}",
            "enqueue": True,
        },
    )


@app.get("/ingest/jobs/{job_id}")
async def ingest_job_status(job_id: str, request: Request):
    await _enforce_security(request)
    record = await ingest_store.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {
        "job_id": record.job_id,
        "session_id": record.session_id,
        "status": record.status,
        "attempts": dict(record.attempts or {}),
        "next_retry_at": record.next_retry_at,
        "last_error": record.last_error,
        "metrics": record.metrics,
    }


@app.post("/ingest/jobs/execute")
async def ingest_job_execute(body: IngestExecuteBody, request: Request):
    # Internal-only compatibility endpoint; external clients should not call this.
    ctx = await _enforce_security(request, require_signature=True)
    scopes = list(ctx.get("scopes") or [])
    if ctx.get("method") != "disabled" and "memory.admin" not in scopes:
        raise HTTPException(status_code=403, detail="insufficient_scope")
    record = await ingest_store.get_job(str(body.job_id).strip())
    if record is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    tenant_id = str(ctx.get("tenant_id") or "")
    if tenant_id and str(record.tenant_id or "") != tenant_id:
        raise HTTPException(status_code=403, detail="forbidden")
    enqueued = await _enqueue_ingest_job(record)
    if not enqueued:
        err = {"stage": "enqueue", "code": "enqueue_failed", "message": "ingest_executor_unavailable"}
        await ingest_store.update_status(record.job_id, status="ENQUEUE_FAILED", error=err)
        return JSONResponse(
            status_code=503,
            content={
                "job_id": record.job_id,
                "session_id": record.session_id,
                "status": "ENQUEUE_FAILED",
                "attempts": dict(record.attempts or {}),
                "next_retry_at": record.next_retry_at,
                "last_error": err,
                "metrics": record.metrics,
            },
        )
    record = await ingest_store.get_job(record.job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return {
        "job_id": record.job_id,
        "session_id": record.session_id,
        "status": record.status,
        "attempts": dict(record.attempts or {}),
        "next_retry_at": record.next_retry_at,
        "last_error": record.last_error,
        "metrics": record.metrics,
        "enqueued": bool(enqueued),
    }


@app.get("/ingest/sessions/{session_id}")
async def ingest_session_status(session_id: str, request: Request):
    await _enforce_security(request)
    data = await ingest_store.get_session(session_id)
    return {
        "session_id": str(session_id),
        "latest_job_id": data.get("latest_job_id"),
        "latest_status": data.get("latest_status"),
        "cursor_committed": data.get("cursor_committed"),
    }


@app.get("/api/list")
async def api_list(request: Request, category: Optional[str] = None, include_internal: bool = False):
    ctx = await _enforce_security(request)
    internal_allowed = (ctx.get("method") == "disabled") or ("memory.admin" in (ctx.get("scopes") or []))

    routes = _iter_api_routes()
    cats: Dict[str, Dict[str, Any]] = {}
    ungrouped: List[Dict[str, Any]] = []

    for item in routes:
        path = str(item.get("path") or "")
        methods = list(item.get("methods") or [])
        required_scope, matched = _lookup_scope_mapping(path)
        is_public = path in PUBLIC_PATHS

        # Hide admin/internal endpoints unless explicitly requested and caller is allowed.
        if required_scope == "memory.admin" and not is_public:
            if not include_internal or not internal_allowed:
                continue

        cat_id, cat_name = _category_for_path(path)
        if category and str(category).strip():
            if str(category).strip().lower() != str(cat_id).lower():
                continue

        ep = {
            "path": path,
            "methods": methods,
            "category": cat_id,
            "auth_scope": required_scope,
            "public": bool(is_public),
            "missing_scope_mapping": (not matched),
        }
        if cat_id == "other":
            ungrouped.append(ep)
        else:
            if cat_id not in cats:
                cats[cat_id] = {"id": cat_id, "name": cat_name, "endpoints": []}
            cats[cat_id]["endpoints"].append(ep)

    # stable category ordering for clients
    categories = list(cats.values())
    categories.sort(key=lambda c: str(c.get("id") or ""))
    for c in categories:
        c["endpoints"].sort(key=lambda e: (e.get("path") or "", ",".join(e.get("methods") or [])))

    from datetime import datetime, timezone

    return {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "service": {
            "name": "memory",
            "base_url": str(request.base_url).rstrip("/"),
        },
        "categories": categories,
        "ungrouped": ungrouped,
    }


@app.post("/memory/v1/clear", response_model=ClearResponse)
async def memory_clear(body: ClearRequest, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    _require_explicit_scope(ctx, "memory.clear")
    tenant_id = str(ctx.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    reason = str(body.reason or "").strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason_required")
    if body.scope != "tenant":
        raise HTTPException(status_code=400, detail="scope_not_supported_yet")

    clear_lock = _get_tenant_clear_lock(tenant_id)
    if clear_lock.locked():
        raise HTTPException(status_code=409, detail="tenant_clear_in_progress")

    dry_run = not bool(body.confirm)
    api_key_id = _resolve_request_api_key_id(request, ctx)
    service = _get_svc()
    vector_store = getattr(service, "vectors", None)
    graph_store = getattr(service, "graph", None)
    audit_store = getattr(service, "audit", None)
    warnings: List[str] = []
    stores: Dict[str, ClearStoreStatus] = {}
    vector_entry_ids: List[str] = []
    estimated_vectors = 0
    estimated_graph_nodes = 0
    estimated_ingest_jobs = 0
    cleared_vectors = 0
    cleared_graph_nodes = 0
    cleared_ingest_jobs = 0
    partial_failure = False

    async with clear_lock:
        async def _estimate(name: str, obj: Any, method_name: str, **kwargs: Any) -> int:
            nonlocal partial_failure
            fn = getattr(obj, method_name, None)
            if not callable(fn):
                partial_failure = True
                detail = f"{method_name}_not_supported"
                warnings.append(f"{name}:{detail}")
                stores[name] = ClearStoreStatus(status="error", count=0, detail=detail)
                return 0
            try:
                count = int(await fn(**kwargs))
                stores[name] = ClearStoreStatus(status="estimated", count=count)
                return count
            except Exception as exc:
                partial_failure = True
                detail = f"{type(exc).__name__}: {str(exc)[:200]}"
                warnings.append(f"{name}:estimate_failed")
                stores[name] = ClearStoreStatus(status="error", count=0, detail=detail)
                return 0

        async def _execute(name: str, obj: Any, method_name: str, **kwargs: Any) -> int:
            nonlocal partial_failure
            fn = getattr(obj, method_name, None)
            if not callable(fn):
                partial_failure = True
                detail = f"{method_name}_not_supported"
                warnings.append(f"{name}:{detail}")
                stores[name] = ClearStoreStatus(status="error", count=0, detail=detail)
                return 0
            try:
                count = int(await fn(**kwargs))
                stores[name] = ClearStoreStatus(status="cleared", count=count)
                return count
            except Exception as exc:
                partial_failure = True
                detail = f"{type(exc).__name__}: {str(exc)[:200]}"
                warnings.append(f"{name}:clear_failed")
                stores[name] = ClearStoreStatus(status="error", count=0, detail=detail)
                return 0

        estimated_vectors = await _estimate("vectors", vector_store, "count_by_filter", tenant_id=tenant_id)
        estimated_ingest_jobs = await _estimate("ingest_jobs", ingest_store, "count_by_tenant", tenant_id=tenant_id)
        try:
            list_ids_fn = getattr(vector_store, "list_entry_ids_by_filter", None)
            if not callable(list_ids_fn):
                list_ids_fn = getattr(vector_store, "list_ids_by_filter", None)
            if callable(list_ids_fn):
                vector_entry_ids = list(await list_ids_fn(tenant_id=tenant_id))
            elif estimated_vectors > 0:
                warnings.append("vectors:list_ids_not_supported")
        except Exception as exc:
            partial_failure = True
            warnings.append("vectors:list_ids_failed")
            stores["vectors"] = ClearStoreStatus(
                status="error",
                count=estimated_vectors,
                detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            vector_entry_ids = []

        graph_direct = await _estimate("graph", graph_store, "count_tenant_nodes", tenant_id=tenant_id)
        graph_legacy = 0
        if vector_entry_ids:
            legacy_count_fn = getattr(graph_store, "count_legacy_memory_nodes_by_ids", None)
            if callable(legacy_count_fn):
                try:
                    graph_legacy = int(await legacy_count_fn(vector_entry_ids))
                except Exception as exc:
                    partial_failure = True
                    warnings.append("graph:legacy_estimate_failed")
                    stores["graph"] = ClearStoreStatus(
                        status="error",
                        count=graph_direct,
                        detail=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
            elif estimated_vectors > 0:
                warnings.append("graph:legacy_estimate_not_supported")
        estimated_graph_nodes = int(graph_direct + graph_legacy)
        if "graph" in stores and stores["graph"].status == "estimated":
            stores["graph"] = ClearStoreStatus(status="estimated", count=estimated_graph_nodes)

        if not dry_run:
            cleared_vectors = await _execute("vectors", vector_store, "delete_by_filter", tenant_id=tenant_id)
            cleared_ingest_jobs = await _execute("ingest_jobs", ingest_store, "clear_by_tenant", tenant_id=tenant_id)
            graph_primary = await _execute("graph", graph_store, "purge_tenant", tenant_id=tenant_id)
            graph_legacy_deleted = 0
            if vector_entry_ids:
                legacy_purge_fn = getattr(graph_store, "purge_legacy_memory_nodes_by_ids", None)
                if callable(legacy_purge_fn):
                    try:
                        graph_legacy_deleted = int(await legacy_purge_fn(vector_entry_ids))
                    except Exception as exc:
                        partial_failure = True
                        warnings.append("graph:legacy_clear_failed")
                        stores["graph"] = ClearStoreStatus(
                            status="error",
                            count=graph_primary,
                            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
                        )
                elif estimated_vectors > 0:
                    warnings.append("graph:legacy_clear_not_supported")
            cleared_graph_nodes = int(graph_primary + graph_legacy_deleted)
            if "graph" in stores and stores["graph"].status == "cleared":
                stores["graph"] = ClearStoreStatus(status="cleared", count=cleared_graph_nodes)

    status: Literal["completed", "partial_failure"] = "partial_failure" if partial_failure else "completed"
    request_id = _request_id_from_request(request)
    audit_payload = {
        "event_type": "memory_clear",
        "tenant_id": tenant_id,
        "api_key_id": api_key_id,
        "scope": body.scope,
        "dry_run": dry_run,
        "reason": reason,
        "request_id": request_id,
        "status": status,
        "estimated_vectors": estimated_vectors,
        "estimated_graph_nodes": estimated_graph_nodes,
        "estimated_ingest_jobs": estimated_ingest_jobs,
        "cleared_vectors": cleared_vectors,
        "cleared_graph_nodes": cleared_graph_nodes,
        "cleared_ingest_jobs": cleared_ingest_jobs,
        "warnings": list(warnings),
    }
    try:
        audit_logger.warning("memory.clear", extra=audit_payload)
    except Exception:
        pass
    try:
        add_one = getattr(audit_store, "add_one", None)
        if callable(add_one):
            event_name = "CLEAR_TENANT_DRY_RUN" if dry_run else "CLEAR_TENANT"
            audit_obj_id = f"{tenant_id}:{request_id or uuid.uuid4().hex[:12]}"
            await add_one(event_name, audit_obj_id, audit_payload, reason=reason)
    except Exception:
        pass

    return ClearResponse(
        tenant_id=tenant_id,
        scope=body.scope,
        api_key_id=api_key_id,
        dry_run=dry_run,
        status=status,
        estimated_vectors=estimated_vectors,
        estimated_graph_nodes=estimated_graph_nodes,
        estimated_ingest_jobs=estimated_ingest_jobs,
        cleared_vectors=cleared_vectors,
        cleared_graph_nodes=cleared_graph_nodes,
        cleared_ingest_jobs=cleared_ingest_jobs,
        warnings=warnings,
        stores=stores,
    )


@app.post("/write")
async def write(body: WriteBody, request: Request):
    # Enforce auth + signature and capture tenant for vector-store scoping.
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id") or "") if ctx else ""
    try:
        entries = [MemoryEntry.model_validate(e) for e in body.entries]
        # Propagate tenant_id into metadata for all entries unless explicitly provided.
        if tenant_id:
            for ent in entries:
                try:
                    md = dict(ent.metadata or {})
                    if not md.get("tenant_id"):
                        md["tenant_id"] = tenant_id
                    ent.metadata = md
                except Exception:
                    # Best-effort; do not block writes on metadata enrichment.
                    pass
        links = [Edge.model_validate(link) for link in body.links] if body.links else None
        res = await svc.write(entries, links, upsert=body.upsert, return_id_map=body.return_id_map)
        if body.return_id_map and isinstance(res, tuple):
            ver, id_map = res
            data = ver.model_dump()
            data["id_map"] = id_map
            return data
        if isinstance(res, tuple):
            # Should not happen unless caller forgot to set return_id_map; fall back gracefully.
            res = res[0]
        return res.model_dump()
    except Exception as e:
        # Expose underlying error to client to simplify diagnostics during experiments
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update(body: UpdateBody, request: Request):
    await _enforce_security(request, require_signature=True)
    try:
        ver = await svc.update(body.id, body.patch, reason=body.reason, confirm=body.confirm)
        return ver.model_dump()
    except SafetyError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/delete")
async def delete(body: DeleteBody, request: Request):
    await _enforce_security(request, require_signature=True)
    try:
        ver = await svc.delete(body.id, soft=body.soft, reason=body.reason, confirm=body.confirm)
        return ver.model_dump()
    except SafetyError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/link")
async def link(body: LinkBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id") or "")
    try:
        ok = await svc.link(
            body.src_id,
            body.dst_id,
            body.rel_type,
            weight=body.weight,
            confirm=body.confirm,
            tenant_id=tenant_id,
        )
        return {"ok": ok}
    except SafetyError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/rollback")
async def rollback(body: RollbackBody, request: Request):
    await _enforce_security(request, require_signature=True)
    ok = await svc.rollback_version(body.version)
    if not ok:
        raise HTTPException(status_code=400, detail=f"cannot rollback version {body.version}")
    return {"ok": True}


# ---- Batch editing (dangerous operations guarded) ----
class BatchDeleteBody(BaseModel):
    ids: List[str]
    soft: bool = True
    reason: Optional[str] = None
    confirm: Optional[bool] = None


@app.post("/batch_delete")
async def batch_delete(body: BatchDeleteBody, request: Request):
    await _enforce_security(request, require_signature=True)
    if not body.ids:
        return {"ok": True, "deleted": 0, "errors": []}
    results = {"deleted": 0, "errors": []}
    for mid in body.ids:
        try:
            await svc.delete(mid, soft=body.soft, reason=body.reason, confirm=body.confirm)
            results["deleted"] += 1
        except SafetyError as e:
            results["errors"].append({"id": mid, "error": str(e)})
    if results["errors"]:
        # partial success → 409 to indicate conflicts
        raise HTTPException(status_code=409, detail=results)
    return {"ok": True, **results}


class BatchLinkBody(BaseModel):
    links: List[LinkBody]
    confirm: Optional[bool] = None


@app.post("/batch_link")
async def batch_link(body: BatchLinkBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not body.links:
        return {"ok": True, "linked": 0, "errors": []}
    results = {"linked": 0, "errors": []}
    for link in body.links:
        try:
            ok = await svc.link(
                link.src_id,
                link.dst_id,
                link.rel_type,
                weight=link.weight,
                confirm=(link.confirm if link.confirm is not None else body.confirm),
                tenant_id=tenant_id,
            )
            if ok:
                results["linked"] += 1
        except SafetyError as e:
            results["errors"].append(
                {"src": link.src_id, "dst": link.dst_id, "rel": link.rel_type, "error": str(e)}
            )
    if results["errors"]:
        raise HTTPException(status_code=409, detail=results)
    return {"ok": True, **results}


# ---- Equivalence pending workflow endpoints ----
class EqPair(BaseModel):
    src_id: str
    dst_id: str
    score: Optional[float] = None
    reason: Optional[str] = None


class EqPairsBody(BaseModel):
    pairs: List[EqPair]
    weight: Optional[float] = None  # for confirm
    limit: Optional[int] = None     # for list


@app.get("/equiv/pending")
async def equiv_pending_list(request: Request, limit: Optional[int] = 50):
    await _enforce_security(request)
    lim = int(limit or 50)
    return await svc.list_pending_equivalence(limit=lim)


@app.post("/equiv/pending/add")
async def equiv_pending_add(body: EqPairsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    pairs = [(p.src_id, p.dst_id) for p in (body.pairs or [])]
    scores = [p.score for p in (body.pairs or [])]
    reasons = [p.reason for p in (body.pairs or [])]
    n = await svc.add_pending_equivalence(pairs, scores=scores, reasons=reasons)
    return {"ok": True, "added": int(n)}


@app.post("/equiv/pending/confirm")
async def equiv_pending_confirm(body: EqPairsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    pairs = [(p.src_id, p.dst_id) for p in (body.pairs or [])]
    n = await svc.confirm_pending_equivalence(pairs, weight=body.weight)
    return {"ok": True, "confirmed": int(n)}


@app.post("/equiv/pending/remove")
async def equiv_pending_remove(body: EqPairsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    pairs = [(p.src_id, p.dst_id) for p in (body.pairs or [])]
    n = await svc.remove_pending_equivalence(pairs)
    return {"ok": True, "removed": int(n)}


# ---- Config overview endpoints ----
_CONFIG_SECRET_HINTS = ("password", "passwd", "secret", "token", "api_key", "apikey", "access_key", "private_key")
_CONFIG_READONLY_PATHS = [
    "memory.llm",
    "memory.vector_store",
    "memory.graph_store",
    "memory.governance",
    "memory.reliability",
    "memory.api",
    "memory.events",
    "memory.write.batch",
    "memory.search.cache",
    "memory.search.character_expansion",
    "memory.search.sampling",
]


def _is_secret_key(name: str) -> bool:
    key = str(name or "").lower()
    return any(hint in key for hint in _CONFIG_SECRET_HINTS)


def _redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(k):
                out[k] = "******"
            else:
                out[k] = _redact_config(v)
        return out
    if isinstance(value, list):
        return [_redact_config(v) for v in value]
    return value


def _get_scoping_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sc = (((cfg.get("memory", {}) or {}).get("search", {}) or {}).get("scoping", {}) or {})
        if not isinstance(sc, dict):
            return {}
        return {
            "default_scope": sc.get("default_scope"),
            "user_match_mode": sc.get("user_match_mode"),
            "fallback_order": sc.get("fallback_order"),
            "require_user": sc.get("require_user"),
        }
    except Exception:
        return {}


def _modality_weights_supported() -> bool:
    try:
        return hasattr(svc, "vectors") and callable(getattr(svc.vectors, "set_modality_weights", None))
    except Exception:
        return False


def _current_modality_weights() -> Dict[str, float]:
    try:
        if hasattr(svc, "vectors") and hasattr(svc.vectors, "_mod_weights"):
            raw = getattr(svc.vectors, "_mod_weights", {}) or {}
            if isinstance(raw, dict):
                weights: Dict[str, float] = {}
                for k, v in raw.items():
                    try:
                        weights[str(k)] = float(v)
                    except Exception:
                        continue
                return weights
    except Exception:
        pass
    return {}


def _effective_modality_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    current = _current_modality_weights()
    if current:
        return current
    try:
        base = (((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("search", {}) or {}).get("modality_weights", {}) or {}
        if isinstance(base, dict):
            weights: Dict[str, float] = {}
            for k, v in base.items():
                try:
                    weights[str(k)] = float(v)
                except Exception:
                    continue
            return weights
    except Exception:
        pass
    return {}


def _hot_update_paths() -> List[Dict[str, Any]]:
    mod_supported = _modality_weights_supported()
    return [
        {"path": "memory.search.rerank", "endpoint": "/config/search/rerank", "supported": True},
        {"path": "memory.search.graph", "endpoint": "/config/graph", "supported": True},
        {"path": "memory.search.scoping", "endpoint": "/config/search/scoping", "supported": True},
        {"path": "memory.search.ann", "endpoint": "/config/search/ann", "supported": True},
        {"path": "memory.search.lexical_hybrid", "endpoint": "/config/search/lexical_hybrid", "supported": True},
        {
            "path": "memory.vector_store.search.modality_weights",
            "endpoint": "/config/search/modality_weights",
            "supported": mod_supported,
        },
    ]


def _hot_update_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rerank_base = get_search_weights(cfg)
    rerank_override = rtconf.get_rerank_weights_override()
    rerank_effective = dict(rerank_base)
    if rerank_override:
        rerank_effective.update(rerank_override)

    graph_base = get_graph_settings(cfg)
    graph_override = rtconf.get_graph_params_override()
    graph_effective = dict(graph_base)
    if graph_override:
        graph_effective.update(graph_override)

    scoping_base = _get_scoping_defaults(cfg)
    scoping_override = rtconf.get_scoping_override()
    scoping_effective = dict(scoping_base)
    if scoping_override:
        scoping_effective.update(scoping_override)

    ann_base = get_ann_settings(cfg)
    ann_override = rtconf.get_ann_override()
    ann_effective = dict(ann_base)
    if ann_override:
        ann_effective.update(ann_override)

    lexical_override = rtconf.get_lexical_hybrid_override()
    lexical_effective = resolve_lexical_hybrid_settings(cfg, lexical_override)

    modality_effective = _effective_modality_weights(cfg)

    return {
        "overrides": {
            "rerank": rerank_override,
            "graph": graph_override,
            "scoping": scoping_override,
            "ann": ann_override,
            "lexical_hybrid": lexical_override,
            "modality_weights": _current_modality_weights(),
        },
        "effective": {
            "search": {
                "rerank": rerank_effective,
                "scoping": scoping_effective,
                "ann": ann_effective,
                "lexical_hybrid": lexical_effective,
            },
            "graph": graph_effective,
            "vector_store": {"search": {"modality_weights": modality_effective}},
        },
    }


@app.get("/config")
async def get_config(request: Request):
    await _enforce_security(request)
    cfg = load_memory_config()
    memory_cfg = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
    core = _redact_config(memory_cfg)
    hot_snapshot = _hot_update_snapshot(cfg)
    return {
        "core": core,
        "hot_update": {
            "paths": _hot_update_paths(),
            "overrides": hot_snapshot.get("overrides", {}),
            "effective": hot_snapshot.get("effective", {}),
        },
        "read_only_paths": list(_CONFIG_READONLY_PATHS),
    }


# ---- Runtime hot-config endpoints ----

class RerankWeightsBody(BaseModel):
    alpha_vector: Optional[float] = None
    beta_bm25: Optional[float] = None
    gamma_graph: Optional[float] = None
    delta_recency: Optional[float] = None
    user_boost: Optional[float] = None
    domain_boost: Optional[float] = None
    session_boost: Optional[float] = None


@app.get("/config/search/rerank")
async def get_rerank_weights(request: Request):
    await _enforce_security(request)
    return rtconf.get_rerank_weights_override()


@app.post("/config/search/rerank")
async def set_rerank_weights(body: RerankWeightsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    data = {k: v for k, v in body.model_dump(exclude_none=True).items()}
    rtconf.set_rerank_weights(data)
    rtconf.save_overrides()
    return {"ok": True, "override": rtconf.get_rerank_weights_override()}


class GraphParamsBody(BaseModel):
    rel_whitelist: Optional[List[str]] = None
    max_hops: Optional[int] = None
    neighbor_cap_per_seed: Optional[int] = None
    restrict_to_user: Optional[bool] = None
    restrict_to_domain: Optional[bool] = None
    allow_cross_user: Optional[bool] = None
    allow_cross_domain: Optional[bool] = None


@app.get("/config/graph")
async def get_graph_params(request: Request):
    await _enforce_security(request)
    return rtconf.get_graph_params_override()


@app.post("/config/graph")
async def set_graph_params(body: GraphParamsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    rtconf.set_graph_params(
        rel_whitelist=body.rel_whitelist,
        max_hops=body.max_hops,
        neighbor_cap_per_seed=body.neighbor_cap_per_seed,
        restrict_to_user=body.restrict_to_user,
        restrict_to_domain=body.restrict_to_domain,
        allow_cross_user=body.allow_cross_user,
        allow_cross_domain=body.allow_cross_domain,
    )
    rtconf.save_overrides()
    return {"ok": True, "override": rtconf.get_graph_params_override()}


# ---- Scoping hot-config endpoints ----
class ScopingParamsBody(BaseModel):
    default_scope: Optional[str] = None  # session|domain|user
    user_match_mode: Optional[str] = None  # any|all
    fallback_order: Optional[List[str]] = None  # e.g., ["session","domain","user"]
    require_user: Optional[bool] = None


@app.get("/config/search/scoping")
async def get_scoping_params(request: Request):
    await _enforce_security(request)
    return rtconf.get_scoping_override()


@app.post("/config/search/scoping")
async def set_scoping_params(body: ScopingParamsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    rtconf.set_scoping_params(
        default_scope=body.default_scope,
        user_match_mode=body.user_match_mode,
        fallback_order=body.fallback_order,
        require_user=body.require_user,
    )
    rtconf.save_overrides()
    return {"ok": True, "override": rtconf.get_scoping_override()}


# ---- ANN default modalities hot-config ----
class AnnParamsBody(BaseModel):
    default_modalities: Optional[List[str]] = None  # e.g., ["text","image","audio"]
    default_all_modalities: Optional[bool] = None   # if true and list omitted, use all


@app.get("/config/search/ann")
async def get_ann_params(request: Request):
    await _enforce_security(request)
    return rtconf.get_ann_override()


@app.post("/config/search/ann")
async def set_ann_params(body: AnnParamsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    rtconf.set_ann_params(default_modalities=body.default_modalities, default_all_modalities=body.default_all_modalities)
    rtconf.save_overrides()
    return {"ok": True, "override": rtconf.get_ann_override()}


class LexicalHybridParamsBody(BaseModel):
    enabled: Optional[bool] = None
    corpus_limit: Optional[int] = None
    lexical_topn: Optional[int] = None
    normalize_scores: Optional[bool] = None


@app.get("/config/search/lexical_hybrid")
async def get_lexical_hybrid_params(request: Request):
    await _enforce_security(request)
    return rtconf.get_lexical_hybrid_override()


@app.post("/config/search/lexical_hybrid")
async def set_lexical_hybrid_params(body: LexicalHybridParamsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    rtconf.set_lexical_hybrid_params(
        enabled=body.enabled,
        corpus_limit=body.corpus_limit,
        lexical_topn=body.lexical_topn,
        normalize_scores=body.normalize_scores,
    )
    rtconf.save_overrides()
    return {"ok": True, "override": rtconf.get_lexical_hybrid_override()}


# ---- Graph v0.1 endpoints ----
class GraphUpsertBody(GraphUpsertRequest):
    """Graph upsert body without requiring tenant_id from client."""

    model_config = ConfigDict(extra="allow")


class GraphSearchV1Body(BaseModel):
    query: str
    topk: Optional[int] = None
    source_id: Optional[str] = None
    include_evidence: bool = True


class StateCurrentBody(BaseModel):
    subject_id: str
    property: str


class StateAtTimeBody(BaseModel):
    subject_id: str
    property: str
    t_iso: str


class StateChangesBody(BaseModel):
    subject_id: str
    property: str
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    limit: int = 200
    order: str = "asc"


class StateTimeSinceBody(BaseModel):
    subject_id: str
    property: str
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None


class StatePendingListBody(BaseModel):
    subject_id: Optional[str] = None
    property: Optional[str] = None
    status: Optional[str] = None
    limit: int = 200


class StatePendingApproveBody(BaseModel):
    pending_id: str
    apply: bool = True
    note: Optional[str] = None


class StatePendingRejectBody(BaseModel):
    pending_id: str
    note: Optional[str] = None


@app.post("/graph/v1/search")
async def graph_search_v1(body: GraphSearchV1Body, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    gate = _gate_high_cost("graph_search")
    if gate:
        return gate
    topk = int(body.topk) if body.topk is not None else _API_DEFAULT_TOPK_GRAPH_SEARCH
    if topk <= 0:
        topk = _API_DEFAULT_TOPK_GRAPH_SEARCH
    if topk > SEARCH_TOPK_HARD_LIMIT:
        topk = SEARCH_TOPK_HARD_LIMIT
    started = time.perf_counter()
    try:
        res = await asyncio.wait_for(
            graph_svc.search_events_v1(
                tenant_id=tenant_id,
                query=body.query,
                topk=topk,
                source_id=body.source_id,
                include_evidence=body.include_evidence,
            ),
            timeout=HIGH_COST_TIMEOUT,
        )
        _record_breaker_outcome("graph_search", None)
        _record_graph_metrics("search.v1", started)
        return res
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("graph_search", exc)
        _record_graph_metrics("search.v1", started, "timeout")
        raise HTTPException(status_code=504, detail="graph_search_timeout") from None
    except HTTPException as exc:
        if exc.status_code >= 500:
            _record_breaker_outcome("graph_search", exc)
        else:
            _record_breaker_outcome("graph_search", None)
        _record_graph_metrics("search.v1", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception as exc:
        _record_breaker_outcome("graph_search", exc)
        _record_graph_metrics("search.v1", started, "server_error")
        raise



@app.post("/graph/v0/upsert")
async def graph_upsert(body: GraphUpsertBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    try:
        req_model = _inject_tenant(body, tenant_id)
        await graph_svc.upsert(req_model)
        return {"ok": True}
    except GraphValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - runtime safety net
        raise HTTPException(status_code=500, detail=str(e))


class PendingEquivBody(BaseModel):
    pending_equivs: List[PendingEquiv]


@app.post("/graph/v0/admin/equiv/pending")
async def graph_equiv_pending(body: PendingEquivBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    try:
        items = [pe.model_copy(update={"tenant_id": tenant_id}) for pe in body.pending_equivs]
        equiv_store.upsert_pending(tenant_id=tenant_id, records=items)
        return {"ok": True, "count": len(items)}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/v0/admin/equiv/pending")
async def graph_equiv_list(request: Request, status: str = "pending", limit: int = 200):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    items = equiv_store.list_pending(tenant_id=tenant_id, status=status, limit=limit)
    return {"items": items}


class ApproveEquivBody(BaseModel):
    pending_id: str
    reviewer: Optional[str] = None


@app.post("/graph/v0/admin/equiv/approve")
async def graph_equiv_approve(body: ApproveEquivBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    res = equiv_store.approve(tenant_id=tenant_id, pending_id=body.pending_id, reviewer=body.reviewer)
    if res.get("merged", 0) == 0:
        raise HTTPException(status_code=404, detail="pending_equiv_not_found")
    return {"ok": True, "merged": res.get("merged", 0)}


class RejectEquivBody(BaseModel):
    pending_id: str
    reviewer: Optional[str] = None


@app.post("/graph/v0/admin/equiv/reject")
async def graph_equiv_reject(body: RejectEquivBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    res = equiv_store.reject(tenant_id=tenant_id, pending_id=body.pending_id, reviewer=body.reviewer)
    if res.get("updated", 0) == 0:
        raise HTTPException(status_code=404, detail="pending_equiv_not_found")
    return {"ok": True, "updated": res.get("updated", 0)}


@app.get("/graph/v0/semantic_summary")
async def graph_semantic_summary(request: Request, source_id: Optional[str] = None, limit: int = 100):
    """Get VLM semantic summary: episodic facts, semantic facts, and equivalences.
    
    This endpoint aggregates VLM-generated semantic outputs for display in the demo UI.
    """
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        # 1. Get episodic facts from events with VLM descriptions
        events = await graph_svc.list_events(
            tenant_id=tenant_id,
            source_id=source_id,
            limit=limit,
        )
        episodic: list[str] = []
        semantic: list[str] = []
        
        for ev in events:
            summary = ev.get("summary") or ""
            source = ev.get("source") or ""
            event_type = ev.get("event_type")
            
            # VLM-generated events (source=llm) with Chinese/rich descriptions
            if source == "llm" and summary:
                # Check if this looks like a VLM semantic description (not just detection summary)
                if not summary.startswith(("object(s)", "face(s)", "voice segment")):
                    episodic.append(summary)
            
            # Events with event_type are semantic facts
            if event_type and summary:
                semantic.append(f"[{event_type}] {summary}")
        
        # 2. Get equivalences (PendingEquiv has entity_id/candidate_id, not source_id/target_name)
        pending_equivs = equiv_store.list_pending(tenant_id=tenant_id, status="pending", limit=50)
        approved_equivs = equiv_store.list_pending(tenant_id=tenant_id, status="approved", limit=50)
        
        equivalence: list[dict] = []
        for eq in pending_equivs + approved_equivs:
            if isinstance(eq, dict):
                # Frontend expects targetName for display; use candidate_id as fallback name
                candidate_id = eq.get("candidate_id") or ""
                target_name = eq.get("target_name") or eq.get("name") or candidate_id
                equivalence.append({
                    "id": eq.get("id") or eq.get("pending_id"),
                    "sourceId": eq.get("entity_id"),  # PendingEquiv.entity_id
                    "targetId": candidate_id,  # PendingEquiv.candidate_id
                    "targetName": target_name,  # For frontend display (fallback to candidate_id)
                    "confidence": eq.get("confidence", 1.0),
                    "status": eq.get("status", "pending"),
                    "evidenceId": eq.get("evidence_id"),
                    "reviewer": eq.get("reviewer"),
                    "reviewedAt": eq.get("reviewed_at"),
                })
        
        _record_graph_metrics("semantic_summary", started)
        return {
            "episodic": episodic[:50],  # Limit to 50 items
            "semantic": semantic[:50],
            "equivalence": equivalence,
        }
    except Exception as exc:
        _record_graph_metrics("semantic_summary", started, "server_error")
        raise HTTPException(status_code=500, detail=str(exc))


class CleanupTTLBody(BaseModel):
    buffer_hours: float = 24.0
    limit: int = 500
    dry_run: bool = False


@app.post("/graph/v0/admin/ttl/cleanup")
async def graph_ttl_cleanup(body: CleanupTTLBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.store.cleanup_expired(
            tenant_id=tenant_id,
            buffer_hours=body.buffer_hours,
            limit=body.limit,
            dry_run=body.dry_run,
        )
        # TTL cleanup metrics (treat dry_run as a separate status)
        status = "dry_run" if body.dry_run else "ok"
        record_ttl_cleanup(status, res.get("nodes", 0), res.get("edges", 0))
        add_graph_latency("ttl.cleanup", int((time.perf_counter() - started) * 1000))
        return {"ok": True, "deleted": res}
    except Exception:
        record_ttl_cleanup("error", 0, 0)
        add_graph_latency("ttl.cleanup", int((time.perf_counter() - started) * 1000))
        raise


@app.get("/graph/v0/admin/export_srot")
async def graph_export_srot(
    request: Request,
    rel: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 1000,
    cursor: Optional[str] = None,
):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    rels = [r.strip() for r in rel.split(",")] if rel else None
    page = await graph_svc.store.export_srot(
        tenant_id=tenant_id,
        rel_types=rels,
        min_confidence=min_confidence,
        limit=limit,
        cursor=cursor,
    )
    return page


@app.get("/graph/v0/segments")
async def graph_list_segments(
    request: Request,
    source_id: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    modality: Optional[str] = None,
    limit: int = 200,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_segments(
            tenant_id=tenant_id,
            source_id=source_id,
            start=start,
            end=end,
            modality=modality,
            limit=limit,
        )
        _record_graph_metrics("segments.list", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("segments.list", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("segments.list", started, "server_error")
        raise


@app.get("/graph/v0/entities/{entity_id}/timeline")
async def graph_entity_timeline(request: Request, entity_id: str, limit: int = 200):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 500))
    started = time.perf_counter()
    try:
        res = await graph_svc.entity_timeline(
            tenant_id=tenant_id,
            entity_id=entity_id,
            limit=bounded_limit,
        )
        logger.info(
            "graph.entity.timeline",
            extra={
                "event": "graph.entity.timeline",
                "tenant_id": tenant_id,
                "entity_id": entity_id,
                "limit": bounded_limit,
                "count": len(res),
                "status": "ok",
            },
        )
        _record_graph_metrics("entity.timeline", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("entity.timeline", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("entity.timeline", started, "server_error")
        raise


@app.get("/graph/v0/entities/{entity_id}/evidences")
async def graph_entity_evidences(
    request: Request,
    entity_id: str,
    subtype: Optional[str] = None,
    source_id: Optional[str] = None,
    limit: int = 50,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 200))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_entity_evidences(
            tenant_id=tenant_id,
            entity_id=entity_id,
            subtype=subtype,
            source_id=source_id,
            limit=bounded_limit,
        )
        _record_graph_metrics("entity.evidences", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("entity.evidences", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("entity.evidences", started, "server_error")
        raise


@app.get("/graph/v0/entities/resolve")
async def graph_entities_resolve(
    request: Request,
    name: str,
    entity_type: Optional[str] = Query(None, alias="type"),
    limit: int = 20,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 200))
    started = time.perf_counter()
    try:
        res = await graph_svc.resolve_entities(
            tenant_id=tenant_id,
            name=name,
            entity_type=entity_type,
            limit=bounded_limit,
        )
        _record_graph_metrics("entities.resolve", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("entities.resolve", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("entities.resolve", started, "server_error")
        raise


@app.get("/graph/v0/events")
async def graph_list_events(
    request: Request,
    segment_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    place_id: Optional[str] = None,
    source_id: Optional[str] = None,
    relation: Optional[str] = None,  # NEXT_EVENT|CAUSES|CO_OCCURS_WITH
    layer: Optional[str] = None,  # fact|semantic|hypothesis
    status: Optional[str] = None,  # candidate|accepted|rejected
    limit: int = 100,
): 
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 500))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_events(
            tenant_id=tenant_id,
            segment_id=segment_id,
            entity_id=entity_id,
            place_id=place_id,
            source_id=source_id,
            relation=relation,
            layer=layer,
            status=status,
            limit=bounded_limit,
        )
        logger.info(
            "graph.events.list",
            extra={
                "event": "graph.events.list",
                "tenant_id": tenant_id,
                "segment_id": segment_id,
                "entity_id": entity_id,
                "place_id": place_id,
                "source_id": source_id,
                "limit": bounded_limit,
                "count": len(res),
                "status": "ok",
            },
        )
        _record_graph_metrics("events.list", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("events.list", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("events.list", started, "server_error")
        raise


@app.get("/graph/v0/places")
async def graph_list_places(
    request: Request,
    name: Optional[str] = None,
    segment_id: Optional[str] = None,
    covers_timeslice: Optional[str] = None,
    limit: int = 100,
): 
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 500))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_places(
            tenant_id=tenant_id,
            name=name,
            segment_id=segment_id,
            covers_timeslice=covers_timeslice,
            limit=bounded_limit,
        )
        logger.info(
            "graph.places.list",
            extra={
                "event": "graph.places.list",
                "tenant_id": tenant_id,
                "place_name": name,
                "segment_id": segment_id,
                "limit": bounded_limit,
                "count": len(res),
                "status": "ok",
            },
        )
        _record_graph_metrics("places.list", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("places.list", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("places.list", started, "server_error")
        raise


@app.get("/graph/v0/events/{event_id}")
async def graph_get_event(request: Request, event_id: str):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.event_detail(tenant_id=tenant_id, event_id=event_id)
        if not data:
            _record_graph_metrics("events.detail", started, "not_found")
            raise HTTPException(status_code=404, detail="event_not_found")
        _record_graph_metrics("events.detail", started)
        return {"item": data}
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("events.detail", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("events.detail", started, "server_error")
        raise


@app.get("/graph/v0/explain/first_meeting")
async def graph_explain_first_meeting(request: Request, me_id: str, other_id: str):
    """Explain the first meeting between two entities within the current tenant."""
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.explain_first_meeting(
            tenant_id=tenant_id,
            me_id=me_id,
            other_id=other_id,
        )
        _record_graph_metrics("explain.first_meeting", started)
        return {"item": data}
    except Exception:
        _record_graph_metrics("explain.first_meeting", started, "server_error")
        raise


@app.get("/graph/v0/explain/event/{event_id}")
async def graph_explain_event(request: Request, event_id: str):
    """Return a structured evidence chain for a given event."""
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.explain_event_evidence(tenant_id=tenant_id, event_id=event_id)
        if not data.get("event"):
            _record_graph_metrics("explain.event", started, "not_found")
            raise HTTPException(status_code=404, detail="event_not_found")
        _record_graph_metrics("explain.event", started)
        return {"item": data}
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("explain.event", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("explain.event", started, "server_error")
        raise


@app.get("/graph/v0/places/{place_id}")
async def graph_get_place(request: Request, place_id: str):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.place_detail(tenant_id=tenant_id, place_id=place_id)
        if not data:
            _record_graph_metrics("places.detail", started, "not_found")
            raise HTTPException(status_code=404, detail="place_not_found")
        _record_graph_metrics("places.detail", started)
        return {"item": data}
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("places.detail", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("places.detail", started, "server_error")
        raise


# ---- Graph admin helpers ----
class BuildEventRelationsBody(BaseModel):
    source_id: Optional[str] = None
    place_id: Optional[str] = None
    limit: int = 1000
    create_causes: bool = True


@app.post("/graph/v0/admin/build_event_relations")
async def graph_build_event_relations(body: BuildEventRelationsBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.build_event_relations(
            tenant_id=tenant_id,
            source_id=body.source_id,
            place_id=body.place_id,
            limit=body.limit,
            create_causes=body.create_causes,
        )
        _record_graph_metrics("admin.event_relations", started)
        return {"result": res}
    except Exception:
        _record_graph_metrics("admin.event_relations", started, "server_error")
        raise


class BuildTimeSlicesBody(BaseModel):
    window_seconds: float = 3600.0
    source_id: Optional[str] = None
    modality: Optional[str] = None
    modes: Optional[List[str]] = None  # e.g., ["media_window","day","hour"]


@app.post("/graph/v0/admin/build_timeslices")
async def graph_build_timeslices(body: BuildTimeSlicesBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.build_time_slices_from_segments(
            tenant_id=tenant_id,
            window_seconds=body.window_seconds,
            source_id=body.source_id,
            modality=body.modality,
            modes=body.modes,
        )
        _record_graph_metrics("admin.timeslices", started)
        return {"result": res}
    except Exception:
        _record_graph_metrics("admin.timeslices", started, "server_error")
        raise


class BuildCooccursBody(BaseModel):
    min_weight: float = 1.0
    mode: Optional[str] = None  # timeslice|event


class BuildFirstMeetingsBody(BaseModel):
    limit: int = 5000


@app.post("/graph/v0/admin/build_cooccurs")
async def graph_build_cooccurs(body: BuildCooccursBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        mode = str(body.mode or "timeslice").strip().lower()
        if mode in {"event", "events"}:
            res = await graph_svc.build_cooccurs_from_events(
                tenant_id=tenant_id,
                min_weight=body.min_weight,
            )
        elif mode in {"timeslice", "timeslices"}:
            res = await graph_svc.build_cooccurs_from_timeslices(
                tenant_id=tenant_id,
                min_weight=body.min_weight,
            )
        else:
            raise HTTPException(status_code=400, detail="invalid_cooccurs_mode")
        _record_graph_metrics("admin.cooccurs", started)
        return {"result": res}
    except Exception:
        _record_graph_metrics("admin.cooccurs", started, "server_error")
        raise


@app.post("/graph/v0/admin/build_first_meetings")
async def graph_build_first_meetings(body: BuildFirstMeetingsBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.build_first_meetings(
            tenant_id=tenant_id,
            limit=body.limit,
        )
        _record_graph_metrics("admin.first_meetings", started)
        return {"result": res}
    except Exception:
        _record_graph_metrics("admin.first_meetings", started, "server_error")
        raise


class PurgeSourceBody(BaseModel):
    source_id: str
    delete_orphans: bool = False


@app.post("/graph/v0/admin/purge_source")
async def graph_purge_source(body: PurgeSourceBody, request: Request):
    ctx = await _enforce_security(request, require_signature=True)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        res = await graph_svc.purge_source(
            tenant_id=tenant_id,
            source_id=body.source_id,
            delete_orphans=body.delete_orphans,
        )
        _record_graph_metrics("admin.purge_source", started)
        return {"result": res}
    except GraphValidationError as exc:
        _record_graph_metrics("admin.purge_source", started, "bad_request")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        _record_graph_metrics("admin.purge_source", started, "server_error")
        raise


@app.get("/graph/v0/timeslices")
async def graph_list_timeslices(
    request: Request,
    kind: Optional[str] = None,
    covers_segment: Optional[str] = None,
    covers_event: Optional[str] = None,
    limit: int = 200,
): 
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 500))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_time_slices(
            tenant_id=tenant_id,
            kind=kind,
            covers_segment=covers_segment,
            covers_event=covers_event,
            limit=bounded_limit,
        )
        _record_graph_metrics("timeslices.list", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("timeslices.list", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("timeslices.list", started, "server_error")
        raise


@app.get("/graph/v0/timeslices/range")
async def graph_list_timeslices_range(
    request: Request,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 200,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    bounded_limit = max(1, min(limit, 500))
    started = time.perf_counter()
    try:
        res = await graph_svc.list_time_slices_by_range(
            tenant_id=tenant_id,
            start_iso=start_iso,
            end_iso=end_iso,
            kind=kind,
            limit=bounded_limit,
        )
        _record_graph_metrics("timeslices.range", started)
        return {"items": res}
    except HTTPException as exc:
        _record_graph_metrics("timeslices.range", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("timeslices.range", started, "server_error")
        raise


# ---- Memory discovery (v2.0) ----
@app.get("/memory/v1/entities")
async def memory_entities(
    request: Request,
    user_tokens: Optional[List[str]] = Query(None),
    type: Optional[str] = Query(None, alias="type"),
    query: Optional[str] = None,
    mentioned_since: Optional[str] = None,
    limit: int = 20,
    cursor: Optional[str] = None,
    memory_domain: Optional[str] = None,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    tokens = _parse_user_tokens_param(user_tokens)
    if not tokens:
        tokens = [f"u:{tenant_id}"]
    bounded_limit = max(1, min(int(limit or 20), 100))
    offset = _decode_cursor(cursor)
    start_dt = _parse_iso_datetime(mentioned_since)
    memory_domain = str(memory_domain or "").strip() or None

    total = await graph_svc.count_entities_overview(
        tenant_id=tenant_id,
        entity_type=type,
        query=query,
        mentioned_since=start_dt,
        user_ids=tokens,
        memory_domain=memory_domain,
    )
    rows = await graph_svc.list_entities_overview(
        tenant_id=tenant_id,
        entity_type=type,
        query=query,
        mentioned_since=start_dt,
        user_ids=tokens,
        memory_domain=memory_domain,
        limit=bounded_limit,
        offset=offset,
    )
    entities: List[Dict[str, Any]] = []
    for row in rows:
        ent = row.get("entity") or {}
        ent_id = ent.get("id")
        name = ent.get("name") or ent.get("manual_name") or ent.get("cluster_label")
        aliases = []
        for key in ("name", "manual_name", "cluster_label"):
            val = ent.get(key)
            if val and str(val).strip():
                aliases.append(str(val).strip())
        aliases = list(dict.fromkeys(aliases))
        entities.append(
            {
                "id": ent_id,
                "name": name,
                "type": ent.get("type"),
                "aliases": aliases,
                "first_mentioned": row.get("first_mentioned"),
                "last_mentioned": row.get("last_mentioned"),
                "mention_count": row.get("mention_count") or 0,
            }
        )

    has_more = (offset + len(entities)) < total
    resp: Dict[str, Any] = {
        "entities": entities,
        "total": total,
        "has_more": has_more,
    }
    if has_more:
        resp["next_cursor"] = _encode_cursor(offset + len(entities))
    return resp


@app.get("/memory/v1/topics")
async def memory_topics(
    request: Request,
    user_tokens: Optional[List[str]] = Query(None),
    query: Optional[str] = None,
    parent_path: Optional[str] = None,
    min_events: Optional[int] = None,
    limit: int = 20,
    cursor: Optional[str] = None,
    memory_domain: Optional[str] = None,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    tokens = _parse_user_tokens_param(user_tokens)
    if not tokens:
        tokens = [f"u:{tenant_id}"]
    bounded_limit = max(1, min(int(limit or 20), 100))
    offset = _decode_cursor(cursor)
    memory_domain = str(memory_domain or "").strip() or None

    total = await graph_svc.count_topics_overview(
        tenant_id=tenant_id,
        query=query,
        parent_path=parent_path,
        min_events=min_events,
        user_ids=tokens,
        memory_domain=memory_domain,
    )
    rows = await graph_svc.list_topics_overview(
        tenant_id=tenant_id,
        query=query,
        parent_path=parent_path,
        min_events=min_events,
        user_ids=tokens,
        memory_domain=memory_domain,
        limit=bounded_limit,
        offset=offset,
    )
    topics: List[Dict[str, Any]] = []
    for row in rows:
        last_ts = _parse_iso_datetime(row.get("last_mentioned"))
        topics.append(
            {
                "topic_path": row.get("topic_path"),
                "display_name": None,
                "event_count": row.get("event_count") or 0,
                "first_mentioned": row.get("first_mentioned"),
                "last_mentioned": row.get("last_mentioned"),
                "status": _topic_status_from_last(last_ts),
            }
        )

    has_more = (offset + len(topics)) < total
    resp: Dict[str, Any] = {
        "topics": topics,
        "total": total,
        "has_more": has_more,
        "status_thresholds": {
            "ongoing_days": int(TOPIC_STATUS_ONGOING_DAYS),
            "paused_days": int(TOPIC_STATUS_PAUSED_DAYS),
        },
    }
    if has_more:
        resp["next_cursor"] = _encode_cursor(offset + len(topics))
    return resp


@app.post("/memory/v1/resolve-entity")
async def memory_resolve_entity(body: ResolveEntityBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    name = str(body.name or "").strip()
    if not name:
        return {"found": False, "resolved_entity": None, "candidates": []}
    _ = _parse_user_tokens_param(body.user_tokens)
    entity_id, resolved, candidates = await _resolve_entity_candidates(
        tenant_id=tenant_id,
        name=name,
        entity_type=str(body.type or "").strip() or None,
        limit=max(1, min(int(body.limit or 5), 20)),
    )
    resp: Dict[str, Any] = {"found": bool(entity_id), "resolved_entity": resolved}
    if candidates:
        resp["candidates"] = candidates
    return resp


@app.get("/memory/v1/state/properties")
async def memory_state_properties(
    request: Request,
    user_tokens: Optional[List[str]] = Query(None),
    limit: int = 50,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")
    _ = _parse_user_tokens_param(user_tokens)
    data = _load_state_properties()
    props = []
    for it in data.get("properties") or []:
        if not isinstance(it, dict):
            continue
        props.append(
            {
                "name": it.get("name"),
                "description": it.get("description"),
                "value_type": it.get("value_type"),
                "allowed_values": it.get("allowed_values"),
                "allow_raw_value": bool(it.get("allow_raw_value")),
            }
        )
    bounded_limit = max(1, min(int(limit or 50), 200))
    return {"vocab_version": data.get("version"), "properties": props[:bounded_limit]}


# ---- Memory semantic (Phase 4) ----
@app.post("/memory/v1/explain")
async def memory_explain(body: ExplainBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    event_id = str(body.event_id or "").strip()
    empty_resp: Dict[str, Any] = {
        "found": False,
        "event_id": event_id or None,
        "event": None,
        "entities": [],
        "places": [],
        "timeslices": [],
        "evidences": [],
        "utterances": [],
        "utterance_speakers": [],
        "knowledge": [],
    }
    if not event_id:
        return empty_resp

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        # Keep explain compatible with existing callers while applying tenant-scoped filtering.
        user_tokens = [f"u:{tenant_id}"]
    memory_domain = str(body.memory_domain or "").strip() or None

    bundle = await graph_svc.explain_event_evidence(
        tenant_id=tenant_id,
        event_id=event_id,
        user_ids=(list(user_tokens) if user_tokens else None),
        memory_domain=memory_domain,
    )
    resp: Dict[str, Any] = dict(empty_resp)
    resp.update(
        {
            "event": bundle.get("event"),
            "entities": list(bundle.get("entities") or []),
            "places": list(bundle.get("places") or []),
            "timeslices": list(bundle.get("timeslices") or []),
            "evidences": list(bundle.get("evidences") or []),
            "utterances": list(bundle.get("utterances") or []),
            "utterance_speakers": list(bundle.get("utterance_speakers") or []),
            "knowledge": list(bundle.get("knowledge") or []),
        }
    )
    if resp.get("event"):
        resp["found"] = True
        resp["event_id"] = str(resp["event"].get("id") or event_id)
    if body.debug:
        resp["trace"] = {"source": "graph_svc.explain_event_evidence"}
    return resp


@app.post("/memory/v1/topic-timeline")
async def memory_topic_timeline(body: TopicTimelineBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("topic_timeline")
    if gate:
        return gate
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    topic = str(body.topic or "").strip() or None
    topic_id = str(body.topic_id or "").strip() or None
    topic_path = str(body.topic_path or "").strip() or None
    explicit_topic_path = bool(topic_path)
    explicit_topic_id = bool(topic_id)
    raw_keywords = [str(k).strip() for k in (body.keywords or []) if str(k).strip()]

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]

    missing: List[str] = []
    if not topic and not topic_id and not topic_path and not raw_keywords:
        missing.append("topic|keywords")
    if not user_tokens:
        missing.append("user_tokens")
    if missing:
        return _missing_core_requirements(missing, target="topic_timeline")

    limit = int(body.limit or 50)
    if limit <= 0:
        limit = 1
    if limit > SEARCH_TOPK_HARD_LIMIT:
        limit = SEARCH_TOPK_HARD_LIMIT

    start_dt, end_dt = _parse_time_range_dict(body.time_range)
    memory_domain = str(body.memory_domain or "").strip() or None

    norm_tags: Optional[List[str]] = None
    norm_keywords: Optional[List[str]] = None
    if not topic_path and topic:
        try:
            normalized = normalize_topic_text(topic)
            topic_path = str(normalized.topic_path or "").strip() or None
            if not topic_id:
                topic_id = str(normalized.topic_id or topic).strip() or None
            norm_tags = list(normalized.tags) or None
            norm_keywords = list(normalized.keywords) or None
        except Exception:
            pass

    keywords: Optional[List[str]] = None
    if raw_keywords:
        keywords = raw_keywords
    elif norm_keywords:
        keywords = list(norm_keywords)

    # If topic_path is derived and falls into _uncategorized, fallback to retrieval.
    force_retrieval = False
    if not explicit_topic_path and not explicit_topic_id and topic_path and topic_path.startswith("_uncategorized/"):
        topic_path = None
        force_retrieval = True

    async def _run_graph_query(event_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return await graph_svc.topic_timeline(
            tenant_id=tenant_id,
            topic_id=topic_id if not topic_path else None,
            topic_path=topic_path,
            tags=norm_tags,
            keywords=keywords,
            start=start_dt,
            end=end_dt,
            user_ids=user_tokens,
            memory_domain=memory_domain,
            limit=limit,
            event_ids=event_ids,
        )

    try:
        trace: Dict[str, Any] = {}
        if (topic_path or topic_id) and not force_retrieval:
            events = await asyncio.wait_for(_run_graph_query(), timeout=HIGH_COST_TIMEOUT)
            trace["source"] = "graph_topic_filter"
        else:
            time_hints = {}
            if start_dt:
                time_hints["start_iso"] = start_dt.isoformat()
            if end_dt:
                time_hints["end_iso"] = end_dt.isoformat()
            retrieval_res = await asyncio.wait_for(
                retrieval(
                    svc,
                    tenant_id=str(tenant_id),
                    user_tokens=list(user_tokens),
                    query=str(topic or topic_id or " ".join(raw_keywords)).strip(),
                    strategy="dialog_v2",
                    topk=limit,
                    memory_domain=str(memory_domain or "dialog"),
                    user_match="all",
                    run_id=(str(body.session_id) if body.session_id else None),
                    debug=bool(body.debug),
                    with_answer=False,
                    task="GENERAL",
                    llm_policy="best_effort",
                    backend="tkg",
                    tkg_explain=False,
                    time_hints=time_hints if time_hints else None,
                    request_id=_request_id_from_request(request),
                    trace_id=str(request.headers.get("X-Trace-ID") or "").strip() or None,
                    byok_route=None,
                ),
                timeout=HIGH_COST_TIMEOUT,
            )
            evidence_details = list((retrieval_res or {}).get("evidence_details") or [])
            event_ids: List[str] = []
            for item in evidence_details:
                ev_id = str(item.get("tkg_event_id") or item.get("event_id") or "").strip()
                if ev_id:
                    event_ids.append(ev_id)
            events = await asyncio.wait_for(_run_graph_query(event_ids=event_ids), timeout=HIGH_COST_TIMEOUT)
            trace["source"] = "retrieval_dialog_v2"
            trace["retrieval_evidence"] = len(evidence_details)

        timeline: List[Dict[str, Any]] = []
        for ev in events or []:
            when = ev.get("t_abs_start") or ev.get("t_abs_end")
            evidence_count = ev.get("evidence_count")
            if evidence_count is None:
                evidence_count = len(ev.get("source_turn_ids") or [])
            ev_id = ev.get("id")
            item: Dict[str, Any] = {
                "event_id": ev_id,
                "logical_event_id": ev.get("logical_event_id"),
                "when": when,
                "summary": ev.get("summary") or ev.get("desc") or "",
                "confidence": ev.get("event_confidence") or ev.get("importance"),
                "evidence_count": evidence_count,
            }
            if (body.with_quotes or body.with_entities) and ev_id:
                bundle = await graph_svc.explain_event_evidence(tenant_id=tenant_id, event_id=str(ev_id))
                if body.with_quotes:
                    item["quotes"] = _quotes_from_bundle(bundle)
                if body.with_entities:
                    ents = list(bundle.get("entities") or [])
                    item["entities"] = [
                        {
                            "id": e.get("id") or e.get("entity_id"),
                            "name": _entity_display_name(e),
                            "type": e.get("type"),
                        }
                        for e in ents
                    ]
            timeline.append(item)

        resp: Dict[str, Any] = {
            "topic": topic,
            "topic_id": topic_id,
            "topic_path": topic_path,
            "status": _timeline_status(events or []),
            "timeline": timeline,
            "total": len(timeline),
        }
        if body.debug:
            trace["count"] = len(timeline)
            trace["normalized_tags"] = norm_tags
            trace["normalized_keywords"] = norm_keywords
            resp["trace"] = trace
        _record_breaker_outcome("topic_timeline", None)
        return resp
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("topic_timeline", exc)
        raise HTTPException(status_code=504, detail="timeline_timeout") from None
    except Exception as exc:
        _record_breaker_outcome("topic_timeline", exc)
        raise


@app.post("/memory/v1/entity-profile")
async def memory_entity_profile(body: EntityProfileBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("entity_profile")
    if gate:
        return gate
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    entity_name = str(body.entity or "").strip() or None
    entity_id = str(body.entity_id or "").strip() or None

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]

    missing: List[str] = []
    if not entity_name and not entity_id:
        missing.append("entity")
    if not user_tokens:
        missing.append("user_tokens")
    if missing:
        return _missing_core_requirements(missing, target="entity_profile")

    facts_limit = max(0, min(int(body.facts_limit or 0), 200))
    relations_limit = max(0, min(int(body.relations_limit or 0), 200))
    events_limit = max(0, min(int(body.events_limit or 0), 200))
    quotes_limit = max(0, min(int(body.quotes_limit or 0), 200))
    memory_domain = str(body.memory_domain or "").strip() or None

    trace: Dict[str, Any] = {}
    resolved = None
    if not entity_id and entity_name:
        entity_id, resolved_entity, candidates = await _resolve_entity_candidates(
            tenant_id=tenant_id,
            name=entity_name,
            entity_type=None,
            limit=5,
        )
        resolved = resolved_entity
        if candidates:
            trace["resolved_candidates"] = len(candidates)

    if not entity_id:
        return {
            "found": False,
            "entity": None,
            "facts": [],
            "relations": [],
            "recent_events": [],
        }

    entity_detail = await graph_svc.entity_detail(tenant_id=tenant_id, entity_id=entity_id)
    if not entity_detail and isinstance(resolved, dict):
        entity_detail = {
            "id": entity_id,
            "name": resolved.get("name"),
            "type": resolved.get("type"),
        }

    aliases: List[str] = []
    for key in ("name", "manual_name", "cluster_label"):
        val = entity_detail.get(key) if isinstance(entity_detail, dict) else None
        if val and str(val).strip():
            aliases.append(str(val).strip())
    aliases = list(dict.fromkeys(aliases))

    facts = []
    if facts_limit > 0:
        facts = await graph_svc.entity_facts(tenant_id=tenant_id, entity_id=entity_id, limit=facts_limit)

    relations = []
    if body.include_relations and relations_limit > 0:
        raw_relations = await graph_svc.entity_relations(
            tenant_id=tenant_id,
            entity_id=entity_id,
            limit=relations_limit,
        )
        relations = []
        for r in raw_relations or []:
            strength = int(r.get("weight") or r.get("strength") or 0)
            relations.append(
                {
                    "entity_id": r.get("entity_id") or r.get("id"),
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "relation_type": "co_occurs_with",
                    "relation_source": "inferred",
                    "relation_semantics": f"在 {strength} 个事件中共同出现",
                    "strength": strength,
                }
            )

    recent_events = []
    if body.include_events and events_limit > 0:
        recent_events = await graph_svc.list_events(
            tenant_id=tenant_id,
            entity_id=entity_id,
            user_ids=list(user_tokens),
            memory_domain=memory_domain,
            limit=events_limit,
        )

    recent_events_out = []
    for ev in recent_events or []:
        recent_events_out.append(
            {
                "event_id": ev.get("id"),
                "summary": ev.get("summary"),
                "t_abs_start": ev.get("t_abs_start"),
                "t_abs_end": ev.get("t_abs_end"),
            }
        )

    resp: Dict[str, Any] = {
        "found": True,
        "resolved_entity": resolved if resolved else {"id": entity_id, "name": entity_detail.get("name") if isinstance(entity_detail, dict) else None},
        "entity": {
            "id": entity_id,
            "name": entity_detail.get("name") if isinstance(entity_detail, dict) else None,
            "type": entity_detail.get("type") if isinstance(entity_detail, dict) else None,
            "aliases": aliases,
        },
        "facts": facts,
        "relations": relations,
        "recent_events": recent_events_out,
    }
    if body.include_quotes and quotes_limit > 0:
        quotes: List[Dict[str, Any]] = []
        events_for_quotes = list(recent_events or [])
        if not events_for_quotes:
            events_for_quotes = await graph_svc.list_events(
                tenant_id=tenant_id,
                entity_id=entity_id,
                user_ids=list(user_tokens),
                memory_domain=memory_domain,
                limit=max(events_limit, min(quotes_limit * 3, 200)),
            )
        for ev in events_for_quotes:
            if len(quotes) >= quotes_limit:
                break
            ev_id = str(ev.get("id") or "").strip()
            if not ev_id:
                continue
            bundle = await graph_svc.explain_event_evidence(tenant_id=tenant_id, event_id=ev_id)
            quotes.extend(_quotes_from_bundle(bundle, entity_id=entity_id))
        quotes.sort(
            key=lambda x: (
                _parse_iso_datetime(x.get("when")) or datetime.min.replace(tzinfo=timezone.utc),
                float(x.get("t_media_start") or 0.0),
            ),
            reverse=True,
        )
        resp["quotes"] = quotes[:quotes_limit]

    if body.include_states:
        states: List[Dict[str, Any]] = []
        vocab = _load_state_properties()
        props = [p for p in (vocab.get("properties") or []) if isinstance(p, dict)]
        mvp_props = [p for p in props if p.get("mvp")] or props
        for prop in mvp_props:
            name = str(prop.get("name") or "").strip()
            if not name:
                continue
            try:
                item = await graph_svc.get_current_state(
                    tenant_id=tenant_id,
                    subject_id=entity_id,
                    property=name,
                )
            except Exception:
                item = None
            if item:
                states.append(item)
        resp["states"] = states
    if body.debug:
        trace["facts_count"] = len(facts)
        trace["relations_count"] = len(relations)
        trace["events_count"] = len(recent_events_out)
        resp["trace"] = trace
    _record_breaker_outcome("entity_profile", None)
    return resp


@app.post("/memory/v1/quotes")
async def memory_quotes(body: QuotesBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("quotes")
    if gate:
        return gate
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    entity_name = str(body.entity or "").strip() or None
    entity_id = str(body.entity_id or "").strip() or None
    topic = str(body.topic or "").strip() or None
    topic_id = str(body.topic_id or "").strip() or None
    topic_path = str(body.topic_path or "").strip() or None

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]

    missing: List[str] = []
    if not (entity_name or entity_id or topic or topic_id or topic_path):
        missing.append("entity|topic")
    if not user_tokens:
        missing.append("user_tokens")
    if missing:
        return _missing_core_requirements(missing, target="quotes")

    limit = int(body.limit or 50)
    if limit <= 0:
        limit = 1
    if limit > SEARCH_TOPK_HARD_LIMIT:
        limit = SEARCH_TOPK_HARD_LIMIT

    memory_domain = str(body.memory_domain or "").strip() or None
    start_dt, end_dt = _parse_time_range_dict(body.time_range)
    trace: Dict[str, Any] = {}

    # Resolve entity if needed
    resolved_entity = None
    candidates: List[Dict[str, Any]] = []
    if not entity_id and entity_name:
        entity_id, resolved_entity, candidates = await _resolve_entity_candidates(
            tenant_id=tenant_id,
            name=entity_name,
            entity_type=None,
            limit=5,
        )
        if candidates:
            trace["resolved_candidates"] = len(candidates)

    # Normalize topic if needed
    explicit_topic_path = bool(topic_path)
    explicit_topic_id = bool(topic_id)
    if not topic_path and topic:
        try:
            normalized = normalize_topic_text(topic)
            topic_path = str(normalized.topic_path or "").strip() or None
            if not topic_id:
                topic_id = str(normalized.topic_id or topic).strip() or None
        except Exception:
            pass

    force_retrieval = False
    if not explicit_topic_path and not explicit_topic_id and topic_path and topic_path.startswith("_uncategorized/"):
        topic_path = None
        force_retrieval = True

    quotes: List[Dict[str, Any]] = []

    async def _collect_from_events(events: List[Dict[str, Any]]) -> None:
        for ev in events:
            if len(quotes) >= limit:
                break
            ev_id = str(ev.get("id") or "").strip()
            if not ev_id:
                continue
            bundle = await graph_svc.explain_event_evidence(tenant_id=tenant_id, event_id=ev_id)
            quotes.extend(_quotes_from_bundle(bundle, entity_id=entity_id))

    try:
        events: List[Dict[str, Any]] = []
        if topic_path or topic_id:
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=topic_id if not topic_path else None,
                topic_path=topic_path,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=max(1, min(limit * 2, 200)),
                event_ids=None,
            )
            trace["source"] = "graph_topic_filter"
        elif topic and not force_retrieval:
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=topic_id,
                topic_path=topic_path,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=max(1, min(limit * 2, 200)),
                event_ids=None,
            )
            trace["source"] = "graph_topic_filter"
        elif topic or force_retrieval:
            time_hints = {}
            if start_dt:
                time_hints["start_iso"] = start_dt.isoformat()
            if end_dt:
                time_hints["end_iso"] = end_dt.isoformat()
            retrieval_res = await asyncio.wait_for(
                retrieval(
                    svc,
                    tenant_id=str(tenant_id),
                    user_tokens=list(user_tokens),
                    query=str(topic or topic_id or "").strip(),
                    strategy="dialog_v2",
                    topk=limit,
                    memory_domain=str(memory_domain or "dialog"),
                    user_match="all",
                    run_id=None,
                    debug=bool(body.debug),
                    with_answer=False,
                    task="GENERAL",
                    llm_policy="best_effort",
                    backend="tkg",
                    tkg_explain=False,
                    time_hints=time_hints if time_hints else None,
                    request_id=_request_id_from_request(request),
                    trace_id=str(request.headers.get("X-Trace-ID") or "").strip() or None,
                    byok_route=None,
                ),
                timeout=HIGH_COST_TIMEOUT,
            )
            evidence_details = list((retrieval_res or {}).get("evidence_details") or [])
            event_ids = []
            for item in evidence_details:
                ev_id = str(item.get("tkg_event_id") or item.get("event_id") or "").strip()
                if ev_id:
                    event_ids.append(ev_id)
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=None,
                topic_path=None,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=max(1, min(limit * 2, 200)),
                event_ids=event_ids,
            )
            trace["source"] = "retrieval_dialog_v2"
            trace["retrieval_evidence"] = len(evidence_details)
        elif entity_id:
            events = await graph_svc.list_events(
                tenant_id=tenant_id,
                entity_id=entity_id,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=max(1, min(limit * 3, 200)),
            )
            trace["source"] = "graph_entity_events"

        # Entity-only path (no topic)
        if entity_id and not (topic or topic_id or topic_path):
            await _collect_from_events(events)
        else:
            await _collect_from_events(events)

        # Post-filter by time range if needed
        if start_dt or end_dt:
            filtered: List[Dict[str, Any]] = []
            for q in quotes:
                ts = _parse_iso_datetime(q.get("when"))
                if start_dt and ts and ts < start_dt:
                    continue
                if end_dt and ts and ts > end_dt:
                    continue
                filtered.append(q)
            quotes = filtered

        quotes.sort(
            key=lambda x: (
                _parse_iso_datetime(x.get("when")) or datetime.min.replace(tzinfo=timezone.utc),
                float(x.get("t_media_start") or 0.0),
            ),
            reverse=True,
        )
        quotes = quotes[:limit]

        resp: Dict[str, Any] = {
            "resolved_entity": resolved_entity if resolved_entity else ({"id": entity_id, "name": entity_name} if entity_id else None),
            "entity_id": entity_id,
            "topic_id": topic_id,
            "topic_path": topic_path,
            "quotes": quotes,
            "total": len(quotes),
        }
        if candidates:
            resp["candidates"] = candidates
        if body.debug:
            trace["count"] = len(quotes)
            resp["trace"] = trace
        _record_breaker_outcome("quotes", None)
        return resp
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("quotes", exc)
        raise HTTPException(status_code=504, detail="quotes_timeout") from None
    except Exception as exc:
        _record_breaker_outcome("quotes", exc)
        raise


@app.post("/memory/v1/relations")
async def memory_relations(body: RelationsBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("relations")
    if gate:
        return gate
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    entity_name = str(body.entity or "").strip() or None
    entity_id = str(body.entity_id or "").strip() or None

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]

    missing: List[str] = []
    if not entity_name and not entity_id:
        missing.append("entity")
    if not user_tokens:
        missing.append("user_tokens")
    if missing:
        return _missing_core_requirements(missing, target="relations")

    limit = int(body.limit or 50)
    if limit <= 0:
        limit = 1
    if limit > SEARCH_TOPK_HARD_LIMIT:
        limit = SEARCH_TOPK_HARD_LIMIT

    start_dt, end_dt = _parse_time_range_dict(body.time_range)
    trace: Dict[str, Any] = {}

    resolved_entity = None
    candidates: List[Dict[str, Any]] = []
    if not entity_id and entity_name:
        entity_id, resolved_entity, candidates = await _resolve_entity_candidates(
            tenant_id=tenant_id,
            name=entity_name,
            entity_type=None,
            limit=5,
        )
        if candidates:
            trace["resolved_candidates"] = len(candidates)

    if not entity_id:
        return {"found": False, "entity_id": None, "relations": []}

    relation_type = str(body.relation_type or "").strip().lower() or "co_occurs_with"
    if relation_type not in {"co_occurs_with", "co-occurs-with"}:
        return {"found": True, "entity_id": entity_id, "relations": [], "total": 0}

    try:
        rels = await graph_svc.entity_relations_by_events(
            tenant_id=tenant_id,
            entity_id=entity_id,
            start=start_dt,
            end=end_dt,
            limit=limit,
        )
        out = []
        for r in rels:
            strength = int(r.get("strength") or 0)
            out.append(
                {
                    "entity_id": r.get("entity_id"),
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "relation_type": "co_occurs_with",
                    "relation_source": "inferred",
                    "relation_semantics": f"在 {strength} 个事件中共同出现",
                    "strength": strength,
                    "first_mentioned": r.get("first_mentioned"),
                    "last_mentioned": r.get("last_mentioned"),
                    "evidence_event_ids": list(r.get("evidence_event_ids") or []),
                }
            )
        resp: Dict[str, Any] = {
            "found": True,
            "resolved_entity": resolved_entity if resolved_entity else ({"id": entity_id, "name": entity_name} if entity_id else None),
            "entity_id": entity_id,
            "relations": out,
            "total": len(out),
        }
        if candidates:
            resp["candidates"] = candidates
        if body.debug:
            trace["count"] = len(out)
            resp["trace"] = trace
        _record_breaker_outcome("relations", None)
        return resp
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("relations", exc)
        raise HTTPException(status_code=504, detail="relations_timeout") from None
    except Exception as exc:
        _record_breaker_outcome("relations", exc)
        raise


@app.post("/memory/v1/time-since")
async def memory_time_since(body: TimeSinceBody, request: Request):
    ctx = await _enforce_security(request)
    gate = _gate_high_cost("time_since")
    if gate:
        return gate
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    topic = str(body.topic or "").strip() or None
    topic_id = str(body.topic_id or "").strip() or None
    topic_path = str(body.topic_path or "").strip() or None
    entity_name = str(body.entity or "").strip() or None
    entity_id = str(body.entity_id or "").strip() or None

    user_tokens = [str(x).strip() for x in (body.user_tokens or []) if str(x).strip()]
    if not user_tokens and tenant_id:
        user_tokens = [f"u:{tenant_id}"]

    missing: List[str] = []
    if not (topic or topic_id or topic_path or entity_name or entity_id):
        missing.append("topic|entity")
    if not user_tokens:
        missing.append("user_tokens")
    if missing:
        return _missing_core_requirements(missing, target="time_since")

    memory_domain = str(body.memory_domain or "").strip() or None
    start_dt, end_dt = _parse_time_range_dict(body.time_range)
    limit = max(1, min(int(body.limit or 50), SEARCH_TOPK_HARD_LIMIT))

    trace: Dict[str, Any] = {}

    resolved_entity = None
    candidates: List[Dict[str, Any]] = []
    if not entity_id and entity_name:
        entity_id, resolved_entity, candidates = await _resolve_entity_candidates(
            tenant_id=tenant_id,
            name=entity_name,
            entity_type=None,
            limit=5,
        )
        if candidates:
            trace["resolved_candidates"] = len(candidates)

    explicit_topic_path = bool(topic_path)
    explicit_topic_id = bool(topic_id)
    resolved_topic: Optional[Dict[str, Any]] = None
    if not topic_path and topic:
        try:
            normalized = normalize_topic_text(topic)
            topic_path = str(normalized.topic_path or "").strip() or None
            if not topic_id:
                topic_id = str(normalized.topic_id or topic).strip() or None
            resolved_topic = {
                "topic_path": topic_path,
                "topic_id": topic_id,
                "confidence": None,
                "source": "normalizer",
            }
        except Exception:
            resolved_topic = None
    elif topic_path or topic_id:
        resolved_topic = {
            "topic_path": topic_path,
            "topic_id": topic_id,
            "confidence": 1.0,
            "source": "explicit",
        }

    force_retrieval = False
    if not explicit_topic_path and not explicit_topic_id and topic_path and topic_path.startswith("_uncategorized/"):
        topic_path = None
        force_retrieval = True

    try:
        events: List[Dict[str, Any]] = []
        source = None
        if entity_id and not (topic or topic_id or topic_path):
            events = await graph_svc.list_events(
                tenant_id=tenant_id,
                entity_id=entity_id,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=limit,
            )
            source = "graph_entity_events"
        elif topic_path or topic_id:
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=topic_id if not topic_path else None,
                topic_path=topic_path,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=limit,
                event_ids=None,
            )
            source = "graph_topic_filter"
        elif topic and not force_retrieval:
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=topic_id,
                topic_path=topic_path,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=limit,
                event_ids=None,
            )
            source = "graph_topic_filter"
        else:
            time_hints = {}
            if start_dt:
                time_hints["start_iso"] = start_dt.isoformat()
            if end_dt:
                time_hints["end_iso"] = end_dt.isoformat()
            retrieval_res = await asyncio.wait_for(
                retrieval(
                    svc,
                    tenant_id=str(tenant_id),
                    user_tokens=list(user_tokens),
                    query=str(topic or topic_id or "").strip(),
                    strategy="dialog_v2",
                    topk=limit,
                    memory_domain=str(memory_domain or "dialog"),
                    user_match="all",
                    run_id=None,
                    debug=bool(body.debug),
                    with_answer=False,
                    task="GENERAL",
                    llm_policy="best_effort",
                    backend="tkg",
                    tkg_explain=False,
                    time_hints=time_hints if time_hints else None,
                    request_id=_request_id_from_request(request),
                    trace_id=str(request.headers.get("X-Trace-ID") or "").strip() or None,
                    byok_route=None,
                ),
                timeout=HIGH_COST_TIMEOUT,
            )
            evidence_details = list((retrieval_res or {}).get("evidence_details") or [])
            event_ids = []
            for item in evidence_details:
                ev_id = str(item.get("tkg_event_id") or item.get("event_id") or "").strip()
                if ev_id:
                    event_ids.append(ev_id)
            events = await graph_svc.topic_timeline(
                tenant_id=tenant_id,
                topic_id=None,
                topic_path=None,
                tags=None,
                keywords=None,
                start=start_dt,
                end=end_dt,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=limit,
                event_ids=event_ids,
            )
            source = "retrieval_dialog_v2"
            trace["retrieval_evidence"] = len(evidence_details)

        # AND semantics: entity + topic together -> intersect event sets
        if entity_id and (topic_path or topic_id or topic or force_retrieval):
            entity_events = await graph_svc.list_events(
                tenant_id=tenant_id,
                entity_id=entity_id,
                user_ids=user_tokens,
                memory_domain=memory_domain,
                limit=max(limit, 200),
            )
            entity_ids = {str(e.get("id") or "") for e in (entity_events or []) if str(e.get("id") or "")}
            events = [ev for ev in events or [] if str(ev.get("id") or "") in entity_ids]

        last = None
        for ev in reversed(events or []):
            ts = _parse_iso_datetime(ev.get("t_abs_start") or ev.get("t_abs_end"))
            if ts is not None:
                last = ev
                break

        last_ts = _parse_iso_datetime(last.get("t_abs_start") or last.get("t_abs_end")) if isinstance(last, dict) else None
        now = datetime.now(timezone.utc)
        days_ago = None
        if last_ts is not None:
            delta = now - last_ts
            days_ago = max(0.0, delta.total_seconds() / 86400.0)

        resp: Dict[str, Any] = {
            "resolved_entity": resolved_entity if resolved_entity else ({"id": entity_id, "name": entity_name} if entity_id else None),
            "resolved_topic": resolved_topic,
            "topic_id": topic_id,
            "topic_path": topic_path,
            "entity_id": entity_id,
            "last_mentioned": last_ts.isoformat() if last_ts is not None else None,
            "days_ago": days_ago,
            "summary": (last.get("summary") if isinstance(last, dict) else None),
        }
        if candidates:
            resp["candidates"] = candidates
        if body.debug:
            trace["source"] = source
            trace["events_count"] = len(events or [])
            resp["trace"] = trace
        _record_breaker_outcome("time_since", None)
        return resp
    except asyncio.TimeoutError as exc:
        _record_breaker_outcome("time_since", exc)
        raise HTTPException(status_code=504, detail="time_since_timeout") from None
    except Exception as exc:
        _record_breaker_outcome("time_since", exc)
        raise


@app.get("/memory/agentic/tools")
async def memory_agentic_tools(
    request: Request,
    format: str = "openai",
    tool_whitelist: Optional[List[str]] = Query(None),
    include_disabled_default: bool = False,
):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    tool_names = _agentic_effective_tool_names(
        tool_whitelist,
        include_disabled_default=bool(include_disabled_default),
    )
    tools = _agentic_tools_payload(format_name=format, names=tool_names)
    return {
        "format": str(format or "openai").strip().lower(),
        "tool_names": tool_names,
        "tools": tools,
        "count": len(tools),
    }


@app.post("/memory/agentic/execute")
async def memory_agentic_execute(body: AgenticExecuteBody, request: Request):
    from modules.memory.adk import ToolResult

    started = time.perf_counter()
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    tool_name = str(body.tool_name or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="missing_tool_name")
    try:
        _agentic_effective_tool_names([tool_name], include_disabled_default=True)
    except HTTPException:
        result = ToolResult.no_match(message=f"unknown tool: {tool_name}")
        return {
            "tool_used": tool_name,
            "tool_args": {},
            "result": result.to_wire_dict(include_debug=bool(body.include_debug)),
            "meta": {
                "router_mode": "direct_execute_v1",
                "request_id": _agentic_request_id(body.request_id, request),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "tool_found": False,
            },
        }

    tool_args = _agentic_validate_tool_args(tool_name, body.args or {})
    runtime = _create_agentic_runtime(
        request=request,
        tenant_id=tenant_id,
        user_tokens=[f"u:{tenant_id}"],
    )
    try:
        result = await _execute_agentic_tool(runtime=runtime, tool_name=tool_name, args=tool_args)
    finally:
        with suppress(Exception):
            await runtime.aclose()

    return {
        "tool_used": tool_name,
        "tool_args": tool_args,
        "result": result.to_wire_dict(include_debug=bool(body.include_debug)),
        "meta": {
            "router_mode": "direct_execute_v1",
            "request_id": _agentic_request_id(body.request_id, request),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "tool_found": True,
        },
    }


@app.post("/memory/agentic/query")
async def memory_agentic_query(body: AgenticQueryBody, request: Request):
    from modules.memory.adk import ToolResult

    started = time.perf_counter()
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id") or "")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    query = str(body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="missing_query")

    request_id = _agentic_request_id(body.request_id, request)
    trace_id = str(request.headers.get("X-Trace-ID") or "").strip() or None
    tool_names = _agentic_effective_tool_names(body.tool_whitelist, include_disabled_default=False)
    tools = _agentic_tools_payload(format_name="openai", names=tool_names)

    route = await _agentic_route_tool_call(
        query=query,
        tools=tools,
        request_id=request_id,
        trace_id=trace_id,
        model=None,
    )

    selected_tool = str(route.get("tool_name") or "").strip() or None
    selected_args: Dict[str, Any] = {}
    result: ToolResult
    if not bool(route.get("has_tool_call")):
        result = ToolResult.no_match(message="未识别到可执行的记忆工具")
    elif not selected_tool or selected_tool not in tool_names:
        result = ToolResult.no_match(message="router_selected_unknown_tool")
    elif bool(route.get("tool_args_invalid")):
        result = ToolResult.no_match(message="router_returned_invalid_tool_args")
    else:
        raw_args = dict(route.get("tool_args") or {})
        try:
            selected_args = _agentic_validate_tool_args(selected_tool, raw_args)
        except HTTPException:
            result = ToolResult.no_match(message="router_returned_invalid_tool_args")
        else:
            runtime = _create_agentic_runtime(
                request=request,
                tenant_id=tenant_id,
                user_tokens=[f"u:{tenant_id}"],
            )
            try:
                result = await _execute_agentic_tool(runtime=runtime, tool_name=selected_tool, args=selected_args)
            finally:
                with suppress(Exception):
                    await runtime.aclose()

    meta: Dict[str, Any] = {
        "router_mode": "single_tool_v1",
        "request_id": request_id,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "provider": route.get("provider"),
        "model": route.get("model"),
        "finish_reason": route.get("finish_reason"),
        "has_tool_call": bool(route.get("has_tool_call")),
    }
    if body.include_debug:
        meta["tool_whitelist"] = list(tool_names)
        meta["selected_tool"] = selected_tool
        meta["tool_call_id"] = route.get("tool_call_id")
        meta["tool_args_raw"] = route.get("tool_args_raw")
        meta["tool_args_invalid"] = bool(route.get("tool_args_invalid"))

    return {
        "tool_used": selected_tool,
        "tool_args": selected_args,
        "result": result.to_wire_dict(include_debug=bool(body.include_debug)),
        "meta": meta,
    }


# ---- Memory state (Phase 3) ----
@app.post("/memory/state/current")
async def memory_state_current(body: StateCurrentBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.get_current_state(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
        )
        if not data:
            _record_graph_metrics("state.current", started, "not_found")
            raise HTTPException(status_code=404, detail="state_not_found")
        _record_graph_metrics("state.current", started)
        return {"item": data}
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("state.current", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("state.current", started, "server_error")
        raise


@app.post("/memory/state/at_time")
async def memory_state_at_time(body: StateAtTimeBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    t_val = _parse_iso_datetime(body.t_iso)
    if not t_val:
        _record_graph_metrics("state.at_time", started, "bad_request")
        raise HTTPException(status_code=400, detail="invalid_time")
    try:
        data = await graph_svc.get_state_at_time(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
            t=t_val,
        )
        if not data:
            _record_graph_metrics("state.at_time", started, "not_found")
            raise HTTPException(status_code=404, detail="state_not_found")
        _record_graph_metrics("state.at_time", started)
        return {"item": data}
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("state.at_time", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("state.at_time", started, "server_error")
        raise


@app.post("/memory/state/changes")
async def memory_state_changes(body: StateChangesBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    start_dt = _parse_iso_datetime(body.start_iso)
    end_dt = _parse_iso_datetime(body.end_iso)
    try:
        data = await graph_svc.get_state_changes(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
            start=start_dt,
            end=end_dt,
            limit=body.limit,
            order=body.order,
        )
        _record_graph_metrics("state.changes", started)
        return {"items": data}
    except Exception:
        _record_graph_metrics("state.changes", started, "server_error")
        raise


@app.post("/memory/state/what-changed")
async def memory_state_what_changed(body: StateChangesBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.get_state_changes(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
            start=_parse_iso_datetime(body.start_iso),
            end=_parse_iso_datetime(body.end_iso),
            limit=body.limit,
            order=body.order,
        )
        _record_graph_metrics("state.what_changed", started)
        return {"items": data}
    except Exception:
        _record_graph_metrics("state.what_changed", started, "server_error")
        raise


@app.post("/memory/state/time-since")
async def memory_state_time_since(body: StateTimeSinceBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.get_state_changes(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
            start=_parse_iso_datetime(body.start_iso),
            end=_parse_iso_datetime(body.end_iso),
            limit=1,
            order="desc",
        )
        if not data:
            _record_graph_metrics("state.time_since", started, "not_found")
            raise HTTPException(status_code=404, detail="state_not_found")
        last = data[0]
        ts = _parse_iso_datetime(last.get("valid_from")) if isinstance(last, dict) else None
        now = datetime.now(timezone.utc)
        seconds_ago = int((now - ts).total_seconds()) if ts else None
        _record_graph_metrics("state.time_since", started)
        return {
            "subject_id": body.subject_id,
            "property": body.property,
            "last_changed_at": last.get("valid_from"),
            "value": last.get("value"),
            "seconds_ago": seconds_ago,
        }
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("state.time_since", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("state.time_since", started, "server_error")
        raise


@app.post("/memory/state/pending/list")
async def memory_state_pending_list(body: StatePendingListBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.list_pending_states(
            tenant_id=tenant_id,
            subject_id=body.subject_id,
            property=body.property,
            status=body.status,
            limit=body.limit,
        )
        _record_graph_metrics("state.pending.list", started)
        return {"items": data}
    except Exception:
        _record_graph_metrics("state.pending.list", started, "server_error")
        raise


@app.post("/memory/state/pending/approve")
async def memory_state_pending_approve(body: StatePendingApproveBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.approve_pending_state(
            tenant_id=tenant_id,
            pending_id=body.pending_id,
            apply=body.apply,
            note=body.note,
        )
        if not data:
            _record_graph_metrics("state.pending.approve", started, "not_found")
            raise HTTPException(status_code=404, detail="pending_state_not_found")
        _record_graph_metrics("state.pending.approve", started)
        return data
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("state.pending.approve", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("state.pending.approve", started, "server_error")
        raise


@app.post("/memory/state/pending/reject")
async def memory_state_pending_reject(body: StatePendingRejectBody, request: Request):
    ctx = await _enforce_security(request)
    tenant_id = str(ctx.get("tenant_id"))
    started = time.perf_counter()
    try:
        data = await graph_svc.reject_pending_state(
            tenant_id=tenant_id,
            pending_id=body.pending_id,
            note=body.note,
        )
        if not data:
            _record_graph_metrics("state.pending.reject", started, "not_found")
            raise HTTPException(status_code=404, detail="pending_state_not_found")
        _record_graph_metrics("state.pending.reject", started)
        return data
    except HTTPException as exc:
        if exc.status_code != 404:
            _record_graph_metrics("state.pending.reject", started, _graph_status_from_code(exc.status_code))
        raise
    except Exception:
        _record_graph_metrics("state.pending.reject", started, "server_error")
        raise


# ---- Qdrant modality weights hot-config ----
class ModalityWeightsBody(BaseModel):
    weights: Optional[Dict[str, float]] = None


@app.get("/config/search/modality_weights")
async def get_modality_weights(request: Request):
    await _enforce_security(request)
    try:
        if hasattr(svc, "vectors") and hasattr(svc.vectors, "_mod_weights"):
            return {"weights": getattr(svc.vectors, "_mod_weights", {})}
    except Exception:
        pass
    return {"weights": {}}


@app.post("/config/search/modality_weights")
async def set_modality_weights(body: ModalityWeightsBody, request: Request):
    await _enforce_security(request, require_signature=True)
    try:
        if hasattr(svc, "vectors"):
            setter = getattr(svc.vectors, "set_modality_weights", None)
            if callable(setter):
                setter(body.weights)
                return {"ok": True, "weights": getattr(svc.vectors, "_mod_weights", {})}
    except Exception:
        pass
    return {"ok": False}


class ConfigSearchPatch(BaseModel):
    rerank: Optional[RerankWeightsBody] = None
    scoping: Optional[ScopingParamsBody] = None
    ann: Optional[AnnParamsBody] = None
    lexical_hybrid: Optional[LexicalHybridParamsBody] = None
    model_config = ConfigDict(extra="forbid")


class ConfigVectorStorePatch(BaseModel):
    modality_weights: Optional[Dict[str, float]] = None
    model_config = ConfigDict(extra="forbid")


class ConfigPatchBody(BaseModel):
    search: Optional[ConfigSearchPatch] = None
    graph: Optional[GraphParamsBody] = None
    vector_store: Optional[ConfigVectorStorePatch] = None
    model_config = ConfigDict(extra="forbid")


@app.patch("/config")
async def patch_config(body: ConfigPatchBody, request: Request):
    await _enforce_security(request, require_signature=True)
    applied: Dict[str, Any] = {}
    persist = False

    if body.search and body.search.rerank:
        data = body.search.rerank.model_dump(exclude_none=True)
        if data:
            try:
                rtconf.set_rerank_weights(data)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            applied["memory.search.rerank"] = data
            persist = True

    if body.graph:
        rtconf.set_graph_params(
            rel_whitelist=body.graph.rel_whitelist,
            max_hops=body.graph.max_hops,
            neighbor_cap_per_seed=body.graph.neighbor_cap_per_seed,
            restrict_to_user=body.graph.restrict_to_user,
            restrict_to_domain=body.graph.restrict_to_domain,
            allow_cross_user=body.graph.allow_cross_user,
            allow_cross_domain=body.graph.allow_cross_domain,
        )
        applied["memory.search.graph"] = body.graph.model_dump(exclude_none=True)
        persist = True

    if body.search and body.search.scoping:
        rtconf.set_scoping_params(
            default_scope=body.search.scoping.default_scope,
            user_match_mode=body.search.scoping.user_match_mode,
            fallback_order=body.search.scoping.fallback_order,
            require_user=body.search.scoping.require_user,
        )
        applied["memory.search.scoping"] = body.search.scoping.model_dump(exclude_none=True)
        persist = True

    if body.search and body.search.ann:
        rtconf.set_ann_params(
            default_modalities=body.search.ann.default_modalities,
            default_all_modalities=body.search.ann.default_all_modalities,
        )
        applied["memory.search.ann"] = body.search.ann.model_dump(exclude_none=True)
        persist = True

    if body.search and body.search.lexical_hybrid:
        rtconf.set_lexical_hybrid_params(
            enabled=body.search.lexical_hybrid.enabled,
            corpus_limit=body.search.lexical_hybrid.corpus_limit,
            lexical_topn=body.search.lexical_hybrid.lexical_topn,
            normalize_scores=body.search.lexical_hybrid.normalize_scores,
        )
        applied["memory.search.lexical_hybrid"] = body.search.lexical_hybrid.model_dump(exclude_none=True)
        persist = True

    if body.vector_store and body.vector_store.modality_weights is not None:
        if not _modality_weights_supported():
            raise HTTPException(status_code=400, detail="modality_weights_not_supported")
        try:
            if hasattr(svc, "vectors"):
                setter = getattr(svc.vectors, "set_modality_weights", None)
                if callable(setter):
                    setter(body.vector_store.modality_weights)
                    applied["memory.vector_store.search.modality_weights"] = body.vector_store.modality_weights
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if not applied:
        raise HTTPException(status_code=400, detail="no_hot_update_fields")

    if persist:
        rtconf.save_overrides()

    cfg = load_memory_config()
    hot_snapshot = _hot_update_snapshot(cfg)
    return {"ok": True, "applied": applied, "hot_update": hot_snapshot}


@app.post("/admin/ensure_collections")
async def admin_ensure_collections(request: Request):
    await _enforce_security(request, require_signature=True)
    try:
        await svc.vectors.ensure_collections()  # type: ignore[attr-defined]
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/admin/run_ttl")
async def admin_run_ttl(request: Request):
    await _enforce_security(request, require_signature=True)
    changed = await svc.run_ttl_cleanup_now()
    return {"ok": True, "changed": int(changed)}


class DecayBody(BaseModel):
    factor: float = 0.9
    rel_whitelist: Optional[List[str]] = None
    min_weight: float = 0.0


@app.post("/admin/decay_edges")
async def admin_decay_edges(body: DecayBody):
    ok = await svc.decay_graph_edges(factor=body.factor, rel_whitelist=body.rel_whitelist, min_weight=body.min_weight)
    return {"ok": bool(ok)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
