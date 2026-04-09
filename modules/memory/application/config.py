from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import os
from pathlib import Path
import yaml

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

try:
    from dotenv import load_dotenv  # type: ignore

    _ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"
    _MOD_ENV = Path(__file__).resolve().parents[1] / "config" / ".env"
    if _ROOT_ENV.exists():
        load_dotenv(_ROOT_ENV, override=False)
    if _MOD_ENV.exists():
        load_dotenv(_MOD_ENV, override=False)
except Exception:
    pass


DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "memory.config.yaml")
)
HYDRA_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "hydra"

# ---- Hard safety limits for search/graph expansion（v0.6 防护层）----
# 通过环境变量可在部署时调整上限；调用方不能绕过这些硬限制。
SEARCH_TOPK_HARD_LIMIT: int = int(os.getenv("MEMORY_SEARCH_TOPK_HARD_LIMIT", "200"))
GRAPH_MAX_HOPS_HARD_LIMIT: int = int(os.getenv("MEMORY_GRAPH_MAX_HOPS_HARD_LIMIT", "4"))
GRAPH_NEIGHBOR_CAP_HARD_LIMIT: int = int(os.getenv("MEMORY_GRAPH_NEIGHBOR_CAP_HARD_LIMIT", "64"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml_with_env(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    expanded = os.path.expandvars(raw)
    data = yaml.safe_load(expanded) or {}
    return data if isinstance(data, dict) else {}


def _apply_overrides(base: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    """Merge dotlist overrides (Hydra style) on top of base config.

    - Allows runtime overrides without changing YAML.
    - Keeps return type as plain dict for downstream compatibility.
    - Uses `resolve=False` to avoid blowing up on unresolved ${VAR} placeholders
      (consistent with历史行为：未设置的 env 留空字符串/占位符）。
    """

    if not overrides:
        return base
    try:
        base_conf = OmegaConf.create(base)
        override_conf = OmegaConf.from_dotlist(list(overrides))
        merged = OmegaConf.merge(base_conf, override_conf)
        return OmegaConf.to_container(merged, resolve=False)  # type: ignore[return-value]
    except Exception:
        # 回退：如果 Hydra 解析失败，保留原配置，避免影响调用方。
        return base


def _load_hydra_config(overrides: List[str] | None = None) -> Dict[str, Any]:
    """Load config via Hydra compose from `config/hydra` directory.

    - Uses defaults.yaml (-> memory) as entrypoint.
    - Clears global Hydra state to allow repeated calls in tests.
    - Returns plain dict; env values resolved via `${oc.env:VAR,default}`.
    """

    overrides = overrides or []
    if not HYDRA_CONFIG_DIR.exists():
        return {}
    try:
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()
        with initialize_config_dir(str(HYDRA_CONFIG_DIR), version_base=None):
            cfg = compose(config_name="defaults", overrides=list(overrides))
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        return {}


def load_memory_config(
    path: str | None = None,
    overrides: List[str] | None = None,
    use_hydra: bool | None = None,
) -> Dict[str, Any]:
    """Load memory config with optional Hydra-style overrides.

    - `USE_HYDRA_CONFIG=1` 或显式 `use_hydra=True` 时启用 Hydra (OmegaConf) 合并逻辑。
    - 默认行为保持向后兼容：直接读取 YAML + os.environ 展开。
    - 始终返回普通 dict，避免调用方感知配置实现方式。
    """

    profile = os.getenv("MEMORY_CONFIG_PROFILE")
    if path is None and profile:
        candidate = Path(DEFAULT_CONFIG_PATH).with_name(f"memory.config.{profile}.yaml")
        path = str(candidate) if candidate.exists() else None
    p = Path(path or DEFAULT_CONFIG_PATH)
    hydra_enabled = use_hydra if use_hydra is not None else _env_flag("USE_HYDRA_CONFIG")
    try:
        if hydra_enabled:
            hydra_cfg = _load_hydra_config(overrides)
            if hydra_cfg:
                return hydra_cfg
        base = _load_yaml_with_env(p)
        if hydra_enabled:
            return _apply_overrides(base, overrides)
        return base
    except Exception:
        return {}


def get_search_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    try:
        weights = cfg.get("memory", {}).get("search", {}).get("rerank", {})
        return {
            "alpha_vector": float(weights.get("alpha_vector", 0.6)),
            "beta_bm25": float(weights.get("beta_bm25", 0.2)),
            "gamma_graph": float(weights.get("gamma_graph", 0.15)),
            "delta_recency": float(weights.get("delta_recency", 0.05)),
            "user_boost": float(weights.get("user_boost", 0.20)),
            "domain_boost": float(weights.get("domain_boost", 0.10)),
            "session_boost": float(weights.get("session_boost", 0.10)),
        }
    except Exception:
        return {
            "alpha_vector": 0.6,
            "beta_bm25": 0.2,
            "gamma_graph": 0.15,
            "delta_recency": 0.05,
            "user_boost": 0.20,
            "domain_boost": 0.10,
            "session_boost": 0.10,
        }


def get_dialog_v2_reranker_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dialog_v2 reranker settings.

    YAML keys under memory.search.dialog_v2.reranker:
      - enabled: bool
      - engine: noop | native | llm
      - llm_kind: config key under memory.llm.<kind>
      - stage: preselect | final_only
      - score_mode: base_only | rerank_only | fused  (only applies when stage=preselect)
      - rerank_pool_size: int
      - weight_base: float
      - weight_rerank: float
      - instruct: optional reranker instruction
    """
    default = {
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
        "type_bias": {
            "fact": 0.15,
            "reference": 0.10,
            "event": 0.0,
            "unknown": 0.0,
        },
    }
    try:
        node = cfg.get("memory", {}).get("search", {}).get("dialog_v2", {}).get("reranker", {}) or {}
        out = dict(default)
        out["enabled"] = bool(node.get("enabled", default["enabled"]))
        out["engine"] = str(node.get("engine", default["engine"]) or default["engine"]).strip().lower() or "native"
        out["llm_kind"] = str(node.get("llm_kind", default["llm_kind"]) or default["llm_kind"]).strip() or "reranker"
        out["stage"] = str(node.get("stage", default["stage"]) or default["stage"]).strip().lower() or "preselect"
        raw_score_mode = str(node.get("score_mode", default["score_mode"]) or default["score_mode"]).strip().lower()
        out["score_mode"] = raw_score_mode if raw_score_mode in {"base_only", "rerank_only", "fused"} else str(default["score_mode"])
        try:
            out["rerank_pool_size"] = max(1, int(node.get("rerank_pool_size", default["rerank_pool_size"]) or default["rerank_pool_size"]))
        except Exception:
            out["rerank_pool_size"] = int(default["rerank_pool_size"])
        raw_instruct = str(node.get("instruct", default["instruct"]) or default["instruct"]).strip()
        out["instruct"] = raw_instruct or str(default["instruct"])
        for key in ("weight_base", "weight_rerank", "weight_type"):
            try:
                out[key] = float(node.get(key, default[key]))
            except Exception:
                out[key] = float(default[key])
        type_bias = node.get("type_bias")
        if isinstance(type_bias, dict):
            merged_bias = dict(default["type_bias"])
            for k, v in type_bias.items():
                try:
                    merged_bias[str(k)] = float(v)
                except Exception:
                    continue
            out["type_bias"] = merged_bias
        return out
    except Exception:
        return dict(default)


def get_dialog_v2_ranking_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dialog_v2 ranking defaults.

    YAML keys under memory.search.dialog_v2.ranking:
      - rrf_k: int
      - weights.event_vec / vec / knowledge / entity / time
      - weights.match / recency / signal
      - weights.score_blend_alpha
    """
    default = {
        "rrf_k": 60,
        "weights": {
            "event_vec": 0.6,
            "vec": 0.6,
            "knowledge": 0.9,
            "entity": 0.15,
            "time": 0.15,
            "match": 1.0,
            "recency": 0.0,
            "signal": 0.0,
            "score_blend_alpha": 0.7,
        },
    }
    try:
        node = cfg.get("memory", {}).get("search", {}).get("dialog_v2", {}).get("ranking", {}) or {}
        out = {
            "rrf_k": int(default["rrf_k"]),
            "weights": dict(default["weights"]),
        }
        try:
            out["rrf_k"] = max(1, int(node.get("rrf_k", default["rrf_k"]) or default["rrf_k"]))
        except Exception:
            out["rrf_k"] = int(default["rrf_k"])
        raw_weights = node.get("weights")
        if isinstance(raw_weights, dict):
            for key, value in raw_weights.items():
                try:
                    out["weights"][str(key)] = float(value)
                except Exception:
                    continue
        return out
    except Exception:
        return {
            "rrf_k": int(default["rrf_k"]),
            "weights": dict(default["weights"]),
        }


def resolve_dialog_v2_reranker_settings(
    cfg: Dict[str, Any],
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge runtime override on top of config-backed dialog_v2 reranker settings."""
    base = get_dialog_v2_reranker_settings(cfg)
    if not isinstance(override, dict):
        return base
    out = dict(base)
    for key in ("enabled", "engine", "llm_kind", "stage", "score_mode", "rerank_pool_size", "weight_base", "weight_rerank", "weight_type", "instruct"):
        if key not in override:
            continue
        value = override.get(key)
        if key == "enabled":
            out[key] = bool(value)
        elif key == "instruct":
            raw = str(value or "").strip()
            if raw:
                out[key] = raw
        elif key in {"rerank_pool_size"}:
            try:
                out[key] = max(1, int(value))
            except Exception:
                continue
        elif key == "score_mode":
            raw = str(value or "").strip().lower()
            if raw in {"base_only", "rerank_only", "fused"}:
                out[key] = raw
        elif key in {"weight_base", "weight_rerank", "weight_type"}:
            try:
                out[key] = float(value)
            except Exception:
                continue
        else:
            raw = str(value or "").strip()
            if raw:
                out[key] = raw.lower() if key in {"engine", "stage"} else raw
    if isinstance(override.get("type_bias"), dict):
        merged_bias = dict(out.get("type_bias") or {})
        for k, v in dict(override.get("type_bias") or {}).items():
            try:
                merged_bias[str(k)] = float(v)
            except Exception:
                continue
        out["type_bias"] = merged_bias
    return out


def get_graph_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract graph expansion defaults from config.

    Returns keys: expand(bool), max_hops(int), rel_whitelist(list[str]), neighbor_cap_per_seed(int)
    """
    try:
        g = cfg.get("memory", {}).get("search", {}).get("graph", {})
        expand = bool(g.get("expand", True))
        # clamp hops 和邻居 cap 到安全上限，防止 YAML 配置被误设过大
        raw_max_hops = int(g.get("max_hops", 1))
        max_hops = max(0, min(raw_max_hops, GRAPH_MAX_HOPS_HARD_LIMIT))
        rel_whitelist = list(g.get("rel_whitelist", []) or [])
        raw_cap = int(g.get("neighbor_cap_per_seed", 5))
        cap = max(0, min(raw_cap, GRAPH_NEIGHBOR_CAP_HARD_LIMIT))
        hop1 = float(g.get("hop1_boost", 1.0))
        hop2 = float(g.get("hop2_boost", 0.5))
        r_user = bool(g.get("restrict_to_user", True))
        r_dom = bool(g.get("restrict_to_domain", True))
        allow_cross_user = bool(g.get("allow_cross_user", False))
        allow_cross_domain = bool(g.get("allow_cross_domain", False))
        return {
            "expand": expand,
            "max_hops": max_hops,
            "rel_whitelist": rel_whitelist,
            "neighbor_cap_per_seed": cap,
            "hop1_boost": hop1,
            "hop2_boost": hop2,
            "restrict_to_user": r_user,
            "restrict_to_domain": r_dom,
            "allow_cross_user": allow_cross_user,
            "allow_cross_domain": allow_cross_domain,
        }
    except Exception:
        return {
            "expand": True,
            "max_hops": 1,
            "rel_whitelist": [],
            "neighbor_cap_per_seed": 5,
            "hop1_boost": 1.0,
            "hop2_boost": 0.5,
            "restrict_to_user": True,
            "restrict_to_domain": True,
            "allow_cross_user": False,
            "allow_cross_domain": False,
        }


def get_ann_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ANN default settings, including default modalities.

    YAML keys under memory.search.ann:
      - default_modalities: [text,image,audio] | omitted
      - default_all_modalities: bool (if true and default_modalities omitted, use all)
    """
    try:
        ann = cfg.get("memory", {}).get("search", {}).get("ann", {})
        dm = ann.get("default_modalities")
        if dm is not None:
            try:
                dm = list(dm)
            except Exception:
                dm = None
        all_flag = bool(ann.get("default_all_modalities", False))
        return {"default_modalities": dm, "default_all_modalities": all_flag}
    except Exception:
        return {"default_modalities": None, "default_all_modalities": False}


def get_lexical_hybrid_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract lexical-hybrid search settings.

    YAML keys under memory.search.lexical_hybrid:
      - enabled: bool
      - corpus_limit: int
      - lexical_topn: int
      - normalize_scores: bool

    Environment overrides are supported for local experiments:
      - MEMORY_SEARCH_LEXICAL_HYBRID_ENABLED
      - MEMORY_SEARCH_LEXICAL_CORPUS_LIMIT
      - MEMORY_SEARCH_LEXICAL_TOPN
      - MEMORY_SEARCH_LEXICAL_NORMALIZE
    """
    try:
        hybrid = cfg.get("memory", {}).get("search", {}).get("lexical_hybrid", {}) or {}
    except Exception:
        hybrid = {}

    enabled = bool(hybrid.get("enabled", False))
    corpus_limit = int(hybrid.get("corpus_limit", 500) or 500)
    lexical_topn = int(hybrid.get("lexical_topn", 50) or 50)
    normalize_scores = bool(hybrid.get("normalize_scores", True))

    raw_enabled = os.getenv("MEMORY_SEARCH_LEXICAL_HYBRID_ENABLED")
    raw_limit = os.getenv("MEMORY_SEARCH_LEXICAL_CORPUS_LIMIT")
    raw_topn = os.getenv("MEMORY_SEARCH_LEXICAL_TOPN")
    raw_norm = os.getenv("MEMORY_SEARCH_LEXICAL_NORMALIZE")

    if raw_enabled not in (None, "", "${MEMORY_SEARCH_LEXICAL_HYBRID_ENABLED}"):
        enabled = str(raw_enabled).strip().lower() in {"1", "true", "yes", "on"}
    if raw_limit not in (None, "", "${MEMORY_SEARCH_LEXICAL_CORPUS_LIMIT}"):
        try:
            corpus_limit = int(raw_limit)
        except Exception:
            pass
    if raw_topn not in (None, "", "${MEMORY_SEARCH_LEXICAL_TOPN}"):
        try:
            lexical_topn = int(raw_topn)
        except Exception:
            pass
    if raw_norm not in (None, "", "${MEMORY_SEARCH_LEXICAL_NORMALIZE}"):
        normalize_scores = str(raw_norm).strip().lower() in {"1", "true", "yes", "on"}

    return {
        "enabled": enabled,
        "corpus_limit": max(1, corpus_limit),
        "lexical_topn": max(1, lexical_topn),
        "normalize_scores": normalize_scores,
    }


def resolve_lexical_hybrid_settings(
    cfg: Dict[str, Any],
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve lexical-hybrid settings after applying runtime overrides.

    This keeps the merge and type-normalization logic in one place so the search
    path and config snapshot path cannot silently drift apart.
    """

    lexical = get_lexical_hybrid_settings(cfg)
    ov = dict(override or {})

    if ov.get("enabled") is not None:
        lexical["enabled"] = bool(ov.get("enabled"))
    if ov.get("corpus_limit") is not None:
        try:
            lexical["corpus_limit"] = max(1, int(ov.get("corpus_limit")))
        except Exception:
            pass
    if ov.get("lexical_topn") is not None:
        try:
            lexical["lexical_topn"] = max(1, int(ov.get("lexical_topn")))
        except Exception:
            pass
    if ov.get("normalize_scores") is not None:
        lexical["normalize_scores"] = bool(ov.get("normalize_scores"))

    return lexical


def get_dialog_event_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dialog event extraction/alignment settings."""
    defaults = {
        "alignment_top_k": 3,
        "alignment_min_score": 0.35,
        "alignment_embed_batch_size": 128,
        "alignment_embed_concurrency": 8,
        "min_event_confidence": 0.4,
        "max_events_per_session": 0,
        "supported_by_topk": 5,
        "ttl_mapped": 0,
        "ttl_weak": "7d",
        "ttl_unmapped": "discard",
        "segment_max_turns": 120,
        "event_extract_concurrency": 8,
        "fact_extract_concurrency": 4,
    }
    try:
        dlg = cfg.get("memory", {}).get("dialog", {}) or {}
        align = dlg.get("event_alignment", {}) or {}
        gate = dlg.get("event_gate", {}) or {}
        ttl = dlg.get("event_ttl", {}) or {}
        seg = dlg.get("event_segmentation", {}) or {}
        out = dict(defaults)
        if align.get("top_k") is not None:
            out["alignment_top_k"] = int(align.get("top_k"))
        if align.get("min_score") is not None:
            out["alignment_min_score"] = float(align.get("min_score"))
        if align.get("embed_batch_size") is not None:
            out["alignment_embed_batch_size"] = int(align.get("embed_batch_size"))
        if align.get("embed_concurrency") is not None:
            out["alignment_embed_concurrency"] = int(align.get("embed_concurrency"))
        if gate.get("min_event_confidence") is not None:
            out["min_event_confidence"] = float(gate.get("min_event_confidence"))
        if gate.get("max_events_per_session") is not None:
            out["max_events_per_session"] = int(gate.get("max_events_per_session"))
        if dlg.get("event_supported_by_topk") is not None:
            out["supported_by_topk"] = int(dlg.get("event_supported_by_topk"))
        if ttl.get("mapped") is not None:
            out["ttl_mapped"] = ttl.get("mapped")
        if ttl.get("weak") is not None:
            out["ttl_weak"] = ttl.get("weak")
        if ttl.get("unmapped") is not None:
            out["ttl_unmapped"] = ttl.get("unmapped")
        if seg.get("max_turns") is not None:
            out["segment_max_turns"] = int(seg.get("max_turns"))
        if dlg.get("event_extract_concurrency") is not None:
            out["event_extract_concurrency"] = int(dlg.get("event_extract_concurrency"))
        if dlg.get("fact_extract_concurrency") is not None:
            out["fact_extract_concurrency"] = int(dlg.get("fact_extract_concurrency"))
        return out
    except Exception:
        return dict(defaults)


def get_api_topk_defaults(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Default topk for high-level API responses.

    YAML keys under memory.api.topk_defaults:
      - search: int (default for POST /search when topk omitted)
      - retrieval: int (default for POST /retrieval/dialog/v2 when topk omitted)
      - graph_search: int (default for POST /graph/v1/search when topk omitted)

    Note: This is NOT the ANN seed/candidate topk (`memory.search.ann.default_topk`).
    """

    defaults = {"search": 10, "retrieval": 15, "graph_search": 10}
    try:
        raw = cfg.get("memory", {}).get("api", {}).get("topk_defaults", {})
        if isinstance(raw, dict):
            for k in list(defaults.keys()):
                if raw.get(k) is None:
                    continue
                try:
                    defaults[k] = int(raw.get(k))
                except Exception:
                    continue
    except Exception:
        pass
    for k, v in list(defaults.items()):
        if v <= 0:
            defaults[k] = 10 if k != "retrieval" else 15
        if defaults[k] > SEARCH_TOPK_HARD_LIMIT:
            defaults[k] = SEARCH_TOPK_HARD_LIMIT
    return defaults


def get_ingest_executor_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest executor settings.

    YAML keys under memory.ingest.executor:
      - enabled: bool (default true)
      - worker_count: int (default 2)
      - queue_maxsize: int (default 0 = unbounded)
      - global_concurrency: int (default 2)
      - per_tenant_concurrency: int (default 1)
      - job_timeout_s: int (default 300)
      - shutdown_grace_s: int (default 30)
      - recover_stale_s: int (default 3600)
      - retry_delay_s: int (default 60)
    """

    out: Dict[str, Any] = {
        "enabled": True,
        "worker_count": 10,
        "queue_maxsize": 0,
        "global_concurrency": 10,
        "per_tenant_concurrency": 3,
        "job_timeout_s": 900,
        "shutdown_grace_s": 30,
        "recover_stale_s": 3600,
        "retry_delay_s": 60,
    }
    try:
        node = cfg.get("memory", {}).get("ingest", {}).get("executor", {})
        if isinstance(node, dict):
            if node.get("enabled") is not None:
                out["enabled"] = bool(node.get("enabled"))
            for k in [
                "worker_count",
                "queue_maxsize",
                "global_concurrency",
                "per_tenant_concurrency",
                "job_timeout_s",
                "shutdown_grace_s",
                "recover_stale_s",
                "retry_delay_s",
            ]:
                if node.get(k) is None:
                    continue
                try:
                    out[k] = int(node.get(k))
                except Exception:
                    continue
    except Exception:
        return out
    if int(out.get("worker_count") or 0) <= 0:
        out["worker_count"] = 2
    if int(out.get("global_concurrency") or 0) <= 0:
        out["global_concurrency"] = 2
    if int(out.get("per_tenant_concurrency") or 0) <= 0:
        out["per_tenant_concurrency"] = 1
    if int(out.get("job_timeout_s") or 0) <= 0:
        out["job_timeout_s"] = 900
    if int(out.get("shutdown_grace_s") or 0) <= 0:
        out["shutdown_grace_s"] = 30
    if int(out.get("recover_stale_s") or 0) < 0:
        out["recover_stale_s"] = 0
    if int(out.get("retry_delay_s") or 0) <= 0:
        out["retry_delay_s"] = 60
    return out


def get_ingest_auto_execute_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for ingest executor settings."""
    return get_ingest_executor_settings(cfg)

def get_llm_multimodal_mapping(cfg: Dict[str, Any]) -> str:
    """Return the multimodal mapping strategy for LLM messages.

    Allowed values (suggested):
      - generic_image_url (default): map media → OpenAI-style content parts {type:text/image_url}
      - none: do not map; send messages as-is
    """
    try:
        m = cfg.get("memory", {}).get("llm", {}).get("multimodal", {})
        strategy = str(m.get("mapping_strategy", "generic_image_url")).strip().lower()
        if strategy in ("generic_image_url", "none"):
            return strategy
        return "generic_image_url"
    except Exception:
        return "generic_image_url"

def get_llm_selection(cfg: Dict[str, Any], kind: str = "text") -> Dict[str, str]:
    """Return LLM selection for given kind: 'text' | 'multimodal'.

    Expected YAML:
      memory.llm.text: { provider: openai, model: gpt-4o-mini }
      memory.llm.multimodal: { provider: openai, model: gpt-4o }

    Provider names follow LiteLLM conventions where possible. Env keys (API keys/base) should
    be provided via .env. For openrouter, model should be the upstream name and we will prefix
    with 'openrouter/' automatically.
    """
    try:
        node = cfg.get("memory", {}).get("llm", {}).get(kind, {})
        provider = str(node.get("provider", "")).strip().lower()
        model = str(node.get("model", "")).strip()
        return {"provider": provider, "model": model}
    except Exception:
        return {"provider": "", "model": ""}
