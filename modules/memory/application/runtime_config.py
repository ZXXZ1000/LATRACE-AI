from __future__ import annotations

"""
运行时配置覆盖（热更新）

提供在不修改 YAML 的情况下，通过 API 热更新的搜索重排权重、图邻域参数等。
MemoryService.search 在每次调用时会优先读取这里的覆盖值。
"""

from typing import Any, Dict, List, Optional
from threading import RLock
import os
import json

from modules.memory.application.config import (
    GRAPH_MAX_HOPS_HARD_LIMIT,
    GRAPH_NEIGHBOR_CAP_HARD_LIMIT,
)


_lock = RLock()

# 覆盖的重排权重（alpha/beta/gamma/delta）
_rerank_weights_override: Dict[str, float] = {}

# 覆盖的图邻域参数
_graph_params_override: Dict[str, Any] = {}

# 覆盖的检索作用域参数
_scoping_override: Dict[str, Any] = {}

# 覆盖的 ANN 参数（默认模态等）
_ann_override: Dict[str, Any] = {}

# 覆盖的写入行为（如去重开关）
_write_override: Dict[str, Any] = {}

# 覆盖 lexical hybrid 搜索参数
_lexical_hybrid_override: Dict[str, Any] = {}


def set_rerank_weights(weights: Dict[str, float]) -> None:
    """设置重排权重，包含验证。

    Args:
        weights: 权重字典，包含alpha_vector/beta_bm25/gamma_graph/delta_recency等
        核心权重（alpha/beta/gamma/delta）的和应该为1.0

    Raises:
        ValueError: 如果核心权重和不等于1.0或权重为负数
    """
    with _lock:
        # 验证权重并设置
        valid_weights = {}
        for k, v in (weights or {}).items():
            if k in {"alpha_vector", "beta_bm25", "gamma_graph", "delta_recency", "user_boost", "domain_boost", "session_boost"}:
                try:
                    value = float(v)
                    if value < 0:
                        raise ValueError(f"权重 {k} 不能为负数: {value}")
                    valid_weights[k] = value
                except Exception:
                    pass

        # 验证核心权重和
        core_keys = {"alpha_vector", "beta_bm25", "gamma_graph", "delta_recency"}
        core_weights = {k: v for k, v in valid_weights.items() if k in core_keys}

        if core_weights:
            total = sum(core_weights.values())
            # 允许小误差（0.001），处理浮点数精度问题
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"核心权重（alpha_vector/beta_bm25/gamma_graph/delta_recency）的和必须为1.0，当前为: {total:.3f} "
                    f"（权重: {core_weights}）"
                )

        # 设置有效权重
        _rerank_weights_override.update(valid_weights)


def validate_rerank_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """验证权重但不设置。

    Args:
        weights: 要验证的权重字典

    Returns:
        验证后的权重字典

    Raises:
        ValueError: 如果权重无效
    """
    if not weights:
        return weights

    # 核心权重和验证
    core_keys = {"alpha_vector", "beta_bm25", "gamma_graph", "delta_recency"}
    core_weights = {k: float(v) for k, v in weights.items() if k in core_keys}

    if core_weights:
        total = sum(core_weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"核心权重和必须为1.0，当前为: {total:.3f}"
            )

    # 非负验证
    for k, v in weights.items():
        if float(v) < 0:
            raise ValueError(f"权重 {k} 不能为负数: {v}")

    return weights


def get_rerank_weights_override() -> Dict[str, float]:
    with _lock:
        return dict(_rerank_weights_override)


def clear_rerank_weights_override() -> None:
    with _lock:
        _rerank_weights_override.clear()


def set_graph_params(
    *,
    rel_whitelist: Optional[List[str]] = None,
    max_hops: Optional[int] = None,
    neighbor_cap_per_seed: Optional[int] = None,
    restrict_to_user: Optional[bool] = None,
    restrict_to_domain: Optional[bool] = None,
    allow_cross_user: Optional[bool] = None,
    allow_cross_domain: Optional[bool] = None,
) -> None:
    with _lock:
        if rel_whitelist is not None:
            _graph_params_override["rel_whitelist"] = list(rel_whitelist)
        if max_hops is not None:
            try:
                raw = int(max_hops)
                # clamp 到安全上限，防止恶意/误用配置
                _graph_params_override["max_hops"] = max(0, min(raw, GRAPH_MAX_HOPS_HARD_LIMIT))
            except Exception:
                pass
        if neighbor_cap_per_seed is not None:
            try:
                raw_cap = int(neighbor_cap_per_seed)
                _graph_params_override["neighbor_cap_per_seed"] = max(0, min(raw_cap, GRAPH_NEIGHBOR_CAP_HARD_LIMIT))
            except Exception:
                pass
        if restrict_to_user is not None:
            _graph_params_override["restrict_to_user"] = bool(restrict_to_user)
        if restrict_to_domain is not None:
            _graph_params_override["restrict_to_domain"] = bool(restrict_to_domain)
        if allow_cross_user is not None:
            _graph_params_override["allow_cross_user"] = bool(allow_cross_user)
        if allow_cross_domain is not None:
            _graph_params_override["allow_cross_domain"] = bool(allow_cross_domain)


def get_graph_params_override() -> Dict[str, Any]:
    with _lock:
        return dict(_graph_params_override)


def clear_graph_params_override() -> None:
    with _lock:
        _graph_params_override.clear()


def set_scoping_params(*, default_scope: Optional[str] = None, user_match_mode: Optional[str] = None, fallback_order: Optional[List[str]] = None, require_user: Optional[bool] = None) -> None:
    with _lock:
        if default_scope is not None:
            _scoping_override["default_scope"] = str(default_scope)
        if user_match_mode is not None:
            _scoping_override["user_match_mode"] = str(user_match_mode)
        if fallback_order is not None:
            _scoping_override["fallback_order"] = list(fallback_order)
        if require_user is not None:
            _scoping_override["require_user"] = bool(require_user)


def get_scoping_override() -> Dict[str, Any]:
    with _lock:
        return dict(_scoping_override)


def clear_scoping_override() -> None:
    with _lock:
        _scoping_override.clear()


# ---- ANN params (default modalities) ----
def set_ann_params(*, default_modalities: Optional[list] = None, default_all_modalities: Optional[bool] = None) -> None:
    with _lock:
        if default_modalities is not None:
            try:
                _ann_override["default_modalities"] = list(default_modalities)
            except Exception:
                pass
        if default_all_modalities is not None:
            _ann_override["default_all_modalities"] = bool(default_all_modalities)


def get_ann_override() -> Dict[str, Any]:
    with _lock:
        return dict(_ann_override)


def clear_ann_override() -> None:
    with _lock:
        _ann_override.clear()


def set_lexical_hybrid_params(
    *,
    enabled: Optional[bool] = None,
    corpus_limit: Optional[int] = None,
    lexical_topn: Optional[int] = None,
    normalize_scores: Optional[bool] = None,
) -> None:
    with _lock:
        if enabled is not None:
            _lexical_hybrid_override["enabled"] = bool(enabled)
        if corpus_limit is not None:
            try:
                _lexical_hybrid_override["corpus_limit"] = max(1, int(corpus_limit))
            except Exception:
                pass
        if lexical_topn is not None:
            try:
                _lexical_hybrid_override["lexical_topn"] = max(1, int(lexical_topn))
            except Exception:
                pass
        if normalize_scores is not None:
            _lexical_hybrid_override["normalize_scores"] = bool(normalize_scores)


def get_lexical_hybrid_override() -> Dict[str, Any]:
    with _lock:
        return dict(_lexical_hybrid_override)


def clear_lexical_hybrid_override() -> None:
    with _lock:
        _lexical_hybrid_override.clear()


# ---- Write params (e.g., dedup toggle) ----
def set_write_params(*, dedup_enabled: Optional[bool] = None) -> None:
    with _lock:
        if dedup_enabled is None:
            _write_override.pop("dedup_enabled", None)
        else:
            _write_override["dedup_enabled"] = bool(dedup_enabled)


def get_write_override() -> Dict[str, Any]:
    with _lock:
        return dict(_write_override)


def clear_write_override() -> None:
    with _lock:
        _write_override.clear()


# ---- Persistence (optional) ----
_DEFAULT_OVERRIDES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "runtime_overrides.json")
)


def _path() -> str:
    return os.getenv("MEMORY_RUNTIME_OVERRIDES", _DEFAULT_OVERRIDES_PATH)


def save_overrides() -> None:
    with _lock:
        data = {
            "rerank": dict(_rerank_weights_override),
            "graph": dict(_graph_params_override),
            "scoping": dict(_scoping_override),
            "ann": dict(_ann_override),
            "lexical_hybrid": dict(_lexical_hybrid_override),
            "write": dict(_write_override),
        }
    try:
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        with open(_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_overrides() -> None:
    # Always start from a clean slate to avoid leaking stale overrides
    # when the overrides file is missing or partially defined.
    clear_rerank_weights_override()
    clear_graph_params_override()
    clear_scoping_override()
    clear_ann_override()
    clear_lexical_hybrid_override()
    clear_write_override()
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if isinstance(data.get("rerank"), dict):
            set_rerank_weights(data.get("rerank") or {})
        g = data.get("graph") or {}
        set_graph_params(
            rel_whitelist=g.get("rel_whitelist"),
            max_hops=g.get("max_hops"),
            neighbor_cap_per_seed=g.get("neighbor_cap_per_seed"),
            restrict_to_user=g.get("restrict_to_user"),
            restrict_to_domain=g.get("restrict_to_domain"),
            allow_cross_user=g.get("allow_cross_user"),
            allow_cross_domain=g.get("allow_cross_domain"),
        )
        s = data.get("scoping") or {}
        set_scoping_params(
            default_scope=s.get("default_scope"),
            user_match_mode=s.get("user_match_mode"),
            fallback_order=s.get("fallback_order"),
            require_user=s.get("require_user"),
        )
        a = data.get("ann") or {}
        set_ann_params(default_modalities=a.get("default_modalities"), default_all_modalities=a.get("default_all_modalities"))
        lex = data.get("lexical_hybrid") or {}
        set_lexical_hybrid_params(
            enabled=lex.get("enabled"),
            corpus_limit=lex.get("corpus_limit"),
            lexical_topn=lex.get("lexical_topn"),
            normalize_scores=lex.get("normalize_scores"),
        )
        w = data.get("write") or {}
        set_write_params(dedup_enabled=w.get("dedup_enabled"))
    except Exception:
        # ignore if not present
        pass
