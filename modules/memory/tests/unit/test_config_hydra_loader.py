from pathlib import Path

from modules.memory.application.config import get_dialog_v2_ranking_settings, load_memory_config


def test_hydra_loader_env_and_overrides(monkeypatch):
    # 保证 env 展开仍生效
    monkeypatch.setenv("QDRANT_HOST", "qdrant.local")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    # 使用 Hydra 路径并覆盖一个值
    cfg = load_memory_config(
        use_hydra=True,
        overrides=["memory.search.ann.default_topk=99"],
    )
    assert cfg["memory"]["vector_store"]["host"] == "qdrant.local"
    assert str(cfg["memory"]["vector_store"]["port"]) == "6333"
    assert cfg["memory"]["search"]["ann"]["default_topk"] == 99


def test_hydra_flag_default(monkeypatch):
    monkeypatch.delenv("USE_HYDRA_CONFIG", raising=False)
    cfg_plain = load_memory_config(use_hydra=False)
    cfg_hydra = load_memory_config(use_hydra=True, overrides=None)
    # Hydra 路径不应破坏基础结构
    assert cfg_plain.get("memory")
    assert cfg_plain["memory"].keys() == cfg_hydra["memory"].keys()


def test_hydra_loader_custom_path(monkeypatch, tmp_path: Path):
    # 创建临时配置并验证覆盖逻辑
    content = """
    memory:
      search:
        ann:
          default_topk: 5
    """
    cfg_file = tmp_path / "memory.config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    cfg = load_memory_config(
        path=str(cfg_file), use_hydra=True, overrides=["memory.search.ann.default_topk=7"]
    )
    assert cfg["memory"]["search"]["ann"]["default_topk"] == 7


def test_api_topk_defaults_and_auth_coexist_in_loaded_config():
    cfg_plain = load_memory_config(use_hydra=False)
    api_plain = (cfg_plain.get("memory") or {}).get("api") or {}
    assert "topk_defaults" in api_plain
    assert "auth" in api_plain

    cfg_hydra = load_memory_config(use_hydra=True, overrides=None)
    api_hydra = (cfg_hydra.get("memory") or {}).get("api") or {}
    assert "topk_defaults" in api_hydra
    assert "auth" in api_hydra


def test_dialog_v2_ranking_defaults_are_loadable():
    cfg_plain = load_memory_config(use_hydra=False)
    ranking_plain = get_dialog_v2_ranking_settings(cfg_plain)
    assert ranking_plain["rrf_k"] == 60
    assert ranking_plain["weights"]["knowledge"] == 0.9
    assert ranking_plain["weights"]["score_blend_alpha"] == 0.7
    assert ranking_plain["weights"]["recency"] == 0.0

    cfg_hydra = load_memory_config(use_hydra=True, overrides=None)
    ranking_hydra = get_dialog_v2_ranking_settings(cfg_hydra)
    assert ranking_hydra == ranking_plain
