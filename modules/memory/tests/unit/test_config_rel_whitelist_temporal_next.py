from __future__ import annotations

import yaml


def test_graph_rel_whitelist_contains_temporal_next(config_dir):
    cfg_path = config_dir / "memory.config.yaml"

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    rels = (((data.get("memory", {}) or {}).get("search", {}) or {}).get("graph", {}) or {}).get("rel_whitelist", [])
    rels_lower = {str(r).lower() for r in rels}
    assert "temporal_next" in rels_lower
