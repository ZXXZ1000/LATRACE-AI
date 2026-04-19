from __future__ import annotations


from modules.memory.scripts.create_neo4j_indexes import build_index_statements
from modules.memory.scripts.verify_qdrant_collections import plan_checks_from_config, _load_memory_config


def test_create_neo4j_indexes_statements():
    stmts = build_index_statements()
    s = "\n".join(stmts)
    assert "CREATE CONSTRAINT entity_id" in s
    assert "CREATE INDEX entity_domain" in s
    assert "CREATE INDEX entity_users" in s
    assert "CREATE INDEX entity_run" in s


def test_qdrant_verify_plan_from_config():
    cfg = _load_memory_config()
    plan = plan_checks_from_config(cfg)
    # Expect three modalities
    kinds = {p["modality"] for p in plan}
    assert kinds == {"text", "image", "audio"}
    # Verify default names and dims as in config
    by_name = {p["collection"]: p for p in plan}
    # text: default embedding model now uses local Jina v5-text-small (1024 dims)
    assert "memory_text" in by_name and by_name["memory_text"]["expected_dim"] == 1024
    # image/clip default 512
    assert "memory_image" in by_name and by_name["memory_image"]["expected_dim"] in (512,)
    # audio dim aligned to ERes2NetV2: 192
    assert "memory_audio" in by_name and by_name["memory_audio"]["expected_dim"] in (192,)
