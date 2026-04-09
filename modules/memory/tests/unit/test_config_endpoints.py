from __future__ import annotations

from fastapi.testclient import TestClient


def _setup_client(monkeypatch, tmp_path) -> TestClient:
    from modules.memory.api import server as srv
    from modules.memory.application import runtime_config as rtconf
    from modules.memory.application.service import MemoryService
    from modules.memory.infra.inmem_vector_store import InMemVectorStore
    from modules.memory.infra.inmem_graph_store import InMemGraphStore
    from modules.memory.infra.audit_store import AuditStore

    monkeypatch.setenv("MEMORY_RUNTIME_OVERRIDES", str(tmp_path / "runtime_overrides.json"))
    rtconf.clear_rerank_weights_override()
    rtconf.clear_graph_params_override()
    rtconf.clear_scoping_override()
    rtconf.clear_ann_override()
    rtconf.clear_lexical_hybrid_override()
    rtconf.clear_write_override()

    srv.svc = MemoryService(InMemVectorStore(), InMemGraphStore(), AuditStore())
    return TestClient(srv.app, headers={"X-Tenant-ID": "test-tenant"})


def test_config_get_and_patch(monkeypatch, tmp_path):
    client = _setup_client(monkeypatch, tmp_path)

    r = client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "core" in data
    assert "hot_update" in data
    assert "read_only_paths" in data

    hot_paths = {item["path"] for item in data["hot_update"]["paths"]}
    assert "memory.search.rerank" in hot_paths
    assert "memory.search.lexical_hybrid" in hot_paths
    mod_item = next(item for item in data["hot_update"]["paths"] if item["path"] == "memory.vector_store.search.modality_weights")
    assert mod_item["supported"] is False

    patch_body = {
        "search": {
            "rerank": {
                "alpha_vector": 0.4,
                "beta_bm25": 0.3,
                "gamma_graph": 0.2,
                "delta_recency": 0.1,
            },
            "scoping": {
                "default_scope": "domain",
                "user_match_mode": "any",
                "fallback_order": ["session", "domain", "user"],
                "require_user": False,
            },
            "ann": {
                "default_modalities": ["text"],
                "default_all_modalities": False,
            },
            "lexical_hybrid": {
                "enabled": True,
                "corpus_limit": 321,
                "lexical_topn": 17,
                "normalize_scores": True,
            },
        },
        "graph": {
            "max_hops": 2,
            "neighbor_cap_per_seed": 7,
            "restrict_to_user": True,
        },
    }
    r2 = client.patch("/config", json=patch_body)
    assert r2.status_code == 200
    applied = r2.json().get("applied") or {}
    assert "memory.search.rerank" in applied
    assert "memory.search.graph" in applied
    assert "memory.search.scoping" in applied
    assert "memory.search.ann" in applied
    assert "memory.search.lexical_hybrid" in applied

    r3 = client.get("/config")
    effective = (r3.json().get("hot_update") or {}).get("effective") or {}
    assert effective.get("search", {}).get("rerank", {}).get("alpha_vector") == 0.4
    assert effective.get("graph", {}).get("max_hops") == 2
    assert effective.get("search", {}).get("scoping", {}).get("default_scope") == "domain"
    assert effective.get("search", {}).get("lexical_hybrid", {}).get("enabled") is True
    assert effective.get("search", {}).get("lexical_hybrid", {}).get("corpus_limit") == 321
