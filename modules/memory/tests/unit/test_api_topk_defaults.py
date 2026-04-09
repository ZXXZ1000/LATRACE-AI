from __future__ import annotations


def test_get_api_topk_defaults_reads_api_config() -> None:
    from modules.memory.application.config import get_api_topk_defaults, SEARCH_TOPK_HARD_LIMIT

    cfg = {
        "memory": {
            "api": {
                "topk_defaults": {
                    "search": 7,
                    "retrieval": 21,
                    "graph_search": 9,
                }
            }
        }
    }
    out = get_api_topk_defaults(cfg)
    assert out["search"] == 7
    assert out["retrieval"] == 21
    assert out["graph_search"] == 9
    assert out["search"] <= SEARCH_TOPK_HARD_LIMIT


def test_get_api_topk_defaults_clamps_and_fallbacks() -> None:
    from modules.memory.application.config import get_api_topk_defaults, SEARCH_TOPK_HARD_LIMIT

    cfg = {"memory": {"api": {"topk_defaults": {"search": -1, "retrieval": 999999, "graph_search": 0}}}}
    out = get_api_topk_defaults(cfg)
    assert out["search"] > 0
    assert out["graph_search"] > 0
    assert out["retrieval"] == SEARCH_TOPK_HARD_LIMIT

