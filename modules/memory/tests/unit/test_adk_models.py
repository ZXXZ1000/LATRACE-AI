from __future__ import annotations

from modules.memory.adk.models import ToolDebugTrace, ToolResult


def test_tool_result_llm_surface_exposes_only_four_fields() -> None:
    res = ToolResult(
        matched=True,
        needs_disambiguation=False,
        message="ok",
        data={"entity": {"id": "e-1"}},
        debug=ToolDebugTrace(
            tool_name="entity_profile",
            source_mode="graph_filter",
            resolution_meta={"entity": {"input": "张三", "resolved_id": "e-1"}},
            raw_api_response_keys=["found", "entity"],
        ),
    )

    payload = res.to_llm_dict()
    assert set(payload.keys()) == {"matched", "needs_disambiguation", "message", "data"}
    assert payload["matched"] is True
    assert payload["data"]["entity"]["id"] == "e-1"
    assert "debug" not in payload
    assert "resolution_meta" not in payload


def test_tool_result_wire_dict_can_include_debug_separately() -> None:
    res = ToolResult.no_match(
        message="not found",
        debug=ToolDebugTrace(
            tool_name="entity_profile",
            error_type="not_found",
            retryable=False,
            fallback_used=False,
        ),
    )
    wire = res.to_wire_dict(include_debug=True)
    assert wire["matched"] is False
    assert wire["needs_disambiguation"] is False
    assert wire["message"] == "not found"
    assert wire["data"] is None
    assert wire["debug"]["tool_name"] == "entity_profile"
    assert wire["debug"]["error_type"] == "not_found"


def test_tool_result_disambiguation_constructor_sets_flags_and_candidates() -> None:
    res = ToolResult.disambiguation(
        candidates=[{"id": "e1", "name": "张三"}, {"id": "e2", "name": "张三（同事）"}],
        message="需要确认你说的是哪位张三",
    )
    llm = res.to_llm_dict()
    assert llm["matched"] is False
    assert llm["needs_disambiguation"] is True
    assert llm["message"]
    assert len(llm["data"]["candidates"]) == 2

