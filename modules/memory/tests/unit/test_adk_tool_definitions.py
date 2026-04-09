from __future__ import annotations

from modules.memory.adk.tool_definitions import TOOL_DEFINITIONS, get_tool_definitions, to_mcp_tools, to_openai_tools


def test_tool_definitions_cover_nine_tools() -> None:
    expected = {
        "entity_profile",
        "topic_timeline",
        "time_since",
        "quotes",
        "relations",
        "list_entities",
        "list_topics",
        "explain",
        "entity_status",
        "status_changes",
        "state_time_since",
    }
    assert set(TOOL_DEFINITIONS.keys()) == expected
    for name, item in TOOL_DEFINITIONS.items():
        assert item.name == name
        assert isinstance(item.description, str) and item.description
        assert item.input_schema.get("type") == "object"
        assert isinstance(item.trigger_keywords, tuple)


def test_get_tool_definitions_default_returns_enabled_subset() -> None:
    defs = get_tool_definitions(enabled_only=True)
    names = [x.name for x in defs]
    assert names == ["entity_profile", "topic_timeline", "time_since", "quotes", "relations"]


def test_openai_and_mcp_export_support_explicit_tool_names() -> None:
    openai_tools = to_openai_tools(names=["entity_profile", "explain"])
    mcp_tools = to_mcp_tools(names=["entity_profile", "explain"])

    assert [t["function"]["name"] for t in openai_tools] == ["entity_profile", "explain"]
    assert [t["name"] for t in mcp_tools] == ["entity_profile", "explain"]
    assert "parameters" in openai_tools[0]["function"]
    assert "inputSchema" in mcp_tools[0]


def test_quotes_limit_schema_aligns_with_runtime_cap() -> None:
    quotes_def = TOOL_DEFINITIONS["quotes"]
    limit_schema = ((quotes_def.input_schema.get("properties") or {}).get("limit") or {})
    assert limit_schema.get("maximum") == 10


def test_tool_export_unknown_name_raises() -> None:
    try:
        _ = to_openai_tools(names=["not_exists"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown tool names" in str(exc)
