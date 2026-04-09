from __future__ import annotations

import json

from modules.memory.contracts.memory_models import MemoryEntry, Edge, SearchFilters


def test_memory_entry_model_defaults():
    e = MemoryEntry(
        kind="semantic",
        modality="text",
        contents=["用户喜欢在晚上看电影"],
        metadata={"source": "mem0"},
    )
    assert e.id is None
    assert e.kind == "semantic"
    assert e.modality == "text"
    assert isinstance(e.contents, list) and len(e.contents) == 1
    assert isinstance(e.metadata, dict) and e.metadata.get("source") == "mem0"


def test_edge_model():
    ed = Edge(src_id="a", dst_id="b", rel_type="prefer", weight=0.8)
    assert ed.rel_type == "prefer"
    assert ed.weight == 0.8


def test_search_filters_model():
    f = SearchFilters(modality=["text"], memory_type=["semantic"], source=["m3", "mem0"])
    assert f.modality == ["text"]
    assert "semantic" in f.memory_type
    assert "m3" in f.source


def test_toolspec_contains_expected_tools(api_dir):
    toolspec_path = api_dir / "memory_toolspec.json"

    data = json.loads(toolspec_path.read_text(encoding="utf-8"))
    names = {t["name"] for t in data.get("tools", [])}
    assert "memory.search" in names
    assert "memory.write" in names
    assert "memory.update" in names
    assert "memory.delete" in names
    assert "memory.link" in names

    # check required field for search
    search = next(t for t in data["tools"] if t["name"] == "memory.search")
    required = set(search["parameters"].get("required", []))
    assert "query" in required

