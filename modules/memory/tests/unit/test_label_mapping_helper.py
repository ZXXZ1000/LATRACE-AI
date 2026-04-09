from __future__ import annotations

from modules.memory.infra.neo4j_store import Neo4jStore


def test_label_from_props_mapping():
    f = Neo4jStore._label_from_props
    assert f("episodic", None) == "Episodic"
    assert f("semantic", "image") == "Image"
    assert f("semantic", "audio") == "Voice"
    assert f("semantic", "structured") == "Structured"
    assert f("semantic", "text") == "Semantic"
    assert f("semantic", None) == "Semantic"
    assert f(None, None) == "Node"

