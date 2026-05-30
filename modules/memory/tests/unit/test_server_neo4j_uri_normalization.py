from __future__ import annotations

from modules.memory.api.server import _normalize_neo4j_uri


def test_normalize_neo4j_uri_keeps_localhost_host() -> None:
    uri, meta = _normalize_neo4j_uri("localhost:7687")

    assert uri == "bolt://localhost:7687"
    assert meta["host"] == "localhost"
    assert meta["port"] == "7687"


def test_normalize_neo4j_uri_keeps_ipv6_loopback_host() -> None:
    uri, meta = _normalize_neo4j_uri("neo4j://[::1]:7687")

    assert uri == "bolt://::1:7687"
    assert meta["host"] == "::1"
    assert meta["port"] == "7687"


def test_normalize_neo4j_uri_normalizes_default_port() -> None:
    uri, meta = _normalize_neo4j_uri("bolt://127.0.0.1:7474")

    assert uri == "bolt://127.0.0.1:7687"
    assert meta["port"] == "7687"
