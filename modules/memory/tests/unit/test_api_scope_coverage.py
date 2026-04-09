from __future__ import annotations

import importlib
from typing import Dict, Set


def _collect_routes() -> Set[str]:
    srv = importlib.import_module("modules.memory.api.server")
    paths = set()
    for route in srv.app.routes:
        path = getattr(route, "path", None)
        if not path or not isinstance(path, str):
            continue
        paths.add(path)
    return paths


def _expand_scope_patterns(requirements: Dict[str, str]) -> Set[str]:
    expanded = set()
    for pattern in requirements.keys():
        if pattern.endswith("/"):
            expanded.add(pattern[:-1])
        expanded.add(pattern)
    return expanded


def test_scope_coverage_has_explicit_rule_for_non_public_routes() -> None:
    srv = importlib.import_module("modules.memory.api.server")
    routes = _collect_routes()
    public = set(srv.PUBLIC_PATHS)
    required = _expand_scope_patterns(srv.PATH_SCOPE_REQUIREMENTS)

    missing = set()
    for path in routes:
        if path in public:
            continue
        # Skip FastAPI docs routes if present.
        if path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/redoc"):
            continue
        # Exact match or prefix pattern in requirements.
        if path in required:
            continue
        matched_prefix = False
        for pattern in srv.PATH_SCOPE_REQUIREMENTS:
            if pattern.endswith("/") and path.startswith(pattern):
                matched_prefix = True
                break
        if not matched_prefix:
            missing.add(path)

    assert not missing, f"missing scope mapping for routes: {sorted(missing)}"
