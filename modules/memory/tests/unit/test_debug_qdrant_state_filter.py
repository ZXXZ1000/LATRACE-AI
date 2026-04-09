from __future__ import annotations

from modules.memory.scripts.debug_qdrant_state import build_qdrant_filter


def test_build_qdrant_filter_none():
    assert build_qdrant_filter(tenant_id=None, user_ids=None) is None


def test_build_qdrant_filter_tenant_and_user_ids():
    flt = build_qdrant_filter(tenant_id="t1", user_ids=["u:demo", "p:app"])
    assert isinstance(flt, dict)
    must = flt.get("must")
    assert isinstance(must, list)
    keys = [m.get("key") for m in must]
    assert keys.count("metadata.tenant_id") == 1
    assert keys.count("metadata.user_id") == 2

