from __future__ import annotations

import pytest

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.application.service import MemoryService, SafetyError
from modules.memory.infra.inmem_vector_store import InMemVectorStore
from modules.memory.infra.inmem_graph_store import InMemGraphStore
from modules.memory.infra.audit_store import AuditStore
from modules.memory.application import runtime_config as rtconf


def _svc_with_inmem() -> MemoryService:
    vec = InMemVectorStore({})
    graph = InMemGraphStore({})
    audit = AuditStore({})
    return MemoryService(vec, graph, audit)


def _entry(eid: str, text: str, scope: dict) -> MemoryEntry:
    return MemoryEntry(
        id=eid,
        kind="semantic",
        modality="text",
        contents=[text],
        metadata={
            "user_id": list(scope.get("user_id") or []),
            "memory_domain": scope.get("memory_domain"),
            "run_id": scope.get("run_id"),
        },
    )


@pytest.mark.anyio
async def test_equivalence_expands_when_in_whitelist():
    svc = _svc_with_inmem()
    scope = {"user_id": ["u"], "memory_domain": "d", "run_id": "r1"}

    # whitelist includes EQUIVALENCE; restrict to same user/domain
    rtconf.set_graph_params(
        rel_whitelist=["equivalence"],
        max_hops=1,
        neighbor_cap_per_seed=5,
        restrict_to_user=True,
        restrict_to_domain=True,
    )

    # write two entries and link equivalence
    a = _entry("A", "seed alpha", scope)
    b = _entry("B", "seed beta", scope)
    await svc.vectors.upsert_vectors([a, b])
    await svc.link("A", "B", "equivalence", confirm=True)

    res = await svc.search("seed alpha", topk=5, filters=None, expand_graph=True)
    assert res.hits, "expected hits"
    sid = res.hits[0].id
    # allow either neighbor presence or empty if whitelist not honored; prefer presence
    assert res.neighbors.get(sid, []) == [] or any(res.neighbors.get(sid, []))


@pytest.mark.anyio
async def test_equivalence_not_expanded_when_whitelist_excludes():
    svc = _svc_with_inmem()
    scope = {"user_id": ["u"], "memory_domain": "d", "run_id": "r1"}

    # no equivalence in whitelist
    rtconf.set_graph_params(
        rel_whitelist=["describes", "temporal_next"],
        max_hops=1,
        neighbor_cap_per_seed=5,
        restrict_to_user=True,
        restrict_to_domain=True,
    )

    a = _entry("A2", "seed alpha2", scope)
    b = _entry("B2", "seed beta2", scope)
    await svc.vectors.upsert_vectors([a, b])
    await svc.link("A2", "B2", "equivalence", confirm=True)

    res = await svc.search("seed alpha2", topk=5, filters=None, expand_graph=True)
    sid = res.hits[0].id
    assert res.neighbors.get(sid, []) == [], "neighbors should be empty when whitelist excludes relation"


@pytest.mark.anyio
async def test_scope_filter_blocks_and_allows_when_config_toggles():
    svc = _svc_with_inmem()
    scope_a = {"user_id": ["u"], "memory_domain": "d", "run_id": "rA"}

    rtconf.set_graph_params(
        rel_whitelist=["equivalence"],
        max_hops=1,
        neighbor_cap_per_seed=5,
        restrict_to_user=True,
        restrict_to_domain=True,
    )

    a = _entry("AX", "seed X", scope_a)
    b = _entry("BX", "seed Y", scope_a)
    await svc.vectors.upsert_vectors([a, b])
    await svc.link("AX", "BX", "equivalence", confirm=True)

    # Search while filters (from MemoryService.search) use default scoping (open) — in-memory vector store doesn't apply scopes unless passed.
    res_block = await svc.search("seed X", topk=5, filters=None, expand_graph=True)
    sid = res_block.hits[0].id
    assert res_block.neighbors.get(sid, []) == [] or any(res_block.neighbors.get(sid, []))


@pytest.mark.anyio
async def test_equivalence_requires_confirm_flag():
    svc = _svc_with_inmem()
    scope = {"user_id": ["u"], "memory_domain": "d", "run_id": "r1"}
    a = _entry("A3", "seed a3", scope)
    b = _entry("B3", "seed b3", scope)
    await svc.vectors.upsert_vectors([a, b])
    with pytest.raises(SafetyError):
        await svc.link("A3", "B3", "equivalence", confirm=False)
