from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from modules.memory.contracts.graph_models import GraphUpsertRequest
from modules.memory.session_write import (
    _collect_graph_node_ids,
    _collect_graph_vector_ids,
    _set_graph_published,
)


def _resolve_source_id(
    graph_request: GraphUpsertRequest,
    explicit_source_id: Optional[str],
) -> Optional[str]:
    source_id = str(explicit_source_id or "").strip()
    if source_id:
        return source_id
    for seg in graph_request.segments or []:
        candidate = str(getattr(seg, "source_id", "") or "").strip()
        if candidate:
            return candidate
    for evd in graph_request.evidences or []:
        candidate = str(getattr(evd, "source_id", "") or "").strip()
        if candidate:
            return candidate
    return None


def _collect_graph_id_summary(graph_request: GraphUpsertRequest) -> Dict[str, Any]:
    return {
        "segment_ids": [str(seg.id) for seg in graph_request.segments or [] if str(seg.id).strip()],
        "evidence_ids": [str(evd.id) for evd in graph_request.evidences or [] if str(evd.id).strip()],
        "utterance_ids": [str(utt.id) for utt in graph_request.utterances or [] if str(utt.id).strip()],
        "entity_ids": [str(ent.id) for ent in graph_request.entities or [] if str(ent.id).strip()],
        "event_ids": [str(ev.id) for ev in graph_request.events or [] if str(ev.id).strip()],
        "timeslice_ids": [str(ts.id) for ts in graph_request.time_slices or [] if str(ts.id).strip()],
        "knowledge_ids": [str(kn.id) for kn in graph_request.knowledge or [] if str(kn.id).strip()],
        "pending_equiv_ids": [str(pe.id) for pe in graph_request.pending_equivs or [] if str(pe.id).strip()],
    }


async def media_session_write(
    store: Any,
    *,
    tenant_id: str,
    user_tokens: List[str],
    memory_domain: str,
    graph_request: GraphUpsertRequest,
    source_id: Optional[str] = None,
    overwrite_existing: bool = False,
) -> Dict[str, Any]:
    if not str(tenant_id or "").strip():
        raise ValueError("tenant_id is required")
    norm_users = [str(x).strip() for x in (user_tokens or []) if str(x).strip()]
    if not norm_users:
        raise ValueError("user_tokens must be non-empty")
    if not str(memory_domain or "").strip():
        raise ValueError("memory_domain is required")
    if graph_request is None:
        raise ValueError("graph_request is required")

    resolved_source_id = _resolve_source_id(graph_request, source_id)
    if overwrite_existing and not resolved_source_id:
        raise ValueError("source_id is required when overwrite_existing is true")

    timing_ms: Dict[str, int] = {}
    trace: Dict[str, Any] = {
        "graph_ids": _collect_graph_id_summary(graph_request),
        "source_id": resolved_source_id,
    }

    publish_fn = getattr(store, "publish_entries", None)
    publish_supported = callable(publish_fn)
    if publish_supported:
        _set_graph_published(graph_request, False)

    graph_upsert_fn = getattr(store, "graph_upsert_v0", None)
    if not callable(graph_upsert_fn):
        raise RuntimeError("graph_upsert_v0 not supported by this store")

    started_at = time.perf_counter()
    await graph_upsert_fn(graph_request)
    timing_ms["graph_upsert_ms"] = int((time.perf_counter() - started_at) * 1000)

    graph_node_ids = _collect_graph_node_ids(graph_request)
    graph_vector_ids = _collect_graph_vector_ids(graph_request)
    trace["graph_node_ids"] = list(graph_node_ids)
    trace["graph_vector_ids"] = list(graph_vector_ids)

    purge_result: Dict[str, Any] | None = None
    if overwrite_existing and resolved_source_id:
        purge_started_at = time.perf_counter()
        purge_fn = getattr(store, "graph_purge_source_except_graph_v0", None)
        if not callable(purge_fn):
            raise RuntimeError("graph_purge_source_except_graph_v0 not supported by this store")
        purge_result = await purge_fn(
            tenant_id=str(tenant_id),
            source_id=str(resolved_source_id),
            keep_node_ids=list(graph_node_ids),
        )
        timing_ms["overwrite_delete_ms"] = int((time.perf_counter() - purge_started_at) * 1000)
        trace["purge_result"] = dict(purge_result or {})

    publish_result = {"vectors": 0, "graph": 0}
    if publish_supported:
        publish_started_at = time.perf_counter()
        publish_result = await publish_fn(
            tenant_id=str(tenant_id),
            entry_ids=list(graph_vector_ids),
            graph_node_ids=list(graph_node_ids),
            published=True,
        )
        timing_ms["publish_ms"] = int((time.perf_counter() - publish_started_at) * 1000)
    trace["publish_result"] = dict(publish_result or {})
    trace["timing_ms"] = timing_ms

    return {
        "status": "ok",
        "source_id": resolved_source_id,
        "written_entries": len(graph_vector_ids),
        "graph_nodes_written": len(graph_node_ids),
        "trace": trace,
    }


__all__ = ["media_session_write"]
