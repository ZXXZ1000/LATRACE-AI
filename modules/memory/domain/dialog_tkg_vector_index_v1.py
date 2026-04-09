from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from modules.memory.contracts.memory_models import MemoryEntry
from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid, make_event_id
from modules.memory.domain.dialog_tkg_graph_v1 import DialogGraphBuildResult


TKG_DIALOG_UTTERANCE_INDEX_SOURCE_V1 = "tkg_dialog_utterance_index_v1"
TKG_DIALOG_EVENT_INDEX_SOURCE_V1 = "tkg_dialog_event_index_v1"


@dataclass(frozen=True)
class DialogTkgVectorIndexBuildResult:
    entries: List[MemoryEntry]
    index_ids: List[str]


def build_dialog_tkg_utterance_index_entries_v1(
    *,
    tenant_id: str,
    session_id: str,
    user_tokens: Sequence[str],
    memory_domain: str,
    turns: Sequence[Dict[str, Any]],
    graph_build: DialogGraphBuildResult,
    source: str = TKG_DIALOG_UTTERANCE_INDEX_SOURCE_V1,
) -> DialogTkgVectorIndexBuildResult:
    """Build vector index entries for TKG utterance evidence (pure, no I/O).

    Design goals:
    - Stable IDs (retry-safe) and strict tenant/user/domain/run scoping.
    - Default excluded from generic /search unless caller explicitly filters by `source`.
    - Avoid dedup/merge: mark `dedup_skip=true` (server strips it before persistence).
    """
    if not str(tenant_id or "").strip():
        raise ValueError("tenant_id is required")
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    if not user_tokens:
        raise ValueError("user_tokens is required")

    utterance_ids = list((graph_build.graph_ids or {}).get("utterance_ids") or [])
    seg_id = str((graph_build.graph_ids or {}).get("segment_id") or "").strip() or None
    ts_id = str((graph_build.graph_ids or {}).get("timeslice_id") or "").strip() or None

    if len(utterance_ids) != len(list(turns)):
        raise ValueError("graph_build.utterance_ids length mismatch with turns")

    # Build mapping from utterance -> event via graph edges (SUPPORTED_BY)
    utt_to_events: Dict[str, List[str]] = {}
    try:
        edges = list(getattr(graph_build.request, "edges", []) or [])
        for e in edges:
            if str(getattr(e, "rel_type", "")).upper() != "SUPPORTED_BY":
                continue
            src_type = str(getattr(e, "src_type", "") or "")
            dst_type = str(getattr(e, "dst_type", "") or "")
            if src_type != "Event" or dst_type != "UtteranceEvidence":
                continue
            src_id = str(getattr(e, "src_id", "") or "").strip()
            dst_id = str(getattr(e, "dst_id", "") or "").strip()
            if not src_id or not dst_id:
                continue
            utt_to_events.setdefault(dst_id, []).append(src_id)
    except Exception:
        utt_to_events = {}

    entries: List[MemoryEntry] = []
    index_ids: List[str] = []
    for idx, t in enumerate(turns, start=1):
        raw_text = str(t.get("text") or t.get("content") or "").strip()
        if not raw_text:
            continue
        speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"
        ts_iso = str(t.get("timestamp_iso") or "").strip() or None
        dia_id = str(t.get("dia_id") or "").strip() or None

        utt_id = str(utterance_ids[idx - 1])
        ev_ids = list(dict.fromkeys(utt_to_events.get(utt_id, []) or []))
        ev_id = ev_ids[0] if len(ev_ids) == 1 else None
        entry_id = generate_uuid("memory.tkg.dialog.utterance_index", f"{tenant_id}|{session_id}|{idx}")
        index_ids.append(entry_id)

        logical_event_id: Optional[str] = None
        if dia_id:
            logical_event_id = make_event_id(str(session_id), dia_id)

        md: Dict[str, Any] = {
            "tenant_id": str(tenant_id),
            "user_id": [str(x) for x in user_tokens],
            "memory_domain": str(memory_domain),
            "run_id": str(session_id),
            "source": str(source),
            "node_type": "tkg_utterance_index",
            "dedup_skip": True,
            "tkg_utterance_id": utt_id,
        }
        if ev_ids:
            md["tkg_event_ids"] = list(ev_ids)
        if ev_id:
            md["tkg_event_id"] = ev_id
        if logical_event_id:
            # Keep benchmark-compatible logical id for evaluation and clients.
            md["event_id"] = logical_event_id
        if dia_id:
            md["dia_id"] = dia_id
        if seg_id:
            md["tkg_segment_id"] = seg_id
        if ts_id:
            md["tkg_timeslice_id"] = ts_id
        md["turn_index"] = int(idx)
        md["speaker"] = speaker
        if ts_iso:
            md["timestamp_iso"] = ts_iso

        # Keep speaker in content to make person-referenced queries easier in embedding space.
        content = f"{speaker}: {raw_text}"
        entries.append(
            MemoryEntry(
                id=entry_id,
                kind="semantic",
                modality="text",
                contents=[content],
                metadata=md,
            )
        )

    return DialogTkgVectorIndexBuildResult(entries=entries, index_ids=index_ids)


__all__ = [
    "DialogTkgVectorIndexBuildResult",
    "TKG_DIALOG_UTTERANCE_INDEX_SOURCE_V1",
    "TKG_DIALOG_EVENT_INDEX_SOURCE_V1",
    "build_dialog_tkg_utterance_index_entries_v1",
]
