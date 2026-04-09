from __future__ import annotations

from typing import Any, Dict, List, Tuple
import uuid
from modules.memory.contracts.memory_models import MemoryEntry, Edge


def build_entries_from_mem0(messages: List[Dict[str, Any]], *, profile: Dict[str, Any] | None = None) -> Tuple[List[MemoryEntry], List[Edge]]:
    """Map mem0 extracted facts (messages) into unified MemoryEntry/Edge.

    Expect messages as: [{"role": "user|assistant", "content": str, ...}]
    This is a placeholder. In production, call mem0 fact extractor + update decision first, then translate.
    """
    entries: List[MemoryEntry] = []
    edges: List[Edge] = []
    profile = profile or {}
    user_id = profile.get("user_id")
    user_entry: MemoryEntry | None = None
    if user_id:
        user_entry = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="structured",
            contents=[str(user_id)],
            metadata={"source": "mem0", "entity_type": "user"},
        )
        entries.append(user_entry)
    for msg in messages:
        text = str(msg.get("content", "")).strip()
        if not text:
            continue
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            kind="semantic",
            modality="text",
            contents=[text],
            metadata={"source": "mem0", "role": msg.get("role")},
        )
        entries.append(entry)
        # Optional: simple preference extraction → edges
        if user_entry is not None:
            if "喜欢" in text or "偏好" in text:
                edges.append(Edge(src_id=user_entry.id, dst_id=entry.id, rel_type="prefer", weight=1.0))
    return entries, edges
