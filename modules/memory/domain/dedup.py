from __future__ import annotations

import hashlib

from modules.memory.contracts.memory_models import MemoryEntry


def text_fingerprint(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def should_merge(existing: MemoryEntry, incoming: MemoryEntry, threshold: float = 0.95) -> bool:
    """Naive merge decision by identical fingerprint. Replace with ANN similarity if needed."""
    if not existing.contents or not incoming.contents:
        return False
    return text_fingerprint(existing.contents[0]) == text_fingerprint(incoming.contents[0])


def merge_entries(base: MemoryEntry, new: MemoryEntry) -> MemoryEntry:
    """Simple merge: append unique contents; prefer newer metadata keys."""
    merged = base.model_copy(deep=True)
    for c in new.contents:
        if c not in merged.contents:
            merged.contents.append(c)
    merged.metadata.update(new.metadata)
    return merged

