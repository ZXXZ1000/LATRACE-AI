from __future__ import annotations

from typing import Dict, Any


def compute_importance(entry_metadata: Dict[str, Any]) -> float:
    """Heuristic importance score in [0,1].

    Placeholder: combine recency, modality weight, user-marked significance.
    To be replaced or augmented by an LLM-based rater if enabled.
    """
    score = 0.5
    if entry_metadata.get("modality") == "text":
        score += 0.1
    if entry_metadata.get("source") == "ctrl":
        score += 0.1
    return max(0.0, min(1.0, score))


def compute_stability(entry_metadata: Dict[str, Any]) -> float:
    """Heuristic stability score for long-term (0..1)."""
    base = 0.5
    if entry_metadata.get("kind") == "semantic":
        base += 0.2
    return max(0.0, min(1.0, base))


def default_ttl_seconds(importance: float) -> int:
    """Map importance to TTL in seconds (0 means keep long)."""
    if importance >= 0.8:
        return 0
    if importance >= 0.6:
        return 7 * 24 * 3600
    return 24 * 3600

