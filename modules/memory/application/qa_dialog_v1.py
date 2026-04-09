from __future__ import annotations

"""Dialog v1 QA generator (benchmark-aligned).

This module intentionally mirrors:
- benchmark/shared/adapters/moyan_memory_qa_adapter.py (QA_SYSTEM_PROMPT_GENERAL + user prompt formatting)

We keep the prompt here (not importing benchmark code) so that production usage does not depend on `benchmark/`.
Tests should assert prompt equality to prevent drift.
"""

from pathlib import Path
from typing import Any, Dict, List


_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_qa_system_prompt_general_v1.txt"
QA_SYSTEM_PROMPT_GENERAL = _PROMPT_PATH.read_text(encoding="utf-8")
QA_EVIDENCE_LIMIT = 30


def build_qa_user_prompt(*, query: str, task: str, evidence: List[Dict[str, Any]]) -> str:
    """Build benchmark-compatible QA user prompt from evidence (best-effort)."""
    q = str(query or "").strip()
    if not evidence:
        return (
            f"Question: {q}\n\n"
            "No evidence was retrieved. Answer that there is insufficient information."
        )

    lines: List[str] = []
    for idx, e in enumerate(evidence[:QA_EVIDENCE_LIMIT], 1):
        eid = e.get("event_id", f"e{idx}")
        ts = e.get("timestamp")
        text = str(e.get("text") or "")
        source = e.get("source", "unknown")
        if source == "fact_search":
            entry_type = "Fact"
        elif source == "reference_trace":
            entry_type = "Reference"
        else:
            entry_type = "Event"
        if text:
            lines.append(f"[{idx}] ({entry_type}) id={eid}, ts={ts}\n    {text}")
    ev_text = "\n".join(lines)
    return (
        f"Question: {q}\n"
        f"Task type: {task}\n\n"
        "Evidence from memory (Fact=summarized memory, Event=original dialogue):\n"
        f"{ev_text}\n\n"
        "Based on the evidence above, provide the best answer. Focus on facts and details mentioned."
    )
