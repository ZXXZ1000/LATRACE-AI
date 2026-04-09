from __future__ import annotations

"""Dialog v1 (benchmark-aligned) fact extractor.

This module intentionally mirrors:
- benchmark/scripts/step2_extract_facts.py (SYSTEM_PROMPT + JSON schema contract)

We keep the prompt here (not importing benchmark code) so that production usage does not depend on `benchmark/`.
Tests should assert prompt equality to prevent drift.
"""

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json

from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_env


# IMPORTANT: keep in sync with `benchmark/scripts/step2_extract_facts.py::SYSTEM_PROMPT`.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_fact_extractor_system_prompt_v1.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def parse_facts_json(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of fact dicts (best-effort)."""
    try:
        data = json.loads(_remove_code_blocks(raw))
    except Exception:
        return []
    facts = data.get("facts")
    if isinstance(facts, list):
        out: List[Dict[str, Any]] = []
        for item in facts:
            if isinstance(item, dict):
                out.append(dict(item))
        return out
    return []


def build_dialogue_context(
    *,
    session_id: str,
    turns: List[Dict[str, Any]],
    reference_time_iso: Optional[str] = None,
) -> str:
    """Build a benchmark-compatible context block for the extractor prompt."""
    speakers = []
    for t in turns or []:
        s = str(t.get("speaker") or t.get("role") or "").strip()
        if s and s not in speakers:
            speakers.append(s)
    speaker_a = speakers[0] if len(speakers) >= 1 else "Speaker_A"
    speaker_b = speakers[1] if len(speakers) >= 2 else "Speaker_B"

    # Follow benchmark format: 'YYYY-MM-DD HH:MM' in session header.
    ref = ""
    if reference_time_iso:
        try:
            dt = datetime.fromisoformat(reference_time_iso)
            ref = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ref = str(reference_time_iso)
    if not ref:
        ref = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: List[str] = []
    lines.append(f"Sample ID: {session_id}")
    lines.append(f"Speaker A Name: {speaker_a}")
    lines.append(f"Speaker B Name: {speaker_b}")
    lines.append("Sessions included: [1]")
    lines.append("\n=== DIALOGUE ===")
    lines.append(f"\n[Session 1] (Reference Time: {ref})")
    for t in turns or []:
        dia_id = str(t.get("dia_id") or "").strip()
        speaker = str(t.get("speaker") or t.get("role") or "Unknown").strip() or "Unknown"
        text = str(t.get("text") or t.get("content") or "").strip()
        if dia_id:
            lines.append(f"{dia_id} {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_dialog_fact_extractor_v1_from_env(
    *,
    session_id: str,
    reference_time_iso: Optional[str] = None,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]:
    """Build a fact extractor using the project LLM adapter (returns None if LLM is not configured)."""
    if adapter is None:
        adapter = build_llm_from_env()
    if adapter is None:
        return None

    def _extract(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ctx = build_dialogue_context(session_id=str(session_id), turns=list(turns or []), reference_time_iso=reference_time_iso)
        raw = adapter.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            response_format={"type": "json_object"},
        )
        return parse_facts_json(raw)

    return _extract
