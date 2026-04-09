from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


LME_SYSTEM_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.

# OUTPUT RULES:
- Use ONLY the provided memories. Do not guess.
- If the memories do not contain enough information, answer exactly: insufficient information
- Provide ONLY the final answer. Do not include analysis, steps, citations, or an "Evidence:" section.
- Be specific. Avoid vague time references like "recently" or "a while ago".
""".strip()


LME_PREFERENCE_SYSTEM_PROMPT = """
You are an intelligent memory assistant tasked with summarizing a user's preferences from conversation memories.

# OUTPUT RULES:
- Output MUST be in this style:
  The user would prefer ... They might not prefer ...
- Use ONLY the provided memories. Do not guess.
- If the memories do not contain enough information, answer exactly: insufficient information
- Provide ONLY the preference summary text. Do not include analysis, steps, or citations.
""".strip()

LME_TEMPORAL_SYSTEM_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# TIME ANCHOR:
- Treat "Current Date" as the reference time for answering relative-time questions.

# OUTPUT RULES:
- Use ONLY the provided memories. Do not guess.
- If the memories do not contain enough information, answer exactly: insufficient information
- Provide ONLY the final answer. Do not include analysis, steps, citations, or an "Evidence:" section.
- For questions like "how many days ago/between", compute based on timestamps in memories and the provided Current Date.
""".strip()

LME_ASSISTANT_SYSTEM_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# OUTPUT RULES:
- Use ONLY the provided memories. Do not guess.
- If the memories do not contain enough information, answer exactly: insufficient information
- Provide ONLY the final answer (a name/number/short phrase). Do not include analysis, steps, citations, or extra formatting.
""".strip()


LME_EVIDENCE_LIMIT = 30


def build_longmemeval_context(*, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "No memories were retrieved."

    lines: List[str] = []
    for idx, e in enumerate(evidence[:LME_EVIDENCE_LIMIT], 1):
        eid = e.get("event_id", f"e{idx}")
        ts = e.get("timestamp")
        source = e.get("source", "unknown")
        text = str(e.get("text") or "")
        lines.append(f"[{idx}] source={source} id={eid} ts={ts}\n{text}")
    return "\n\n".join(lines)


def build_longmemeval_user_prompt_v1(
    *,
    question: str,
    question_date: str,
    evidence: List[Dict[str, Any]],
) -> str:
    ctx = build_longmemeval_context(evidence=evidence)
    return (
        f"{ctx}\n\n"
        f"Current Date: {str(question_date)}\n\n"
        f"Question: {str(question)}\n\n"
        "Answer:"
    )


def build_longmemeval_prompts(
    *,
    question: str,
    question_date: str,
    evidence: List[Dict[str, Any]],
    task: str,
) -> Tuple[str, str]:
    task_norm = str(task or "").strip().lower()
    if task_norm == "single-session-preference":
        system = LME_PREFERENCE_SYSTEM_PROMPT
    elif task_norm == "temporal-reasoning":
        system = LME_TEMPORAL_SYSTEM_PROMPT
    elif task_norm == "single-session-assistant":
        system = LME_ASSISTANT_SYSTEM_PROMPT
    else:
        system = LME_SYSTEM_PROMPT
    user = build_longmemeval_user_prompt_v1(question=question, question_date=question_date, evidence=evidence)
    return system, user


def should_use_longmemeval_prompt(*, memory_domain: str, task: str) -> bool:
    md = str(memory_domain or "").lower()
    if "longmemeval" in md:
        return True
    t = str(task or "").strip().lower()
    return t in {
        "single-session-preference",
        "single-session-assistant",
        "temporal-reasoning",
        "multi-session",
        "knowledge-update",
        "single-session-user",
    }


def extract_question_date_from_time_hints(time_hints: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(time_hints, dict):
        return None
    for k in ("question_date", "now", "now_iso", "current_date"):
        v = time_hints.get(k)
        if v:
            return str(v)
    return None


__all__ = [
    "LME_SYSTEM_PROMPT",
    "LME_PREFERENCE_SYSTEM_PROMPT",
    "LME_TEMPORAL_SYSTEM_PROMPT",
    "LME_ASSISTANT_SYSTEM_PROMPT",
    "LME_EVIDENCE_LIMIT",
    "build_longmemeval_context",
    "build_longmemeval_user_prompt_v1",
    "build_longmemeval_prompts",
    "should_use_longmemeval_prompt",
    "extract_question_date_from_time_hints",
]
