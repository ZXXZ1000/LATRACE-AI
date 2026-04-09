from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from modules.memory.application.llm_adapter import LLMAdapter, build_llm_from_config, build_llm_from_env
from modules.memory.domain.dialog_text_pipeline_v1 import generate_uuid

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "dialog_turn_mark_system_prompt_v1.txt"
SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_SAVE_KEYWORDS = [
    "remember",
    "save this",
    "don't forget",
    "important",
    "keep this",
    "记住",
    "保存",
    "别忘",
    "很重要",
    "以后还要用",
]

_ALLOWED_CATEGORY = {"fact", "preference", "task", "rule", "note"}
_ALLOWED_SUBTYPE = {"profile", "constraint", "commitment", "decision", "tool_grounded_fact", "user_pinned_note"}
_ALLOWED_EVIDENCE = {"S0_user_claim", "S1_ai_inference", "S2_tool_grounded", "S3_user_confirmed"}
_ALLOWED_FORGET = {"permanent", "until_changed", "temporary"}


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def parse_turn_mark_json(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(_remove_code_blocks(raw))
    except Exception:
        return []
    items = data.get("marks") or data.get("turn_marks") or data.get("items")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(dict(it))
    return out


def build_turn_mark_extractor_v1_from_env(
    *,
    adapter: Optional[LLMAdapter] = None,
) -> Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]]:
    if adapter is None:
        adapter = build_llm_from_config("extract")
    if adapter is None:
        adapter = build_llm_from_env()
    if adapter is None:
        return None

    def _extract(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ctx = build_turn_mark_context(turns)
        raw = adapter.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ctx},
            ],
            response_format={"type": "json_object"},
        )
        return parse_turn_mark_json(raw)

    return _extract


def build_turn_mark_context(turns: Sequence[Dict[str, Any]]) -> str:
    lines = ["TURN MARK INPUT", ""]
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "").strip()
        role = str(t.get("role") or "unknown").strip()
        text = str(t.get("text") or "").strip()
        if not tid or not text:
            continue
        lines.append(f"[{tid}] ({role}) {text}")
    return "\n".join(lines)


def validate_and_normalize_marks(
    *,
    turns: Sequence[Dict[str, Any]],
    marks: Sequence[Dict[str, Any]],
    strict: bool = True,
) -> List[Dict[str, Any]]:
    turn_map = {str(t.get("turn_id") or "").strip(): t for t in (turns or []) if str(t.get("turn_id") or "").strip()}
    errors: List[str] = []
    out: List[Dict[str, Any]] = []

    def _record_error(msg: str) -> bool:
        if strict:
            errors.append(msg)
            return True
        return False
    for m in marks or []:
        if not isinstance(m, dict):
            if _record_error("mark_not_dict"):
                continue
            continue
        tid = str(m.get("turn_id") or "").strip()
        if not tid or tid not in turn_map:
            if _record_error(f"unknown_turn_id:{tid}"):
                continue
            continue
        keep = m.get("keep")
        if not isinstance(keep, bool):
            if _record_error(f"keep_invalid:{tid}"):
                continue
            keep = False
        span = m.get("span")
        if span is not None:
            try:
                start = int(span.get("start"))
                end = int(span.get("end"))
            except Exception:
                if _record_error(f"span_invalid:{tid}"):
                    continue
                span = None
            else:
                text = str(turn_map[tid].get("text") or "")
                if start < 0 or end <= start or end > len(text):
                    if _record_error(f"span_range:{tid}"):
                        continue
                    span = None
                else:
                    span = {"start": start, "end": end}

        user_triggered_save = m.get("user_triggered_save")
        if user_triggered_save is None:
            user_triggered_save = False
        if not isinstance(user_triggered_save, bool):
            if _record_error(f"user_triggered_save_invalid:{tid}"):
                continue
            user_triggered_save = False

        role = str(turn_map[tid].get("role") or "").strip().lower()
        category = str(m.get("category") or "").strip().lower() or "note"
        if keep and category not in _ALLOWED_CATEGORY:
            if _record_error(f"category_invalid:{tid}"):
                continue
            category = "note"

        subtype = str(m.get("subtype") or "").strip()
        if not subtype:
            subtype = _default_subtype(category)
        if keep and subtype not in _ALLOWED_SUBTYPE:
            if _record_error(f"subtype_invalid:{tid}"):
                continue
            subtype = _default_subtype(category)

        evidence_level = str(m.get("evidence_level") or "").strip() or _default_evidence_level(role)
        if keep and evidence_level not in _ALLOWED_EVIDENCE:
            if _record_error(f"evidence_invalid:{tid}"):
                continue
            evidence_level = _default_evidence_level(role)

        requires_confirmation = bool(m.get("requires_confirmation")) if m.get("requires_confirmation") is not None else False

        importance = m.get("importance")
        if importance is None:
            importance_val = 0.5
        else:
            try:
                importance_val = float(importance)
            except Exception:
                if _record_error(f"importance_invalid:{tid}"):
                    continue
                importance_val = 0.5
        if importance_val < 0.0 or importance_val > 1.0:
            if _record_error(f"importance_range:{tid}"):
                continue
            importance_val = max(0.0, min(1.0, importance_val))

        ttl_seconds, forget_policy = _default_ttl_policy(category, evidence_level, m.get("status") or m.get("task_status"))
        if user_triggered_save:
            importance_val = max(importance_val, 0.9)
            ttl_seconds = 0
            forget_policy = "permanent"
            keep = True

        out.append(
            {
                "turn_id": tid,
                "keep": keep,
                "span": span,
                "user_triggered_save": user_triggered_save,
                "category": category,
                "subtype": subtype,
                "evidence_level": evidence_level,
                "requires_confirmation": requires_confirmation,
                "importance": importance_val,
                "ttl_seconds": ttl_seconds,
                "forget_policy": forget_policy,
                "reason": str(m.get("reason") or ""),
            }
        )

    if errors:
        raise ValueError(";".join(errors))
    return out


def _default_subtype(category: str) -> str:
    if category == "rule":
        return "constraint"
    if category == "task":
        return "commitment"
    if category == "preference":
        return "profile"
    if category == "fact":
        return "profile"
    return "user_pinned_note"


def _default_evidence_level(role: str) -> str:
    if role == "user":
        return "S0_user_claim"
    if role == "tool":
        return "S2_tool_grounded"
    return "S1_ai_inference"


def _default_ttl_policy(category: str, evidence_level: str, status: Optional[Any]) -> Tuple[int, str]:
    if category == "preference":
        return 0, "until_changed"
    if category == "rule":
        return 0, "permanent"
    if category == "task":
        if str(status or "").lower() in {"done", "completed", "closed"}:
            return 7 * 24 * 3600, "temporary"
        return 30 * 24 * 3600, "temporary"
    if category == "fact":
        if evidence_level in {"S2_tool_grounded", "S3_user_confirmed"}:
            return 0, "permanent"
        return 180 * 24 * 3600, "temporary"
    return 30 * 24 * 3600, "temporary"


def apply_turn_marks(turns: Sequence[Dict[str, Any]], marks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mark_map = {str(m.get("turn_id")): m for m in (marks or [])}
    out: List[Dict[str, Any]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "").strip()
        if not tid:
            continue
        mark = mark_map.get(tid)
        if not mark or not mark.get("keep"):
            continue
        new_t = dict(t)
        span = mark.get("span")
        if isinstance(span, dict):
            text = str(new_t.get("text") or "")
            start = int(span.get("start") or 0)
            end = int(span.get("end") or 0)
            new_t["text"] = text[start:end]
        out.append(new_t)
    return out


def generate_pin_intents(turns: Sequence[Dict[str, Any]], marks: Sequence[Dict[str, Any]], *, window: int = 4) -> List[Dict[str, Any]]:
    target_turn_ids = [str(m.get("turn_id")) for m in (marks or []) if m.get("user_triggered_save")]
    if not target_turn_ids:
        return []
    trigger_turn_id = _find_trigger_turn_id(turns) or target_turn_ids[-1]
    pin_key = f"{trigger_turn_id}|{'|'.join(target_turn_ids)}"
    pin_id = generate_uuid("memory.pin_intent", pin_key)
    return [
        {
            "pin_id": pin_id,
            "trigger_turn_id": trigger_turn_id,
            "target_turn_ids": list(target_turn_ids),
            "reason": "user_explicit_save",
            "importance_boost": 0.9,
            "ttl_seconds": 0,
            "requires_confirmation": False,
        }
    ]


def _find_trigger_turn_id(turns: Sequence[Dict[str, Any]]) -> Optional[str]:
    for t in reversed(list(turns or [])):
        if not isinstance(t, dict):
            continue
        role = str(t.get("role") or "").strip().lower()
        if role != "user":
            continue
        text = str(t.get("text") or "")
        if _contains_save_keyword(text):
            return str(t.get("turn_id") or "").strip() or None
    return None


def _contains_save_keyword(text: str) -> bool:
    hay = text.lower()
    for kw in _SAVE_KEYWORDS:
        if kw.lower() in hay:
            return True
    return False


def default_marks_keep_all(turns: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("turn_id") or "").strip()
        role = str(t.get("role") or "").strip().lower()
        if not tid:
            continue
        evidence = _default_evidence_level(role)
        ttl, forget = _default_ttl_policy("note", evidence, None)
        out.append(
            {
                "turn_id": tid,
                "keep": True,
                "span": None,
                "user_triggered_save": False,
                "category": "note",
                "subtype": "user_pinned_note",
                "evidence_level": evidence,
                "requires_confirmation": False,
                "importance": 0.5,
                "ttl_seconds": ttl,
                "forget_policy": forget,
                "reason": "stage2_skipped",
            }
        )
    return out


def pin_intents_to_facts(
    *,
    pin_intents: Sequence[Dict[str, Any]],
    turns: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not pin_intents:
        return []
    turn_map = {str(t.get("turn_id") or "").strip(): str(t.get("text") or "") for t in (turns or []) if isinstance(t, dict)}
    facts: List[Dict[str, Any]] = []
    for pin in pin_intents:
        target_ids = [str(x) for x in (pin.get("target_turn_ids") or []) if str(x).strip()]
        if not target_ids:
            continue
        texts = [turn_map.get(tid, "") for tid in target_ids if turn_map.get(tid)]
        joined = " | ".join([t for t in texts if t])
        if not joined:
            continue
        statement = f"User pinned note: {joined}"
        statement = re.sub(r"\s+", " ", statement).strip()
        if len(statement) > 600:
            statement = statement[:597] + "..."
        facts.append(
            {
                "statement": statement,
                "type": "note",
                "fact_type": "note",
                "scope": "permanent",
                "status": "n/a",
                "importance": float(pin.get("importance_boost") or 0.9),
                "source_turn_ids": list(target_ids),
                "rationale": "user_triggered_save",
            }
        )
    return facts


__all__ = [
    "build_turn_mark_extractor_v1_from_env",
    "build_turn_mark_context",
    "parse_turn_mark_json",
    "validate_and_normalize_marks",
    "apply_turn_marks",
    "generate_pin_intents",
    "default_marks_keep_all",
    "pin_intents_to_facts",
]
