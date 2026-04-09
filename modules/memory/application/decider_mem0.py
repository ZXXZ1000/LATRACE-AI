from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

try:
    # Prefer using mem0's official prompt builder if available
    from mem0.configs.prompts import get_update_memory_messages  # type: ignore
except Exception:  # pragma: no cover - fallback if mem0 not importable
    def get_update_memory_messages(retrieved_old_memory_dict, response_content, custom_update_memory_prompt=None):
        base = (
            custom_update_memory_prompt
            or "You are a smart memory manager. Decide ADD/UPDATE/DELETE/NONE based on current vs new facts."
        )
        current = (
            f"Current memory:\n```\n{retrieved_old_memory_dict}\n```\n"
            if retrieved_old_memory_dict
            else "Current memory is empty.\n"
        )
        return f"""{base}

{current}
New facts (in triple backticks):
```
{response_content}
```
Return strictly JSON: {{"memory":[{{"id":"<id>","text":"<content>","event":"ADD|UPDATE|DELETE|NONE","old_memory":"<old>"}}]}}
"""


def _remove_code_blocks(content: str) -> str:
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        body = content.strip("`")
        # possible leading language tag
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return content


class Mem0UpdateDecider:
    """LLM-based update decision aligned with mem0 DEFAULT_UPDATE_MEMORY_PROMPT.

    llm: Callable(messages: list[dict], response_format: dict|None) -> str
    decide(existing: list[MemoryEntry], new: MemoryEntry) -> (action, target_id|None)
    """

    def __init__(self, llm: Callable[[List[Dict[str, Any]], Dict[str, Any] | None], str]) -> None:
        self.llm = llm

    def decide(self, existing: List[Any], new: Any) -> Tuple[str, str | None]:
        # Build mem0-style inputs
        old = []
        id_map: Dict[str, str] = {}
        for i, e in enumerate(existing):
            if not getattr(e, "id", None):
                continue
            text = (e.contents[0] if getattr(e, "contents", None) else "")
            key = str(i)
            id_map[key] = e.id
            old.append({"id": key, "text": text})

        facts = []
        if getattr(new, "contents", None):
            facts = [new.contents[0]]

        prompt = get_update_memory_messages(old, facts, None)
        resp = self.llm(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        try:
            import json

            resp = _remove_code_blocks(resp)
            data = json.loads(resp)
            mem = data.get("memory", [])
            for item in mem:
                action = item.get("event")
                if action in ("ADD", "UPDATE", "DELETE", "NONE"):
                    target = item.get("id")
                    if action in ("UPDATE", "DELETE") and target in id_map:
                        return action, id_map[target]
                    # ADD/NONE do not require mapping
                    return action, None
        except Exception:
            pass
        # fallback
        return "ADD", None


def build_mem0_decider_from_env():
    """Build a mem0-style update decider using environment-configured LLM, or None.

    This helps wire a real LLM without hard dependencies or network during tests.
    """
    try:
        from modules.memory.application.llm_adapter import build_llm_from_env

        adapter = build_llm_from_env()
        if adapter is None:
            return None

        def _fn(messages, response_format=None):
            return adapter.generate(messages, response_format)

        return Mem0UpdateDecider(_fn).decide
    except Exception:
        return None
