from __future__ import annotations

"""
mem0 风格的事实抽取器（占位实现，可接入 LLM）。

策略：
- 优先使用本地 LLM 适配（LiteLLM 路由），从 .env 与 memory.config.yaml（通过环境变量展开）获取配置；
- 使用 mem0/mem0/configs/prompts.py 中的 FACT_RETRIEVAL_PROMPT 作为系统提示；
- 将对话 messages 以“User:/Assistant:”格式串联，作为用户消息提交，要求返回 JSON：{"facts": [...]}；
- 若无可用 LLM，则返回 None（由调用方决定回退策略）。
"""

from typing import Any, Callable, Dict, List, Optional
import socket
try:
    from dotenv import load_dotenv  # type: ignore
    import os as _os
    # load root .env
    load_dotenv()
    # load memory module config .env with override to ensure presence
    _MEM_ENV = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "config", ".env"))
    if _os.path.exists(_MEM_ENV):
        load_dotenv(_MEM_ENV, override=True)
except Exception:
    pass


def _remove_code_blocks(content: str) -> str:
    s = (content or "").strip()
    if s.startswith("```") and s.endswith("```"):
        body = s.strip("`")
        if "\n" in body:
            body = body.split("\n", 1)[1]
        return body.strip()
    return s


def build_fact_extractor_from_env() -> Optional[Callable[[List[Dict[str, Any]]], List[str]]]:
    try:
        # 1) 尝试导入 mem0 prompt 与本地 LLM 适配
        from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT  # type: ignore
    except Exception:
        FACT_RETRIEVAL_PROMPT = (
            "Extract user facts/preferences from the following conversation. Return JSON {\"facts\":[...]} only."
        )
    try:
        from modules.memory.application.llm_adapter import build_llm_from_env

        adapter = build_llm_from_env()
        if adapter is None:
            return None

        # 离线/受限环境快速检测：若无法解析主要提供商域名，则视为未配置，返回 None 以便测试跳过
        # 仅做轻量 DNS 检查，避免真实网络请求
        try:
            targets = []
            if _os.getenv("OPENROUTER_API_KEY"):
                targets.append("openrouter.ai")
            if _os.getenv("OPENAI_API_KEY"):
                targets.append("api.openai.com")
            if _os.getenv("DEEPSEEK_API_KEY"):
                targets.append("api.deepseek.com")
            if _os.getenv("MOONSHOT_API_KEY"):
                targets.append("api.moonshot.cn")
            if _os.getenv("DASHSCOPE_API_KEY") or _os.getenv("QWEN_API_KEY"):
                targets.append("dashscope.aliyuncs.com")
            if _os.getenv("ZHIPUAI_API_KEY") or _os.getenv("GLM_API_KEY"):
                targets.append("open.bigmodel.cn")
            if _os.getenv("GOOGLE_API_KEY") or _os.getenv("GEMINI_API_KEY"):
                targets.append("generativelanguage.googleapis.com")

            offline = False
            for host in targets:
                try:
                    socket.gethostbyname(host)
                except Exception:
                    offline = True
                    break
            if offline and targets:
                return None
        except Exception:
            # 检测失败不阻断正常路径
            pass

        def _extract(messages: List[Dict[str, Any]]) -> List[str]:
            # 2) 构造对话文本（仅 user/assistant）
            parts: List[str] = []
            for m in messages or []:
                role = str(m.get("role") or "").lower()
                content = str(m.get("content") or "").strip()
                if not content:
                    continue
                if role in {"user", "assistant"}:
                    cap = "User" if role == "user" else "Assistant"
                    parts.append(f"{cap}: {content}")
            convo = "\n".join(parts)
            prompt = f"{FACT_RETRIEVAL_PROMPT}\n\n### Conversation\n{convo}"
            # 3) 调用 LLM 要求 JSON
            try:
                raw = adapter.generate([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
            except Exception:
                # 网络或限流等异常时，抛出让上层测试可决定 skip
                raise
            try:
                import json

                data = json.loads(_remove_code_blocks(raw))
                facts = data.get("facts")
                if isinstance(facts, list):
                    return [str(x) for x in facts if str(x).strip()]
            except Exception:
                pass
            return []

        return _extract
    except Exception:
        return None
