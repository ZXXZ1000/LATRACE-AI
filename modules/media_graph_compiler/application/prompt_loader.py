#!/usr/bin/env python3
"""
提示词模板加载器。

职责：
- 从prompts/目录加载提示词模板文件
- 支持动态语言和自定义
- 提供fallback机制
"""

from __future__ import annotations

import os
from typing import Dict, Optional


DEFAULT_PROMPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "prompts")
)


class PromptLoader:
    """提示词模板加载器。"""

    def __init__(self, prompts_dir: Optional[str] = None):
        """初始化提示词加载器。

        Args:
            prompts_dir: 自定义提示词目录，如果为None则使用默认目录
        """
        self._prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR

    def load_prompt(self, profile: str) -> str:
        """加载指定profile的提示词。

        Args:
            profile: 提示词profile名称（如'rich_context', 'equivalence_focus', 'strict_json'）

        Returns:
            提示词文本，如果加载失败则返回空字符串
        """
        prompt_file = os.path.join(self._prompts_dir, f"{profile}.txt")
        try:
            if os.path.exists(prompt_file):
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
        except Exception:
            pass

        # Fallback到内置提示词
        return self._get_fallback_prompt(profile)

    def _get_fallback_prompt(self, profile: str) -> str:
        """获取最小化的 fallback 提示词。
        
        注意：真正的提示词应该在 config/prompts/*.txt 文件中定义。
        这里只是在外部文件缺失时的最小化备用。
        """
        # 最小化 fallback - 真正的提示词在 config/prompts/*.txt
        return (
            f"Profile: {profile}. You are a video understanding assistant.\n"
            "Output strict JSON with keys: episodic, semantic, equivalence, semantic_timeline.\n"
            "- semantic_timeline: [{text, event_type, action, actor_tag, place, images}]\n"
            "- Use face_#/voice_# tags for actors. Include dialogue from ASR if provided.\n"
            "Output a single JSON object."
        )

    def list_available_profiles(self) -> list[str]:
        """列出可用的提示词profile。"""
        profiles = []
        try:
            if os.path.exists(self._prompts_dir):
                for file in os.listdir(self._prompts_dir):
                    if file.endswith(".txt"):
                        profiles.append(file[:-4])  # 移除.txt扩展名
        except Exception:
            pass

        # 添加内置profiles
        if not profiles:
            profiles = ["equivalence_focus", "rich_context", "strict_json"]

        return profiles
