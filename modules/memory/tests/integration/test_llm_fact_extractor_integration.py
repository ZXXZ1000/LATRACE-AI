from __future__ import annotations

import asyncio
import os
import pytest

from modules.memory.application.fact_extractor_mem0 import build_fact_extractor_from_env


@pytest.mark.integration
def test_llm_fact_extractor_real_call_if_configured():
    async def _run():
        extractor = build_fact_extractor_from_env()
        if extractor is None or os.getenv("RUN_LLM_INTEGRATION") not in {"1", "true", "TRUE"}:
            pytest.skip("LLM not configured or RUN_LLM_INTEGRATION!=1; skipping real-call integration test")
        try:
            facts = extractor([
                {"role": "user", "content": "我不喜欢恐怖电影，我更喜欢科幻电影"},
                {"role": "assistant", "content": "明白了，我会推荐科幻片。"},
            ])
        except Exception as e:
            print(f"LLM call failed with exception: {e}")
            assert False, f"LLM call failed with exception: {e}"
        assert isinstance(facts, list)
        if not facts:
            print("LLM returned empty facts list.")
            assert False, "LLM returned empty facts list (rate limit or model behavior)."
        # 至少抽取出一个偏好事实
        assert any("喜欢" in str(f) or "偏好" in str(f) for f in facts)
    asyncio.run(_run())
