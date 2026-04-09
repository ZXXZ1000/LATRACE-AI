from __future__ import annotations

from typing import Dict, List

import pytest


def test_llm_usage_hook_increments_call_index() -> None:
    from modules.memory.application.llm_adapter import (
        LLMUsageContext,
        _emit_llm_usage,
        reset_llm_usage_context,
        reset_llm_usage_hook,
        set_llm_usage_context,
        set_llm_usage_hook,
    )

    events: List[Dict[str, object]] = []
    hook_token = set_llm_usage_hook(lambda payload: events.append(dict(payload)))
    ctx_token = set_llm_usage_context(
        LLMUsageContext(
            tenant_id="t1",
            api_key_id="k1",
            request_id="req1",
            stage="stage2",
            job_id="job1",
            session_id="sess1",
            call_index=None,
            source="unit_test",
        )
    )
    try:
        _emit_llm_usage(
            provider="litellm",
            model="gpt-test",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            generation_id="gen_123",
            cost_usd=0.001,
        )
        _emit_llm_usage(
            provider="litellm",
            model="gpt-test",
            usage={"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        )
    finally:
        reset_llm_usage_context(ctx_token)
        reset_llm_usage_hook(hook_token)

    assert len(events) == 2
    assert events[0]["call_index"] == 0
    assert events[1]["call_index"] == 1
    assert events[0]["tokens_missing"] is False
    assert events[1]["tokens_missing"] is True
    assert events[0]["generation_id"] == "gen_123"
    assert events[0]["cost_usd"] == 0.001


def test_llm_usage_event_emits_metrics(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from modules.memory.api import server as srv

    emitted: List[Dict[str, object]] = []
    monkeypatch.setattr(srv, "_emit_usage_event", lambda payload: emitted.append(dict(payload)))

    payload = {
        "tenant_id": "tenant-a",
        "api_key_id": "key-1",
        "request_id": "req-xyz",
        "job_id": "job-123",
        "session_id": "sess-9",
        "stage": "stage2",
        "call_index": 0,
        "provider": "litellm",
        "model": "gpt-test",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "tokens_missing": True,
        "source": "unit_test",
    }

    srv._handle_llm_usage_event(payload)

    assert len(emitted) == 1
    evt = emitted[0]
    seed = "tenant-a|key-1|job-123|stage2|0"
    assert evt["event_id"] == srv._hash_usage_event_id("llm", seed)
    assert evt["event_type"] == "llm"
    assert evt["stage"] == "stage2"
    usage = evt["usage"]
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert evt["meta"]["tokens_missing"] is True
