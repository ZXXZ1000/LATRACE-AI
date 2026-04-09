from modules.memory.contracts.usage_models import UsageEvent, UsageSummary, TokenUsageDetail
from datetime import datetime, timezone
import uuid

def test_usage_event_creation():
    evt = UsageEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        tenant_id="t1",
        event_type="llm",
        provider="openrouter",
        model="gpt-4o",
        usage=TokenUsageDetail(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001
        )
    )
    assert evt.billable is True
    assert evt.status == "ok"
    assert evt.usage.total_tokens == 30

def test_usage_summary_structure():
    s = UsageSummary(
        total=TokenUsageDetail(total_tokens=100),
        llm=TokenUsageDetail(total_tokens=80),
        embedding=TokenUsageDetail(total_tokens=20),
        billable=False
    )
    assert s.billable is False
    assert s.total.total_tokens == 100
    json_out = s.model_dump()
    assert "llm" in json_out
    assert "embedding" in json_out
