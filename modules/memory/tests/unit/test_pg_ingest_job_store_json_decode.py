from __future__ import annotations

from datetime import datetime, timezone

from modules.memory.infra.pg_ingest_job_store import PgIngestJobStore, PgIngestJobStoreSettings


def test_pg_store_row_to_record_parses_str_json() -> None:
    """JSONB 字段返回 str 时也应被解析为 list/dict，避免被清空。"""
    store = PgIngestJobStore(PgIngestJobStoreSettings())
    now = datetime.now(timezone.utc)
    row = {
        "job_id": "job_1",
        "session_id": "s1",
        "commit_id": "c1",
        "tenant_id": "tenant",
        "api_key_id": None,
        "request_id": None,
        "user_tokens": '["u:1"]',
        "memory_domain": "dialog",
        "llm_policy": "require",
        "status": "RECEIVED",
        "attempts": '{"stage2": 2, "stage3": 3}',
        "next_retry_at": None,
        "last_error": '{"code": "x"}',
        "metrics": '{"archived_turns": 5}',
        "created_at": now,
        "updated_at": now,
        "cursor_committed": None,
        "turns": '[{"turn_id": "t1", "text": "hi"}]',
        "client_meta": '{"memory_policy": "user", "user_id": "u:1"}',
        "stage2_marks": '[{"keep": true}]',
        "stage2_pin_intents": "[]",
        "payload_raw": None,
    }

    rec = store._row_to_record(row)  # type: ignore[arg-type]

    assert rec.user_tokens == ["u:1"]
    assert rec.turns and rec.turns[0].get("text") == "hi"
    assert rec.attempts.get("stage2") == 2
    assert rec.attempts.get("stage3") == 3
    assert rec.metrics.get("archived_turns") == 5
    assert rec.last_error.get("code") == "x"
    assert rec.stage2_marks and rec.stage2_marks[0].get("keep") is True
    assert rec.client_meta.get("memory_policy") == "user"
