from __future__ import annotations

from modules.memory.scripts.e2e_longmemeval_oracle_subset_write_and_retrieval import (
    LongMemEvalItem,
    build_workload_items,
)


def _item(qid: str) -> LongMemEvalItem:
    return LongMemEvalItem(
        question_id=qid,
        question_type="single-session-user",
        question="q",
        answer="a",
        question_date="2023/05/30 (Tue) 23:40",
        haystack_sessions=[[{"role": "user", "content": "hello"}]],
        haystack_session_ids=["sess-1"],
        haystack_dates=["2023/05/30 (Tue) 23:40"],
        answer_session_ids=["sess-1"],
    )


def test_build_workload_items_expands_multi_tenant_matrix() -> None:
    workload = build_workload_items(
        items=[_item("q1"), _item("q2")],
        tenant="bench",
        tenant_count=2,
        tenant_prefix="bench-t",
        user_prefix="lm_",
        session_id_mode="tenant_prefixed",
        limit=2,
    )

    assert len(workload) == 4
    assert {work.tenant_id for work in workload} == {"bench-t000", "bench-t001"}
    assert all("::" in work.session_id for work in workload)
    assert len({work.session_id for work in workload}) == 4


def test_build_workload_items_can_reuse_shared_session_ids() -> None:
    workload = build_workload_items(
        items=[_item("q1")],
        tenant="bench",
        tenant_count=2,
        tenant_prefix="bench-t",
        user_prefix="lm_",
        session_id_mode="shared",
        limit=1,
    )

    assert len(workload) == 2
    assert {work.session_id for work in workload} == {"q1"}
