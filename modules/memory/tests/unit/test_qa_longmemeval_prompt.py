from __future__ import annotations

from modules.memory.application.qa_longmemeval import (
    build_longmemeval_prompts,
    build_longmemeval_user_prompt_v1,
    extract_question_date_from_time_hints,
    LME_PREFERENCE_SYSTEM_PROMPT,
    LME_TEMPORAL_SYSTEM_PROMPT,
    LME_ASSISTANT_SYSTEM_PROMPT,
    LME_SYSTEM_PROMPT,
    should_use_longmemeval_prompt,
)


def test_should_use_longmemeval_prompt_by_domain() -> None:
    assert should_use_longmemeval_prompt(memory_domain="longmemeval_oracle", task="GENERAL") is True
    assert should_use_longmemeval_prompt(memory_domain="dialog", task="temporal-reasoning") is True
    assert should_use_longmemeval_prompt(memory_domain="dialog", task="GENERAL") is False


def test_extract_question_date_from_time_hints() -> None:
    assert extract_question_date_from_time_hints(None) is None
    assert extract_question_date_from_time_hints({"question_date": "2023/02/12 (Sun) 05:07"}) == "2023/02/12 (Sun) 05:07"
    assert extract_question_date_from_time_hints({"now_iso": "2023-02-12T05:07:00Z"}) == "2023-02-12T05:07:00Z"


def test_build_longmemeval_user_prompt_includes_date_and_question() -> None:
    p = build_longmemeval_user_prompt_v1(
        question="How many days ago did I watch the Super Bowl?",
        question_date="2023/03/01 (Wed) 00:28",
        evidence=[
            {"event_id": "e1", "timestamp": "2023-02-12T05:07:00Z", "text": "I watched the Super Bowl today", "source": "event_search"},
        ],
    )
    assert "Current Date: 2023/03/01 (Wed) 00:28" in p
    assert "Question: How many days ago did I watch the Super Bowl?" in p
    assert "ts=2023-02-12T05:07:00Z" in p


def test_build_longmemeval_prompts_selects_preference_system_prompt() -> None:
    sys_p, user_p = build_longmemeval_prompts(
        question="Any tips?",
        question_date="2023/05/30 (Tue) 23:40",
        evidence=[],
        task="single-session-preference",
    )
    assert sys_p == LME_PREFERENCE_SYSTEM_PROMPT
    assert "Current Date:" in user_p


def test_build_longmemeval_prompts_selects_general_system_prompt() -> None:
    sys_p, user_p = build_longmemeval_prompts(
        question="What did I buy?",
        question_date="2023/05/30 (Tue) 23:40",
        evidence=[],
        task="multi-session",
    )
    assert sys_p == LME_SYSTEM_PROMPT
    assert "Question:" in user_p


def test_build_longmemeval_prompts_selects_temporal_system_prompt() -> None:
    sys_p, _ = build_longmemeval_prompts(
        question="How many days ago did I watch the Super Bowl?",
        question_date="2023/03/01 (Wed) 00:28",
        evidence=[],
        task="temporal-reasoning",
    )
    assert sys_p == LME_TEMPORAL_SYSTEM_PROMPT


def test_build_longmemeval_prompts_selects_assistant_system_prompt() -> None:
    sys_p, _ = build_longmemeval_prompts(
        question="What was the 7th job?",
        question_date="2023/05/30 (Tue) 23:40",
        evidence=[],
        task="single-session-assistant",
    )
    assert sys_p == LME_ASSISTANT_SYSTEM_PROMPT
