from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from modules.memory.adk.state_property_vocab import (
    StatePropertyVocabLoadError,
    StatePropertyVocabManager,
    map_state_property,
)


class _FetcherStub:
    def __init__(self, responses: List[Dict[str, Any]] | None = None, *, exc: Exception | None = None) -> None:
        self.responses = [dict(x) for x in (responses or [])]
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        if self.responses:
            return self.responses.pop(0)
        return {"vocab_version": "v1", "properties": []}


def _sample_response(*, version: str = "v1") -> Dict[str, Any]:
    return {
        "vocab_version": version,
        "properties": [
            {"name": "occupation", "description": "工作/职位", "value_type": "string", "allow_raw_value": True},
            {"name": "work_status", "description": "工作状态", "value_type": "string", "allow_raw_value": True},
            {"name": "location", "description": "所在地", "value_type": "string", "allow_raw_value": True},
            {"name": "mood", "description": "情绪", "value_type": "string", "allow_raw_value": True},
        ],
    }


def test_state_property_vocab_manager_load_and_cache_hit() -> None:
    fetcher = _FetcherStub(responses=[_sample_response(version="v1")])
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    vocab1 = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1", user_tokens=["u:a"]))
    vocab2 = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))

    assert vocab1.vocab_version == "v1"
    assert vocab1.cache_miss is True
    assert vocab2.cache_hit is True
    assert len(fetcher.calls) == 1
    assert fetcher.calls[0]["tenant_id"] == "t1"


def test_state_property_vocab_manager_force_refresh_and_version_change_marks_refreshed() -> None:
    fetcher = _FetcherStub(responses=[_sample_response(version="v1"), _sample_response(version="v2")])
    mgr = StatePropertyVocabManager(fetcher=fetcher)

    _ = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))
    vocab2 = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1", force_refresh=True))

    assert vocab2.vocab_version == "v2"
    assert vocab2.vocab_refreshed is True
    assert len(fetcher.calls) == 2


def test_map_state_property_supports_canonical_alias_and_normalized_match() -> None:
    fetcher = _FetcherStub(responses=[_sample_response()])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    vocab = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))

    r1 = map_state_property("occupation", vocab=vocab)
    r2 = map_state_property("职位", vocab=vocab)
    r3 = map_state_property("work status", vocab=vocab)

    assert r1.matched and r1.canonical == "occupation" and r1.match_source == "canonical_exact"
    assert r2.matched and r2.canonical == "occupation" and r2.match_source == "alias_exact"
    assert r3.matched and r3.canonical == "work_status" and r3.match_source == "normalized_match"


def test_map_state_property_returns_disambiguation_candidates() -> None:
    fetcher = _FetcherStub(
        responses=[
            {
                "vocab_version": "v1",
                "properties": [
                    {"name": "work status", "allow_raw_value": True},
                    {"name": "work_status", "allow_raw_value": True},
                ],
            }
        ]
    )
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    vocab = asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))
    res = map_state_property("work-status", vocab=vocab)
    assert res.matched is False
    assert res.needs_disambiguation is True
    assert set(res.candidates) == {"work status", "work_status"}


def test_state_property_vocab_load_failure_wraps_error_info() -> None:
    mgr = StatePropertyVocabManager(fetcher=_FetcherStub(exc=TimeoutError("timeout")))
    try:
        asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))
        assert False, "expected StatePropertyVocabLoadError"
    except StatePropertyVocabLoadError as exc:
        assert exc.info.error_type == "timeout"
        assert exc.info.retryable is True


def test_state_property_vocab_inband_http_error_response_maps_via_helper() -> None:
    fetcher = _FetcherStub(responses=[{"status_code": 503, "body": "temporarily unavailable"}])
    mgr = StatePropertyVocabManager(fetcher=fetcher)
    try:
        asyncio.run(mgr.load_state_property_vocab(tenant_id="t1"))
        assert False, "expected StatePropertyVocabLoadError"
    except StatePropertyVocabLoadError as exc:
        assert exc.info.error_type == "rate_limit"
        assert exc.info.retryable is True
