from __future__ import annotations

from modules.memory.adk.errors import normalize_exception, normalize_http_error


def test_normalize_http_503_temporarily_unavailable_maps_retryable_rate_limit() -> None:
    info = normalize_http_error(status_code=503, body="temporarily unavailable")
    assert info.error_type == "rate_limit"
    assert info.retryable is True
    assert info.http_status == 503


def test_normalize_http_404_maps_not_found() -> None:
    info = normalize_http_error(status_code=404, body={"detail": "state_not_found"})
    assert info.error_type == "not_found"
    assert info.retryable is False
    assert info.raw_detail == "state_not_found"


def test_normalize_http_400_maps_invalid_input() -> None:
    info = normalize_http_error(status_code=400, body={"detail": "missing_core_requirements"})
    assert info.error_type == "invalid_input"
    assert info.retryable is False


def test_normalize_exception_timeout_and_connection() -> None:
    timeout_info = normalize_exception(TimeoutError("tool timeout"))
    conn_info = normalize_exception(ConnectionError("socket closed"))
    assert timeout_info.error_type == "timeout"
    assert timeout_info.retryable is True
    assert conn_info.error_type == "transport_error"
    assert conn_info.retryable is True
