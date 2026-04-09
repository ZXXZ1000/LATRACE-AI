from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AdkErrorInfo:
    error_type: str
    retryable: bool
    message: str
    http_status: Optional[int] = None
    raw_detail: Optional[str] = None

    def to_debug_fields(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "retryable": self.retryable,
            "http_status": self.http_status,
            "raw_detail": self.raw_detail,
        }


def _extract_detail_text(body: Any) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail.strip()
        if detail is not None:
            return str(detail).strip()
        return str(body).strip()
    return str(body).strip()


def normalize_http_error(*, status_code: int, body: Any = None) -> AdkErrorInfo:
    text = _extract_detail_text(body)
    text_l = text.lower()

    if status_code == 400:
        return AdkErrorInfo("invalid_input", False, text or "invalid input", http_status=400, raw_detail=text or None)
    if status_code == 401:
        return AdkErrorInfo("unauthorized", False, text or "unauthorized", http_status=401, raw_detail=text or None)
    if status_code == 403:
        return AdkErrorInfo("forbidden", False, text or "forbidden", http_status=403, raw_detail=text or None)
    if status_code == 404:
        return AdkErrorInfo("not_found", False, text or "not found", http_status=404, raw_detail=text or None)
    if status_code == 409:
        return AdkErrorInfo("conflict", False, text or "conflict", http_status=409, raw_detail=text or None)
    if status_code == 429:
        return AdkErrorInfo("rate_limit", True, text or "rate limited", http_status=429, raw_detail=text or None)
    if status_code == 503:
        if "temporarily unavailable" in text_l:
            return AdkErrorInfo(
                "rate_limit",
                True,
                text or "temporarily unavailable",
                http_status=503,
                raw_detail=text or None,
            )
        return AdkErrorInfo("unavailable", True, text or "service unavailable", http_status=503, raw_detail=text or None)
    if status_code == 504:
        return AdkErrorInfo("timeout", True, text or "timeout", http_status=504, raw_detail=text or None)
    if 500 <= int(status_code) < 600:
        return AdkErrorInfo("server_error", True, text or "server error", http_status=status_code, raw_detail=text or None)
    return AdkErrorInfo("http_error", False, text or f"http {status_code}", http_status=status_code, raw_detail=text or None)


def normalize_exception(exc: Exception) -> AdkErrorInfo:
    msg = str(exc).strip() or exc.__class__.__name__
    cname = exc.__class__.__name__.lower()
    mod = str(getattr(exc.__class__, "__module__", "")).lower()

    if isinstance(exc, TimeoutError) or "timeout" in cname:
        return AdkErrorInfo("timeout", True, msg)
    if isinstance(exc, ConnectionError) or "connect" in cname or ("httpx" in mod and "network" in cname):
        return AdkErrorInfo("transport_error", True, msg)
    return AdkErrorInfo("internal_error", False, msg)

