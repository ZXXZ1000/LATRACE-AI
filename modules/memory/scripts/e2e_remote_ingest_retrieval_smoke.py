#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


@dataclass(frozen=True)
class Result:
    ok: bool
    detail: str
    data: Optional[Dict[str, Any]] = None


def _mask(val: Optional[str], keep: int = 6) -> str:
    if not val:
        return ""
    s = str(val)
    if len(s) <= keep:
        return s
    return f"{s[:keep]}…({len(s)} chars)"


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _normalize_base_url(raw: str) -> str:
    base = str(raw or "").strip()
    if not base:
        raise ValueError("base_url is required")
    return base.rstrip("/")


def _join(base: str, path: str) -> str:
    p = str(path or "").strip()
    if not p.startswith("/"):
        p = "/" + p
    return base + p


def _sign_headers(*, secret: str, path: str, body: bytes, sig_header: str, ts_header: str) -> Dict[str, str]:
    ts_val = int(time.time())
    payload = f"{ts_val}.{path}".encode("utf-8") + b"." + (body or b"")
    sig = hmac.new(str(secret).encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return {sig_header: sig, ts_header: str(ts_val)}


def _build_headers(
    *,
    auth_header: Optional[str],
    auth_token: Optional[str],
    tenant_id: Optional[str],
    request_id: Optional[str],
) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if tenant_id:
        headers["X-Tenant-ID"] = str(tenant_id)
    if request_id:
        headers["X-Request-ID"] = str(request_id)
    if auth_header and auth_token:
        if str(auth_header).lower() == "authorization" and " " not in str(auth_token).strip():
            headers[auth_header] = f"Bearer {auth_token}"
        else:
            headers[auth_header] = str(auth_token)
    return headers


def _http_client(*, timeout_s: float, verify_tls: bool, trust_env: bool) -> httpx.Client:
    # Important: many dev machines export HTTP(S)_PROXY which can break localhost requests.
    # Default to trust_env=False to ensure direct connections for smoke tests.
    return httpx.Client(timeout=timeout_s, verify=verify_tls, follow_redirects=True, trust_env=bool(trust_env))


def _request_json(
    client: httpx.Client,
    *,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]] = None,
) -> Result:
    data = _json_dumps(body) if body is not None else b""
    try:
        resp = client.request(method.upper(), url, headers=headers, content=data)
    except Exception as exc:
        return Result(False, f"http_error: {type(exc).__name__}: {str(exc)[:200]}", None)
    try:
        payload = resp.json()
    except Exception:
        payload = None
    if resp.status_code >= 400:
        text = (resp.text or "")[:500]
        return Result(False, f"http_{resp.status_code}: {text}", payload if isinstance(payload, dict) else None)
    return Result(True, "ok", payload if isinstance(payload, dict) else None)


def _poll_job_status(
    client: httpx.Client,
    *,
    base_url: str,
    job_id: str,
    headers: Dict[str, str],
    job_path_tmpl: str,
    timeout_s: float,
    poll_interval_s: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Returns (final_status, last_payload). final_status='unknown' if API not available."""
    deadline = time.time() + float(timeout_s)
    last: Optional[Dict[str, Any]] = None
    path = job_path_tmpl.format(job_id=job_id)
    url = _join(base_url, path)
    while time.time() < deadline:
        res = _request_json(client, method="GET", url=url, headers=headers, body=None)
        if not res.ok:
            # SaaS surface may not expose this endpoint yet (404 at api-dev).
            if "http_404" in res.detail or "Route" in res.detail:
                return "unavailable", None
            last = res.data
            # Other errors: keep retrying briefly to handle transient issues.
            time.sleep(max(0.2, float(poll_interval_s)))
            continue
        last = res.data
        status = str((last or {}).get("status") or "").strip()
        if status in {"COMPLETED", "PAUSED", "STAGE2_FAILED", "STAGE3_FAILED"}:
            return status, last
        time.sleep(max(0.2, float(poll_interval_s)))
    return "timeout", last


def main() -> int:
    ap = argparse.ArgumentParser(description="Remote smoke test: ingest -> (poll job) -> retrieval (+ optional graph).")
    ap.add_argument("--base-url", required=True, help="Base URL, e.g. https://.../api/v1/memory or http://127.0.0.1:8000")
    ap.add_argument("--tenant-id", default=None, help="X-Tenant-ID (core direct often needs this)")
    ap.add_argument("--auth-header", default="Authorization", help="Auth header name (Authorization or X-API-Token)")
    ap.add_argument("--auth-token", default=None, help="Auth token value (Bearer token or raw API key); fallback env MEMORY_SMOKE_AUTH_TOKEN")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    ap.add_argument("--poll-timeout", type=float, default=60.0, help="Job poll timeout seconds")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="Job poll interval seconds")
    ap.add_argument(
        "--wait-after-ingest",
        type=float,
        default=0.0,
        help="If >0, sleep N seconds after ingest (useful when job status API isn't exposed).",
    )
    ap.add_argument("--verify-tls", action="store_true", help="Verify TLS certificates (default: off)")
    ap.add_argument(
        "--trust-env",
        action="store_true",
        help="Trust proxy-related env vars like HTTP_PROXY (default: off; recommended off for localhost).",
    )
    ap.add_argument("--ingest-path", default="/ingest", help="Ingest path (default: /ingest)")
    ap.add_argument("--retrieval-path", default="/retrieval", help="Retrieval path (default: /retrieval)")
    ap.add_argument("--job-path-template", default="/ingest/jobs/{job_id}", help="Job status path template")
    ap.add_argument("--graph-search-path", default="/graph/v1/search", help="Graph search path")
    ap.add_argument("--check-graph", action="store_true", help="Also call graph search endpoint")
    ap.add_argument("--require-job-api", action="store_true", help="Fail if job status API is unavailable")
    ap.add_argument("--signing-secret", default=None, help="If provided, attach X-Signature/X-Signature-Ts for POSTs; fallback env MEMORY_SMOKE_SIGNING_SECRET")
    ap.add_argument("--signature-header", default="X-Signature", help="Signature header name")
    ap.add_argument("--signature-ts-header", default="X-Signature-Ts", help="Signature timestamp header name")
    ap.add_argument("--user-token", action="append", default=[], help="user_tokens (repeatable). If empty, defaults to u:smoke.")
    ap.add_argument("--memory-domain", default="dialog", help="memory_domain")
    ap.add_argument(
        "--llm-policy",
        default="best_effort",
        choices=["best_effort", "require"],
        help="llm_policy for ingest/retrieval (best_effort recommended for smoke).",
    )
    ap.add_argument("--stage2-skip-llm", action="store_true", help="Set client_meta.stage2_skip_llm=true")
    ap.add_argument("--stage3-extract", action="store_true", help="Set client_meta.stage3_extract=true (default false)")
    ap.add_argument("--strategy", default="dialog_v2", help="Retrieval strategy (default dialog_v2)")
    ap.add_argument("--topk", type=int, default=10, help="retrieval topk")
    args = ap.parse_args()

    base_url = _normalize_base_url(args.base_url)
    auth_token = args.auth_token or os.getenv("MEMORY_SMOKE_AUTH_TOKEN")
    signing_secret = args.signing_secret or os.getenv("MEMORY_SMOKE_SIGNING_SECRET")
    request_id = f"req_{uuid.uuid4().hex}"
    session_id = f"smoke_{uuid.uuid4().hex[:10]}"

    print("[smoke] target:", base_url)
    print("[smoke] tenant_id:", args.tenant_id or "(none)")
    print("[smoke] auth:", f"{args.auth_header}={_mask(auth_token)}" if auth_token else "(none)")
    print("[smoke] request_id:", request_id)
    print("[smoke] session_id:", session_id)

    user_tokens = [str(x).strip() for x in (args.user_token or []) if str(x).strip()]
    if not user_tokens:
        user_tokens = ["u:smoke"]

    ingest_body: Dict[str, Any] = {
        "session_id": session_id,
        "user_tokens": user_tokens,
        "memory_domain": str(args.memory_domain),
        "llm_policy": str(args.llm_policy),
        "turns": [
            {"turn_id": "t1", "role": "user", "text": f"hello from {session_id}"},
            {"turn_id": "t2", "role": "assistant", "text": "ack"},
        ],
        "commit_id": f"commit_{uuid.uuid4().hex[:8]}",
        "client_meta": {
            "stage2_skip_llm": bool(args.stage2_skip_llm),
            "stage3_extract": bool(args.stage3_extract),
        },
    }

    retrieval_body: Dict[str, Any] = {
        "tenant_id": args.tenant_id,
        "user_tokens": user_tokens,
        "memory_domain": str(args.memory_domain),
        "query": f"hello from {session_id}",
        "strategy": str(args.strategy),
        "topk": int(args.topk),
        "debug": True,
        "llm_policy": str(args.llm_policy),
    }

    headers = _build_headers(
        auth_header=(str(args.auth_header).strip() if args.auth_header else None),
        auth_token=(str(auth_token).strip() if auth_token else None),
        tenant_id=(str(args.tenant_id).strip() if args.tenant_id else None),
        request_id=request_id,
    )

    verify_tls = bool(args.verify_tls)
    with _http_client(timeout_s=float(args.timeout), verify_tls=verify_tls, trust_env=bool(args.trust_env)) as client:
        # ---- ingest ----
        ingest_path = str(args.ingest_path)
        ingest_url = _join(base_url, ingest_path)
        ingest_headers = dict(headers)
        ingest_bytes = _json_dumps(ingest_body)
        if signing_secret:
            ingest_headers.update(
                _sign_headers(
                    secret=str(signing_secret),
                    path=ingest_path if ingest_path.startswith("/") else f"/{ingest_path}",
                    body=ingest_bytes,
                    sig_header=str(args.signature_header),
                    ts_header=str(args.signature_ts_header),
                )
            )

        print("[smoke] POST", ingest_url)
        try:
            resp = client.post(ingest_url, headers=ingest_headers, content=ingest_bytes)
        except Exception as exc:
            print("[fail] ingest http_error:", type(exc).__name__, str(exc)[:200])
            return 2
        ingest_payload = None
        try:
            ingest_payload = resp.json()
        except Exception:
            ingest_payload = None
        if resp.status_code >= 400:
            print("[fail] ingest http_%d:" % resp.status_code, (resp.text or "")[:500])
            return 2
        job_id = str((ingest_payload or {}).get("job_id") or "").strip()
        status = str((ingest_payload or {}).get("status") or "").strip()
        print("[smoke] ingest ok:", {"job_id": job_id, "status": status})
        if not job_id:
            print("[fail] ingest missing job_id:", ingest_payload)
            return 2

        # ---- poll job ----
        final_status, job_payload = _poll_job_status(
            client,
            base_url=base_url,
            job_id=job_id,
            headers=headers,
            job_path_tmpl=str(args.job_path_template),
            timeout_s=float(args.poll_timeout),
            poll_interval_s=float(args.poll_interval),
        )
        print("[smoke] job status:", final_status)
        if final_status == "unavailable":
            msg = "[warn] job API unavailable on this surface (expected if SaaS hasn't exposed /ingest/jobs/* yet)"
            print(msg)
            if args.require_job_api:
                return 3
            if float(args.wait_after_ingest) > 0:
                wait_s = float(args.wait_after_ingest)
                print(f"[smoke] waiting {wait_s:.1f}s for async processing...")
                time.sleep(wait_s)
        elif final_status in {"STAGE2_FAILED", "STAGE3_FAILED", "PAUSED"}:
            print("[fail] job failed:", job_payload)
            return 4
        elif final_status == "timeout":
            print("[warn] job polling timed out; continue to retrieval")
            if float(args.wait_after_ingest) > 0:
                wait_s = float(args.wait_after_ingest)
                print(f"[smoke] waiting {wait_s:.1f}s for async processing...")
                time.sleep(wait_s)

        # ---- retrieval ----
        retrieval_path = str(args.retrieval_path)
        retrieval_url = _join(base_url, retrieval_path)
        retrieval_headers = dict(headers)
        retrieval_bytes = _json_dumps(retrieval_body)
        if signing_secret:
            retrieval_headers.update(
                _sign_headers(
                    secret=str(signing_secret),
                    path=retrieval_path if retrieval_path.startswith("/") else f"/{retrieval_path}",
                    body=retrieval_bytes,
                    sig_header=str(args.signature_header),
                    ts_header=str(args.signature_ts_header),
                )
            )
        print("[smoke] POST", retrieval_url)
        try:
            resp_r = client.post(retrieval_url, headers=retrieval_headers, content=retrieval_bytes)
        except Exception as exc:
            print("[fail] retrieval http_error:", type(exc).__name__, str(exc)[:200])
            return 5
        try:
            retrieval_payload = resp_r.json()
        except Exception:
            retrieval_payload = None
        if resp_r.status_code >= 400:
            print("[fail] retrieval http_%d:" % resp_r.status_code, (resp_r.text or "")[:500])
            return 5

        ev = retrieval_payload.get("evidence") if isinstance(retrieval_payload, dict) else None
        evd = retrieval_payload.get("evidence_details") if isinstance(retrieval_payload, dict) else None
        ev_len = len(ev) if isinstance(ev, list) else 0
        evd_len = len(evd) if isinstance(evd, list) else 0
        dbg = retrieval_payload.get("debug") if isinstance(retrieval_payload, dict) else None
        calls = (dbg or {}).get("executed_calls") if isinstance(dbg, dict) else None
        print("[smoke] retrieval ok:", {"evidence": ev_len, "evidence_details": evd_len})
        if ev_len == 0 and evd_len == 0:
            print("[fail] retrieval empty; debug:", dbg if isinstance(dbg, dict) else retrieval_payload)
            return 6
        if isinstance(calls, list) and calls:
            print("[smoke] executed_calls:", calls)

        # ---- optional graph ----
        if args.check_graph:
            graph_path = str(args.graph_search_path)
            graph_url = _join(base_url, graph_path)
            graph_headers = dict(headers)
            graph_body = {"query": f"hello from {session_id}", "topk": 5}
            graph_bytes = _json_dumps(graph_body)
            if signing_secret:
                graph_headers.update(
                    _sign_headers(
                        secret=str(signing_secret),
                        path=graph_path if graph_path.startswith("/") else f"/{graph_path}",
                        body=graph_bytes,
                        sig_header=str(args.signature_header),
                        ts_header=str(args.signature_ts_header),
                    )
                )
            print("[smoke] POST", graph_url)
            try:
                resp_g = client.post(graph_url, headers=graph_headers, content=graph_bytes)
            except Exception as exc:
                print("[fail] graph http_error:", type(exc).__name__, str(exc)[:200])
                return 7
            if resp_g.status_code >= 400:
                print("[fail] graph http_%d:" % resp_g.status_code, (resp_g.text or "")[:500])
                return 7
            print("[smoke] graph ok")

    print("[ok] smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
