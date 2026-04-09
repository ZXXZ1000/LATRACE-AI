#!/usr/bin/env python3
from __future__ import annotations

"""E2E sanity run for LoCoMo using HTTP APIs (no SDK):

- Write: POST /ingest/dialog/v1  -> poll /ingest/jobs/{job_id}
- Retrieve: POST /retrieval/dialog/v2

This script intentionally reuses the same LoCoMo parsing + scoring/latency aggregation
as `e2e_dialog_conv26_session_write_and_retrieval.py`, but swaps the call layer to HTTP.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
import hmac
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from modules.memory.application.config import load_memory_config
from modules.memory.domain.dialog_text_pipeline_v1 import parse_datetime as _parse_datetime


@dataclass(frozen=True)
class LocomoQuery:
    query_id: str
    text: str
    ground_truth: str
    task: str
    evidence_ids: List[str]


def _load_locomo_sample(path: Path, sample_id: str) -> Dict[str, Any]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("locomo10.json must be a list")
    for it in items:
        if isinstance(it, dict) and str(it.get("sample_id") or "").strip() == sample_id:
            return it
    raise ValueError(f"sample_id not found: {sample_id}")


def _iter_sessions(conversation: Dict[str, Any]) -> Iterable[Tuple[int, str, List[Dict[str, Any]]]]:
    sessions: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for k, v in conversation.items():
        if not str(k).startswith("session_") or str(k).endswith("_date_time"):
            continue
        try:
            idx = int(str(k).split("_", 1)[1])
        except Exception:
            continue
        if isinstance(v, list):
            dt = str(conversation.get(f"{k}_date_time", "") or "")
            sessions.append((idx, dt, list(v)))
    sessions.sort(key=lambda x: x[0])
    return sessions


def build_turns_for_session_write(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    conv = sample.get("conversation") or {}
    if not isinstance(conv, dict):
        raise ValueError("sample.conversation must be a dict")

    turns: List[Dict[str, Any]] = []
    for sess_idx, dt_str, sess_turns in _iter_sessions(conv):
        base_ts, _ = _parse_datetime(dt_str)
        for i, t in enumerate(sess_turns):
            if not isinstance(t, dict):
                continue
            dia_id = t.get("dia_id")
            speaker = t.get("speaker") or "Unknown"
            text = str(t.get("text") or "").strip()
            if not text:
                continue
            if t.get("blip_caption"):
                cap = str(t.get("blip_caption") or "").strip()
                if cap and "[Image:" not in text:
                    text = f"{text} [Image: {cap}]"
            ts = base_ts + i * 60
            turns.append(
                {
                    "dia_id": str(dia_id) if dia_id else None,
                    "speaker": str(speaker),
                    "text": text,
                    "timestamp_iso": datetime.fromtimestamp(ts).isoformat(),
                    "session_idx": int(sess_idx),
                    "session_date_time": str(dt_str),
                    "blip_caption": (str(t.get("blip_caption")).strip() if t.get("blip_caption") else None),
                }
            )
    if not turns:
        raise ValueError("no turns built")
    return turns


def load_locomo_queries(path: Path) -> List[LocomoQuery]:
    out: List[LocomoQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(
                LocomoQuery(
                    query_id=str(row.get("query_id") or ""),
                    text=str(row.get("text") or ""),
                    ground_truth=str(row.get("ground_truth") or ""),
                    task=str(row.get("task") or "L1"),
                    evidence_ids=[str(x) for x in (row.get("evidence_ids") or []) if str(x).strip()],
                )
            )
    return out


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def _latency_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    xs = [float(x) for x in values]
    return {
        "count": len(xs),
        "sum_ms": float(sum(xs)),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "mean_ms": (sum(xs) / len(xs)),
        "p50_ms": _pct(xs, 50),
        "p90_ms": _pct(xs, 90),
        "p95_ms": _pct(xs, 95),
        "p99_ms": _pct(xs, 99),
    }


def _scalar_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    xs = [float(x) for x in values]
    return {
        "count": len(xs),
        "sum": float(sum(xs)),
        "min": min(xs),
        "max": max(xs),
        "mean": (sum(xs) / len(xs)),
        "p50": _pct(xs, 50),
        "p90": _pct(xs, 90),
        "p95": _pct(xs, 95),
        "p99": _pct(xs, 99),
    }


def _get_usage_field(res: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = res.get("usage")
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    if cur is None:
        return None
    try:
        return float(cur)
    except Exception:
        return None


def _usage_flat(res: Dict[str, Any]) -> Dict[str, Any]:
    u = res.get("usage")
    if not isinstance(u, dict):
        return {}
    total = u.get("total") if isinstance(u.get("total"), dict) else {}
    llm = u.get("llm") if isinstance(u.get("llm"), dict) else {}
    emb = u.get("embedding") if isinstance(u.get("embedding"), dict) else {}
    out: Dict[str, Any] = {
        "billable": u.get("billable"),
        "total_prompt_tokens": total.get("prompt_tokens"),
        "total_completion_tokens": total.get("completion_tokens"),
        "total_total_tokens": total.get("total_tokens"),
        "total_cost_usd": total.get("cost_usd"),
        "llm_prompt_tokens": llm.get("prompt_tokens"),
        "llm_completion_tokens": llm.get("completion_tokens"),
        "llm_total_tokens": llm.get("total_tokens"),
        "llm_cost_usd": llm.get("cost_usd"),
        "embedding_prompt_tokens": emb.get("prompt_tokens"),
        "embedding_total_tokens": emb.get("total_tokens"),
        "embedding_cost_usd": emb.get("cost_usd"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _task_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        task = str(r.get("task") or "UNKNOWN")
        j = r.get("j_score") or {}
        verdict: Optional[bool] = None
        # Accept multiple shapes across judge implementations.
        try:
            if "binary_correct" in j:
                verdict = bool(float(j.get("binary_correct") or 0.0) >= 1.0)
            elif "score" in j:
                verdict = bool(float(j.get("score") or 0.0) >= 1.0)
            elif "label" in j:
                verdict = str(j.get("label") or "").strip().upper() == "CORRECT"
            elif "binary_label" in j:
                verdict = str(j.get("binary_label") or "").strip().upper() == "CORRECT"
        except Exception:
            verdict = None

        bucket = by_task.setdefault(task, {"count": 0, "correct": 0, "skipped": 0})
        if verdict is None:
            bucket["skipped"] += 1
            continue
        bucket["count"] += 1
        bucket["correct"] += int(bool(verdict))
    for t, b in by_task.items():
        c = int(b.get("count") or 0)
        ok = int(b.get("correct") or 0)
        b["accuracy"] = (ok / c) if c else None
    return dict(sorted(by_task.items(), key=lambda kv: kv[0]))


def _render_report_md(aggregate: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# E2E Report: ingest+poll + retrieval ({aggregate.get('sample_id')})")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- sample_id: `{aggregate.get('sample_id')}`")
    lines.append(f"- queries: `{aggregate.get('queries')}`")
    lines.append(f"- judge_enabled: `{aggregate.get('judge_enabled')}`")
    if aggregate.get("accuracy") is not None:
        lines.append(f"- accuracy: `{aggregate.get('accuracy'):.4f}`")
    lines.append("")
    lines.append("## Accuracy By Task")
    for task, b in (aggregate.get("by_task") or {}).items():
        acc = b.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else "n/a"
        lines.append(f"- {task}: {b.get('correct')}/{b.get('count')} ({acc_s})")
    lines.append("")
    lines.append("## Token Usage Stats")
    tok = aggregate.get("token_usage") if isinstance(aggregate.get("token_usage"), dict) else {}
    pt = tok.get("prompt_tokens") if isinstance(tok.get("prompt_tokens"), dict) else {}
    ct = tok.get("completion_tokens") if isinstance(tok.get("completion_tokens"), dict) else {}
    if pt.get("count"):
        lines.append(
            f"- total.prompt_tokens: sum={int(pt.get('sum') or 0)} avg={float(pt.get('mean') or 0):.2f}/query "
            f"p50={float(pt.get('p50') or 0):.2f} p99={float(pt.get('p99') or 0):.2f}"
        )
    if ct.get("count"):
        lines.append(
            f"- total.completion_tokens: sum={int(ct.get('sum') or 0)} avg={float(ct.get('mean') or 0):.2f}/query "
            f"p50={float(ct.get('p50') or 0):.2f} p99={float(ct.get('p99') or 0):.2f}"
        )
    if not pt.get("count") and not ct.get("count"):
        lines.append("- n/a (missing usage.total tokens)")
    lines.append("")
    lines.append("## Cost Analysis")
    cost = aggregate.get("cost") if isinstance(aggregate.get("cost"), dict) else {}
    total_cost = cost.get("total_cost_usd") if isinstance(cost.get("total_cost_usd"), dict) else {}
    llm_cost = cost.get("llm_cost_usd") if isinstance(cost.get("llm_cost_usd"), dict) else {}
    emb_cost = cost.get("embedding_cost_usd") if isinstance(cost.get("embedding_cost_usd"), dict) else {}
    missing = cost.get("missing") if isinstance(cost.get("missing"), dict) else {}
    if total_cost.get("count"):
        lines.append(
            f"- total.cost_usd: sum=${float(total_cost.get('sum') or 0):.6f} avg=${float(total_cost.get('mean') or 0):.6f}/query "
            f"p50=${float(total_cost.get('p50') or 0):.6f} p99=${float(total_cost.get('p99') or 0):.6f}"
        )
    else:
        lines.append("- total.cost_usd: n/a")
    if llm_cost.get("count") or emb_cost.get("count"):
        lines.append(
            f"- breakdown: embedding=${float(emb_cost.get('sum') or 0):.6f} + llm=${float(llm_cost.get('sum') or 0):.6f}"
        )
    if any(int(missing.get(k) or 0) for k in ["total", "embedding", "llm"]):
        lines.append(
            f"- missing: total={int(missing.get('total') or 0)} embedding={int(missing.get('embedding') or 0)} llm={int(missing.get('llm') or 0)}"
        )
    lines.append("")
    lines.append("## Latency (ms)")
    for name, stats in (aggregate.get("latency") or {}).items():
        if name == "per_api_ms" and isinstance(stats, dict):
            for api, s in stats.items():
                if not isinstance(s, dict) or not s.get("count"):
                    continue
                lines.append(
                    f"- api.{api}: count={s['count']} p50={s.get('p50_ms'):.2f} p90={s.get('p90_ms'):.2f} "
                    f"p95={s.get('p95_ms'):.2f} p99={s.get('p99_ms'):.2f} mean={s.get('mean_ms'):.2f}"
                )
            continue
        if not isinstance(stats, dict) or not stats.get("count"):
            continue
        lines.append(
            f"- {name}: count={stats['count']} p50={stats.get('p50_ms'):.2f} p90={stats.get('p90_ms'):.2f} "
            f"p95={stats.get('p95_ms'):.2f} p99={stats.get('p99_ms'):.2f} mean={stats.get('mean_ms'):.2f}"
        )
    lines.append("")
    return "\n".join(lines)


def _load_llm_judge_module() -> Optional[Any]:
    """Load llm_judge.py by file path without importing package __init__."""
    try:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "benchmark" / "shared" / "evaluation" / "llm_judge.py",
            root / "benchmark" / "evaluation" / "llm_judge.py",
        ]
        for llm_path in candidates:
            if not llm_path.exists():
                continue
            spec = importlib.util.spec_from_file_location("llm_judge_local", llm_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)  # type: ignore[call-arg]
            return mod
        return None
    except Exception:
        return None


def build_judge_from_memory_config() -> Optional["object"]:
    """Best-effort build of LLMJudge using memory.config.yaml (llm.judge)."""
    try:
        from modules.memory.application.config import get_llm_selection  # type: ignore
    except Exception:
        return None

    judge_mod = _load_llm_judge_module()
    if judge_mod is None:
        return None
    JudgeConfig = getattr(judge_mod, "JudgeConfig", None)
    LLMJudge = getattr(judge_mod, "LLMJudge", None)
    if JudgeConfig is None or LLMJudge is None:
        return None

    cfg = load_memory_config()
    sel = get_llm_selection(cfg, "judge")
    provider = str(sel.get("provider") or "").strip().lower()
    model = str(sel.get("model") or "").strip()
    if not provider or not model:
        return None

    backend: Optional[str] = None
    if provider in {"openai", "openrouter", "anthropic", "gemini", "dashscope"}:
        backend = provider
    elif provider == "qwen":
        backend = "dashscope"

    if backend is None:
        return None

    try:
        cfg_j = JudgeConfig(backend=backend, model=model)  # type: ignore[call-arg]
        return LLMJudge(cfg_j)
    except Exception:
        return None


def _resolve_queries_path(queries_arg: str, sample_id: str) -> Path:
    p = Path(str(queries_arg))
    if p.is_file():
        return p
    if p.is_dir():
        return p / sample_id / "queries.jsonl"
    if str(p).endswith(".jsonl"):
        return p
    return p / sample_id / "queries.jsonl"


def _discover_sample_ids(queries_arg: str) -> List[str]:
    p = Path(str(queries_arg))
    if p.is_file():
        raise ValueError("--sample-id=0 requires --queries to be a directory, not a single queries.jsonl file.")
    if not p.exists():
        raise ValueError(f"--queries directory not found: {p}")
    ids: List[str] = []
    for qf in sorted(p.glob("*/queries.jsonl")):
        if qf.is_file():
            ids.append(qf.parent.name)
    if not ids:
        raise ValueError(f"no queries.jsonl found under: {p}")
    return ids


def _sign_request(*, secret: str, ts: int, path: str, body: bytes) -> str:
    payload = f"{ts}.{path}".encode() + b"." + body
    return hmac.new(str(secret).encode(), payload, hashlib.sha256).hexdigest()


def _build_client_meta(args: argparse.Namespace, *, user_id: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "memory_policy": str(args.memory_policy),
        "user_id": str(user_id),
    }
    mode = str(args.llm_mode or "platform").strip().lower()
    if mode == "byok":
        meta.update(
            {
                "llm_mode": "byok",
                "llm_provider": str(args.byok_provider or ""),
                "llm_model": str(args.byok_model or ""),
                "llm_api_key": str(args.byok_api_key or ""),
            }
        )
        if args.byok_base_url:
            meta["llm_base_url"] = str(args.byok_base_url)
    else:
        meta["llm_mode"] = "platform"
    # optional tuning knobs (kept compatible with current server)
    if args.stage2_enabled is not None:
        meta["stage2_enabled"] = bool(args.stage2_enabled)
    if args.stage3_extract is not None:
        meta["stage3_extract"] = bool(args.stage3_extract)
    return meta


class MemoryHttpClient:
    def __init__(self, *, base_url: str, tenant_id: str, api_token_header: str, api_token: str, signing_secret: str):
        self.base_url = str(base_url).rstrip("/")
        self.tenant_id = str(tenant_id)
        self.api_token_header = str(api_token_header or "X-API-Token")
        self.api_token = str(api_token or "")
        self.signing_secret = str(signing_secret or "")

    def _headers(self, *, path: str, body: bytes) -> Dict[str, str]:
        h: Dict[str, str] = {"X-Tenant-ID": self.tenant_id, "Content-Type": "application/json"}
        if self.api_token:
            h[self.api_token_header] = self.api_token
        if self.signing_secret:
            ts = int(time.time())
            h["X-Signature-Ts"] = str(ts)
            h["X-Signature"] = _sign_request(secret=self.signing_secret, ts=ts, path=path, body=body)
        return h

    async def ingest_dialog_v1(self, client: httpx.AsyncClient, body_obj: Dict[str, Any]) -> Dict[str, Any]:
        path = "/ingest/dialog/v1"
        body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        r = await client.post(f"{self.base_url}{path}", content=body, headers=self._headers(path=path, body=body))
        r.raise_for_status()
        return dict(r.json())

    async def ingest_job_execute(self, client: httpx.AsyncClient, job_id: str) -> Dict[str, Any]:
        path = "/ingest/jobs/execute"
        body_obj = {"job_id": str(job_id)}
        body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        r = await client.post(f"{self.base_url}{path}", content=body, headers=self._headers(path=path, body=body))
        r.raise_for_status()
        return dict(r.json())

    async def ingest_job_status(self, client: httpx.AsyncClient, job_id: str) -> Dict[str, Any]:
        path = f"/ingest/jobs/{str(job_id).strip()}"
        # GET does not require signature in server; keep it unsigned to avoid body signing ambiguity.
        h: Dict[str, str] = {"X-Tenant-ID": self.tenant_id}
        if self.api_token:
            h[self.api_token_header] = self.api_token
        r = await client.get(f"{self.base_url}{path}", headers=h)
        r.raise_for_status()
        return dict(r.json())

    async def retrieval_dialog_v2(self, client: httpx.AsyncClient, body_obj: Dict[str, Any]) -> Dict[str, Any]:
        path = "/retrieval/dialog/v2"
        body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        r = await client.post(f"{self.base_url}{path}", content=body, headers=self._headers(path=path, body=body))
        r.raise_for_status()
        return dict(r.json())


async def _run_one_sample(args: argparse.Namespace, sample_id: str, out_dir: Path) -> int:
    sample = _load_locomo_sample(Path(args.input), str(sample_id))

    tenant_id = str(args.tenant)
    session_id = str(sample_id)
    user_id = f"{str(args.user_prefix).rstrip('_')}_{session_id}"

    # Optional: chunk multi-session extraction to avoid context overflow (env only; affects platform LLM).
    if int(args.extract_sessions_per_call or 0) > 0:
        os.environ["MEMORY_DIALOG_TKG_EXTRACT_SESSIONS_PER_CALL"] = str(int(args.extract_sessions_per_call))

    client_meta = _build_client_meta(args, user_id=user_id)
    http_client = MemoryHttpClient(
        base_url=str(args.base_url),
        tenant_id=tenant_id,
        api_token_header=str(args.api_token_header),
        api_token=str(args.api_token),
        signing_secret=str(args.signing_secret),
    )

    write_latency_ms = 0.0
    if bool(args.skip_session_write):
        print("↪ ingest skipped (--skip-session-write)", flush=True)
    else:
        turns = build_turns_for_session_write(sample)
        turns_payload_bytes = len(json.dumps(turns, ensure_ascii=False).encode("utf-8"))
        print(
            f"→ ingest: turns={len(turns)} approx_payload={turns_payload_bytes/1024:.1f}KiB "
            f"session_id={session_id} commit_id={(str(args.commit_id) if args.commit_id else session_id)}",
            flush=True,
        )
        t0 = time.perf_counter()
        base_timeout = httpx.Timeout(
            connect=float(args.connect_timeout_seconds),
            read=float(args.ingest_timeout_seconds),
            write=float(args.write_timeout_seconds),
            pool=float(args.pool_timeout_seconds),
        )
        async with httpx.AsyncClient(timeout=base_timeout) as hc:
            ingest_body = {
                "session_id": session_id,
                "commit_id": (str(args.commit_id) if args.commit_id else session_id),
                "user_tokens": [user_id],
                "memory_domain": str(args.memory_domain),
                "turns": turns,
                "client_meta": dict(client_meta),
                "llm_policy": str(args.llm_policy),
            }
            ingest_res: Dict[str, Any]
            attempts = max(1, int(args.ingest_retries or 1))
            for i in range(1, attempts + 1):
                try:
                    ingest_res = await http_client.ingest_dialog_v1(hc, ingest_body)
                    break
                except httpx.ReadTimeout:
                    if i >= attempts:
                        raise
                    wait_s = min(10.0, 1.0 * i)
                    print(
                        f"⚠️ ingest read timeout (attempt {i}/{attempts}); retrying in {wait_s:.1f}s "
                        f"(commit_id={ingest_body.get('commit_id')})",
                        flush=True,
                    )
                    await asyncio.sleep(wait_s)
            job_id = str(ingest_res.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError(f"ingest missing job_id: {ingest_res}")
            async def _poll_progress() -> Dict[str, Any]:
                start = time.time()
                deadline = start + float(args.poll_timeout_seconds)
                last_print = 0.0
                last_fingerprint = ""
                last: Dict[str, Any] = {}
                while True:
                    now = time.time()
                    if now > deadline:
                        raise RuntimeError(f"poll_timeout: job_id={job_id} last={last}")
                    try:
                        last = await http_client.ingest_job_status(hc, job_id)
                    except Exception:
                        last = last or {}
                    st = str(last.get("status") or "").strip().upper() or "UNKNOWN"
                    metrics = last.get("metrics") if isinstance(last.get("metrics"), dict) else {}
                    attempts = last.get("attempts") if isinstance(last.get("attempts"), dict) else {}
                    last_error = last.get("last_error")
                    err_code = None
                    if isinstance(last_error, dict):
                        err_code = last_error.get("code") or last_error.get("error_code")

                    fingerprint = json.dumps(
                        {
                            "status": st,
                            "attempts": attempts,
                            "metrics": metrics,
                            "err_code": err_code,
                            "next_retry_at": last.get("next_retry_at"),
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    should_print = False
                    if fingerprint != last_fingerprint:
                        should_print = True
                    # Heartbeat: print at least every N seconds even if unchanged.
                    if float(args.execute_progress_every_seconds) > 0 and (now - last_print) >= float(
                        args.execute_progress_every_seconds
                    ):
                        should_print = True

                    if should_print:
                        last_fingerprint = fingerprint
                        last_print = now
                        elapsed = now - start
                        kept = metrics.get("kept_turns")
                        g = metrics.get("graph_nodes_written")
                        v = metrics.get("vector_points_written")
                        a2 = attempts.get("stage2")
                        a3 = attempts.get("stage3")
                        msg = f"… ingest progress: {elapsed:.0f}s status={st}"
                        if kept is not None:
                            msg += f" kept={kept}"
                        if a2 is not None or a3 is not None:
                            msg += f" attempts(stage2={a2},stage3={a3})"
                        if g is not None or v is not None:
                            msg += f" written(graph={g},vec={v})"
                        if err_code:
                            msg += f" err={err_code}"
                        print(msg, flush=True)

                    if st in {"COMPLETED", "STAGE2_FAILED", "STAGE3_FAILED", "FAILED"}:
                        return last
                    await asyncio.sleep(float(args.poll_interval_seconds))

            exec_res = await _poll_progress()
        write_latency_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"✓ ingest+poll: job_id={job_id} status={exec_res.get('status')} metrics={exec_res.get('metrics')}",
            flush=True,
        )

    queries_path = _resolve_queries_path(str(args.queries), session_id)
    queries = load_locomo_queries(queries_path)
    if args.limit and int(args.limit) > 0:
        queries = queries[: int(args.limit)]
    print(f"✓ queries loaded: {len(queries)}", flush=True)

    judge = None
    judge_error: Optional[str] = None
    if bool(args.judge) and bool(args.with_answer):
        try:
            judge = build_judge_from_memory_config()
            if judge is None:
                judge_error = "judge_init_failed"
                if bool(args.judge_required):
                    raise RuntimeError(judge_error)
                print(f"⚠️ judge disabled (best-effort): {judge_error}", flush=True)
        except Exception as exc:
            judge = None
            judge_error = f"{type(exc).__name__}: {str(exc)[:240]}"
            if bool(args.judge_required):
                raise
            print(f"⚠️ judge disabled (best-effort): {judge_error}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{session_id}.jsonl"
    print(f"✓ writing results to {out_path}", flush=True)

    results_rows: List[Dict[str, Any]] = []
    retrieval_wall_ms: List[float] = []
    retrieval_stage_ms: List[float] = []
    qa_stage_ms: List[float] = []
    total_stage_ms: List[float] = []
    per_api_ms: Dict[str, List[float]] = {}
    prompt_tokens_per_query: List[float] = []
    completion_tokens_per_query: List[float] = []
    total_cost_usd_per_query: List[float] = []
    embedding_cost_usd_per_query: List[float] = []
    llm_cost_usd_per_query: List[float] = []
    missing_total_cost = 0
    missing_embedding_cost = 0
    missing_llm_cost = 0
    query_failures = 0

    correct = 0
    total = 0
    by_task_running: Dict[str, Dict[str, int]] = {}
    log_every = int(args.log_every or 0)
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))
    completed = 0
    last_print = 0
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    def _maybe_print_progress(*, total_ms: float) -> None:
        nonlocal last_print
        if log_every <= 0:
            return
        if completed == len(queries) or completed == 1 or (completed - last_print) >= log_every:
            last_print = completed
        else:
            return

        lat_msg = f", total_ms={total_ms:.1f}"
        if total > 0:
            acc = correct / total
            acc_msg = f" acc={acc * 100:.1f}%"
            task_bits: List[str] = []
            for t in sorted(by_task_running.keys()):
                b = by_task_running[t]
                if b.get("count"):
                    task_acc = (b["correct"] / b["count"]) * 100
                    task_bits.append(f"{t}={task_acc:.1f}%")
            task_msg = f" tasks[{', '.join(task_bits)}]" if task_bits else ""
        else:
            acc_msg = " acc=n/a"
            task_msg = ""
        print(f"progress {completed}/{len(queries)}{lat_msg}{acc_msg}{task_msg}", flush=True)

    async def _run_one_query(q: LocomoQuery, hc: httpx.AsyncClient) -> None:
        nonlocal completed, correct, total, judge_error, missing_total_cost, missing_embedding_cost, missing_llm_cost
        nonlocal query_failures
        async with sem:
            q0 = time.perf_counter()
            body_obj = {
                "query": q.text,
                "user_tokens": [user_id],
                "memory_domain": str(args.memory_domain),
                "run_id": session_id,
                "strategy": "dialog_v2",
                "topk": int(args.topk),
                "debug": True,
                "with_answer": bool(args.with_answer),
                "task": str(q.task),
                "llm_policy": str(args.llm_policy),
                "backend": str(args.backend),
                "tkg_explain": bool(args.tkg_explain),
                "client_meta": dict(client_meta),
            }
            res: Optional[Dict[str, Any]] = None
            last_exc: Optional[BaseException] = None
            retries = max(1, int(args.query_retries or 1))
            for attempt in range(1, retries + 1):
                try:
                    res = await http_client.retrieval_dialog_v2(hc, body_obj)
                    last_exc = None
                    break
                except httpx.HTTPStatusError as exc:
                    # Retry on 5xx only.
                    status = getattr(exc.response, "status_code", None)
                    if status is None or int(status) < 500 or attempt >= retries:
                        raise
                    last_exc = exc
                except (httpx.ReadTimeout, httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    if attempt >= retries:
                        break
                if attempt < retries:
                    wait_s = float(getattr(args, "retry_backoff_seconds", 1.0) or 1.0) * attempt
                    print(
                        f"⚠️ retrieval transient error (attempt {attempt}/{retries}) query_id={q.query_id}: "
                        f"{type(last_exc).__name__}; retrying in {wait_s:.1f}s",
                        flush=True,
                    )
                    await asyncio.sleep(wait_s)

            q_wall_ms = (time.perf_counter() - q0) * 1000.0
            if res is None:
                msg = f"{type(last_exc).__name__}: {str(last_exc)[:240]}" if last_exc is not None else "unknown_error"
                row_err: Dict[str, Any] = {
                    "sample_id": session_id,
                    "query_id": q.query_id,
                    "task": q.task,
                    "question": q.text,
                    "ground_truth": q.ground_truth,
                    "gold_evidence_ids": list(q.evidence_ids),
                    "answer": "",
                    "evidence": [],
                    "debug": {},
                    "retrieval_error": msg,
                    "retrieval_total_ms": float(q_wall_ms),
                }
                async with lock:
                    query_failures += 1
                    results_rows.append(row_err)
                    f.write(json.dumps(row_err, ensure_ascii=False) + "\n")
                    completed += 1
                    _maybe_print_progress(total_ms=float(q_wall_ms))
                if bool(args.fail_fast):
                    raise RuntimeError(msg)
                return

            row: Dict[str, Any] = {
                "sample_id": session_id,
                "query_id": q.query_id,
                "task": q.task,
                "question": q.text,
                "ground_truth": q.ground_truth,
                "gold_evidence_ids": list(q.evidence_ids),
                "answer": res.get("answer", ""),
                "evidence": list(res.get("evidence") or []),
                "debug": res.get("debug") or {},
            }
            if isinstance(res.get("usage"), dict):
                row["usage"] = res.get("usage")
                row.update(_usage_flat(res))

            plan = (row.get("debug") or {}).get("plan") or {}
            qa_ms = 0.0
            if isinstance(plan, dict) and plan.get("qa_latency_ms") is not None:
                try:
                    qa_ms = max(0.0, float(plan.get("qa_latency_ms") or 0.0))
                except Exception:
                    qa_ms = 0.0
            total_ms = float(q_wall_ms)
            retrieval_only_ms = max(0.0, total_ms - qa_ms)

            calls = (row.get("debug") or {}).get("executed_calls") or []

            verdict: Optional[bool] = None
            if judge is not None and bool(args.with_answer):
                try:
                    jr = await asyncio.to_thread(judge.evaluate_binary, q.text, str(row.get("answer") or ""), q.ground_truth)
                    row["j_score"] = jr.to_dict()
                    verdict = bool(float(jr.score) >= 1.0)
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {str(exc)[:240]}"
                    row["j_score_error"] = msg
                    judge_error = judge_error or msg
                    if bool(args.judge_required):
                        raise

            async with lock:
                retrieval_wall_ms.append(q_wall_ms)
                retrieval_stage_ms.append(retrieval_only_ms)
                qa_stage_ms.append(qa_ms)
                total_stage_ms.append(total_ms)

                if isinstance(calls, list):
                    for c in calls:
                        if not isinstance(c, dict):
                            continue
                        api = str(c.get("api") or "")
                        lat = c.get("latency_ms")
                        if api and lat is not None:
                            per_api_ms.setdefault(api, []).append(float(lat))

                pt = _get_usage_field(res, ["total", "prompt_tokens"])
                if pt is not None:
                    prompt_tokens_per_query.append(pt)
                ct = _get_usage_field(res, ["total", "completion_tokens"])
                if ct is not None:
                    completion_tokens_per_query.append(ct)
                tc = _get_usage_field(res, ["total", "cost_usd"])
                if tc is not None:
                    total_cost_usd_per_query.append(tc)
                else:
                    missing_total_cost += 1
                ec = _get_usage_field(res, ["embedding", "cost_usd"])
                if ec is not None:
                    embedding_cost_usd_per_query.append(ec)
                else:
                    missing_embedding_cost += 1
                lc = _get_usage_field(res, ["llm", "cost_usd"])
                if lc is not None:
                    llm_cost_usd_per_query.append(lc)
                else:
                    missing_llm_cost += 1

                if verdict is not None:
                    total += 1
                    correct += int(bool(verdict))
                    bucket = by_task_running.setdefault(str(q.task), {"count": 0, "correct": 0})
                    bucket["count"] += 1
                    bucket["correct"] += int(bool(verdict))

                results_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                _maybe_print_progress(total_ms=total_ms)

    with out_path.open("w", encoding="utf-8") as f:
        base_timeout = httpx.Timeout(
            connect=float(args.connect_timeout_seconds),
            read=float(args.timeout_seconds),
            write=float(args.write_timeout_seconds),
            pool=float(args.pool_timeout_seconds),
        )
        async with httpx.AsyncClient(timeout=base_timeout) as hc:
            if concurrency <= 1:
                for q in queries:
                    await _run_one_query(q, hc)
            else:
                tasks = [asyncio.create_task(_run_one_query(q, hc)) for q in queries]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, BaseException):
                        raise r

    latency = {
        "write_total_ms": _latency_stats([write_latency_ms]),
        "retrieval_wall_ms": _latency_stats(retrieval_wall_ms),
        "retrieval_stage_ms": _latency_stats(retrieval_stage_ms),
        "qa_stage_ms": _latency_stats(qa_stage_ms),
        "total_stage_ms": _latency_stats(total_stage_ms),
        "per_api_ms": {k: _latency_stats(v) for k, v in sorted(per_api_ms.items(), key=lambda kv: kv[0])},
    }

    agg = {
        "sample_id": session_id,
        "queries": len(queries),
        "query_failures": query_failures,
        "judge_enabled": bool(judge is not None),
        "judge_error": judge_error,
        "judged": total,
        "correct": correct,
        "accuracy": (correct / total if total else None),
        "by_task": (_task_summary(results_rows) if bool(judge is not None) and bool(args.with_answer) else {}),
        "token_usage": {
            "prompt_tokens": _scalar_stats(prompt_tokens_per_query),
            "completion_tokens": _scalar_stats(completion_tokens_per_query),
        },
        "cost": {
            "total_cost_usd": _scalar_stats(total_cost_usd_per_query),
            "embedding_cost_usd": _scalar_stats(embedding_cost_usd_per_query),
            "llm_cost_usd": _scalar_stats(llm_cost_usd_per_query),
            "missing": {
                "total": missing_total_cost,
                "embedding": missing_embedding_cost,
                "llm": missing_llm_cost,
            },
        },
        "latency": latency,
    }
    agg_path = out_dir / f"aggregate_{session_id}.json"
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"report_{session_id}.md").write_text(_render_report_md(agg), encoding="utf-8")
    print(f"✓ wrote {len(queries)} results -> {out_path}", flush=True)
    print(f"✓ aggregate -> {agg_path}", flush=True)
    return 0


async def _run(args: argparse.Namespace) -> int:
    raw_sample_id = str(args.sample_id).strip()
    if raw_sample_id == "0":
        sample_ids = _discover_sample_ids(str(args.queries))
    else:
        sample_ids = [raw_sample_id]

    base_out_dir = Path(args.output_dir)
    for idx, sid in enumerate(sample_ids, 1):
        print(f"=== [{idx}/{len(sample_ids)}] sample_id={sid} ===", flush=True)
        out_dir = (base_out_dir / sid) if raw_sample_id == "0" else base_out_dir
        rc = await _run_one_sample(args, sid, out_dir)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="E2E LoCoMo via HTTP ingest+poll + retrieval.")
    p.add_argument("--input", default="benchmark/data/locomo/raw/locomo10.json")
    p.add_argument(
        "--queries",
        default="benchmark/data/locomo/step1_events",
        help="Path to queries.jsonl OR a directory that contains <sample_id>/queries.jsonl.",
    )
    p.add_argument("--sample-id", default="conv-26", help='LoCoMo sample_id (e.g. conv-26). Use "0" to run all.')

    # HTTP
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Memory API base URL (local server).")
    p.add_argument("--timeout-seconds", default=120, type=float, help="Default request read timeout (seconds).")
    p.add_argument(
        "--ingest-timeout-seconds",
        default=120,
        type=float,
        help="Read timeout for POST /ingest/dialog/v1 (seconds).",
    )
    p.add_argument(
        "--ingest-retries",
        default=2,
        type=int,
        help="Retry count for /ingest/dialog/v1 on ReadTimeout (uses same commit_id for idempotency).",
    )
    p.add_argument(
        "--execute-timeout-seconds",
        default=900,
        type=float,
        help="(Deprecated) Read timeout for /ingest/jobs/execute (seconds). Ingest now auto-executes; polling is used instead.",
    )
    p.add_argument("--connect-timeout-seconds", default=10, type=float)
    p.add_argument("--write-timeout-seconds", default=60, type=float)
    p.add_argument("--pool-timeout-seconds", default=10, type=float)
    p.add_argument("--poll-after-execute-timeout", action="store_true", help="If execute times out, poll job status.")
    p.add_argument("--poll-interval-seconds", default=5, type=float)
    p.add_argument("--poll-timeout-seconds", default=3600, type=float)
    p.add_argument(
        "--execute-progress-every-seconds",
        default=10,
        type=float,
        help="Print ingest progress heartbeat every N seconds (0 disables heartbeat; still prints on changes).",
    )
    p.add_argument("--api-token-header", default="X-API-Token")
    p.add_argument("--api-token", default="")
    p.add_argument("--signing-secret", default="", help="If set, send X-Signature/X-Signature-Ts for signed endpoints.")

    # tenant/user
    p.add_argument("--tenant", default="locomo_bench")
    p.add_argument("--user-prefix", default="locomo_user")
    p.add_argument("--memory-domain", default="dialog")
    p.add_argument("--memory-policy", default="user")
    p.add_argument("--commit-id", default="", help="Optional commit_id for ingest idempotency (defaults to session_id).")

    # behavior
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument(
        "--skip-session-write",
        action="store_true",
        help="Skip ingest; reuse existing DB state.",
    )
    p.add_argument("--backend", default="tkg", choices=["tkg", "memory"])
    p.add_argument("--tkg-explain", action="store_true", help="Enable TKG explain expansion (default off).")
    p.add_argument("--with-answer", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument(
        "--keep-proxies",
        action="store_true",
        help="Keep HTTP(S)/ALL proxy env vars (default: unset like benchmark scripts).",
    )
    p.add_argument(
        "--judge-required",
        action="store_true",
        help="Fail fast if judge cannot be initialized/evaluated (default: best-effort).",
    )
    p.add_argument("--llm-policy", default="require", choices=["require", "best_effort"])
    p.add_argument("--extract-sessions-per-call", default=4, type=int, help="Chunk multi-session extraction (env-only).")

    # LLM selection for request (BYOK vs platform)
    p.add_argument("--llm-mode", default="platform", choices=["platform", "byok"])
    p.add_argument("--byok-provider", default="")
    p.add_argument("--byok-model", default="")
    p.add_argument("--byok-api-key", default="")
    p.add_argument("--byok-base-url", default="")

    # Stage toggles (client_meta)
    p.add_argument("--stage2-enabled", default=None, type=lambda s: str(s).lower() in ("1", "true", "yes"))
    p.add_argument("--stage3-extract", default=None, type=lambda s: str(s).lower() in ("1", "true", "yes"))

    # retrieval loop
    p.add_argument("--topk", default=30, type=int)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--output-dir", default="modules/memory/outputs/e2e_conv26_http")
    p.add_argument("--log-every", default=10, type=int, help="Print progress every N queries (0 disables).")
    p.add_argument("--concurrency", default=1, type=int, help="Concurrent queries (default: 1).")
    p.add_argument("--query-retries", default=2, type=int, help="Retry count per retrieval query on transient network errors.")
    p.add_argument("--retry-backoff-seconds", default=1.0, type=float, help="Base backoff seconds between query retries.")
    p.add_argument("--fail-fast", action="store_true", help="Abort the run on first retrieval query failure.")

    args = p.parse_args()

    if not bool(args.keep_proxies):
        for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(k, None)
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
