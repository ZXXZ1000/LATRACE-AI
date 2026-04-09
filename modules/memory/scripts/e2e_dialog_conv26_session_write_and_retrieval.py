#!/usr/bin/env python3
from __future__ import annotations

"""E2E sanity run for LoCoMo using high-level APIs:

- Write: modules.memory.session_write(...) (TKG-first; real LLM knowledge extraction)
- Retrieve: modules.memory.retrieval(...) (dialog_v1; backend=tkg)

This script is intentionally a thin orchestration layer to validate:
LLM extraction -> TKG graph upsert -> vector indexes -> retrieval(+QA) -> (optional) judge.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modules.memory.application.config import load_memory_config
from modules.memory.application.service import MemoryService
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.retrieval import retrieval
from modules.memory.session_write import session_write
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


def build_service_from_env() -> MemoryService:
    cfg = load_memory_config()
    vcfg = cfg.get("memory", {}).get("vector_store", {})
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    rcfg = cfg.get("memory", {}).get("reliability", {})

    q_host = os.getenv("QDRANT_HOST") or vcfg.get("host", "127.0.0.1")
    q_port = os.getenv("QDRANT_PORT") or vcfg.get("port", 6333)
    q_api = os.getenv("QDRANT_API_KEY") or vcfg.get("api_key", "")
    qdr = QdrantStore(
        {
            "host": str(q_host),
            "port": int(q_port),
            "api_key": str(q_api),
            "collections": vcfg.get("collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}),
            "embedding": vcfg.get("embedding", {}),
            "reliability": rcfg,
        }
    )

    n_uri = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    n_user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    n_pass = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")
    neo = Neo4jStore({"uri": str(n_uri), "user": str(n_user), "password": str(n_pass), "reliability": rcfg})
    audit = AuditStore()
    return MemoryService(qdr, neo, audit)


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
        "min_ms": min(xs),
        "max_ms": max(xs),
        "mean_ms": (sum(xs) / len(xs)),
        "p50_ms": _pct(xs, 50),
        "p90_ms": _pct(xs, 90),
        "p95_ms": _pct(xs, 95),
        "p99_ms": _pct(xs, 99),
    }


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
    lines.append(f"# E2E Report: session_write + retrieval ({aggregate.get('sample_id')})")
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
    lines.append("## Latency (ms)")
    for name, stats in (aggregate.get("latency") or {}).items():
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
        # Qwen via DashScope
        backend = "dashscope"

    if backend is None:
        return None

    try:
        cfg_j = JudgeConfig(backend=backend, model=model)  # type: ignore[call-arg]
        return LLMJudge(cfg_j)
    except Exception:
        return None


def _resolve_queries_path(queries_arg: str, sample_id: str) -> Path:
    """
    Backward compatible:
    - If `--queries` points to a file, use it directly.
    - If it points to a directory, assume LoCoMo layout:
        <queries_dir>/<sample_id>/queries.jsonl
    """
    p = Path(str(queries_arg))
    if p.is_file():
        return p
    if p.is_dir():
        return p / sample_id / "queries.jsonl"
    # If it doesn't exist yet, treat it as a directory path for future output.
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


async def _run_one_sample(args: argparse.Namespace, sample_id: str, out_dir: Path) -> int:
    sample = _load_locomo_sample(Path(args.input), str(sample_id))

    tenant_id = str(args.tenant)
    session_id = str(sample_id)
    user_id = f"{str(args.user_prefix).rstrip('_')}_{session_id}"

    svc = build_service_from_env()

    # Optional: chunk multi-session extraction to avoid context overflow (no API change; env only).
    if int(args.extract_sessions_per_call or 0) > 0:
        os.environ["MEMORY_DIALOG_TKG_EXTRACT_SESSIONS_PER_CALL"] = str(int(args.extract_sessions_per_call))

    write_latency_ms = 0.0
    if bool(args.skip_session_write):
        print("↪ session_write skipped (--skip-session-write)", flush=True)
    else:
        turns = build_turns_for_session_write(sample)
        t0 = asyncio.get_event_loop().time()
        w = await session_write(
            svc,
            tenant_id=tenant_id,
            user_tokens=[user_id],
            session_id=session_id,
            turns=turns,
            memory_domain=str(args.memory_domain),
            overwrite_existing=bool(args.overwrite_existing),
            graph_policy=str(args.graph_policy),
            llm_policy=str(args.llm_policy),
        )
        write_latency_ms = (asyncio.get_event_loop().time() - t0) * 1000.0
        print(f"✓ session_write: status={w['status']} marker={w['marker_id']} version={w.get('version')}", flush=True)

    queries_path = _resolve_queries_path(str(args.queries), session_id)
    queries = load_locomo_queries(queries_path)
    if args.limit and int(args.limit) > 0:
        queries = queries[: int(args.limit)]
    print(f"✓ queries loaded: {len(queries)} ({queries_path})", flush=True)

    judge = None
    judge_error: str | None = None
    if bool(args.judge):
        try:
            judge_mod = _load_llm_judge_module()
            create_judge_from_env = getattr(judge_mod, "create_judge_from_env", None) if judge_mod else None

            # 1) 优先使用 memory.config.yaml 中的 llm.judge 配置
            judge = build_judge_from_memory_config()
            # 2) 若配置缺失或构造失败，则回退到旧的 env 路径
            if judge is None and callable(create_judge_from_env):
                judge = create_judge_from_env()
            
            if judge is None:
                 print("⚠️ judge not configured in memory.config.yaml or env, skipping.", flush=True)
        except ImportError as e:
            import traceback
            traceback.print_exc()
            judge = None
            judge_error = f"{type(e).__name__}: {str(e)[:240]}"
            if bool(args.judge_required):
                raise
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

    async def _run_one_query(q: LocomoQuery) -> None:
        nonlocal completed, correct, total, judge_error
        async with sem:
            q0 = asyncio.get_event_loop().time()
            res = await retrieval(
                svc,
                tenant_id=tenant_id,
                user_tokens=[user_id],
                query=q.text,
                strategy="dialog_v2",
                memory_domain=str(args.memory_domain),
                user_match="all",
                run_id=session_id,
                topk=int(args.topk),
                debug=True,
                with_answer=bool(args.with_answer),
                task=str(q.task),
                llm_policy=str(args.llm_policy),
                backend=str(args.backend),
                tkg_explain=bool(args.tkg_explain),
            )
            q_wall_ms = (asyncio.get_event_loop().time() - q0) * 1000.0

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
        if concurrency <= 1:
            for q in queries:
                await _run_one_query(q)
        else:
            tasks = [asyncio.create_task(_run_one_query(q)) for q in queries]
            for t in asyncio.as_completed(tasks):
                await t

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
        "judge_enabled": bool(judge is not None),
        "judge_error": judge_error,
        "judged": total,
        "correct": correct,
        "accuracy": (correct / total if total else None),
        "by_task": (_task_summary(results_rows) if bool(judge is not None) and bool(args.with_answer) else {}),
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
    p = argparse.ArgumentParser(description="E2E LoCoMo using memory.session_write + memory.retrieval (TKG-first).")
    p.add_argument("--input", default="benchmark/data/locomo/raw/locomo10.json")
    p.add_argument(
        "--queries",
        default="benchmark/data/locomo/step1_events",
        help="Path to queries.jsonl OR a directory that contains <sample_id>/queries.jsonl.",
    )
    p.add_argument("--sample-id", default="conv-26", help='LoCoMo sample_id (e.g. conv-26). Use "0" to run all.')
    p.add_argument("--tenant", default="locomo_bench")
    p.add_argument("--user-prefix", default="locomo_user")
    p.add_argument("--memory-domain", default="dialog")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument(
        "--skip-session-write",
        action="store_true",
        help="Skip session_write (no LLM extraction / no vector+graph writes); reuse existing DB state.",
    )
    p.add_argument("--backend", default="tkg", choices=["tkg", "memory"])
    p.add_argument("--tkg-explain", action="store_true", help="Enable TKG explain expansion (default off in this script).")
    p.add_argument("--with-answer", action="store_true")
    p.add_argument("--judge", action="store_true")
    p.add_argument(
        "--keep-proxies",
        action="store_true",
        help="Keep HTTP(S)/ALL proxy env vars (default: unset like benchmark scripts to avoid SOCKS dependency issues).",
    )
    p.add_argument(
        "--judge-required",
        action="store_true",
        help="Fail fast if judge cannot be initialized/evaluated (default: best-effort, keep results without j_score).",
    )
    p.add_argument("--llm-policy", default="require", choices=["require", "best_effort"])
    p.add_argument("--graph-policy", default="require", choices=["require", "best_effort"])
    p.add_argument("--extract-sessions-per-call", default=4, type=int, help="Chunk multi-session extraction (env-only).")
    p.add_argument("--topk", default=30, type=int)
    p.add_argument("--limit", default=0, type=int)
    p.add_argument("--output-dir", default="modules/memory/outputs/e2e_conv26")
    p.add_argument("--log-every", default=10, type=int, help="Print progress every N queries (0 disables).")
    p.add_argument("--concurrency", default=1, type=int, help="Concurrent queries (default: 1). Start with 20 to probe rate limits.")
    args = p.parse_args()

    if not bool(args.keep_proxies):
        # Match benchmark scripts: disable proxy env to avoid local/LLM requests being routed unexpectedly.
        for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(k, None)
        # Keep local services always direct.
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
