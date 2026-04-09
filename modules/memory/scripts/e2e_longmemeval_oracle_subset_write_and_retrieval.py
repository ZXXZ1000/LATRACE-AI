#!/usr/bin/env python3
from __future__ import annotations

"""
E2E experiment for LongMemEval (oracle subset):

- Build a reproducible subset file first:
    benchmark/data/LongMemEval/make_oracle_subset.py

- Ingest (LLM extract + graph upsert + vector indexes) only for the subset:
    - tenant isolation via --tenant
    - domain isolation via --memory-domain

- Benchmark:
    - retrieval(strategy=dialog_v2, backend=tkg)
    - optional scoring (cheap string match by default; optional LLM judge if configured)
"""

if __package__ in (None, ""):
    raise SystemExit(
        "Please run as a module from repo root:\n"
        "  python -m modules.memory.scripts.e2e_longmemeval_oracle_subset_write_and_retrieval --mode ingest --extract\n"
    )

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.memory.application.config import load_memory_config
from modules.memory.application.service import MemoryService
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.retrieval import retrieval
from modules.memory.session_write import session_write


@dataclass(frozen=True)
class LongMemEvalItem:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_sessions: List[List[Dict[str, Any]]]
    haystack_session_ids: List[str]
    haystack_dates: List[str]
    answer_session_ids: List[str]


@dataclass(frozen=True)
class WorkloadItem:
    tenant_id: str
    tenant_index: int
    session_id: str
    user_id: str
    item: LongMemEvalItem


def _parse_longmemeval_datetime(dt_str: str) -> datetime:
    """
    LongMemEval date examples:
    - '2023/05/30 (Tue) 23:40'
    - '2023/02/01 (Wed) 10:20'
    """
    s = str(dt_str or "").strip()
    m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2}).*?(\d{1,2}):(\d{2})", s)
    if not m:
        return datetime(2023, 5, 1, 12, 0, tzinfo=timezone.utc)
    y, mo, d, h, mi = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def _normalize_text(s: str) -> str:
    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    # Remove common punctuation noise for cheap matching.
    t = re.sub(r"[.,;:!\?\(\)\[\]\{\}\"'`]", "", t)
    return t.strip()


def _cheap_correct(pred: str, gold: str) -> Optional[bool]:
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    if not p or not g:
        return None
    if p == g:
        return True
    # Containment heuristic: acceptable for a cheap sanity score only.
    return (g in p) or (p in g)


def _judge_verdict(j: Dict[str, Any]) -> Optional[bool]:
    """Accept multiple shapes across judge implementations (match conv26 e2e script)."""
    try:
        if "binary_correct" in j:
            return bool(float(j.get("binary_correct") or 0.0) >= 1.0)
        if "score" in j:
            return bool(float(j.get("score") or 0.0) >= 1.0)
        if "label" in j:
            return str(j.get("label") or "").strip().upper() == "CORRECT"
        if "binary_label" in j:
            return str(j.get("binary_label") or "").strip().upper() == "CORRECT"
    except Exception:
        return None
    return None


def _load_subset_items(path: Path) -> Tuple[Dict[str, Any], List[LongMemEvalItem]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        meta = dict(raw.get("meta") or {})
        items_raw = list(raw.get("items") or [])
    elif isinstance(raw, list):
        meta = {}
        items_raw = list(raw)
    else:
        raise ValueError("subset file must be a dict(meta+items) or a list(items)")

    items: List[LongMemEvalItem] = []
    for i, it in enumerate(items_raw):
        if not isinstance(it, dict):
            raise ValueError(f"item[{i}] is not a dict")
        items.append(
            LongMemEvalItem(
                question_id=str(it.get("question_id") or "").strip(),
                question_type=str(it.get("question_type") or "").strip(),
                question=str(it.get("question") or "").strip(),
                answer=str(it.get("answer") or "").strip(),
                question_date=str(it.get("question_date") or "").strip(),
                haystack_sessions=list(it.get("haystack_sessions") or []),
                haystack_session_ids=[str(x) for x in (it.get("haystack_session_ids") or [])],
                haystack_dates=[str(x) for x in (it.get("haystack_dates") or [])],
                answer_session_ids=[str(x) for x in (it.get("answer_session_ids") or [])],
            )
        )
    # Basic sanity: ids required.
    for it in items:
        if not it.question_id:
            raise ValueError("question_id is required")
    return meta, items


def _safe_label(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return out or "default"


def _build_tenant_ids(*, tenant: str, tenant_count: int, tenant_prefix: str) -> List[str]:
    count = max(1, int(tenant_count or 1))
    base = str(tenant or "").strip() or "bench-tenant"
    if count <= 1:
        return [base]
    prefix = str(tenant_prefix or "").strip() or f"{base}-t"
    return [f"{prefix}{idx:03d}" for idx in range(count)]


def build_workload_items(
    *,
    items: Sequence[LongMemEvalItem],
    tenant: str,
    tenant_count: int,
    tenant_prefix: str,
    user_prefix: str,
    session_id_mode: str,
    limit: int,
) -> List[WorkloadItem]:
    selected = list(items)[: max(0, int(limit))]
    tenants = _build_tenant_ids(tenant=tenant, tenant_count=tenant_count, tenant_prefix=tenant_prefix)
    mode = str(session_id_mode or "tenant_prefixed").strip().lower()
    if mode not in {"tenant_prefixed", "shared"}:
        mode = "tenant_prefixed"
    out: List[WorkloadItem] = []
    for tenant_index, tenant_id in enumerate(tenants):
        tenant_label = _safe_label(tenant_id)
        for item in selected:
            qid = str(item.question_id)
            session_id = qid if mode == "shared" else f"{tenant_label}::{qid}"
            user_id = f"{str(user_prefix)}{tenant_label}_{qid}"
            out.append(
                WorkloadItem(
                    tenant_id=str(tenant_id),
                    tenant_index=int(tenant_index),
                    session_id=session_id,
                    user_id=user_id,
                    item=item,
                )
            )
    return out


def build_turns_for_session_write(item: LongMemEvalItem) -> List[Dict[str, Any]]:
    hs = list(item.haystack_sessions or [])
    hs_ids = list(item.haystack_session_ids or [])
    hs_dates = list(item.haystack_dates or [])

    if not (len(hs) == len(hs_ids) == len(hs_dates)):
        raise ValueError(
            f"haystack lists must align: sessions={len(hs)} ids={len(hs_ids)} dates={len(hs_dates)}"
        )

    turns: List[Dict[str, Any]] = []
    for sess_idx, (session_msgs, sess_id, dt_str) in enumerate(zip(hs, hs_ids, hs_dates), start=1):
        base = _parse_longmemeval_datetime(dt_str)
        for i, msg in enumerate(list(session_msgs or []), start=1):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "Unknown").strip() or "Unknown"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            ts = base + timedelta(seconds=(i - 1) * 60)
            turns.append(
                {
                    "dia_id": f"{sess_id}_{i}",
                    "speaker": role,
                    "text": content,
                    "timestamp_iso": ts.isoformat(),
                    "session_idx": int(sess_idx),
                    "session_date_time": str(dt_str),
                    "blip_caption": None,
                }
            )
    if not turns:
        raise ValueError(f"no turns built for question_id={item.question_id}")
    return turns


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
            "transport": vcfg.get("transport", {}),
            "sharding": vcfg.get("sharding", {}),
            "reliability": rcfg,
        }
    )

    n_uri = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    n_user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    n_pass = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")
    neo = Neo4jStore({"uri": str(n_uri), "user": str(n_user), "password": str(n_pass), "reliability": rcfg})
    audit_path = os.getenv("MEMORY_AUDIT_SQLITE_PATH") or ":memory:"
    audit = AuditStore({"sqlite_path": str(audit_path)})
    return MemoryService(qdr, neo, audit)


def _load_llm_judge_module() -> Optional[Any]:
    # Load llm_judge.py by file path without importing package __init__.
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


def _summarize_accuracy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        qt = str(r.get("question_type") or "UNKNOWN")
        verdict = r.get("verdict")
        b = by_type.setdefault(qt, {"count": 0, "correct": 0, "skipped": 0})
        if verdict is None:
            b["skipped"] += 1
            continue
        b["count"] += 1
        b["correct"] += int(bool(verdict))
    out: Dict[str, Any] = {}
    total = 0
    ok = 0
    for qt, b in sorted(by_type.items(), key=lambda kv: kv[0]):
        c = int(b["count"])
        k = int(b["correct"])
        total += c
        ok += k
        out[qt] = {"correct": k, "count": c, "accuracy": (k / c) if c else None, "skipped": int(b["skipped"])}
    out["overall"] = {"correct": ok, "count": total, "accuracy": (ok / total) if total else None}
    return out


async def run_ingest(
    *,
    svc: MemoryService,
    items: Sequence[LongMemEvalItem],
    tenant_id: str,
    tenant_count: int,
    tenant_prefix: str,
    memory_domain: str,
    user_prefix: str,
    session_id_mode: str,
    overwrite_existing: bool,
    extract: bool,
    reuse_facts_from_artifacts: bool,
    reuse_artifact_dir: Path,
    concurrency: int,
    llm_policy: str,
    graph_policy: str,
    out_path: Path,
    artifact_dir: Path,
    limit: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workload = build_workload_items(
        items=items,
        tenant=str(tenant_id),
        tenant_count=int(tenant_count),
        tenant_prefix=str(tenant_prefix),
        user_prefix=str(user_prefix),
        session_id_mode=str(session_id_mode),
        limit=int(limit),
    )
    total_items = len(workload)
    out_path.write_text("", encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, int(concurrency or 1)))
    completed = 0

    # Thread-local service to avoid sharing requests.Session / DB clients across threads.
    _tls = threading.local()

    def _get_thread_service() -> MemoryService:
        svc0 = getattr(_tls, "svc", None)
        if svc0 is not None:
            return svc0
        # Avoid sqlite file contention in parallel ingest; artifacts are already persisted on disk.
        os.environ["MEMORY_AUDIT_SQLITE_PATH"] = ":memory:"
        svc1 = build_service_from_env()
        _tls.svc = svc1
        return svc1

    def _load_reuse_facts(qid: str) -> tuple[list[dict], list[dict]]:
        trace: list[dict] = []
        facts: list[dict] = []
        prior = Path(reuse_artifact_dir) / f"{qid}.json"
        if not prior.exists():
            matches = sorted(Path(reuse_artifact_dir).glob(f"*__{_safe_label(qid)}.json"))
            prior = matches[0] if matches else prior
        if not prior.exists():
            trace.append({"stage": "reuse_facts_missing"})
            return facts, trace
        try:
            prev = json.loads(prior.read_text(encoding="utf-8"))
            prev_facts = prev.get("extracted_facts") or []
            if isinstance(prev_facts, list):
                facts = [dict(x) for x in prev_facts if isinstance(x, dict)]
            trace.append({"stage": "reuse_facts", "facts": len(facts), "path": str(prior)})
            return facts, trace
        except Exception as exc:
            trace.append({"stage": "reuse_facts_error", "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
            return facts, trace

    def _ingest_one_sync(work: WorkloadItem) -> Dict[str, Any]:
        it = work.item
        user_id = str(work.user_id)
        session_id = str(work.session_id)
        tenant_value = str(work.tenant_id)
        turns = build_turns_for_session_write(it)
        reference_time_iso = _parse_longmemeval_datetime(it.question_date).isoformat()

        extract_trace: List[Dict[str, Any]] = []
        extracted_facts: List[Dict[str, Any]] = []
        extra_facts: Optional[List[Dict[str, Any]]] = None
        tkg_extractor = None

        if bool(reuse_facts_from_artifacts):
            facts, trace = _load_reuse_facts(str(it.question_id))
            extra_facts = list(facts) if facts else None
            extract_trace.extend(list(trace))
            extracted_facts = list(facts)

        if bool(extract):
            try:
                from modules.memory.application.dialog_tkg_unified_extractor_v1 import (
                    build_dialog_tkg_unified_extractor_v1_from_env,
                )

                def _trace_hook(payload: Dict[str, Any]) -> None:
                    extract_trace.append(dict(payload))

                base_extractor = build_dialog_tkg_unified_extractor_v1_from_env(
                    session_id=session_id,
                    reference_time_iso=str(reference_time_iso),
                    trace_hook=_trace_hook,
                    trace_include_context=bool(os.getenv("LONGMEMEVAL_TRACE_INCLUDE_CONTEXT") == "1"),
                )
                if base_extractor is not None:

                    def _extractor(turns_in: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
                        nonlocal extracted_facts
                        payload = base_extractor(list(turns_in))
                        if isinstance(payload, dict):
                            extracted_facts = list(payload.get("knowledge") or [])
                        else:
                            extracted_facts = []
                        return {"events": list(payload.get("events") or []) if isinstance(payload, dict) else [], "knowledge": list(extracted_facts)}

                    tkg_extractor = _extractor
            except Exception as exc:
                extract_trace.append({"stage": "extractor_build_error", "error": f"{type(exc).__name__}: {str(exc)[:240]}"})

        # Run async session_write in this thread.
        svc_t = _get_thread_service()

        async def _call() -> Dict[str, Any]:
            return await session_write(
                svc_t,
                tenant_id=tenant_value,
                user_tokens=[user_id],
                session_id=session_id,
                turns=turns,
                memory_domain=str(memory_domain),
                overwrite_existing=bool(overwrite_existing),
                extract=bool(extract),
                llm_policy=str(llm_policy),
                graph_policy=str(graph_policy),
                tkg_extractor=tkg_extractor,
                reference_time_iso=str(reference_time_iso),
                extra_facts=extra_facts,
            )

        started = time.time()
        try:
            res = asyncio.run(_call())
        except Exception as exc:
            res = {
                "status": "failed",
                "session_id": session_id,
                "trace": {"error": f"{type(exc).__name__}: {str(exc)[:240]}", "elapsed_s": time.time() - started},
            }

        return {
            "workload": work,
            "user_id": user_id,
            "session_id": session_id,
            "tenant_id": tenant_value,
            "turns": turns,
            "reference_time_iso": reference_time_iso,
            "extract_trace": extract_trace,
            "extracted_facts": extracted_facts,
            "session_write": res,
        }

    async def _run_one(work: WorkloadItem) -> None:
        nonlocal completed
        async with sem:
            result = await asyncio.to_thread(_ingest_one_sync, work)
            work0: WorkloadItem = result["workload"]
            it0: LongMemEvalItem = work0.item
            user_id = str(result["user_id"])
            session_id = str(result["session_id"])
            tenant_value = str(result["tenant_id"])
            turns = list(result["turns"])
            reference_time_iso = str(result["reference_time_iso"])
            extract_trace = list(result["extract_trace"])
            extracted_facts = list(result["extracted_facts"])
            res = result["session_write"]

            artifact_label = f"{_safe_label(tenant_value)}__{_safe_label(session_id)}"
            artifact_path = artifact_dir / f"{artifact_label}.json"
            artifact: Dict[str, Any] = {
                "question_id": it0.question_id,
                "question_type": it0.question_type,
                "question": it0.question,
                "question_date": it0.question_date,
                "ground_truth": it0.answer,
                "tenant_id": tenant_value,
                "tenant_index": int(work0.tenant_index),
                "memory_domain": str(memory_domain),
                "user_id": str(user_id),
                "session_id": session_id,
                "facts_source": ("llm" if bool(extract) else ("reuse" if bool(reuse_facts_from_artifacts) else "none")),
                "extract_trace": list(extract_trace),
                "extracted_facts": list(extracted_facts),
                "session_write": res,
            }

            try:
                from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1

                g = build_dialog_graph_upsert_v1(
                    tenant_id=tenant_value,
                    session_id=session_id,
                    user_tokens=[user_id],
                    turns=list(turns),
                    memory_domain=str(memory_domain),
                    facts_raw=list(extracted_facts),
                    turn_marks_by_index=None,
                    reference_time_iso=str(reference_time_iso),
                    turn_interval_seconds=60,
                    tenant_scoped_fact_ids=bool(getattr(svc.vectors, "tenant_scoped_ids_enabled", lambda: False)()),
                )
                req = g.request
                try:
                    artifact["graph_upsert_request"] = req.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        artifact["graph_upsert_request"] = dict(req)  # type: ignore[arg-type]
                    except Exception:
                        artifact["graph_upsert_request"] = str(req)
                artifact["graph_ids"] = g.graph_ids
            except Exception as exc:
                artifact["graph_build_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"

            artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

            row = {
                "question_id": it0.question_id,
                "question_type": it0.question_type,
                "tenant_id": tenant_value,
                "session_id": session_id,
                "user_id": user_id,
                "artifact_path": str(artifact_path),
                "artifact_sha256": sha,
                "session_write": res,
            }

            async with lock:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                completed += 1
                if completed == 1 or completed == total_items or completed % 10 == 0:
                    print(f"ingest progress {completed}/{total_items}", flush=True)

    await asyncio.gather(*[_run_one(work) for work in workload])


async def run_benchmark(
    *,
    svc: MemoryService,
    items: Sequence[LongMemEvalItem],
    tenant_id: str,
    tenant_count: int,
    tenant_prefix: str,
    memory_domain: str,
    user_prefix: str,
    session_id_mode: str,
    topk: int,
    strategy: str,
    backend: str,
    with_answer: bool,
    llm_policy: str,
    judge_enabled: bool,
    judge_required: bool,
    concurrency: int,
    use_question_date_hint: bool,
    out_path: Path,
    limit: int,
) -> Dict[str, Any]:
    judge = build_judge_from_memory_config() if judge_enabled else None
    if judge_enabled and judge is None:
        msg = "judge not configured; skipping accuracy scoring (no fallback)"
        if judge_required:
            raise RuntimeError(msg)
        print(f"⚠️ {msg}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, int(concurrency or 1)))
    workload = build_workload_items(
        items=items,
        tenant=str(tenant_id),
        tenant_count=int(tenant_count),
        tenant_prefix=str(tenant_prefix),
        user_prefix=str(user_prefix),
        session_id_mode=str(session_id_mode),
        limit=int(limit),
    )
    total_items = len(workload)
    completed = 0
    started = asyncio.get_event_loop().time()
    last_print = 0
    log_every = int(os.getenv("LONGMEMEVAL_LOG_EVERY", "10"))

    judged_total = 0
    judged_correct = 0
    by_type_running: Dict[str, Dict[str, int]] = {}

    def _maybe_print_progress() -> None:
        nonlocal last_print
        if log_every <= 0:
            return
        if completed == total_items or completed == 1 or (completed - last_print) >= log_every:
            last_print = completed
        else:
            return

        elapsed_s = max(0.0, asyncio.get_event_loop().time() - started)
        total_ms = elapsed_s * 1000.0
        if judged_total > 0:
            acc = judged_correct / judged_total
            acc_msg = f" acc={acc * 100:.1f}%"
            bits: List[str] = []
            for qt in sorted(by_type_running.keys()):
                b = by_type_running[qt]
                c = int(b.get("count") or 0)
                ok = int(b.get("correct") or 0)
                if c:
                    bits.append(f"{qt}={(ok / c) * 100:.1f}%")
            task_msg = f" types[{', '.join(bits)}]" if bits else ""
        else:
            acc_msg = " acc=n/a"
            task_msg = ""
        print(f"progress {completed}/{total_items} total_ms={total_ms:.1f}{acc_msg}{task_msg}", flush=True)

    async def _run_one(work: WorkloadItem) -> None:
        nonlocal completed, judged_total, judged_correct, by_type_running
        async with sem:
            it = work.item
            user_id = str(work.user_id)
            tenant_value = str(work.tenant_id)
            session_id = str(work.session_id)
            score_meta: Dict[str, Any] = {}
            res: Dict[str, Any] = {}
            pred = ""
            verdict: Optional[bool] = None
            error: Optional[str] = None

            try:
                th = {"question_date": str(it.question_date)} if bool(use_question_date_hint) else None
                res = await retrieval(
                    svc,
                    tenant_id=tenant_value,
                    user_tokens=[user_id],
                    query=str(it.question),
                    strategy=str(strategy),
                    memory_domain=str(memory_domain),
                    user_match="all",
                    run_id=session_id,
                    topk=int(topk),
                    debug=True,
                    with_answer=bool(with_answer),
                    task=str(it.question_type or "GENERAL"),
                    llm_policy=str(llm_policy),
                    backend=str(backend),
                    tkg_explain=True,
                    time_hints=th,
                )
                pred = str(res.get("answer") or "")
            except Exception as exc:
                error = f"retrieval_error: {type(exc).__name__}: {str(exc)[:240]}"
                pred = ""

            if error is None and judge is not None and with_answer:
                try:
                    jr = await asyncio.to_thread(judge.evaluate_binary, it.question, pred, it.answer)
                    j_score = jr.to_dict()
                    score_meta["j_score"] = j_score
                    verdict = _judge_verdict(j_score)
                except Exception as exc:
                    score_meta["j_score_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                    if judge_required:
                        raise
                    verdict = None
            elif error is None and not judge_enabled:
                # Debug-only fallback when explicitly requested. Do NOT silently use this for "accuracy".
                verdict = _cheap_correct(pred, it.answer)

            row: Dict[str, Any] = {
                "question_id": it.question_id,
                "question_type": it.question_type,
                "tenant_id": tenant_value,
                "tenant_index": int(work.tenant_index),
                "session_id": session_id,
                "user_id": user_id,
                "question": it.question,
                "ground_truth": it.answer,
                "pred_answer": pred,
                "verdict": verdict,
                "error": error,
                "meta": {
                    "answer_session_ids": list(it.answer_session_ids),
                    "haystack_session_ids": list(it.haystack_session_ids),
                },
                "debug": (res.get("debug") or {}) if isinstance(res, dict) else {},
                **score_meta,
            }

            async with lock:
                rows.append(row)
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                completed += 1
                if verdict is None:
                    pass
                else:
                    judged_total += 1
                    judged_correct += int(bool(verdict))
                    bucket = by_type_running.setdefault(str(it.question_type or "UNKNOWN"), {"count": 0, "correct": 0})
                    bucket["count"] += 1
                    bucket["correct"] += int(bool(verdict))
                _maybe_print_progress()

    out_path.write_text("", encoding="utf-8")
    await asyncio.gather(*[_run_one(work) for work in workload])
    return {"summary": _summarize_accuracy(rows), "rows": len(rows), "judge_used": bool(judge is not None)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LongMemEval oracle subset: ingest + retrieval benchmark (E2E).")
    ap.add_argument(
        "--subset",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "benchmark" / "data" / "LongMemEval" / "longmemeval_oracle.subset_200_seed42.json"),
        help="Path to subset JSON (generated by benchmark/data/LongMemEval/make_oracle_subset.py)",
    )
    ap.add_argument("--mode", type=str, default="all", choices=["ingest", "benchmark", "all"])
    ap.add_argument("--tenant", type=str, default="bench-longmemeval-oracle-200-seed42")
    ap.add_argument("--tenant-count", type=int, default=1, help="Number of isolated tenants to generate in the workload.")
    ap.add_argument(
        "--tenant-prefix",
        type=str,
        default="",
        help="Tenant prefix when --tenant-count > 1 (default: <tenant>-t).",
    )
    ap.add_argument("--memory-domain", type=str, default="longmemeval_oracle")
    ap.add_argument("--user-prefix", type=str, default="lm_user_")
    ap.add_argument(
        "--session-id-mode",
        type=str,
        default="tenant_prefixed",
        choices=["tenant_prefixed", "shared"],
        help="Use tenant-prefixed session ids by default to avoid cross-tenant point-id collisions in stress runs.",
    )
    ap.add_argument(
        "--config-profile",
        type=str,
        default="",
        help="Set MEMORY_CONFIG_PROFILE for this run (loads modules/memory/config/memory.config.<profile>.yaml).",
    )
    ap.add_argument("--limit", type=int, default=200, help="Max questions to run from the subset (default: 200)")
    ap.add_argument("--overwrite-existing", action="store_true")

    ap.add_argument("--extract", action="store_true", help="Enable LLM fact extraction during ingest")
    ap.add_argument(
        "--reuse-facts-from-artifacts",
        action="store_true",
        help="During ingest, reuse extracted_facts from artifacts to rebuild graph/index without calling the LLM.",
    )
    ap.add_argument(
        "--reuse-artifact-dir",
        type=str,
        default="",
        help="Directory to read prior artifacts from (default: <out-dir>/artifacts).",
    )
    ap.add_argument("--ingest-concurrency", type=int, default=1, help="Parallelism for ingest (default: 1)")
    ap.add_argument("--extract-sessions-per-call", type=int, default=4, help="MEMORY_DIALOG_TKG_EXTRACT_SESSIONS_PER_CALL")
    ap.add_argument("--llm-policy", type=str, default="best_effort", choices=["require", "best_effort"])
    ap.add_argument("--graph-policy", type=str, default="best_effort", choices=["require", "best_effort"])

    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--strategy", type=str, default="dialog_v2", choices=["dialog_v1", "dialog_v2"])
    ap.add_argument("--backend", type=str, default="tkg", choices=["tkg", "memory"])
    ap.add_argument(
        "--no-with-answer",
        action="store_true",
        help="Disable QA answer generation (retrieval returns evidence only; accuracy will be skipped).",
    )

    jg = ap.add_mutually_exclusive_group()
    jg.add_argument("--judge", dest="judge", action="store_true", help="Enable LLM judge for accuracy scoring (default).")
    jg.add_argument(
        "--no-judge",
        dest="judge",
        action="store_false",
        help="Disable LLM judge; use cheap string-match scoring for quick sanity only.",
    )
    ap.set_defaults(judge=True)
    ap.add_argument("--judge-required", action="store_true", help="Fail if judge isn't available")
    ap.add_argument("--benchmark-concurrency", type=int, default=1, help="Parallelism for retrieval benchmark (default: 1)")
    ap.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print running accuracy every N items during benchmark (0 disables).",
    )
    ap.add_argument(
        "--no-question-date-hint",
        action="store_true",
        help="Do not pass LongMemEval question_date into retrieval time_hints (disables Current Date injection).",
    )
    ap.add_argument(
        "--keep-proxies",
        action="store_true",
        help="Keep HTTP(S)/ALL proxy env vars (default: unset like other benchmark scripts to avoid SOCKS dependency issues).",
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "benchmark" / "outputs" / "longmemeval_oracle_subset"),
    )
    ap.add_argument(
        "--audit-db",
        type=str,
        default="",
        help="SQLite audit db path (default: <out-dir>/audit.db). Stores artifact index; large payloads stay on disk.",
    )
    args = ap.parse_args()

    if str(getattr(args, "config_profile", "") or "").strip():
        os.environ["MEMORY_CONFIG_PROFILE"] = str(args.config_profile).strip()

    if not bool(args.keep_proxies):
        # Disable proxy env to avoid OpenAI/httpx treating local SOCKS proxy as required (needs socksio).
        for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
            os.environ.pop(k, None)
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    # Used inside run_benchmark; keep the function signature minimal.
    os.environ["LONGMEMEVAL_LOG_EVERY"] = str(int(args.log_every or 0))

    if int(args.extract_sessions_per_call or 0) > 0:
        os.environ["MEMORY_DIALOG_TKG_EXTRACT_SESSIONS_PER_CALL"] = str(int(args.extract_sessions_per_call))

    subset_path = Path(str(args.subset)).expanduser().resolve()
    meta, items = _load_subset_items(subset_path)

    limit = min(int(args.limit), len(items))
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_db = Path(str(args.audit_db)).expanduser().resolve() if str(args.audit_db).strip() else (out_dir / "audit.db")
    os.environ["MEMORY_AUDIT_SQLITE_PATH"] = str(audit_db)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "subset_meta": meta,
                "workload": {
                    "tenant": str(args.tenant),
                    "tenant_count": int(args.tenant_count),
                    "tenant_prefix": str(args.tenant_prefix),
                    "session_id_mode": str(args.session_id_mode),
                    "limit_per_tenant": int(limit),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    svc = build_service_from_env()

    async def _run() -> Dict[str, Any]:
        if str(args.mode) in ("ingest", "all"):
            ingest_path = out_dir / "ingest.jsonl"
            artifact_dir = out_dir / "artifacts"
            reuse_artifact_dir = (
                Path(str(args.reuse_artifact_dir)).expanduser().resolve()
                if str(args.reuse_artifact_dir).strip()
                else artifact_dir
            )
            await run_ingest(
                svc=svc,
                items=items,
                tenant_id=str(args.tenant),
                tenant_count=int(args.tenant_count),
                tenant_prefix=str(args.tenant_prefix),
                memory_domain=str(args.memory_domain),
                user_prefix=str(args.user_prefix),
                session_id_mode=str(args.session_id_mode),
                overwrite_existing=bool(args.overwrite_existing),
                extract=bool(args.extract),
                reuse_facts_from_artifacts=bool(args.reuse_facts_from_artifacts),
                reuse_artifact_dir=reuse_artifact_dir,
                concurrency=int(args.ingest_concurrency),
                llm_policy=str(args.llm_policy),
                graph_policy=str(args.graph_policy),
                out_path=ingest_path,
                artifact_dir=artifact_dir,
                limit=limit,
            )
            print(f"✓ ingest done: {ingest_path}", flush=True)
            print(f"✓ artifacts: {artifact_dir}", flush=True)
            if bool(args.reuse_facts_from_artifacts):
                print(f"✓ reuse artifacts from: {reuse_artifact_dir}", flush=True)
            print(f"✓ audit db: {audit_db}", flush=True)

        if str(args.mode) in ("benchmark", "all"):
            bench_path = out_dir / "results.jsonl"
            agg = await run_benchmark(
                svc=svc,
                items=items,
                tenant_id=str(args.tenant),
                tenant_count=int(args.tenant_count),
                tenant_prefix=str(args.tenant_prefix),
                memory_domain=str(args.memory_domain),
                user_prefix=str(args.user_prefix),
                session_id_mode=str(args.session_id_mode),
                topk=int(args.topk),
                strategy=str(args.strategy),
                backend=str(args.backend),
                with_answer=not bool(args.no_with_answer),
                llm_policy=str(args.llm_policy),
                judge_enabled=bool(args.judge),
                judge_required=bool(args.judge_required),
                concurrency=int(args.benchmark_concurrency),
                use_question_date_hint=not bool(args.no_question_date_hint),
                out_path=bench_path,
                limit=limit,
            )
            (out_dir / "summary.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✓ benchmark done: {bench_path}", flush=True)
            print(f"✓ summary: {out_dir / 'summary.json'}", flush=True)
            return agg
        return {"status": "ok"}

    agg = asyncio.run(_run())
    if isinstance(agg, dict) and isinstance(agg.get("summary"), dict):
        overall = (agg.get("summary") or {}).get("overall") or {}
        acc = overall.get("accuracy")
        if isinstance(acc, float):
            print(f"overall_accuracy={acc * 100:.2f}% (count={overall.get('count')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
