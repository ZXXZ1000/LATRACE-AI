from __future__ import annotations

"""
E2E contract probes for ingest pipeline semantics.

This script is intentionally small and "human-verifiable":
  A) Failure Path: graph upsert succeeds, vector write fails -> data must remain unpublished/unqueryable.
  B) Incremental Commit: same session commits twice -> server cursor advances; no re-submit of old turns.

Notes:
- Uses best-effort flags via client_meta to avoid calling external LLMs.
- Requires local Qdrant + Neo4j running, and memory FastAPI server reachable.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
import socket

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase  # type: ignore

from modules.memory.application.config import load_memory_config
from modules.memory.application.service import MemoryService
from modules.memory.application.graph_service import GraphService
from modules.memory.api.server import ingest_store as _ingest_store
from modules.memory.api.server import _run_ingest_job as _run_ingest_job  # type: ignore
from modules.memory.contracts.memory_models import SearchFilters
from modules.memory.domain.dialog_tkg_graph_v1 import build_dialog_graph_upsert_v1
from modules.memory.infra.audit_store import AuditStore
from modules.memory.infra.neo4j_store import Neo4jStore
from modules.memory.infra.qdrant_store import QdrantStore
from modules.memory.session_write import session_write


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cfg() -> Dict[str, Any]:
    return load_memory_config()


def _qdrant_store() -> QdrantStore:
    cfg = _cfg()
    vcfg = cfg.get("memory", {}).get("vector_store", {})
    rcfg = cfg.get("memory", {}).get("reliability", {})
    q_host = os.getenv("QDRANT_HOST") or vcfg.get("host", "127.0.0.1")
    q_port = os.getenv("QDRANT_PORT") or vcfg.get("port", 6333)
    q_api = os.getenv("QDRANT_API_KEY") or vcfg.get("api_key", "")
    return QdrantStore(
        {
            "host": str(q_host),
            "port": int(q_port),
            "api_key": str(q_api),
            "collections": vcfg.get(
                "collections", {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"}
            ),
            "embedding": vcfg.get("embedding", {}),
            "reliability": rcfg,
        }
    )


def _neo4j_store() -> Neo4jStore:
    cfg = _cfg()
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    rcfg = cfg.get("memory", {}).get("reliability", {})
    n_uri = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    n_user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    n_pass = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")
    neo = Neo4jStore(
        {
            "uri": str(n_uri),
            "user": str(n_user),
            "password": str(n_pass),
            "reliability": rcfg,
        }
    )
    try:
        neo.ensure_schema_v0()
    except Exception:
        pass
    return neo


class _FailOnUpsertVectors:
    def __init__(self, inner: QdrantStore) -> None:
        self._inner = inner

    async def ensure_collections(self) -> None:
        return await self._inner.ensure_collections()

    async def search_vectors(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.search_vectors(*args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.get(*args, **kwargs)

    async def set_published(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.set_published(*args, **kwargs)

    async def delete_ids(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.delete_ids(*args, **kwargs)

    async def health(self) -> Any:
        return await self._inner.health()

    async def upsert_vectors(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected_failure: upsert_vectors")


async def probe_failure_path(*, tenant_id: str, user_token: str) -> Dict[str, Any]:
    """Graph succeeds, vectors fail; graph data must be unpublished and filtered from explain/search."""
    qdr = _qdrant_store()
    neo = _neo4j_store()
    svc = MemoryService(_FailOnUpsertVectors(qdr), neo, AuditStore())

    if hasattr(qdr, "ensure_collections"):
        await qdr.ensure_collections()

    session_id = f"e2e_fail_{uuid.uuid4().hex[:6]}"
    turns = [{"turn_id": "t0001", "role": "user", "text": "我不吃香菜。"}]

    # Pre-compute deterministic event_id for assertions.
    build = build_dialog_graph_upsert_v1(
        tenant_id=tenant_id,
        session_id=session_id,
        user_tokens=[user_token],
        turns=list(turns),
        memory_domain="dialog",
        facts_raw=[],
        reference_time_iso=None,
        turn_interval_seconds=60,
    )
    event_id = str((build.graph_ids.get("event_ids") or [""])[0])

    error: Optional[str] = None
    try:
        await session_write(
            svc,
            tenant_id=tenant_id,
            user_tokens=[user_token],
            session_id=session_id,
            turns=list(turns),
            memory_domain="dialog",
            llm_policy="best_effort",
            extract=False,
            write_facts=False,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"

    # 1) Direct Neo4j check: node exists and published=false
    cfg = _cfg()
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    n_uri = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    n_user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    n_pass = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")

    exists = False
    published = None
    with GraphDatabase.driver(str(n_uri), auth=(str(n_user), str(n_pass))) as driver:
        with driver.session(database=str(getattr(neo, "_database", "neo4j"))) as sess:  # type: ignore[attr-defined]
            row = sess.run(
                "MATCH (e:Event {tenant_id:$t, id:$id}) RETURN count(e) AS c, e.published AS p",
                t=str(tenant_id),
                id=str(event_id),
            ).single()
            if row:
                exists = int(row.get("c") or 0) > 0
                published = row.get("p")

    # 2) Public graph view must filter it out (published=false)
    graph = GraphService(svc.graph)
    explain = await graph.explain_event_evidence(tenant_id=tenant_id, event_id=str(event_id))
    filtered_out = explain.get("event") is None

    # 3) Search should not return anything for this session scope (vectors failed anyway, but filter still holds)
    try:
        sr = await svc.search(
            "香菜",
            topk=3,
            filters=SearchFilters(
                modality=["text"],
                tenant_id=tenant_id,
                user_id=[user_token],
                user_match="all",
                memory_domain="dialog",
                run_id=session_id,
            ),
        )
        search_hits = len(sr.hits or [])
    except Exception:
        search_hits = -1

    # 4) Direct Qdrant count for run_id (should be 0 when vector write failed)
    vector_count = None
    try:
        filt = {
            "must": [
                {"key": "metadata.tenant_id", "match": {"value": str(tenant_id)}},
                {"key": "metadata.user_id", "match": {"value": str(user_token)}},
                {"key": "metadata.memory_domain", "match": {"value": "dialog"}},
                {"key": "metadata.run_id", "match": {"value": str(session_id)}},
            ],
            "must_not": [{"key": "published", "match": {"value": False}}],
        }
        url = f"{qdr.base}/collections/{qdr.collections.get('text', 'memory_text')}/points/count"
        resp = qdr.session.post(url, json={"filter": filt, "exact": True}, timeout=10)
        if resp.ok:
            vector_count = int((resp.json().get("result") or {}).get("count") or 0)
    except Exception:
        vector_count = None

    return {
        "probe": "failure_path",
        "tenant_id": tenant_id,
        "session_id": session_id,
        "event_id": event_id,
        "write_error": error,
        "neo4j_event_exists": exists,
        "neo4j_event_published": published,
        "graph_explain_filtered_out": filtered_out,
        "search_hits": search_hits,
        "vector_count": vector_count,
    }


@dataclass(frozen=True)
class _CommitResult:
    job_id: str
    session_id: str


async def _http_post_json(
    client: httpx.AsyncClient, path: str, body: Dict[str, Any], *, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    r = await client.post(path, json=body, headers=headers)
    r.raise_for_status()
    return dict(r.json() or {})


async def _http_get_json(
    client: httpx.AsyncClient, path: str, *, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    r = await client.get(path, headers=headers)
    r.raise_for_status()
    return dict(r.json() or {})


async def _check_health_json(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=2.5, trust_env=False) as client:
            r = await client.get("/health")
            if r.status_code >= 400:
                return False
            data = r.json()
            return isinstance(data, dict) and "vectors" in data and "graph" in data
    except Exception:
        return False


async def _ensure_server(base_url: str) -> tuple[str, Optional[subprocess.Popen]]:
    """Ensure a FastAPI server is reachable; if not, start an in-process uvicorn."""
    ok = await _check_health_json(base_url)
    if ok:
        return base_url.rstrip("/"), None

    def _port_open(p: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", int(p)), timeout=0.2):
                return True
        except Exception:
            return False

    port = 18080
    candidate = ""
    for _ in range(50):
        if not _port_open(port):
            candidate = f"http://127.0.0.1:{port}"
            break
        port += 1
    if not candidate:
        raise RuntimeError("cannot_find_free_port_for_uvicorn")

    py = sys.executable or "python"
    proc = subprocess.Popen(
        [py, "-m", "uvicorn", "modules.memory.api.server:app", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for readiness
    t0 = time.time()
    while time.time() - t0 < 60.0:
        if proc.poll() is not None:
            out = ""
            try:
                out = (proc.stdout.read() if proc.stdout else "")[:1000]  # type: ignore[union-attr]
            except Exception:
                out = ""
            raise RuntimeError(f"uvicorn_failed_to_start: rc={proc.returncode} out={out}")
        if await _check_health_json(candidate):
            return candidate, proc
        await asyncio.sleep(0.5)
    raise RuntimeError("uvicorn_start_timeout")


async def probe_incremental_commit(
    *,
    base_url: str,
    tenant_id: str,
    user_token: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    session_id = f"e2e_inc_{uuid.uuid4().hex[:6]}"
    owns_client = False
    if client is None:
        client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30.0, trust_env=False)
        owns_client = True
    headers = {"X-Tenant-ID": str(tenant_id)}
    use_http = client is not None and str(base_url).strip().lower() != "local"
    try:
        # Commit 1: 3 turns
        body1 = {
            "session_id": session_id,
            "user_tokens": [user_token],
            "memory_domain": "dialog",
            "turns": [
                {"turn_id": "t0001", "role": "user", "text": "我不吃香菜。"},
                {"turn_id": "t0002", "role": "assistant", "text": "好的。"},
                {"turn_id": "t0003", "role": "user", "text": "我也不吃葱。"},
            ],
            "commit_id": f"c1_{uuid.uuid4().hex[:8]}",
            "client_meta": {"stage2_skip_llm": True, "stage3_extract": False, "stage2_strict": False},
            "llm_policy": "best_effort",
        }
        job1 = ""
        cursor_after_1 = None
        if use_http:
            try:
                out1 = await _http_post_json(client, "/ingest/dialog/v1", body1, headers=headers)
                job1 = str(out1.get("job_id") or "")
            except Exception:
                use_http = False
        if not use_http:
            record, created = _ingest_store.create_job(
                session_id=session_id,
                commit_id=body1.get("commit_id"),
                tenant_id=str(tenant_id),
                api_key_id=None,
                request_id=None,
                turns=list(body1.get("turns") or []),
                user_tokens=[user_token],
                base_turn_id=None,
                client_meta=body1.get("client_meta"),
                memory_domain=str(body1.get("memory_domain")),
                llm_policy=str(body1.get("llm_policy")),
            )
            job1 = record.job_id
            # Local fallback only validates ingest idempotency/cursor; skip heavy Stage2/Stage3.
            cursor_after_1 = _ingest_store.get_session(session_id).get("cursor_committed")

        # wait best-effort for cursor update (we don't require full job completion here)
        if use_http:
            t0 = time.time()
            cursor_after_1 = None
            while time.time() - t0 < 20.0:
                sess = await _http_get_json(client, f"/ingest/sessions/{session_id}", headers=headers)
                cursor_after_1 = sess.get("cursor_committed")
                if cursor_after_1 == "t0003":
                    break
                await asyncio.sleep(0.5)

        # Commit 2: only NEW turns (t0004..t0005)
        body2 = {
            "session_id": session_id,
            "user_tokens": [user_token],
            "memory_domain": "dialog",
            "turns": [
                {"turn_id": "t0004", "role": "user", "text": "记住这个，很重要。"},
                {"turn_id": "t0005", "role": "assistant", "text": "收到。"},
            ],
            "commit_id": f"c2_{uuid.uuid4().hex[:8]}",
            "cursor": {"base_turn_id": "t0003"},
            "client_meta": {"stage2_skip_llm": True, "stage3_extract": False, "stage2_strict": False},
            "llm_policy": "best_effort",
        }
        job2 = ""
        cursor_after_2 = None
        if use_http:
            out2 = await _http_post_json(client, "/ingest/dialog/v1", body2, headers=headers)
            job2 = str(out2.get("job_id") or "")
            t1 = time.time()
            while time.time() - t1 < 20.0:
                sess = await _http_get_json(client, f"/ingest/sessions/{session_id}", headers=headers)
                cursor_after_2 = sess.get("cursor_committed")
                if cursor_after_2 == "t0005":
                    break
                await asyncio.sleep(0.5)
        else:
            record2, created2 = _ingest_store.create_job(
                session_id=session_id,
                commit_id=body2.get("commit_id"),
                tenant_id=str(tenant_id),
                api_key_id=None,
                request_id=None,
                turns=list(body2.get("turns") or []),
                user_tokens=[user_token],
                base_turn_id=str(body2.get("cursor", {}).get("base_turn_id") or ""),
                client_meta=body2.get("client_meta"),
                memory_domain=str(body2.get("memory_domain")),
                llm_policy=str(body2.get("llm_policy")),
            )
            job2 = record2.job_id
            # Local fallback only validates ingest idempotency/cursor; skip heavy Stage2/Stage3.
            cursor_after_2 = _ingest_store.get_session(session_id).get("cursor_committed")

        return {
            "probe": "incremental_commit",
            "tenant_id": tenant_id,
            "session_id": session_id,
            "job_id_1": job1,
            "job_id_2": job2,
            "cursor_after_commit_1": cursor_after_1,
            "cursor_after_commit_2": cursor_after_2,
            "submitted_turns_commit_1": len(body1["turns"]),
            "submitted_turns_commit_2": len(body2["turns"]),
            "path": ("http" if use_http else "local"),
        }
    finally:
        if owns_client:
            await client.aclose()


async def main() -> None:
    load_dotenv()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="Write JSON report to this path")
    ap.add_argument("--base-url", default=os.getenv("MEMORY_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--tenant-id", default=os.getenv("MEMORY_API_TENANT_ID", "tenant_e2e"))
    ap.add_argument("--user-token", default=os.getenv("MEMORY_E2E_USER", "user_e2e"))
    args = ap.parse_args()

    started = _now_ms()
    report: Dict[str, Any] = {"started_ms": started, "base_url_requested": str(args.base_url)}

    proc: Optional[subprocess.Popen] = None
    base_url = ""
    client: Optional[httpx.AsyncClient] = None
    base_url_raw = str(args.base_url).strip()
    force_asgi = bool(os.getenv("E2E_FORCE_ASGI")) or base_url_raw.lower() in {"asgi", "http://asgi"}
    use_local = base_url_raw.lower() == "local"
    try:
        if use_local:
            report["base_url"] = "local"
            report["server_started_by_script"] = False
            report["server_mode"] = "local"
            base_url = "local"
        elif force_asgi:
            raise RuntimeError("force_asgi")
        else:
            base_url, proc = await _ensure_server(base_url_raw)
            report["base_url"] = base_url
            report["server_started_by_script"] = bool(proc is not None)
    except Exception as exc:
        report["base_url"] = "http://asgi"
        report["server_started_by_script"] = False
        report["server_mode"] = "asgi"
        report["server_start_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        from modules.memory.api.server import app
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://asgi", timeout=30.0, trust_env=False)

    try:
        report["failure_path"] = await probe_failure_path(tenant_id=str(args.tenant_id), user_token=str(args.user_token))
        report["incremental_commit"] = await probe_incremental_commit(
            base_url=str(report["base_url"]),
            tenant_id=str(args.tenant_id),
            user_token=str(args.user_token),
            client=client,
        )
    finally:
        if client is not None:
            await client.aclose()
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    report["finished_ms"] = _now_ms()

    s = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    asyncio.run(main())
