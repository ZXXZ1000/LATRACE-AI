from __future__ import annotations

"""
Clear a memory domain (and optionally user_id/run_id) from Qdrant + Neo4j.

Usage:
  python -m modules.memory.scripts.clear_memory_domain --domain <name> [--user <uid> --run <runid> --dry-run]

Behavior:
  - Deletes vector points from all configured collections in Qdrant by filter on metadata.memory_domain
    and optional metadata.user_id / metadata.run_id (user_id uses ANY semantics).
  - Deletes graph nodes from Neo4j where n.memory_domain = $domain (and optional user_id/run_id constraints).
  - If --dry-run, only prints counts/queries without executing destructive operations.

Requirements:
  - Qdrant reachable at host/port from memory.config.yaml or env overrides.
  - Neo4j reachable if configured; otherwise graph deletion is skipped.
"""

from typing import Any, Dict, List
import argparse
import sys


def _load_memory_cfg() -> Dict[str, Any]:
    from modules.memory.application.config import load_memory_config
    return load_memory_config()


def _qdrant_delete_by_filter(cfg: Dict[str, Any], flt: Dict[str, Any], *, dry_run: bool = False) -> Dict[str, int]:
    import requests
    vs = ((cfg.get("memory", {}) or {}).get("vector_store", {}) or {})
    host = vs.get("host", "127.0.0.1")
    port = int(vs.get("port", 6333))
    base = f"http://{host}:{port}"
    collections = (vs.get("collections") or {"text": "memory_text", "image": "memory_image", "audio": "memory_audio"})
    api_key = vs.get("api_key")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    deleted: Dict[str, int] = {}
    for mod, coll in collections.items():
        if not coll:
            continue
        url = f"{base}/collections/{coll}/points/delete"
        payload = {"filter": flt}
        if dry_run:
            print(f"[DRY-RUN] Would delete Qdrant points in '{coll}' with filter={flt}")
            deleted[coll] = 0
            continue
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code >= 400:
            print(f"[WARN] Qdrant delete failed for {coll}: {r.status_code} {r.text[:200]}")
            continue
        # Qdrant returns operation status, not count; best-effort report -1
        deleted[coll] = -1
    return deleted


def _neo4j_delete_domain(cfg: Dict[str, Any], domain: str, *, user: List[str] | None, run: str | None, dry_run: bool = False) -> int:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception:
        print("[INFO] neo4j driver not installed; skipping graph deletion")
        return 0
    gcfg = ((cfg.get("memory", {}) or {}).get("graph_store", {}) or {})
    uri = gcfg.get("uri", "bolt://127.0.0.1:7687")
    user_name = gcfg.get("user", "neo4j")
    pwd = gcfg.get("password", "password")
    drv = GraphDatabase.driver(uri, auth=(user_name, pwd))
    where = ["n.memory_domain = $domain"]
    params: Dict[str, Any] = {"domain": domain}
    if user:
        where.append("ANY(u IN n.user_id WHERE u IN $uids)")
        params["uids"] = [str(x) for x in user]
    if run:
        where.append("n.run_id = $run")
        params["run"] = str(run)
    where_clause = " AND ".join(where)
    cypher = f"MATCH (n:Entity) WHERE {where_clause} DETACH DELETE n"
    if dry_run:
        print(f"[DRY-RUN] Would execute Cypher: {cypher} with params={params}")
        return 0
    with drv.session() as sess:
        # Neo4j does not return deleted count by default; we can run an EXPLAIN/PROFILE or count first
        count_query = f"MATCH (n:Entity) WHERE {where_clause} RETURN count(n) AS cnt"
        rec = sess.run(count_query, **params).single()
        cnt = int(rec["cnt"]) if rec and rec["cnt"] is not None else 0
        sess.run(cypher, **params)
        return cnt


def _build_qdrant_filter(domain: str, user: List[str] | None, run: str | None) -> Dict[str, Any]:
    must = [{"key": "metadata.memory_domain", "match": {"value": domain}}]
    if user:
        should = [{"key": "metadata.user_id", "match": {"value": u}} for u in user]
        return {"must": must, "should": should}
    if run is not None:
        must.append({"key": "metadata.run_id", "match": {"value": run}})
    return {"must": must}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Clear a memory domain from Qdrant+Neo4j")
    ap.add_argument("--domain", required=True, help="memory_domain to clear")
    ap.add_argument("--user", action="append", help="user_id to restrict (repeatable)")
    ap.add_argument("--run", help="run_id to restrict")
    ap.add_argument("--dry-run", action="store_true", help="do not perform destructive operations")
    args = ap.parse_args(argv)

    cfg = _load_memory_cfg()
    qflt = _build_qdrant_filter(args.domain, args.user, args.run)
    print(f"[INFO] Clearing Qdrant collections for domain='{args.domain}', user={args.user}, run={args.run}")
    qres = _qdrant_delete_by_filter(cfg, qflt, dry_run=bool(args.dry_run))
    for coll, cnt in (qres or {}).items():
        print(f"  - Qdrant/{coll}: deleted={cnt if cnt>=0 else 'submitted'}")
    print(f"[INFO] Clearing Neo4j nodes for domain='{args.domain}', user={args.user}, run={args.run}")
    gdel = _neo4j_delete_domain(cfg, args.domain, user=args.user, run=args.run, dry_run=bool(args.dry_run))
    print(f"  - Neo4j: deleted_nodes~={gdel}")
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

