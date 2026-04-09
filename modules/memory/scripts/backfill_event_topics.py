from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from modules.memory.application.config import load_memory_config
from modules.memory.application.topic_normalizer import TopicNormalizer
from modules.memory.infra.qdrant_store import QdrantStore

QDRANT_DEFAULT_COLLECTION = "memory_text"


def _to_native_dt(val: Any) -> Optional[datetime]:
    try:
        if val is None:
            return None
        to_native = getattr(val, "to_native", None)
        if callable(to_native):
            val = to_native()
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except Exception:
                return None
        return None
    except Exception:
        return None


def _derive_time_bucket(t_start: Any, t_end: Any) -> List[str]:
    ts = _to_native_dt(t_start) or _to_native_dt(t_end)
    if not ts:
        return []
    try:
        return [ts.strftime("%Y"), ts.strftime("%Y-%m"), ts.strftime("%Y-%m-%d")]
    except Exception:
        return []


def _build_update_fields(ev: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    topic_path = str(ev.get("topic_path") or "").strip()
    if topic_path:
        out["topic_path"] = topic_path
    tags = ev.get("tags") or []
    if isinstance(tags, list) and tags:
        out["tags"] = tags
    keywords = ev.get("keywords") or []
    if isinstance(keywords, list) and keywords:
        out["keywords"] = keywords
    time_bucket = ev.get("time_bucket") or []
    if isinstance(time_bucket, list) and time_bucket:
        out["time_bucket"] = time_bucket
    tv = str(ev.get("tags_vocab_version") or "").strip()
    if tv:
        out["tags_vocab_version"] = tv
    return out


def _scroll_qdrant_points(host: str, port: int, collection: str, batch: int = 512) -> Iterable[Dict[str, Any]]:
    url = f"http://{host}:{port}/collections/{collection}/points/scroll"
    offset = None
    while True:
        payload: Dict[str, Any] = {"with_payload": True, "limit": int(batch)}
        if offset is not None:
            payload["offset"] = offset
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - local tool
            data = json.loads(resp.read().decode("utf-8"))
        points = ((data.get("result") or {}).get("points") or [])
        for p in points:
            if isinstance(p, dict):
                yield p
        next_page = (data.get("result") or {}).get("next_page_offset")
        if not next_page:
            break
        offset = next_page


def _index_qdrant_points(host: str, port: int, collection: str) -> tuple[Dict[tuple[str, str], List[str]], Dict[str, Dict[str, Any]]]:
    index: Dict[tuple[str, str], List[str]] = {}
    meta_by_point: Dict[str, Dict[str, Any]] = {}

    for p in _scroll_qdrant_points(host, port, collection):
        pid = str(p.get("id") or "").strip()
        payload = p.get("payload") or {}
        meta = payload.get("metadata") or {}
        if not pid or not isinstance(meta, dict):
            continue
        tenant = str(meta.get("tenant_id") or "").strip()
        if not tenant:
            continue
        meta_by_point[pid] = meta
        event_ids: List[str] = []
        for key in ("node_id", "event_id", "tkg_event_id"):
            val = meta.get(key)
            if val:
                event_ids.append(str(val))
        tkg_ids = meta.get("tkg_event_ids")
        if isinstance(tkg_ids, list):
            event_ids.extend([str(x) for x in tkg_ids if x])
        for eid in event_ids:
            if not eid:
                continue
            index.setdefault((tenant, eid), []).append(pid)

    return index, meta_by_point


def _qdrant_set_payload_points(host: str, port: int, collection: str, point_id: str, payload: Dict[str, Any]) -> None:
    url = f"http://{host}:{port}/collections/{collection}/points/payload"
    body = {"points": [point_id], "payload": payload}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - local tool
        resp.read()


def _fetch_events(session, tenant_id: Optional[str], limit: int) -> Iterable[Dict[str, Any]]:
    cond = "WHERE e.tenant_id IS NOT NULL"
    params: Dict[str, Any] = {}
    if tenant_id:
        cond += " AND e.tenant_id = $tenant"
        params["tenant"] = tenant_id
    cypher = (
        "MATCH (e:Event) "
        f"{cond} "
        "RETURN e.id AS id, e.tenant_id AS tenant_id, e.summary AS summary, e.desc AS desc, e.topic_id AS topic_id, "
        "e.topic_path AS topic_path, e.tags AS tags, e.keywords AS keywords, e.time_bucket AS time_bucket, "
        "e.tags_vocab_version AS tags_vocab_version, e.t_abs_start AS t_abs_start, e.t_abs_end AS t_abs_end, "
        "e.user_id AS user_id, e.memory_domain AS memory_domain "
        "ORDER BY e.id "
        "LIMIT $limit"
    )
    params["limit"] = int(limit)
    for row in session.run(cypher, **params):
        yield dict(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill Event topic fields in Neo4j/Qdrant.")
    ap.add_argument("--tenant", help="Tenant id filter")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--apply-neo4j", action="store_true")
    ap.add_argument("--apply-qdrant", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_memory_config()
    neo_cfg = cfg.get("memory", {}).get("graph_store", {})
    uri = str(neo_cfg.get("uri") or "bolt://127.0.0.1:7687")
    user = str(neo_cfg.get("user") or "neo4j")
    password = str(neo_cfg.get("password") or "neo4j")

    normalizer = TopicNormalizer()
    updates: List[Dict[str, Any]] = []

    from neo4j import GraphDatabase  # type: ignore

    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with drv.session() as sess:
            for ev in _fetch_events(sess, args.tenant, args.limit):
                normalized = normalizer.normalize_event(ev)
                if not normalized.get("time_bucket"):
                    normalized["time_bucket"] = _derive_time_bucket(ev.get("t_abs_start"), ev.get("t_abs_end"))
                fields = _build_update_fields(normalized)
                if not fields:
                    continue
                updates.append({"event_id": ev["id"], "tenant_id": ev["tenant_id"], **fields})
                if args.apply_neo4j and not args.dry_run:
                    sets = []
                    params = {"tenant": ev["tenant_id"], "id": ev["id"]}
                    for k, v in fields.items():
                        sets.append(f"e.{k} = ${k}")
                        params[k] = v
                    cypher = "MATCH (e:Event {tenant_id: $tenant, id: $id}) SET " + ", ".join(sets)
                    sess.run(cypher, **params)
    finally:
        drv.close()

    if args.apply_qdrant and not args.dry_run:
        store = QdrantStore(cfg.get("memory", {}).get("vector_store", {}))
        host = str(store.settings.get("host") or "127.0.0.1")
        port = int(store.settings.get("port") or 6333)
        collection = str((store.collections or {}).get("text") or QDRANT_DEFAULT_COLLECTION)

        index, meta_by_point = _index_qdrant_points(host, port, collection)
        # aggregate per point_id (supports points linked to multiple events)
        agg: Dict[str, Dict[str, Any]] = {}

        for row in updates:
            tenant = str(row.get("tenant_id") or "").strip()
            event_id = str(row.get("event_id") or "").strip()
            if not tenant or not event_id:
                continue
            point_ids = index.get((tenant, event_id), [])
            if not point_ids:
                continue
            for pid in point_ids:
                slot = agg.setdefault(pid, {"topic_path": set(), "tags": set(), "keywords": set(), "time_bucket": set(), "tags_vocab_version": None})
                if row.get("topic_path"):
                    slot["topic_path"].add(row["topic_path"])  # type: ignore[union-attr]
                for key in ("tags", "keywords", "time_bucket"):
                    vals = row.get(key) or []
                    if isinstance(vals, list):
                        slot[key].update([str(v) for v in vals if v])  # type: ignore[union-attr]
                tv = row.get("tags_vocab_version")
                if tv and not slot["tags_vocab_version"]:
                    slot["tags_vocab_version"] = tv

        for pid, slot in agg.items():
            existing = dict(meta_by_point.get(pid) or {})
            # merge aggregated fields into metadata
            if slot["topic_path"]:
                tp = sorted(slot["topic_path"])  # type: ignore[arg-type]
                existing["topic_path"] = tp[0] if len(tp) == 1 else tp
            for key in ("tags", "keywords", "time_bucket"):
                vals = sorted(slot[key])  # type: ignore[arg-type]
                if vals:
                    existing[key] = vals
            if slot.get("tags_vocab_version"):
                existing["tags_vocab_version"] = slot["tags_vocab_version"]
            _qdrant_set_payload_points(host, port, collection, pid, {"metadata": existing})

    print(f"updates={len(updates)} applied_neo4j={bool(args.apply_neo4j and not args.dry_run)} applied_qdrant={bool(args.apply_qdrant and not args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
