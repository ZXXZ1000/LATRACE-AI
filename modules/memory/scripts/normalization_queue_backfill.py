from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

from modules.memory.application.config import load_memory_config
from modules.memory.application.topic_normalizer import TopicNormalizer
from modules.memory.infra.qdrant_store import QdrantStore

QDRANT_DEFAULT_COLLECTION = "memory_text"


def _iter_queue(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


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


def _build_update_fields(ev: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    topic_path = str(ev.get("topic_path") or "").strip()
    if not topic_path or topic_path.startswith("_uncategorized/"):
        return {}
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


def _update_neo4j(session, tenant_id: str, event_id: str, fields: Dict[str, Any]) -> bool:
    if not fields:
        return False
    sets = []
    params = {"tenant": tenant_id, "id": event_id}
    for k, v in fields.items():
        sets.append(f"e.{k} = ${k}")
        params[k] = v
    cypher = "MATCH (e:Event {tenant_id: $tenant, id: $id}) SET " + ", ".join(sets)
    session.run(cypher, **params)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill topic normalization from queue JSONL.")
    parser.add_argument("--input", "-i", help="Queue JSONL path")
    parser.add_argument("--output", "-o", help="Write normalized updates JSONL")
    parser.add_argument("--apply-neo4j", action="store_true", help="Apply updates to Neo4j")
    parser.add_argument("--apply-qdrant", action="store_true", help="Apply payload updates to Qdrant")
    args = parser.parse_args()

    cfg = load_memory_config()
    queue_path = Path(args.input) if args.input else None
    if queue_path is None:
        queue_path = Path(
            cfg.get("memory", {})
            .get("outputs", {})
            .get("topic_queue_path", "")
        )
    if queue_path is None or not str(queue_path).strip():
        queue_path = Path("modules/memory/outputs/topic_normalization_queue.jsonl")

    normalizer = TopicNormalizer()
    updates: List[Dict[str, Any]] = []

    for raw in _iter_queue(queue_path):
        event_id = str(raw.get("event_id") or "").strip()
        tenant_id = str(raw.get("tenant_id") or "").strip()
        if not event_id or not tenant_id:
            continue
        normalized = normalizer.normalize_event(raw)
        fields = _build_update_fields(normalized)
        if not fields:
            continue
        updates.append({"event_id": event_id, "tenant_id": tenant_id, **fields})

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in updates:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.apply_neo4j:
        from neo4j import GraphDatabase  # type: ignore
        neo_cfg = cfg.get("memory", {}).get("graph_store", {})
        uri = str(neo_cfg.get("uri") or "bolt://127.0.0.1:7687")
        user = str(neo_cfg.get("user") or "neo4j")
        password = str(neo_cfg.get("password") or "neo4j")
        drv = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with drv.session() as sess:
                for row in updates:
                    fields = {k: v for k, v in row.items() if k not in {"event_id", "tenant_id"}}
                    _update_neo4j(sess, row["tenant_id"], row["event_id"], fields)
        finally:
            drv.close()

    if args.apply_qdrant:
        q_cfg = cfg.get("memory", {}).get("vector_store", {})
        store = QdrantStore(q_cfg)
        host = str(store.settings.get("host") or "127.0.0.1")
        port = int(store.settings.get("port") or 6333)
        collection = str((store.collections or {}).get("text") or QDRANT_DEFAULT_COLLECTION)

        index, meta_by_point = _index_qdrant_points(host, port, collection)
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

    print(f"backfill_updates={len(updates)} queue={queue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
