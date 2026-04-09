from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from datetime import datetime, timezone

from modules.memory.application.config import load_memory_config
from modules.memory.application.graph_service import GraphService
from modules.memory.infra.neo4j_store import Neo4jStore


def _load_env(env_path: Optional[str]) -> None:
    if load_dotenv is None:
        return
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


def _build_store() -> Neo4jStore:
    cfg = load_memory_config()
    gcfg = cfg.get("memory", {}).get("graph_store", {})
    rcfg = cfg.get("memory", {}).get("reliability", {})

    uri = os.getenv("NEO4J_URI") or gcfg.get("uri", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER") or gcfg.get("user", "neo4j")
    password = os.getenv("NEO4J_PASSWORD") or gcfg.get("password", "password")
    return Neo4jStore({"uri": str(uri), "user": str(user), "password": str(password), "reliability": rcfg})


def _dt_to_iso(store: Neo4jStore, val: Any) -> Optional[str]:
    try:
        return store._dt_to_iso(val)  # type: ignore[attr-defined]
    except Exception:
        return None


def _query_top_topic_paths(
    store: Neo4jStore,
    *,
    tenant_id: str,
    user_ids: Optional[List[str]],
    max_topics: int,
    min_count: int,
    allow_uncategorized: bool,
) -> List[Tuple[str, int]]:
    cypher = """
MATCH (ev:Event {tenant_id:$tenant})
WHERE ev.topic_path IS NOT NULL
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
RETURN ev.topic_path as topic_path, count(*) as cnt
ORDER BY cnt DESC
"""
    items: List[Tuple[str, int]] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, user_ids=user_ids)
        for row in rows:
            tp = str(row.get("topic_path") or "").strip()
            cnt = int(row.get("cnt") or 0)
            if not tp:
                continue
            if not allow_uncategorized and tp.startswith("_uncategorized/"):
                continue
            if cnt < min_count:
                continue
            items.append((tp, cnt))
            if len(items) >= max_topics:
                break
    return items


def _query_events_for_topic(
    store: Neo4jStore,
    *,
    tenant_id: str,
    user_ids: Optional[List[str]],
    topic_path: str,
    limit: int,
) -> List[Dict[str, Any]]:
    cypher = """
MATCH (ev:Event {tenant_id:$tenant})
WHERE ev.topic_path IS NOT NULL
  AND ev.topic_path STARTS WITH $topic_path
  AND ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
RETURN ev.id as event_id, ev.t_abs_start as t_abs_start, ev.t_abs_end as t_abs_end
ORDER BY coalesce(ev.t_abs_start, datetime({epochMillis: 0})) ASC
LIMIT $limit
"""
    items: List[Dict[str, Any]] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(
            cypher,
            tenant=tenant_id,
            user_ids=user_ids,
            topic_path=str(topic_path),
            limit=int(limit),
        )
        for row in rows:
            items.append(
                {
                    "event_id": row.get("event_id"),
                    "t_abs_start": _dt_to_iso(store, row.get("t_abs_start")),
                    "t_abs_end": _dt_to_iso(store, row.get("t_abs_end")),
                }
            )
    return items


def _query_top_entity_names(
    store: Neo4jStore,
    *,
    tenant_id: str,
    user_ids: Optional[List[str]],
    max_entities: int,
    min_count: int,
) -> List[str]:
    cypher = """
MATCH (ev:Event {tenant_id:$tenant})-[:INVOLVES]->(ent:Entity {tenant_id:$tenant})
WHERE ($user_ids IS NULL OR ANY(x IN coalesce(ev.user_id, []) WHERE x IN $user_ids))
WITH coalesce(ent.name, ent.manual_name, ent.cluster_label) AS name, count(distinct ev) AS cnt
WHERE name IS NOT NULL AND name <> ""
RETURN name, cnt
ORDER BY cnt DESC
"""
    out: List[str] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, user_ids=user_ids)
        for row in rows:
            name = str(row.get("name") or "").strip()
            cnt = int(row.get("cnt") or 0)
            if not name or cnt < min_count:
                continue
            if name not in out:
                out.append(name)
            if len(out) >= max_entities:
                break
    return out


def _resolve_entity_id(store: Neo4jStore, *, tenant_id: str, name: str) -> Optional[str]:
    cypher = """
CALL db.index.fulltext.queryNodes('tkg_entity_name_v1', $q) YIELD node, score
WHERE node.tenant_id = $tenant
RETURN node AS ent, score
ORDER BY score DESC
LIMIT 1
"""
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        row = sess.run(cypher, tenant=tenant_id, q=name).single()
        ent = row.get("ent") if row else None
        if ent:
            props = dict(ent)
            return str(props.get("id") or "").strip() or None
    return None


def _query_entity_relations_by_events(
    store: Neo4jStore,
    *,
    tenant_id: str,
    entity_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (ent)<-[:INVOLVES]-(ev:Event {tenant_id: $tenant})-[:INVOLVES]->(other:Entity {tenant_id: $tenant})
WHERE (ev.expires_at IS NULL OR ev.expires_at > datetime())
  AND (ev.published IS NULL OR ev.published = true)
  AND (other.published IS NULL OR other.published = true)
  AND other.id <> ent.id
WITH other,
     collect(DISTINCT ev.id) AS event_ids,
     count(DISTINCT ev) AS strength,
     min(ev.t_abs_start) AS first_ts,
     max(ev.t_abs_start) AS last_ts
RETURN other AS entity,
       event_ids,
       strength,
       first_ts,
       last_ts
ORDER BY strength DESC
LIMIT $limit
"""
    out: List[Dict[str, Any]] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, entity_id=entity_id, limit=int(limit))
        for row in rows:
            ent = row.get("entity")
            if not ent:
                continue
            props = dict(ent)
            out.append(
                {
                    "entity_id": props.get("id"),
                    "name": props.get("name") or props.get("manual_name") or props.get("cluster_label"),
                    "strength": int(row.get("strength") or 0),
                    "first_mentioned": _dt_to_iso(store, row.get("first_ts")),
                    "last_mentioned": _dt_to_iso(store, row.get("last_ts")),
                    "evidence_event_ids": list(row.get("event_ids") or [])[:10],
                }
            )
    return out


def _query_entity_relations_cooccur(
    store: Neo4jStore,
    *,
    tenant_id: str,
    entity_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (ent)-[r:CO_OCCURS_WITH]-(other:Entity {tenant_id: $tenant})
WHERE (other.published IS NULL OR other.published = true)
RETURN other AS entity, r AS rel
ORDER BY coalesce(r.weight, 0.0) DESC
LIMIT $limit
"""
    out: List[Dict[str, Any]] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, entity_id=entity_id, limit=int(limit))
        for row in rows:
            ent = row.get("entity")
            if not ent:
                continue
            rel = row.get("rel")
            props = dict(ent)
            out.append(
                {
                    "entity_id": props.get("id"),
                    "name": props.get("name") or props.get("manual_name") or props.get("cluster_label"),
                    "weight": float(rel.get("weight") or 0.0) if hasattr(rel, "get") else 0.0,
                }
            )
    return out


def _parse_iso(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val)
    except Exception:
        return None


async def _build_quotes_via_graph(
    graph_svc: GraphService,
    *,
    tenant_id: str,
    entity_id: str,
    user_ids: Optional[List[str]],
    memory_domain: Optional[str],
    limit: int,
) -> List[Dict[str, str]]:
    events = await graph_svc.list_events(
        tenant_id=tenant_id,
        entity_id=entity_id,
        user_ids=user_ids,
        memory_domain=memory_domain,
        limit=max(1, min(limit * 3, 200)),
    )
    quotes: List[Dict[str, Any]] = []
    for ev in events:
        if len(quotes) >= limit:
            break
        ev_id = str(ev.get("id") or "").strip()
        if not ev_id:
            continue
        bundle = await graph_svc.explain_event_evidence(tenant_id=tenant_id, event_id=ev_id)
        utterances = list((bundle or {}).get("utterances") or [])
        utterance_speakers = list((bundle or {}).get("utterance_speakers") or [])
        speaker_map: Dict[str, str] = {}
        for item in utterance_speakers:
            uid = str(item.get("utterance_id") or "").strip()
            sid = str(item.get("entity_id") or "").strip()
            if uid and sid:
                speaker_map[uid] = sid
        event_time = None
        event_obj = (bundle or {}).get("event")
        if isinstance(event_obj, dict):
            event_time = event_obj.get("t_abs_start") or event_obj.get("t_abs_end")
        for utt in utterances:
            utt_id = str(utt.get("id") or utt.get("utterance_id") or "").strip()
            text = str(utt.get("raw_text") or utt.get("text") or "").strip()
            if not utt_id or not text:
                continue
            speaker_id = speaker_map.get(utt_id)
            if speaker_id != entity_id:
                continue
            quotes.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": speaker_id,
                    "when": event_time,
                    "t_media_start": utt.get("t_media_start"),
                }
            )
            if len(quotes) >= limit:
                break
    quotes.sort(
        key=lambda x: (
            _parse_iso(x.get("when")) or datetime.min.replace(tzinfo=timezone.utc),
            float(x.get("t_media_start") or 0.0),
        ),
        reverse=True,
    )
    return [{"utterance_id": q["utterance_id"], "speaker_id": q["speaker_id"]} for q in quotes[:limit]]


def _query_entity_facts(
    store: Neo4jStore,
    *,
    tenant_id: str,
    entity_id: str,
    limit: int,
) -> List[str]:
    cypher = """
MATCH (ent:Entity {id: $entity_id, tenant_id: $tenant})
MATCH (k:Knowledge {tenant_id: $tenant})-[:STATED_BY]->(ent)
WHERE (k.expires_at IS NULL OR k.expires_at > datetime())
  AND (k.published IS NULL OR k.published = true)
RETURN k AS knowledge
ORDER BY coalesce(k.updated_at, k.created_at, datetime({epochMillis: 0})) DESC
LIMIT $limit
"""
    out: List[str] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, entity_id=entity_id, limit=int(limit))
        for row in rows:
            k = row.get("knowledge")
            if not k:
                continue
            props = dict(k)
            summary = props.get("summary") or props.get("statement")
            if summary:
                out.append(str(summary))
    return out


def _query_entity_quotes(
    store: Neo4jStore,
    *,
    tenant_id: str,
    entity_id: str,
    limit: int,
) -> List[Dict[str, str]]:
    cypher = """
MATCH (utt:UtteranceEvidence {tenant_id:$tenant})-[:SPOKEN_BY]->(ent:Entity {tenant_id:$tenant, id:$entity_id})
RETURN utt.id as utterance_id, ent.id as speaker_id, utt.t_media_start as t_media_start
ORDER BY coalesce(utt.t_media_start, 0.0) DESC
LIMIT $limit
"""
    out: List[Dict[str, str]] = []
    with store._driver.session(database=store._database) as sess:  # type: ignore[attr-defined]
        rows = sess.run(cypher, tenant=tenant_id, entity_id=entity_id, limit=int(limit))
        for row in rows:
            utt_id = str(row.get("utterance_id") or "").strip()
            speaker_id = str(row.get("speaker_id") or "").strip()
            if utt_id and speaker_id:
                out.append({"utterance_id": utt_id, "speaker_id": speaker_id})
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Phase5 ground-truth from LoCoMo-conv26 ingest.")
    ap.add_argument("--tenant-id", default="locomo_bench")
    ap.add_argument("--user-token", action="append", default=[], help="user_tokens (repeatable)")
    ap.add_argument("--output-dir", default="modules/memory/data/phase5/ground_truth/locomo_conv26")
    ap.add_argument("--max-topics", type=int, default=5)
    ap.add_argument("--min-topic-count", type=int, default=2)
    ap.add_argument("--max-entities", type=int, default=2)
    ap.add_argument("--min-entity-count", type=int, default=2)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--facts-limit", type=int, default=5)
    ap.add_argument("--relations-limit", type=int, default=5)
    ap.add_argument("--events-limit", type=int, default=5)
    ap.add_argument("--quotes-limit", type=int, default=5)
    ap.add_argument("--allow-uncategorized", action="store_true")
    ap.add_argument("--env", default=None, help="Path to .env (default: auto)")
    args = ap.parse_args()

    _load_env(args.env)
    store = _build_store()
    graph_svc = GraphService(store)

    user_tokens = [str(x).strip() for x in (args.user_token or []) if str(x).strip()]
    user_ids = user_tokens or None

    out_dir = Path(args.output_dir)

    # Topic timeline + time-since
    topic_rows = _query_top_topic_paths(
        store,
        tenant_id=str(args.tenant_id),
        user_ids=user_ids,
        max_topics=int(args.max_topics),
        min_count=int(args.min_topic_count),
        allow_uncategorized=bool(args.allow_uncategorized),
    )
    topic_samples: List[Dict[str, Any]] = []
    time_since_samples: List[Dict[str, Any]] = []
    for idx, (topic_path, _cnt) in enumerate(topic_rows, start=1):
        events = _query_events_for_topic(
            store,
            tenant_id=str(args.tenant_id),
            user_ids=user_ids,
            topic_path=topic_path,
            limit=int(args.limit),
        )
        event_ids = [e.get("event_id") for e in events if e.get("event_id")]
        if not event_ids:
            continue
        start_iso = None
        if events and events[0].get("t_abs_start"):
            start_iso = events[0].get("t_abs_start")
        topic_samples.append(
            {
                "query_id": f"tt-{idx:03d}",
                "topic_path": topic_path,
                "user_tokens": user_tokens,
                "expected_event_ids": event_ids,
                "expected_order": event_ids,
                "time_range": {"start": start_iso} if start_iso else None,
                "limit": int(args.limit),
            }
        )
        last = events[-1] if events else None
        last_ts = last.get("t_abs_start") if last else None
        if last_ts:
            time_since_samples.append(
                {
                    "query_id": f"ts-{idx:03d}",
                    "topic_path": topic_path,
                    "user_tokens": user_tokens,
                    "expected_last_mentioned": last_ts,
                }
            )

    # Entity samples
    names = _query_top_entity_names(
        store,
        tenant_id=str(args.tenant_id),
        user_ids=user_ids,
        max_entities=int(args.max_entities),
        min_count=int(args.min_entity_count),
    )
    entity_profile_samples: List[Dict[str, Any]] = []
    relations_samples: List[Dict[str, Any]] = []
    quotes_samples: List[Dict[str, Any]] = []
    for idx, name in enumerate(names, start=1):
        ent_id = _resolve_entity_id(store, tenant_id=str(args.tenant_id), name=name)
        if not ent_id:
            continue
        rels_by_events = _query_entity_relations_by_events(
            store, tenant_id=str(args.tenant_id), entity_id=ent_id, limit=int(args.relations_limit)
        )
        rel_ids_by_events = [r.get("entity_id") or r.get("name") for r in rels_by_events if r.get("entity_id") or r.get("name")]
        rels_cooccur = _query_entity_relations_cooccur(
            store, tenant_id=str(args.tenant_id), entity_id=ent_id, limit=int(args.relations_limit)
        )
        rel_ids_cooccur = [r.get("entity_id") or r.get("name") for r in rels_cooccur if r.get("entity_id") or r.get("name")]
        facts = _query_entity_facts(
            store, tenant_id=str(args.tenant_id), entity_id=ent_id, limit=int(args.facts_limit)
        )
        # Use GraphService list_events (decay ordering) to match entity_profile semantics.
        # Apply decay ordering internally; we only need event IDs for evaluation.
        try:
            import asyncio

            async def _get_events() -> List[str]:
                evs = await graph_svc.list_events(
                    tenant_id=str(args.tenant_id),
                    entity_id=ent_id,
                    user_ids=user_ids,
                    memory_domain=None,
                    limit=int(args.events_limit),
                )
                return [str(e.get("id") or "") for e in evs if str(e.get("id") or "").strip()]

            recent_events = asyncio.run(_get_events())
        except Exception:
            recent_events = []

        # Build quotes via graph API path to match runtime behavior.
        try:
            import asyncio

            quotes = asyncio.run(
                _build_quotes_via_graph(
                    graph_svc,
                    tenant_id=str(args.tenant_id),
                    entity_id=ent_id,
                    user_ids=user_ids,
                    memory_domain=None,
                    limit=int(args.quotes_limit),
                )
            )
        except Exception:
            quotes = []

        entity_profile_samples.append(
            {
                "query_id": f"ep-{idx:03d}",
                "entity": name,
                "entity_id": ent_id,
                "user_tokens": user_tokens,
                "expected_facts": facts,
                "expected_relations": rel_ids_cooccur,
                "expected_recent_event_ids": recent_events,
            }
        )
        relations_samples.append(
            {
                "query_id": f"rel-{idx:03d}",
                "entity": name,
                "entity_id": ent_id,
                "user_tokens": user_tokens,
                "expected_related_entities": rel_ids_by_events,
            }
        )
        quotes_samples.append(
            {
                "query_id": f"qt-{idx:03d}",
                "entity": name,
                "entity_id": ent_id,
                "user_tokens": user_tokens,
                "expected_quotes": quotes,
            }
        )

    _write_jsonl(out_dir / "topic_timeline.jsonl", topic_samples)
    _write_jsonl(out_dir / "time_since.jsonl", time_since_samples)
    _write_jsonl(out_dir / "entity_profile.jsonl", entity_profile_samples)
    _write_jsonl(out_dir / "relations.jsonl", relations_samples)
    _write_jsonl(out_dir / "quotes.jsonl", quotes_samples)

    store.close()
    print(f"✅ ground truth written to: {out_dir}")
    print(f"  topics: {len(topic_samples)} | entities: {len(entity_profile_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
