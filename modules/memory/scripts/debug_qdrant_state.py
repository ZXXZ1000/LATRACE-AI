from __future__ import annotations

"""
Debug helper: inspect Qdrant collections and payloads for a given tenant/user scope.

This is intentionally a script (not an HTTP endpoint) to avoid expanding the Memory service API surface.
Run it locally while Memory/Qdrant are up.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import requests

from modules.memory.application.config import load_memory_config
from modules.memory.application.embedding_adapter import (
    build_embedding_from_settings,
    build_image_embedding_from_settings,
    build_audio_embedding_from_settings,
)


def _as_int(val: Any, default: int) -> int:
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val).strip()
        if not s:
            return default
        return int(s)
    except Exception:
        return default


def _resolve_env_like_str(val: Any, default: str) -> str:
    """Match Memory API server behavior for placeholders like '${QDRANT_HOST}'."""
    try:
        s = str(val).strip()
        if not s or s.startswith("${"):
            return default
        return s
    except Exception:
        return default


def _qdrant_base(cfg: Dict[str, Any]) -> str:
    vcfg = (cfg.get("memory", {}) or {}).get("vector_store", {}) or {}
    host_raw = os.getenv("QDRANT_HOST") or vcfg.get("host", "127.0.0.1")
    port_raw = os.getenv("QDRANT_PORT") or vcfg.get("port", 6333)
    host = _resolve_env_like_str(host_raw, "127.0.0.1")
    port = _as_int(port_raw, 6333)
    return f"http://{host}:{port}"


def _collections(cfg: Dict[str, Any]) -> Dict[str, str]:
    vcfg = (cfg.get("memory", {}) or {}).get("vector_store", {}) or {}
    col = vcfg.get("collections") or {}
    if not isinstance(col, dict):
        col = {}
    # include optional extended collections if present
    out: Dict[str, str] = {}
    for k, v in col.items():
        try:
            if v:
                out[str(k)] = str(v)
        except Exception:
            continue
    # fallback defaults
    out.setdefault("text", "memory_text")
    out.setdefault("image", "memory_image")
    out.setdefault("audio", "memory_audio")
    return out


def build_qdrant_filter(*, tenant_id: Optional[str], user_ids: List[str] | None) -> Dict[str, Any] | None:
    must: list[Dict[str, Any]] = []
    if tenant_id:
        must.append({"key": "metadata.tenant_id", "match": {"value": str(tenant_id)}})
    if user_ids:
        for uid in user_ids:
            if uid:
                must.append({"key": "metadata.user_id", "match": {"value": str(uid)}})
    return {"must": must} if must else None


def _post(session: requests.Session, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = session.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json() if r.content else {}


def _embedding_fns(cfg: Dict[str, Any]):
    vcfg = (cfg.get("memory", {}) or {}).get("vector_store", {}) or {}
    emb_cfg = vcfg.get("embedding", {}) or {}
    text_fn = build_embedding_from_settings(emb_cfg if isinstance(emb_cfg, dict) else {})
    img_cfg = emb_cfg.get("clip_image") if isinstance(emb_cfg, dict) else None
    aud_cfg = emb_cfg.get("audio") if isinstance(emb_cfg, dict) else None
    clip_img_fn = build_image_embedding_from_settings(img_cfg)
    audio_fn = build_audio_embedding_from_settings(aud_cfg)
    return text_fn, clip_img_fn, audio_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant-id", default="default")
    ap.add_argument("--user-id", action="append", dest="user_ids", default=[])
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--query", default=None, help="Optional: run a live points/search for this query")
    ap.add_argument("--threshold", type=float, default=0.1, help="score_threshold for points/search when --query is set")
    args = ap.parse_args()

    cfg = load_memory_config()
    base = _qdrant_base(cfg)
    cols = _collections(cfg)
    flt = build_qdrant_filter(tenant_id=str(args.tenant_id or "").strip() or None, user_ids=list(args.user_ids or []))

    sess = requests.Session()
    # keep it simple: api-key header if configured
    api_key = os.getenv("QDRANT_API_KEY") or (((cfg.get("memory", {}) or {}).get("vector_store", {}) or {}).get("api_key") or "")
    if api_key:
        sess.headers.update({"api-key": str(api_key)})

    print(f"QDRANT base={base}")
    print(f"collections={json.dumps(cols, ensure_ascii=False)}")
    print(f"filter={json.dumps(flt, ensure_ascii=False)}")

    text_fn, clip_img_fn, audio_fn = _embedding_fns(cfg)

    for logical, name in cols.items():
        try:
            info = sess.get(f"{base}/collections/{name}", timeout=10).json()
        except Exception as e:
            print(f"[{logical}] {name}: collection_info_error={e}")
            continue

        # Print vector distance/size if available (helps diagnose score_threshold issues)
        try:
            vectors = (((info.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors")
            if isinstance(vectors, dict):
                print(f"\n[{logical}] {name}")
                print(f"  vectors.size={vectors.get('size')} vectors.distance={vectors.get('distance')}")
            else:
                print(f"\n[{logical}] {name}")
        except Exception:
            print(f"\n[{logical}] {name}")

        # count points (optionally filtered)
        try:
            count_payload: Dict[str, Any] = {"exact": True}
            if flt:
                count_payload["filter"] = flt
            counted = _post(sess, f"{base}/collections/{name}/points/count", count_payload)
        except Exception as e:
            print(f"[{logical}] {name}: count_error={e}")
            counted = {}
        try:
            points = (((info.get("result") or {}).get("points_count")) if isinstance(info, dict) else None)
            print(f"  points_total={points}")
        except Exception:
            pass
        if counted:
            print(f"  points_filtered={((counted.get('result') or {}).get('count'))}")

        # Optional: run a live ANN search to see whether score_threshold is filtering everything.
        if args.query:
            try:
                q = str(args.query)
                vec: list[float] = []
                if logical == "text":
                    vec = list(text_fn(q) or [])
                elif logical == "clip_image":
                    vec = list(clip_img_fn(q) or [])
                elif logical == "audio":
                    vec = list(audio_fn(q) or [])
                else:
                    vec = []
                if vec:
                    search_payload: Dict[str, Any] = {
                        "vector": vec,
                        "limit": 5,
                        "with_payload": False,
                        "score_threshold": float(args.threshold),
                    }
                    if flt:
                        search_payload["filter"] = flt
                    out = _post(sess, f"{base}/collections/{name}/points/search", search_payload)
                    hits = (out.get("result") or []) if isinstance(out, dict) else []
                    top_score = None
                    try:
                        if hits:
                            top_score = float(hits[0].get("score"))
                    except Exception:
                        top_score = None
                    print(f"  live_search query='{q}' vec_len={len(vec)} threshold={args.threshold} hits={len(hits)} top_score={top_score}")
                else:
                    print(f"  live_search skipped (no embedder) for logical={logical}")
            except Exception as e:
                print(f"  live_search_error={e}")

        # scroll a few samples to see payload keys
        try:
            scroll_payload: Dict[str, Any] = {"limit": int(args.limit), "with_payload": True, "with_vector": False}
            if flt:
                scroll_payload["filter"] = flt
            sc = _post(sess, f"{base}/collections/{name}/points/scroll", scroll_payload)
            pts = (sc.get("result") or {}).get("points") or []
            print(f"  sample_points={len(pts)}")
            for p in pts[: int(args.limit)]:
                payload = p.get("payload") if isinstance(p, dict) else None
                md = (payload or {}).get("metadata") if isinstance(payload, dict) else None
                print(f"    id={p.get('id')} modality={(payload or {}).get('modality')} kind={(payload or {}).get('kind')}")
                if isinstance(md, dict):
                    print(
                        f"    metadata.tenant_id={md.get('tenant_id')} user_id={md.get('user_id')} memory_domain={md.get('memory_domain')} run_id={md.get('run_id')}"
                    )
        except Exception as e:
            print(f"  scroll_error={e}")


if __name__ == "__main__":
    main()
