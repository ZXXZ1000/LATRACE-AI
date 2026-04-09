from __future__ import annotations

"""
Verify Qdrant collections against memory.config.yaml (dims/metric) and optionally check live service.

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/verify_qdrant_collections.py --dry-run

  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/verify_qdrant_collections.py \
      --host 127.0.0.1 --port 6333 --check

Notes:
  - Dry-run uses only config; no network required (suitable for CI).
  - Live check performs HTTP GET /collections/<name> and compares vector_size/distance.
"""

import argparse
import os
import json
from typing import Any, Dict, List


def _load_memory_config() -> Dict[str, Any]:
    import yaml
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "memory.config.yaml"))
    if not os.path.exists(base):
        return {}
    raw = open(base, "r", encoding="utf-8").read()
    raw = os.path.expandvars(raw)
    return yaml.safe_load(raw) or {}


def plan_checks_from_config(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    mem = cfg.get("memory", {}) or {}
    vs = mem.get("vector_store", {}) or {}
    cols = (vs.get("collections", {}) or {})
    emb = vs.get("embedding", {}) or {}
    checks: List[Dict[str, Any]] = []
    # text
    checks.append({
        "collection": cols.get("text", "memory_text"),
        "expected_dim": int(emb.get("dim", 768)),
        "expected_distance": str(emb.get("distance", "cosine")),
        "modality": "text",
    })
    # image
    i = (emb.get("image", {}) or {})
    checks.append({
        "collection": cols.get("image", "memory_image"),
        "expected_dim": int(i.get("dim", 512)),
        "expected_distance": str(emb.get("distance", "cosine")),
        "modality": "image",
    })
    # audio
    a = (emb.get("audio", {}) or {})
    checks.append({
        "collection": cols.get("audio", "memory_audio"),
        "expected_dim": int(a.get("dim", 192)),
        "expected_distance": str(emb.get("distance", "cosine")),
        "modality": "audio",
    })
    return checks


def live_fetch_collection(host: str, port: int, name: str) -> Dict[str, Any]:
    import urllib.request
    url = f"http://{host}:{port}/collections/{name}"
    with urllib.request.urlopen(url, timeout=3.0) as resp:  # nosec - internal/local invocation
        data = resp.read().decode("utf-8")
        return json.loads(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Qdrant collections against config")
    ap.add_argument("--host", default=os.getenv("QDRANT_HOST", "127.0.0.1"))
    ap.add_argument("--port", default=int(os.getenv("QDRANT_PORT", "6333")))
    ap.add_argument("--check", action="store_true", help="Perform live HTTP checks against Qdrant")
    ap.add_argument("--dry-run", action="store_true", help="Print expected plan only (no network)")
    args = ap.parse_args()

    cfg = _load_memory_config()
    plan = plan_checks_from_config(cfg)
    if args.dry_run and not args.check:
        print(json.dumps({"plan": plan}, ensure_ascii=False, indent=2))
        return

    if not args.check:
        print("No live check requested. Use --check or --dry-run.")
        return

    # live check
    failures = 0
    for item in plan:
        name = item["collection"]
        try:
            info = live_fetch_collection(args.host, int(args.port), name)
            vspec = (((info or {}).get("result") or {}).get("vectors") or {})
            # can be dict or list; handle dict with 'size' and 'distance'
            size = None
            dist = None
            if isinstance(vspec, dict):
                size = vspec.get("size")
                dist = vspec.get("distance")
            if size is None:
                raise ValueError("Cannot parse vector spec from response")
            ok_dim = int(size) == int(item["expected_dim"])
            ok_dist = (str(dist).lower() == str(item["expected_distance"]).lower()) if dist is not None else True
            status = "OK" if (ok_dim and ok_dist) else "MISMATCH"
            if status != "OK":
                failures += 1
            print(f"[{status}] {name}: expected dim={item['expected_dim']} distance={item['expected_distance']}; got size={size} distance={dist}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {name}: {e}")

    if failures:
        raise SystemExit(f"Checks finished with {failures} failure(s)")
    print("All collections match expected configuration.")


if __name__ == "__main__":
    main()
