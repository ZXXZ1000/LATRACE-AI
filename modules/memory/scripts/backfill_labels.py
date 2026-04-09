from __future__ import annotations

"""
Backfill type labels on existing Neo4j nodes based on properties (kind/modality).

Usage:
  python -m modules.memory.scripts.backfill_labels [--dry-run]

Reads Neo4j connection from memory.config.yaml and env overrides (NEO4J_URI/USER/PASSWORD).
"""

import argparse
from typing import Any, Dict

from modules.memory.application.config import load_memory_config


def _connect(cfg: Dict[str, Any]):
    from neo4j import GraphDatabase  # type: ignore
    gcfg = (cfg.get("memory", {}) or {}).get("graph_store", {})
    uri = str(gcfg.get("uri", "bolt://127.0.0.1:7687"))
    user = str(gcfg.get("user", "neo4j"))
    pwd = str(gcfg.get("password", "password"))
    return GraphDatabase.driver(uri, auth=(user, pwd))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_memory_config()
    drv = None
    try:
        drv = _connect(cfg)
    except Exception as e:
        print(f"[ERROR] cannot connect neo4j: {e}")
        return 2

    cql = [
        "MATCH (n:Entity) WHERE coalesce(n.kind,'')='episodic' SET n:Episodic",
        "MATCH (n:Entity) WHERE coalesce(n.kind,'')='semantic' AND coalesce(n.modality,'')='image' SET n:Image",
        "MATCH (n:Entity) WHERE coalesce(n.kind,'')='semantic' AND coalesce(n.modality,'')='audio' SET n:Voice",
        "MATCH (n:Entity) WHERE coalesce(n.kind,'')='semantic' AND coalesce(n.modality,'')='structured' SET n:Structured",
        "MATCH (n:Entity) WHERE coalesce(n.kind,'')='semantic' AND NOT coalesce(n.modality,'') IN ['image','audio','structured'] SET n:Semantic",
    ]
    if args.dry_run:
        print("[DRY-RUN] Cypher to run:")
        for q in cql:
            print("  ", q)
        return 0

    with drv.session() as sess:  # default db
        for q in cql:
            sess.run(q)
    print("[OK] labels backfilled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

