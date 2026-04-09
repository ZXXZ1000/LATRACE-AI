from __future__ import annotations

"""
Export Neo4j nodes and relations (Entity graph) to JSONL files.

Usage (dry-run):
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/export_neo4j_graph.py --dry-run

Usage (live):
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/export_neo4j_graph.py \
      --uri bolt://127.0.0.1:7687 --user neo4j --password pass --nodes nodes.jsonl --rels rels.jsonl

Notes:
  - Requires Neo4j Python driver when not in dry-run.
  - Queries: Nodes with labels and properties; Relations with src/dst/rel and properties.
"""

import argparse
import json
import os
from typing import Tuple


def build_queries() -> Tuple[str, str]:
    q_nodes = "MATCH (n:Entity) RETURN n.id AS id, labels(n) AS labels, properties(n) AS props"
    q_rels = "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id AS src, type(r) AS rel, b.id AS dst, properties(r) AS props"
    return q_nodes, q_rels


def run_export(uri: str, user: str, password: str, nodes_out: str, rels_out: str, database: str = "neo4j") -> None:
    from neo4j import GraphDatabase  # type: ignore
    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with drv.session(database=database) as s:
            q_nodes, q_rels = build_queries()
            with open(nodes_out, "w", encoding="utf-8") as fn:
                for rec in s.run(q_nodes):
                    obj = {"id": rec["id"], "labels": rec["labels"], "props": rec["props"]}
                    fn.write(json.dumps(obj, ensure_ascii=False) + "\n")
            with open(rels_out, "w", encoding="utf-8") as fr:
                for rec in s.run(q_rels):
                    obj = {"src": rec["src"], "rel": rec["rel"], "dst": rec["dst"], "props": rec["props"]}
                    fr.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        try:
            drv.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Neo4j Entity graph to JSONL")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"))
    ap.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    ap.add_argument("--nodes", default="nodes.jsonl")
    ap.add_argument("--rels", default="rels.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        qn, qr = build_queries()
        print(json.dumps({"nodes_query": qn, "rels_query": qr}, ensure_ascii=False, indent=2))
        return

    if not (args.uri and args.user and args.password):
        raise SystemExit("Missing uri/user/password; use --dry-run to preview queries")
    run_export(str(args.uri), str(args.user), str(args.password), str(args.nodes), str(args.rels), database=str(args.database))
    print(f"Exported to {args.nodes} and {args.rels}")


if __name__ == "__main__":
    main()

