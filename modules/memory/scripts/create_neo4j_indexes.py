from __future__ import annotations

"""
Create recommended Neo4j schema (constraints/indexes) for Memory graph.

Usage:
  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/create_neo4j_indexes.py \
      --dry-run

  PYTHONPATH=MOYAN_Agent_Infra:. python3 MOYAN_Agent_Infra/modules/memory/scripts/create_neo4j_indexes.py \
      --uri bolt://127.0.0.1:7687 --user neo4j --password pass

By default reads env vars NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD if CLI args omitted.
Safe to run multiple times: uses IF NOT EXISTS.
"""

import argparse
import os
from typing import List


def build_index_statements() -> List[str]:
    return [
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX entity_domain IF NOT EXISTS FOR (n:Entity) ON (n.memory_domain)",
        "CREATE INDEX entity_users IF NOT EXISTS FOR (n:Entity) ON (n.user_id)",
        "CREATE INDEX entity_run IF NOT EXISTS FOR (n:Entity) ON (n.run_id)",
    ]


def run_apply(uri: str, user: str, password: str, statements: List[str], database: str = "neo4j") -> None:
    from neo4j import GraphDatabase  # type: ignore
    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with drv.session(database=database) as s:
            for q in statements:
                s.run(q)
    finally:
        try:
            drv.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Neo4j schema for Memory graph")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI"), help="Neo4j URI, e.g. bolt://127.0.0.1:7687")
    ap.add_argument("--user", default=os.getenv("NEO4J_USER"), help="Neo4j user")
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), help="Neo4j password")
    ap.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Database name")
    ap.add_argument("--dry-run", action="store_true", help="Print statements only, do not connect")
    args = ap.parse_args()

    stmts = build_index_statements()
    if args.dry_run:
        print("-- Neo4j index/constraint statements --")
        for q in stmts:
            print(q)
        return

    uri = os.path.expandvars(args.uri or "")
    user = os.path.expandvars(args.user or "")
    password = os.path.expandvars(args.password or "")
    if not (uri and user and password):
        raise SystemExit("Missing Neo4j connection parameters; use --dry-run to preview statements.")
    run_apply(uri, user, password, stmts, database=str(args.database or "neo4j"))
    print("Applied indexes/constraints successfully.")


if __name__ == "__main__":
    main()

