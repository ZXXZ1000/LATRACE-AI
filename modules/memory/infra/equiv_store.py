from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modules.memory.contracts.graph_models import PendingEquiv
from modules.memory.infra.neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)


class EquivStore:
    """Identity registry store for PendingEquiv lifecycle (pending -> approved/rejected)."""

    def __init__(self, neo: Neo4jStore):
        self.neo = neo

    def _ensure_driver(self) -> bool:
        return bool(getattr(self.neo, "_driver", None))

    def list_pending(
        self,
        *,
        tenant_id: str,
        status: str = "pending",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not self._ensure_driver():
            return []
        cypher = """
MATCH (p:PendingEquiv {tenant_id:$t})
WHERE ($status IS NULL OR p.status = $status)
RETURN p
LIMIT $limit
"""
        try:
            with self.neo._driver.session(database=self.neo._database) as sess:  # type: ignore[attr-defined]
                rows = sess.run(cypher, t=tenant_id, status=status, limit=int(limit))
                return [dict(row["p"]) for row in rows]
        except Exception as exc:
            logger.error("equiv.list.error", extra={"event": "equiv.list", "status": "error", "reason": str(exc)}, exc_info=True)
            return []

    def upsert_pending(self, *, tenant_id: str, records: List[PendingEquiv]) -> None:
        if not self._ensure_driver():
            return None
        peq_dicts = [r.model_dump(mode="python") for r in records]
        cypher = (
            "UNWIND $peq AS p "
            "MERGE (n:PendingEquiv {id: p.id, tenant_id: p.tenant_id}) "
            "SET n += p "
        )
        with self.neo._driver.session(database=self.neo._database) as sess:  # type: ignore[attr-defined]
            sess.run(cypher, peq=peq_dicts)

    def approve(
        self,
        *,
        tenant_id: str,
        pending_id: str,
        reviewer: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._ensure_driver():
            return {"merged": 0}
        cypher = """
MATCH (p:PendingEquiv {id:$id, tenant_id:$t})
SET p.status = 'approved',
    p.reviewer = $reviewer,
    p.reviewed_at = datetime()
WITH p
MATCH (a:Entity {id:p.entity_id, tenant_id:$t})
MATCH (b:Entity {id:p.candidate_id, tenant_id:$t})
MERGE (a)-[r:EQUIV]->(b)
SET r.tenant_id = $t,
    r.confidence = coalesce(p.confidence, r.confidence),
    r.source = coalesce(p.provenance.source, r.source)
RETURN p, r
"""
        try:
            with self.neo._driver.session(database=self.neo._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, id=pending_id, t=tenant_id, reviewer=reviewer).single()
                return {"merged": 1 if row else 0}
        except Exception as exc:
            logger.error("equiv.approve.error", extra={"event": "equiv.approve", "status": "error", "reason": str(exc)}, exc_info=True)
            return {"merged": 0}

    def reject(self, *, tenant_id: str, pending_id: str, reviewer: Optional[str] = None) -> Dict[str, Any]:
        if not self._ensure_driver():
            return {"updated": 0}
        cypher = """
MATCH (p:PendingEquiv {id:$id, tenant_id:$t})
SET p.status = 'rejected',
    p.reviewer = $reviewer,
    p.reviewed_at = datetime()
RETURN p
"""
        try:
            with self.neo._driver.session(database=self.neo._database) as sess:  # type: ignore[attr-defined]
                row = sess.run(cypher, id=pending_id, t=tenant_id, reviewer=reviewer).single()
                return {"updated": 1 if row else 0}
        except Exception as exc:
            logger.error("equiv.reject.error", extra={"event": "equiv.reject", "status": "error", "reason": str(exc)}, exc_info=True)
            return {"updated": 0}
