import os
import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from modules.memory.infra.neo4j_store import Neo4jStore


@pytest.mark.skipif(os.getenv("NEO4J_URI") is None, reason="Neo4j not configured")
def test_state_chain_update_basic() -> None:
    store = Neo4jStore(
        {
            "uri": os.getenv("NEO4J_URI"),
            "user": os.getenv("NEO4J_USER"),
            "password": os.getenv("NEO4J_PASSWORD"),
        }
    )
    health = asyncio.run(store.health())
    if health.get("status") != "ok":
        store.close()
        pytest.skip("Neo4j not reachable")
    store.ensure_schema_v0()
    tenant = f"test_state_chain_{uuid.uuid4().hex[:6]}"
    subject = f"entity::{uuid.uuid4().hex[:6]}"
    prop = "job_status"
    t0 = datetime.now(timezone.utc)

    asyncio.run(
        store.apply_state_update(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            value="employed",
            valid_from=t0,
        )
    )
    asyncio.run(
        store.apply_state_update(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            value="employed",
            valid_from=t0 + timedelta(minutes=1),
        )
    )
    asyncio.run(
        store.apply_state_update(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            value="unemployed",
            valid_from=t0 + timedelta(minutes=2),
        )
    )

    cur = asyncio.run(
        store.get_current_state(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
        )
    )
    assert cur is not None
    assert cur.get("value") == "unemployed"

    at_t = asyncio.run(
        store.get_state_at_time(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            t=t0 + timedelta(seconds=30),
        )
    )
    assert at_t is not None
    assert at_t.get("value") == "employed"

    changes = asyncio.run(
        store.get_state_changes(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            start=t0,
            end=t0 + timedelta(minutes=3),
            order="asc",
        )
    )
    assert len(changes) == 2
    assert changes[0].get("value") == "employed"
    assert changes[1].get("value") == "unemployed"

    pending = asyncio.run(
        store.apply_state_update(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
            value="job_seeking",
            valid_from=t0 - timedelta(minutes=5),
        )
    )
    assert pending.get("pending") is True
    cur2 = asyncio.run(
        store.get_current_state(
            tenant_id=tenant,
            subject_id=subject,
            property=prop,
        )
    )
    assert cur2 is not None
    assert cur2.get("value") == "unemployed"

    store.close()
