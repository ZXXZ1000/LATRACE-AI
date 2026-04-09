"""Unit tests for PostgreSQL ingest job store.

These tests verify the PgIngestJobStore implementation works correctly.
For integration tests, use a real PostgreSQL database via testcontainers or docker.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import datetime, timezone

from modules.memory.infra.pg_ingest_job_store import (
    PgIngestJobStore,
    PgIngestJobStoreSettings,
)
from modules.memory.infra.async_ingest_job_store import IngestJobRecord


class TestPgIngestJobStoreSettings:
    """Tests for PgIngestJobStoreSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = PgIngestJobStoreSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.user == "postgres"
        assert settings.password == ""
        assert settings.database == "memory"
        assert settings.pool_min == 2
        assert settings.pool_max == 10

    def test_from_env(self, monkeypatch):
        """Test loading settings from environment."""
        monkeypatch.setenv("MEMORY_PG_HOST", "db.example.com")
        monkeypatch.setenv("MEMORY_PG_PORT", "5433")
        monkeypatch.setenv("MEMORY_PG_USER", "testuser")
        monkeypatch.setenv("MEMORY_PG_PASSWORD", "testpass")
        monkeypatch.setenv("MEMORY_PG_DATABASE", "testdb")
        monkeypatch.setenv("MEMORY_PG_POOL_MIN", "5")
        monkeypatch.setenv("MEMORY_PG_POOL_MAX", "20")

        settings = PgIngestJobStoreSettings.from_env()
        assert settings.host == "db.example.com"
        assert settings.port == 5433
        assert settings.user == "testuser"
        assert settings.password == "testpass"
        assert settings.database == "testdb"
        assert settings.pool_min == 5
        assert settings.pool_max == 20


class TestPgIngestJobStoreRowToRecord:
    """Tests for row to record conversion."""

    def test_row_to_record(self):
        """Test converting database row to IngestJobRecord."""
        store = PgIngestJobStore.__new__(PgIngestJobStore)
        store._settings = PgIngestJobStoreSettings()
        store._pool = None
        store._schema_initialized = False

        # Create a mock row
        row = {
            "job_id": "job_abc123",
            "session_id": "session_1",
            "commit_id": "commit_1",
            "tenant_id": "tenant_1",
            "api_key_id": "key_1",
            "request_id": "req_1",
            "user_tokens": ["user:alice"],
            "memory_domain": "dialog",
            "llm_policy": "default",
            "status": "RECEIVED",
            "attempts": {"stage2": 0, "stage3": 0},
            "next_retry_at": None,
            "last_error": None,
            "metrics": {"archived_turns": 5},
            "created_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "cursor_committed": "t0005",
            "turns": [{"turn_id": "t0001", "text": "Hello"}],
            "client_meta": {"source": "test"},
            "stage2_marks": [],
            "stage2_pin_intents": [],
            "payload_raw": None,
        }

        # Mock asyncpg.Record behavior
        class MockRecord(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

        mock_row = MockRecord(row)
        record = store._row_to_record(mock_row)

        assert record.job_id == "job_abc123"
        assert record.session_id == "session_1"
        assert record.tenant_id == "tenant_1"
        assert record.status == "RECEIVED"
        assert record.user_tokens == ["user:alice"]
        assert record.memory_domain == "dialog"


class TestPgIngestJobStorePayloadCheck:
    """Tests for payload validation."""

    def test_payload_has_core_with_turns(self):
        """Test payload check with valid turns."""
        store = PgIngestJobStore.__new__(PgIngestJobStore)

        job = IngestJobRecord(
            job_id="job_1",
            session_id="session_1",
            commit_id=None,
            tenant_id="tenant_1",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:alice"],
            memory_domain="dialog",
            llm_policy="default",
            status="RECEIVED",
            attempts={"stage2": 0, "stage3": 0},
            next_retry_at=None,
            last_error=None,
            metrics={},
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            cursor_committed=None,
            turns=[{"turn_id": "t1", "text": "Hello"}],
            client_meta={},
        )

        assert store._payload_has_core(job) is True

    def test_payload_has_core_empty_turns(self):
        """Test payload check with empty turns."""
        store = PgIngestJobStore.__new__(PgIngestJobStore)

        job = IngestJobRecord(
            job_id="job_1",
            session_id="session_1",
            commit_id=None,
            tenant_id="tenant_1",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:alice"],
            memory_domain="dialog",
            llm_policy="default",
            status="RECEIVED",
            attempts={"stage2": 0, "stage3": 0},
            next_retry_at=None,
            last_error=None,
            metrics={},
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            cursor_committed=None,
            turns=[],
            client_meta={},
        )

        assert store._payload_has_core(job) is False

    def test_payload_has_core_no_user_tokens(self):
        """Test payload check with no user tokens."""
        store = PgIngestJobStore.__new__(PgIngestJobStore)

        job = IngestJobRecord(
            job_id="job_1",
            session_id="session_1",
            commit_id=None,
            tenant_id="tenant_1",
            api_key_id=None,
            request_id=None,
            user_tokens=[],
            memory_domain="dialog",
            llm_policy="default",
            status="RECEIVED",
            attempts={"stage2": 0, "stage3": 0},
            next_retry_at=None,
            last_error=None,
            metrics={},
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            cursor_committed=None,
            turns=[{"turn_id": "t1", "text": "Hello"}],
            client_meta={},
        )

        assert store._payload_has_core(job) is False


@pytest.mark.asyncio
class TestPgIngestJobStoreIntegration:
    """Integration tests requiring a PostgreSQL connection.

    These tests are skipped by default. Run with:
        MEMORY_PG_HOST=localhost MEMORY_PG_PASSWORD=test pytest -m integration
    """

    @pytest_asyncio.fixture
    async def store(self):
        """Create a store instance for testing."""
        import os
        if not os.getenv("MEMORY_PG_HOST"):
            pytest.skip("PostgreSQL not available (set MEMORY_PG_HOST)")

        store = PgIngestJobStore()
        yield store
        await store.close()

    @pytest.mark.integration
    async def test_create_and_get_job(self, store):
        """Test creating and retrieving a job."""
        import uuid
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"

        job, created = await store.create_job(
            session_id=session_id,
            commit_id=None,
            tenant_id="test_tenant",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:test"],
            memory_domain="dialog",
            llm_policy="default",
            turns=[{"turn_id": "t1", "text": "Hello"}],
            base_turn_id=None,
            client_meta=None,
        )

        assert created is True
        assert job.session_id == session_id
        assert job.status == "RECEIVED"

        # Retrieve the job
        retrieved = await store.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    @pytest.mark.integration
    async def test_commit_id_idempotency(self, store):
        """Test that duplicate commit_id returns existing job."""
        import uuid
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        commit_id = f"commit_{uuid.uuid4().hex[:8]}"

        job1, created1 = await store.create_job(
            session_id=session_id,
            commit_id=commit_id,
            tenant_id="test_tenant",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:test"],
            memory_domain="dialog",
            llm_policy="default",
            turns=[{"turn_id": "t1", "text": "Hello"}],
            base_turn_id=None,
            client_meta=None,
        )

        assert created1 is True

        # Second call with same commit_id should return existing job
        job2, created2 = await store.create_job(
            session_id=session_id,
            commit_id=commit_id,
            tenant_id="test_tenant",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:test"],
            memory_domain="dialog",
            llm_policy="default",
            turns=[{"turn_id": "t2", "text": "World"}],
            base_turn_id=None,
            client_meta=None,
        )

        assert created2 is False
        assert job2.job_id == job1.job_id

    @pytest.mark.integration
    async def test_update_status(self, store):
        """Test updating job status."""
        import uuid
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"

        job, _ = await store.create_job(
            session_id=session_id,
            commit_id=None,
            tenant_id="test_tenant",
            api_key_id=None,
            request_id=None,
            user_tokens=["user:test"],
            memory_domain="dialog",
            llm_policy="default",
            turns=[{"turn_id": "t1", "text": "Hello"}],
            base_turn_id=None,
            client_meta=None,
        )

        await store.update_status(
            job.job_id,
            status="PROCESSING",
            stage="stage2",
            attempt_inc=True,
        )

        updated = await store.get_job(job.job_id)
        assert updated is not None
        assert updated.status == "PROCESSING"
        assert updated.attempts["stage2"] == 1
