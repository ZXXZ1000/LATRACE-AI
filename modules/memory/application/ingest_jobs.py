"""Application-facing re-export for ingest job storage."""

from modules.memory.infra.ingest_job_store import IngestJobRecord, IngestJobStore

__all__ = ["IngestJobRecord", "IngestJobStore"]

