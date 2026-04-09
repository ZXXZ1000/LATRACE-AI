from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple
import logging

from modules.memory.infra.async_ingest_job_store import AsyncIngestJobStore, IngestJobRecord


RunJobFn = Callable[[IngestJobRecord], Awaitable[None]]


@dataclass(frozen=True)
class IngestExecutorConfig:
    enabled: bool = True
    worker_count: int = 2
    queue_maxsize: int = 0
    global_concurrency: int = 2
    per_tenant_concurrency: int = 1
    job_timeout_s: int = 900
    shutdown_grace_s: int = 30
    recover_stale_s: int = 3600
    retry_delay_s: int = 60
    max_retries: int = 3


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class IngestExecutor:
    def __init__(
        self,
        *,
        store: AsyncIngestJobStore,
        run_job: RunJobFn,
        config: IngestExecutorConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._store = store
        self._run_job = run_job
        self._cfg = config
        maxsize = int(config.queue_maxsize) if int(config.queue_maxsize) > 0 else 0
        self._queue: asyncio.Queue[Optional[Tuple[str, Optional[str]]]] = asyncio.Queue(maxsize=maxsize)
        self._workers: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()
        self._global_sem = asyncio.Semaphore(max(1, int(config.global_concurrency)))
        self._tenant_limit = max(1, int(config.per_tenant_concurrency))
        self._tenant_sems: Dict[str, asyncio.Semaphore] = {}
        self._tenant_lock = asyncio.Lock()
        self._running_jobs: Set[str] = set()
        self._running_lock = asyncio.Lock()
        self._delayed_tasks: Set[asyncio.Task[None]] = set()
        self._logger = logger or logging.getLogger(__name__)

    async def start(self) -> None:
        if self._workers:
            return
        self._stopping.clear()
        for idx in range(max(1, int(self._cfg.worker_count))):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))
        await self._recover_pending_jobs()

    async def stop(self) -> None:
        if not self._workers:
            return
        self._stopping.set()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except Exception:
                await self._queue.put(None)
        grace = max(1, int(self._cfg.shutdown_grace_s))
        try:
            await asyncio.wait_for(asyncio.gather(*self._workers), timeout=grace)
        except asyncio.TimeoutError:
            for t in self._workers:
                t.cancel()
        finally:
            self._workers.clear()
        await self._rollback_running_jobs()
        for task in list(self._delayed_tasks):
            task.cancel()
        self._delayed_tasks.clear()

    async def enqueue(self, job_id: str, *, tenant_id: Optional[str] = None, delay_s: float = 0.0) -> bool:
        if not self._workers or self._stopping.is_set():
            return False
        if delay_s and delay_s > 0:
            task = asyncio.create_task(self._delayed_enqueue(job_id, tenant_id, delay_s))
            self._delayed_tasks.add(task)
            task.add_done_callback(lambda t: self._delayed_tasks.discard(t))
            return True
        try:
            self._queue.put_nowait((str(job_id), tenant_id))
            return True
        except asyncio.QueueFull:
            return False

    async def _delayed_enqueue(self, job_id: str, tenant_id: Optional[str], delay_s: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay_s))
            await self.enqueue(job_id, tenant_id=tenant_id, delay_s=0.0)
        except asyncio.CancelledError:
            return

    async def _worker_loop(self, idx: int) -> None:
        while not self._stopping.is_set():
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            job_id, tenant_id = item
            try:
                await self._execute_one(job_id, tenant_id)
            except Exception as exc:
                self._logger.exception("ingest.executor.worker_failed", extra={"job_id": job_id, "reason": str(exc)})
            finally:
                self._queue.task_done()

    async def _execute_one(self, job_id: str, tenant_id: Optional[str]) -> None:
        record = await self._store.get_job(str(job_id))
        if record is None:
            return
        if record.next_retry_at:
            due_in = self._seconds_until(record.next_retry_at)
            if due_in is not None and due_in > 0:
                await self.enqueue(record.job_id, tenant_id=str(record.tenant_id), delay_s=due_in)
                return
        tenant = tenant_id or str(record.tenant_id)
        tenant_sem = await self._tenant_sem(tenant)
        async with self._global_sem:
            async with tenant_sem:
                await self._mark_running(record.job_id)
                try:
                    await asyncio.wait_for(self._run_job(record), timeout=max(1, int(self._cfg.job_timeout_s)))
                except asyncio.TimeoutError:
                    await self._handle_timeout(record)
                finally:
                    await self._unmark_running(record.job_id)

    async def _handle_timeout(self, record: IngestJobRecord) -> None:
        delay = max(1, int(self._cfg.retry_delay_s))
        latest = await self._store.get_job(record.job_id)
        attempts_src = latest.attempts if (latest is not None and latest.attempts is not None) else (record.attempts or {})
        attempts = int(attempts_src.get("stage3", 0))
        attempts = max(1, attempts)
        max_retries = max(0, int(self._cfg.max_retries))
        max_attempts = max(1, max_retries + 1)
        err = {"stage": "stage3", "code": "timeout", "message": "ingest_job_timeout"}
        if attempts >= max_attempts:
            await self._store.update_status(
                record.job_id,
                status="STAGE3_FAILED",
                error=err,
                next_retry_at="",
            )
            return
        next_retry = (_now_utc() + timedelta(seconds=delay)).isoformat()
        await self._store.update_status(
            record.job_id,
            status="RECEIVED",
            error=err,
            next_retry_at=next_retry,
        )
        await self.enqueue(record.job_id, tenant_id=str(record.tenant_id), delay_s=delay)

    async def _tenant_sem(self, tenant_id: str) -> asyncio.Semaphore:
        if not tenant_id:
            return asyncio.Semaphore(self._tenant_limit)
        async with self._tenant_lock:
            sem = self._tenant_sems.get(tenant_id)
            if sem is None:
                sem = asyncio.Semaphore(self._tenant_limit)
                self._tenant_sems[tenant_id] = sem
            return sem

    async def _mark_running(self, job_id: str) -> None:
        async with self._running_lock:
            self._running_jobs.add(str(job_id))

    async def _unmark_running(self, job_id: str) -> None:
        async with self._running_lock:
            self._running_jobs.discard(str(job_id))

    async def _rollback_running_jobs(self) -> None:
        async with self._running_lock:
            jobs = list(self._running_jobs)
            self._running_jobs.clear()
        for job_id in jobs:
            try:
                await self._store.update_status(job_id, status="RECEIVED")
            except Exception:
                continue

    async def _recover_pending_jobs(self) -> None:
        statuses = ["RECEIVED", "STAGE2_RUNNING", "STAGE3_RUNNING"]
        jobs = await self._store.list_jobs_by_status(statuses)
        if not jobs:
            return
        now = _now_utc()
        stale_s = max(0, int(self._cfg.recover_stale_s))
        for job in jobs:
            updated_at = _parse_iso(job.updated_at)
            if stale_s > 0 and updated_at is not None:
                if (now - updated_at).total_seconds() > stale_s:
                    err = {"stage": "recovery", "code": "stale_job", "message": "recovered_job_too_old"}
                    await self._store.update_status(job.job_id, status="STAGE3_FAILED", error=err)
                    continue
            if job.status in {"STAGE2_RUNNING", "STAGE3_RUNNING"}:
                transitioned = await self._store.try_transition_status(
                    job.job_id, from_statuses=[job.status], to_status="RECEIVED"
                )
                if not transitioned:
                    continue
            delay = self._seconds_until(job.next_retry_at)
            if delay is not None and delay > 0:
                await self.enqueue(job.job_id, tenant_id=str(job.tenant_id), delay_s=delay)
            else:
                await self.enqueue(job.job_id, tenant_id=str(job.tenant_id))

    def _seconds_until(self, ts: Optional[str]) -> Optional[float]:
        if not ts:
            return None
        dt = _parse_iso(ts)
        if dt is None:
            return None
        return max(0.0, (dt - _now_utc()).total_seconds())
