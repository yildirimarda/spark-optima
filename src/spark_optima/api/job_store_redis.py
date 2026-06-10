# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Redis-backed job store for asynchronous optimization runs.

This module provides RedisJobStore, a shared implementation of the job
store interface defined in :mod:`spark_optima.api.jobs`. It is enabled
with ``SPARK_OPTIMA_JOB_STORE=redis`` and is the recommended backend for
multi-replica deployments: every API replica reads and writes the same
Redis keys, so ``GET /api/v1/jobs/{id}`` works no matter which replica
accepted the job.

The ``redis`` package is an **optional dependency** (like boto3): this
module always imports cleanly, and the store factory in
:mod:`spark_optima.api.jobs` logs a warning and falls back to the
in-memory store when the package is missing or the server is unreachable
at startup — a misconfigured environment never crashes the API.

Storage model:

- One key per job: ``spark_optima:job:<job_id>`` holding the job record
  as a JSON string.
- TTL semantics match the other stores: when a job finishes
  (completed/failed), the key gets a Redis ``EXPIRE`` of ``ttl_hours``
  so Redis evicts it natively — no cleanup sweeps are needed.
- Listing uses ``SCAN`` over the key prefix and sorts newest-first by
  the ``submitted_ts`` stored in the payload.

The optimization work itself still runs in-process on a thread pool. If
the process executing a job dies mid-run, the record would stay
"running" forever, so the same staleness rule as the SQLite store
applies: unfinished jobs older than ``stale_after_hours`` (default 2)
are marked failed with a "worker lost" error when they are next read.

Connection configuration comes from the ``SPARK_OPTIMA_REDIS_URL``
environment variable (default ``redis://localhost:6379/0``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from spark_optima.api.job_store_sqlite import DEFAULT_STALE_AFTER_HOURS, WORKER_LOST_ERROR
from spark_optima.api.jobs import DEFAULT_JOB_TTL_HOURS, BaseJobStore, Job, _utc_now_iso

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from spark_optima.api.jobs import WebhookStatus

try:  # pragma: no cover — exercised via REDIS_AVAILABLE in tests
    import redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    redis = None

#: Whether the optional ``redis`` package is importable.
REDIS_AVAILABLE = redis is not None

logger = logging.getLogger(__name__)

#: Environment variable holding the Redis connection URL.
REDIS_URL_ENV_VAR = "SPARK_OPTIMA_REDIS_URL"

#: Default Redis connection URL.
DEFAULT_REDIS_URL = "redis://localhost:6379/0"

#: Prefix for all job keys in Redis.
REDIS_KEY_PREFIX = "spark_optima:job:"


def _job_to_json(job: Job) -> str:
    """Serialize a Job to its JSON payload.

    Args:
        job: The job record to serialize.

    Returns:
        JSON string with all Job fields.
    """
    return json.dumps(asdict(job))


def _job_from_json(raw: str | bytes) -> Job:
    """Deserialize a Job from its JSON payload.

    Args:
        raw: JSON string (or bytes) previously produced by _job_to_json.

    Returns:
        Populated Job instance.
    """
    data = json.loads(raw)
    return Job(**data)


class RedisJobStore(BaseJobStore):
    """Redis-backed registry of asynchronous optimization jobs.

    Implements the same interface as the in-memory store but persists job
    state as JSON strings in Redis, so jobs survive restarts and are
    visible to every API replica sharing the same Redis instance. Finished
    jobs expire natively via Redis ``EXPIRE`` instead of cleanup sweeps.

    Job state transitions are read-modify-write on a single key. They are
    safe in practice because each job is only ever written by the process
    that accepted it, and its transitions run sequentially on one worker
    thread.

    Attributes:
        url: Redis connection URL the store was created with.
    """

    def __init__(
        self,
        url: str | None = None,
        ttl_hours: float = DEFAULT_JOB_TTL_HOURS,
        executor: Executor | None = None,
        stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
        client: Any | None = None,
    ) -> None:
        """Initialize the store and verify connectivity.

        Args:
            url: Optional Redis connection URL. Falls back to the
                ``SPARK_OPTIMA_REDIS_URL`` environment variable, then to
                ``redis://localhost:6379/0``.
            ttl_hours: Hours to retain finished (completed/failed) jobs,
                applied as a Redis EXPIRE. Non-positive disables expiry.
            executor: Optional executor override; defaults to the shared
                module-level thread pool when None.
            stale_after_hours: Hours after which an unfinished job is
                reported as failed ("worker lost"). Non-positive disables
                the staleness rule.
            client: Optional pre-built Redis client (dependency injection
                for tests). Must implement get/set/delete/scan_iter/
                expire/ping with ``decode_responses=True`` semantics.

        Raises:
            RuntimeError: If the optional ``redis`` package is not
                installed and no client was injected.
            Exception: Any connection error raised by ``ping()`` when the
                server is unreachable (the factory catches this and falls
                back to the in-memory store).
        """
        super().__init__(ttl_hours=ttl_hours, executor=executor)
        self._stale_seconds = stale_after_hours * 3600.0
        self.url = url or os.environ.get(REDIS_URL_ENV_VAR) or DEFAULT_REDIS_URL
        if client is not None:
            self._client = client
        else:
            if redis is None:
                raise RuntimeError(
                    "The 'redis' package is required for the redis job store; install it with `uv add redis`"
                )
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        # Fail fast at store creation so the factory can fall back to the
        # in-memory store instead of failing on the first request.
        self._client.ping()

    @staticmethod
    def _key(job_id: str) -> str:
        """Build the Redis key for a job id.

        Args:
            job_id: Job identifier.

        Returns:
            Namespaced Redis key.
        """
        return f"{REDIS_KEY_PREFIX}{job_id}"

    def _save(self, job: Job) -> None:
        """Write a job payload, re-applying expiry for finished jobs.

        Redis SET clears any existing TTL, so the remaining TTL (relative
        to ``finished_ts``) is recomputed and re-applied on every write of
        a finished job. Jobs whose TTL already elapsed are deleted.

        Args:
            job: The job record to persist.
        """
        key = self._key(job.job_id)
        if job.finished_ts is None or self._ttl_seconds <= 0:
            self._client.set(key, _job_to_json(job))
            return
        remaining = int(job.finished_ts + self._ttl_seconds - time.time())
        if remaining <= 0:
            self._client.delete(key)
            return
        self._client.set(key, _job_to_json(job))
        self._client.expire(key, remaining)

    def _load(self, raw: str | bytes) -> Job:
        """Deserialize a payload and apply the staleness rule.

        Unfinished jobs older than the staleness window are marked failed
        with a "worker lost" error and the transition is persisted, exactly
        like the SQLite store's cleanup.

        Args:
            raw: JSON payload read from Redis.

        Returns:
            The (possibly transitioned) job record.
        """
        job = _job_from_json(raw)
        if self._stale_seconds <= 0 or job.status not in ("pending", "running"):
            return job
        cutoff = time.time() - self._stale_seconds
        if job.status == "running" and job.started_at is not None:
            reference = datetime.fromisoformat(job.started_at).timestamp()
        else:
            reference = job.submitted_ts
        if reference >= cutoff:
            return job
        job.status = "failed"
        job.error = WORKER_LOST_ERROR
        job.finished_at = _utc_now_iso()
        job.finished_ts = time.time()
        self._save(job)
        return job

    def get(self, job_id: str) -> Job | None:
        """Look up a job by identifier.

        Args:
            job_id: Job identifier returned by submit().

        Returns:
            The Job if known, None otherwise (including after TTL expiry).
        """
        raw = self._client.get(self._key(job_id))
        if raw is None:
            return None
        return self._load(raw)

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List known jobs, newest first.

        Iterates the key prefix with SCAN (never KEYS) and sorts by the
        submission timestamp stored in each payload.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            Jobs ordered by submission time descending.
        """
        jobs: list[Job] = []
        for key in self._client.scan_iter(match=f"{REDIS_KEY_PREFIX}*"):
            raw = self._client.get(key)
            if raw is None:  # pragma: no cover — expired between SCAN and GET
                continue
            jobs.append(self._load(raw))
        jobs.sort(key=lambda job: (-job.submitted_ts, job.job_id))
        return jobs[:limit]

    def _add_job(self, job: Job) -> None:
        """Persist a freshly submitted job.

        Args:
            job: The new job record to store.
        """
        self._save(job)

    def _update(self, job_id: str, **changes: Any) -> None:
        """Apply field changes to a stored job and persist them.

        Args:
            job_id: Identifier of the job to update.
            **changes: Job attribute names mapped to their new values.
        """
        raw = self._client.get(self._key(job_id))
        if raw is None:  # pragma: no cover — job expired mid-transition
            logger.warning(f"Job {job_id} disappeared from Redis before its state could be updated")
            return
        job = _job_from_json(raw)
        for name, value in changes.items():
            setattr(job, name, value)
        self._save(job)

    def _mark_running(self, job_id: str) -> None:
        """Transition a job to the running state.

        Args:
            job_id: Identifier of the job to update.
        """
        self._update(job_id, status="running", started_at=_utc_now_iso())

    def _mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        """Transition a job to the completed state.

        Args:
            job_id: Identifier of the job to update.
            result: The optimization result payload.
        """
        self._update(
            job_id, status="completed", result=result, error=None, finished_at=_utc_now_iso(), finished_ts=time.time()
        )

    def _mark_failed(self, job_id: str, error: str) -> None:
        """Transition a job to the failed state.

        Args:
            job_id: Identifier of the job to update.
            error: Human-readable failure message.
        """
        self._update(job_id, status="failed", error=error, finished_at=_utc_now_iso(), finished_ts=time.time())

    def set_webhook_status(self, job_id: str, webhook_status: WebhookStatus) -> None:
        """Record the webhook delivery outcome on a finished job.

        Args:
            job_id: Identifier of the job to update.
            webhook_status: "delivered" or "failed".
        """
        self._update(job_id, webhook_status=webhook_status)
