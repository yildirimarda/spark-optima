# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""SQLite-backed job store for asynchronous optimization runs.

This module provides SQLiteJobStore, a persistent implementation of the
job store interface defined in :mod:`spark_optima.api.jobs`. It is enabled
with ``SPARK_OPTIMA_JOB_STORE=sqlite`` and persists job *state* only:

- Job records survive API restarts, so history is not lost.
- Multiple uvicorn workers on one node share the same database file (WAL
  journal mode allows concurrent readers alongside a writer), so any
  worker can answer polling requests for jobs submitted to another.

The optimization work itself still runs in-process on a thread pool. If
the process executing a job dies mid-run, no other process picks the work
up — the row would stay "running" forever. A staleness rule covers this:
unfinished jobs older than ``stale_after_hours`` (default
``DEFAULT_STALE_AFTER_HOURS`` = 2) are marked failed with a "worker lost"
error during opportunistic cleanup. True multi-node deployments need an
external job queue instead of this store.

The database location is resolved in the following order:

1. Explicit ``db_path`` constructor argument.
2. The ``SPARK_OPTIMA_JOB_DB`` environment variable.
3. The default ``~/.spark_optima/jobs.db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spark_optima.api.jobs import DEFAULT_JOB_TTL_HOURS, BaseJobStore, Job, _utc_now_iso

if TYPE_CHECKING:
    from concurrent.futures import Executor

JOB_DB_ENV_VAR = "SPARK_OPTIMA_JOB_DB"

#: Hours after which an unfinished job is presumed lost (its worker
#: process died or was restarted mid-run) and is reported as failed.
DEFAULT_STALE_AFTER_HOURS = 2.0

#: Error message recorded on jobs failed by the staleness rule.
WORKER_LOST_ERROR = "worker lost: the job did not finish in time; the worker process likely died or was restarted"

#: SQLite busy timeout in seconds. Generous because WAL still serializes
#: writers and optimization jobs may finish concurrently.
_CONNECT_TIMEOUT_SECONDS = 30.0

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    platform TEXT NOT NULL,
    spark_version TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    submitted_ts REAL NOT NULL DEFAULT 0.0,
    finished_ts REAL
)
"""


def _default_db_path() -> Path:
    """Resolve the default job database path.

    Returns:
        Path from the ``SPARK_OPTIMA_JOB_DB`` environment variable if set,
        otherwise ``~/.spark_optima/jobs.db``.
    """
    env_path = os.environ.get(JOB_DB_ENV_VAR)
    if env_path:
        return Path(env_path)
    return Path.home() / ".spark_optima" / "jobs.db"


def _job_from_row(row: sqlite3.Row) -> Job:
    """Build a Job from a database row.

    Args:
        row: SQLite row with all ``jobs`` columns.

    Returns:
        Populated Job instance.
    """
    result_json = row["result_json"]
    return Job(
        job_id=str(row["job_id"]),
        platform=str(row["platform"]),
        spark_version=str(row["spark_version"]),
        status=row["status"],
        submitted_at=str(row["submitted_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        result=json.loads(result_json) if result_json else None,
        error=row["error"],
        submitted_ts=float(row["submitted_ts"]),
        finished_ts=row["finished_ts"],
    )


class SQLiteJobStore(BaseJobStore):
    """SQLite-backed registry of asynchronous optimization jobs.

    Implements the same interface as the in-memory store but persists job
    state to a SQLite database, so jobs survive restarts and are visible
    to every worker process on the node. Each call opens a short-lived
    connection — no connection object is shared across threads. All
    queries are parameterized.

    Cleanup runs opportunistically on every access, exactly like the
    in-memory store:

    - Finished (completed/failed) jobs older than the TTL are deleted.
    - Unfinished jobs older than the staleness window are marked failed
      with a "worker lost" error, since the process that owned them can
      no longer report progress (the executor runs in-process).

    Attributes:
        db_path: Resolved path of the SQLite database file.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        ttl_hours: float = DEFAULT_JOB_TTL_HOURS,
        executor: Executor | None = None,
        stale_after_hours: float = DEFAULT_STALE_AFTER_HOURS,
    ) -> None:
        """Initialize the store and ensure the schema exists.

        Args:
            db_path: Optional database file path. Falls back to the
                ``SPARK_OPTIMA_JOB_DB`` environment variable, then to
                ``~/.spark_optima/jobs.db``. Parent directories are
                created as needed.
            ttl_hours: Hours to retain finished (completed/failed) jobs.
            executor: Optional executor override; defaults to the shared
                module-level thread pool when None.
            stale_after_hours: Hours after which an unfinished job is
                reported as failed ("worker lost"). Non-positive disables
                the staleness rule.
        """
        super().__init__(ttl_hours=ttl_hours, executor=executor)
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stale_seconds = stale_after_hours * 3600.0
        with closing(self._connect()) as conn:
            # WAL allows concurrent readers while one process writes,
            # which is what makes multi-worker single-node setups safe.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection for the current call.

        Connections are intentionally per-call: sqlite3 connections must
        not be shared across threads, and the executor runs jobs on a
        thread pool.

        Returns:
            A new sqlite3 connection with row access by column name.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=_CONNECT_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, job_id: str) -> Job | None:
        """Look up a job by identifier.

        Args:
            job_id: Job identifier returned by submit().

        Returns:
            The Job if known, None otherwise (including after TTL eviction).
        """
        with closing(self._connect()) as conn:
            self._cleanup(conn)
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return _job_from_row(row) if row is not None else None

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """List known jobs, newest first.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            Jobs ordered by submission time descending.
        """
        with closing(self._connect()) as conn:
            self._cleanup(conn)
            cursor = conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_ts DESC, job_id LIMIT ?",
                (limit,),
            )
            return [_job_from_row(row) for row in cursor.fetchall()]

    def _add_job(self, job: Job) -> None:
        """Persist a freshly submitted job.

        Args:
            job: The new job record to store.
        """
        with closing(self._connect()) as conn:
            self._cleanup(conn)
            conn.execute(
                "INSERT INTO jobs "
                "(job_id, status, submitted_at, started_at, finished_at, platform, "
                "spark_version, result_json, error, submitted_ts, finished_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.status,
                    job.submitted_at,
                    job.started_at,
                    job.finished_at,
                    job.platform,
                    job.spark_version,
                    json.dumps(job.result) if job.result is not None else None,
                    job.error,
                    job.submitted_ts,
                    job.finished_ts,
                ),
            )
            conn.commit()

    def _mark_running(self, job_id: str) -> None:
        """Transition a job to the running state.

        Args:
            job_id: Identifier of the job to update.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
                (_utc_now_iso(), job_id),
            )
            conn.commit()

    def _mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        """Transition a job to the completed state.

        Args:
            job_id: Identifier of the job to update.
            result: The optimization result payload.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'completed', result_json = ?, error = NULL, "
                "finished_at = ?, finished_ts = ? WHERE job_id = ?",
                (json.dumps(result), _utc_now_iso(), time.time(), job_id),
            )
            conn.commit()

    def _mark_failed(self, job_id: str, error: str) -> None:
        """Transition a job to the failed state.

        Args:
            job_id: Identifier of the job to update.
            error: Human-readable failure message.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ?, finished_ts = ? WHERE job_id = ?",
                (error, _utc_now_iso(), time.time(), job_id),
            )
            conn.commit()

    def _cleanup(self, conn: sqlite3.Connection) -> None:
        """Apply the staleness rule and evict expired finished jobs.

        Runs opportunistically on every store access, mirroring the
        in-memory store's behaviour, so no background threads are needed.

        Args:
            conn: Open connection to run the cleanup statements on.
        """
        now = time.time()
        if self._stale_seconds > 0:
            # Timestamps are uniform ISO-8601 UTC strings produced by
            # _utc_now_iso(), so lexicographic comparison is chronological.
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self._stale_seconds)).isoformat()
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ?, finished_ts = ? "
                "WHERE (status = 'running' AND started_at < ?) "
                "OR (status = 'pending' AND submitted_at < ?)",
                (WORKER_LOST_ERROR, _utc_now_iso(), now, stale_cutoff, stale_cutoff),
            )
        if self._ttl_seconds > 0:
            ttl_cutoff = now - self._ttl_seconds
            conn.execute(
                "DELETE FROM jobs WHERE finished_ts IS NOT NULL AND finished_ts < ?",
                (ttl_cutoff,),
            )
        conn.commit()
