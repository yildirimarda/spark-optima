# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for job progress persistence across all three job stores.

Covers ``set_progress`` on the in-memory, SQLite, and Redis stores
(Workstream Z), including the idempotent SQLite schema migration for
databases created before v1.5. The Redis store is exercised against the
existing in-memory FakeRedis — no network access.
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from spark_optima.api.job_store_redis import REDIS_KEY_PREFIX, RedisJobStore
from spark_optima.api.job_store_sqlite import SQLiteJobStore
from spark_optima.api.jobs import JobStore
from tests.unit.api.test_job_store_redis import FakeRedis

PROGRESS = {"trial_number": 4, "n_trials": 10, "trials_completed": 5, "best_value": 9.2, "state": "COMPLETE"}


class ManualExecutor(Executor):
    """Executor that defers execution until run_all() is called."""

    def __init__(self) -> None:
        self.pending: list[tuple[Any, tuple, dict]] = []

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Queue the callable without executing it."""
        future: Future[Any] = Future()
        self.pending.append((fn, args, kwargs))
        return future

    def run_all(self) -> None:
        """Execute all queued callables synchronously."""
        for fn, args, kwargs in self.pending:
            fn(*args, **kwargs)
        self.pending.clear()


class TestMemoryStoreProgress:
    """Tests for set_progress on the in-memory store."""

    def test_set_progress_visible_on_get(self) -> None:
        """Progress written during a run is readable via get()."""
        store = JobStore(executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        store.set_progress(job_id, PROGRESS)

        job = store.get(job_id)
        assert job is not None
        assert job.progress == PROGRESS

    def test_set_progress_overwrites_previous_snapshot(self) -> None:
        """Each write replaces the previous snapshot."""
        store = JobStore(executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        store.set_progress(job_id, {"trials_completed": 1})
        store.set_progress(job_id, {"trials_completed": 2})

        job = store.get(job_id)
        assert job is not None
        assert job.progress == {"trials_completed": 2}

    def test_set_progress_unknown_job_is_ignored(self) -> None:
        """Unknown job ids are silently ignored (job may have been evicted)."""
        store = JobStore(executor=ManualExecutor())
        store.set_progress("does-not-exist", PROGRESS)  # Must not raise

    def test_progress_defaults_to_none(self) -> None:
        """A fresh job carries no progress."""
        store = JobStore(executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        job = store.get(job_id)
        assert job is not None
        assert job.progress is None


class TestOnAcceptedHook:
    """Tests for the on_accepted submission hook used to bind recorders."""

    def test_on_accepted_runs_before_work(self) -> None:
        """The hook receives the job id before the work callable executes."""
        from tests.unit.api.test_job_store_redis import InlineExecutor

        seen: list[str] = []
        order: list[str] = []
        store = JobStore(executor=InlineExecutor())

        def work() -> dict[str, Any]:
            order.append("work")
            return {}

        def accepted(job_id: str) -> None:
            seen.append(job_id)
            order.append("accepted")

        job_id = store.submit(work, platform="local", spark_version="3.5.0", on_accepted=accepted)

        assert seen == [job_id]
        assert order == ["accepted", "work"]

    def test_on_accepted_exception_does_not_break_the_job(self) -> None:
        """A raising hook is logged and the job still runs to completion."""
        from tests.unit.api.test_job_store_redis import InlineExecutor

        def explode(_job_id: str) -> None:
            raise RuntimeError("hook crashed")

        store = JobStore(executor=InlineExecutor())
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0", on_accepted=explode)

        job = store.get(job_id)
        assert job is not None
        assert job.status == "completed"


class TestSQLiteStoreProgress:
    """Tests for set_progress and the progress_json migration on SQLite."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Provide a per-test database file path."""
        return tmp_path / "jobs.db"

    def test_set_progress_round_trip(self, db_path: Path) -> None:
        """Progress survives the SQLite round trip as JSON."""
        store = SQLiteJobStore(db_path=db_path, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        store.set_progress(job_id, PROGRESS)

        job = store.get(job_id)
        assert job is not None
        assert job.progress == PROGRESS

    def test_progress_persisted_in_database_column(self, db_path: Path) -> None:
        """The snapshot lands in the progress_json column, not in memory."""
        store = SQLiteJobStore(db_path=db_path, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        store.set_progress(job_id, PROGRESS)

        conn = sqlite3.connect(str(db_path))
        try:
            raw = conn.execute("SELECT progress_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0]
        finally:
            conn.close()
        assert json.loads(raw) == PROGRESS

    def test_progress_visible_to_second_store_instance(self, db_path: Path) -> None:
        """Another store (worker process) on the same file sees the progress."""
        store1 = SQLiteJobStore(db_path=db_path, executor=ManualExecutor())
        job_id = store1.submit(lambda: {}, platform="local", spark_version="3.5.0")
        store1.set_progress(job_id, PROGRESS)

        store2 = SQLiteJobStore(db_path=db_path, executor=ManualExecutor())
        job = store2.get(job_id)

        assert job is not None
        assert job.progress == PROGRESS

    def test_migration_adds_progress_column_to_pre_v15_database(self, db_path: Path) -> None:
        """Opening a pre-v1.5 database adds progress_json without data loss."""
        # Build a database with the pre-v1.5 schema (no progress_json column)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE jobs ("
                "job_id TEXT PRIMARY KEY, status TEXT NOT NULL, submitted_at TEXT NOT NULL, "
                "started_at TEXT, finished_at TEXT, platform TEXT NOT NULL, spark_version TEXT NOT NULL, "
                "result_json TEXT, error TEXT, webhook_status TEXT, "
                "submitted_ts REAL NOT NULL DEFAULT 0.0, finished_ts REAL)"
            )
            conn.execute(
                "INSERT INTO jobs (job_id, status, submitted_at, platform, spark_version, submitted_ts) "
                "VALUES ('old-job', 'pending', '2026-06-11T00:00:00+00:00', 'local', '3.5.0', 0.0)"
            )
            conn.commit()
        finally:
            conn.close()

        store = SQLiteJobStore(db_path=db_path, executor=ManualExecutor(), stale_after_hours=0.0)

        # The old row is still readable and reports no progress
        job = store.get("old-job")
        assert job is not None
        assert job.progress is None

        # The new column is writable
        store.set_progress("old-job", PROGRESS)
        job = store.get("old-job")
        assert job is not None
        assert job.progress == PROGRESS

    def test_migration_is_idempotent(self, db_path: Path) -> None:
        """Re-opening the store repeatedly never fails on the ALTER TABLE."""
        for _ in range(3):
            SQLiteJobStore(db_path=db_path, executor=ManualExecutor())

        conn = sqlite3.connect(str(db_path))
        try:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        finally:
            conn.close()
        assert columns.count("progress_json") == 1
        assert columns.count("webhook_status") == 1


class TestRedisStoreProgress:
    """Tests for set_progress on the Redis store (via FakeRedis)."""

    @pytest.fixture
    def fake_redis(self) -> FakeRedis:
        """Provide a fresh fake redis client."""
        return FakeRedis()

    def test_set_progress_round_trip(self, fake_redis: FakeRedis) -> None:
        """Progress survives the Redis JSON round trip."""
        store = RedisJobStore(client=fake_redis, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        store.set_progress(job_id, PROGRESS)

        job = store.get(job_id)
        assert job is not None
        assert job.progress == PROGRESS

    def test_progress_stored_in_json_payload(self, fake_redis: FakeRedis) -> None:
        """The snapshot is a field of the job's JSON payload (replica-visible)."""
        store = RedisJobStore(client=fake_redis, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        store.set_progress(job_id, PROGRESS)

        payload = json.loads(fake_redis.data[f"{REDIS_KEY_PREFIX}{job_id}"])
        assert payload["progress"] == PROGRESS

    def test_pre_v15_payload_without_progress_key_loads(self, fake_redis: FakeRedis) -> None:
        """Payloads written before v1.5 (no progress key) deserialize fine."""
        store = RedisJobStore(client=fake_redis, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        key = f"{REDIS_KEY_PREFIX}{job_id}"
        payload = json.loads(fake_redis.data[key])
        del payload["progress"]
        fake_redis.set(key, json.dumps(payload))

        job = store.get(job_id)
        assert job is not None
        assert job.progress is None

    def test_progress_visible_to_second_store_instance(self, fake_redis: FakeRedis) -> None:
        """Another store (replica) on the same client sees the progress."""
        store1 = RedisJobStore(client=fake_redis, executor=ManualExecutor())
        job_id = store1.submit(lambda: {}, platform="local", spark_version="3.5.0")
        store1.set_progress(job_id, PROGRESS)

        store2 = RedisJobStore(client=fake_redis, executor=ManualExecutor())
        job = store2.get(job_id)

        assert job is not None
        assert job.progress == PROGRESS
