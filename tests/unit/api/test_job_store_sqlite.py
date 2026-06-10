# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the SQLite-backed job store and the store selection factory."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.api import jobs as jobs_module
from spark_optima.api.job_store_sqlite import (
    JOB_DB_ENV_VAR,
    WORKER_LOST_ERROR,
    SQLiteJobStore,
    _default_db_path,
)
from spark_optima.api.jobs import JOB_STORE_ENV_VAR, JobStore, create_job_store, get_job_store, reset_job_store


class InlineExecutor(Executor):
    """Executor that runs submitted callables synchronously in the caller thread."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Run the callable immediately and return a resolved future."""
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - the store handles errors itself
            future.set_exception(exc)
        return future


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


def _iso_hours_ago(hours: float) -> str:
    """Return an ISO-8601 UTC timestamp `hours` in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _insert_row(
    db_path: Path,
    *,
    job_id: str | None = None,
    status: str = "running",
    submitted_at: str | None = None,
    started_at: str | None = None,
    finished_ts: float | None = None,
) -> str:
    """Insert a raw job row, bypassing the store (simulates another process)."""
    job_id = job_id or uuid.uuid4().hex
    submitted_at = submitted_at or _iso_hours_ago(0)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO jobs "
            "(job_id, status, submitted_at, started_at, finished_at, platform, "
            "spark_version, result_json, error, submitted_ts, finished_ts) "
            "VALUES (?, ?, ?, ?, NULL, 'local', '3.5.0', NULL, NULL, ?, ?)",
            (job_id, status, submitted_at, started_at, time.time(), finished_ts),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a per-test database file path."""
    return tmp_path / "jobs.db"


class TestSQLiteJobStoreLifecycle:
    """Tests for submit/get state transitions backed by SQLite."""

    def test_completed_job_lifecycle(self, db_path: Path) -> None:
        """A successful job ends up completed with its result and timestamps."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())

        job_id = store.submit(lambda: {"answer": 42}, platform="local", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "completed"
        assert job.result == {"answer": 42}
        assert job.error is None
        assert job.platform == "local"
        assert job.spark_version == "3.5.0"
        assert job.submitted_at
        assert job.started_at is not None
        assert job.finished_at is not None

    def test_job_starts_pending(self, db_path: Path) -> None:
        """A job is pending until the executor picks it up."""
        executor = ManualExecutor()
        store = SQLiteJobStore(db_path=db_path, executor=executor)

        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "pending"
        assert job.started_at is None

        executor.run_all()
        job = store.get(job_id)
        assert job is not None
        assert job.status == "completed"

    def test_failed_job_records_error(self, db_path: Path) -> None:
        """A job whose work raises is marked failed with the error message."""

        def boom() -> dict[str, Any]:
            raise ValueError("kaboom")

        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        job_id = store.submit(boom, platform="databricks", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == "kaboom"
        assert job.result is None
        assert job.finished_at is not None

    def test_get_unknown_job_returns_none(self, db_path: Path) -> None:
        """Unknown identifiers return None."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        assert store.get("does-not-exist") is None

    def test_list_jobs_newest_first_and_limited(self, db_path: Path) -> None:
        """list_jobs orders by submission time descending and honors limit."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        ids = [store.submit(lambda: {}, platform="local", spark_version="3.5.0") for _ in range(3)]

        # Force distinct, increasing submission timestamps for deterministic order
        conn = sqlite3.connect(str(db_path))
        try:
            for index, job_id in enumerate(ids):
                conn.execute("UPDATE jobs SET submitted_ts = ? WHERE job_id = ?", (float(index), job_id))
            conn.commit()
        finally:
            conn.close()

        listed = [job.job_id for job in store.list_jobs()]
        assert listed == list(reversed(ids))
        assert len(store.list_jobs(limit=2)) == 2

    def test_uses_wal_journal_mode(self, db_path: Path) -> None:
        """The database is switched to WAL mode at initialization."""
        SQLiteJobStore(db_path=db_path, executor=InlineExecutor())

        conn = sqlite3.connect(str(db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert mode == "wal"


class TestSQLiteJobStorePersistence:
    """Tests for persistence across store instances (restart behaviour)."""

    def test_jobs_survive_new_store_instance(self, db_path: Path) -> None:
        """A second store on the same file sees jobs from the first one."""
        store1 = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        job_id = store1.submit(lambda: {"answer": 42}, platform="local", spark_version="3.5.0")

        store2 = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        job = store2.get(job_id)

        assert job is not None
        assert job.status == "completed"
        assert job.result == {"answer": 42}
        assert [j.job_id for j in store2.list_jobs()] == [job_id]

    def test_db_path_from_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPARK_OPTIMA_JOB_DB overrides the default database location."""
        env_db = tmp_path / "custom" / "env-jobs.db"
        monkeypatch.setenv(JOB_DB_ENV_VAR, str(env_db))

        assert _default_db_path() == env_db
        store = SQLiteJobStore(executor=InlineExecutor())
        store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        assert store.db_path == env_db
        assert env_db.exists()

    def test_default_db_path_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env variable the default lives under ~/.spark_optima."""
        monkeypatch.delenv(JOB_DB_ENV_VAR, raising=False)
        assert _default_db_path() == Path.home() / ".spark_optima" / "jobs.db"


class TestSQLiteJobStoreCleanup:
    """Tests for TTL eviction and the worker-lost staleness rule."""

    def test_expired_finished_job_is_evicted_on_access(self, db_path: Path) -> None:
        """Finished jobs older than the TTL disappear on the next access."""
        store = SQLiteJobStore(db_path=db_path, ttl_hours=1.0, executor=InlineExecutor())
        job_id = _insert_row(db_path, status="completed", finished_ts=time.time() - 2 * 3600)

        assert store.get(job_id) is None
        assert store.list_jobs() == []

    def test_recent_finished_job_is_kept(self, db_path: Path) -> None:
        """Finished jobs within the TTL are retained."""
        store = SQLiteJobStore(db_path=db_path, ttl_hours=1.0, executor=InlineExecutor())
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        assert store.get(job_id) is not None

    def test_zero_ttl_disables_eviction(self, db_path: Path) -> None:
        """A non-positive TTL disables TTL eviction entirely."""
        store = SQLiteJobStore(db_path=db_path, ttl_hours=0.0, executor=InlineExecutor())
        job_id = _insert_row(db_path, status="completed", finished_ts=time.time() - 100 * 3600)

        assert store.get(job_id) is not None

    def test_stale_running_job_reported_as_worker_lost(self, db_path: Path) -> None:
        """A running job started before the staleness window is reported failed."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        # Simulate a job that a now-dead process left in "running" 3h ago
        job_id = _insert_row(db_path, status="running", started_at=_iso_hours_ago(3.0))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == WORKER_LOST_ERROR
        assert job.finished_at is not None
        # The transition is persisted, not just reported
        conn = sqlite3.connect(str(db_path))
        try:
            status = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0]
        finally:
            conn.close()
        assert status == "failed"

    def test_stale_pending_job_reported_as_worker_lost(self, db_path: Path) -> None:
        """A pending job submitted before the staleness window is reported failed."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        job_id = _insert_row(db_path, status="pending", submitted_at=_iso_hours_ago(3.0))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == WORKER_LOST_ERROR

    def test_fresh_running_job_is_not_marked_stale(self, db_path: Path) -> None:
        """A recently started running job is left untouched."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor())
        job_id = _insert_row(db_path, status="running", started_at=_iso_hours_ago(0.5))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "running"
        assert job.error is None

    def test_staleness_window_is_configurable(self, db_path: Path) -> None:
        """A custom stale_after_hours moves the cutoff."""
        store = SQLiteJobStore(db_path=db_path, executor=InlineExecutor(), stale_after_hours=6.0)
        job_id = _insert_row(db_path, status="running", started_at=_iso_hours_ago(3.0))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "running"


class TestSQLiteJobStoreConcurrency:
    """Concurrency smoke tests for the SQLite store."""

    def test_concurrent_submissions_and_reads(self, db_path: Path) -> None:
        """Jobs submitted from many threads all complete consistently."""
        executor = ThreadPoolExecutor(max_workers=4)
        store = SQLiteJobStore(db_path=db_path, executor=executor)
        job_count = 10

        def make_work(index: int):
            def work() -> dict[str, Any]:
                time.sleep(0.001)
                return {"index": index}

            return work

        try:
            submitter = ThreadPoolExecutor(max_workers=4)
            try:
                futures = [submitter.submit(store.submit, make_work(i), "local", "3.5.0") for i in range(job_count)]
                ids = [future.result() for future in futures]
            finally:
                submitter.shutdown(wait=True)

            deadline = time.time() + 15.0
            while time.time() < deadline:
                jobs = [store.get(job_id) for job_id in ids]
                store.list_jobs(limit=job_count)
                if all(job is not None and job.status == "completed" for job in jobs):
                    break
                time.sleep(0.005)

            results = [store.get(job_id) for job_id in ids]
            assert all(job is not None and job.status == "completed" for job in results)
            indices = sorted(job.result["index"] for job in results if job is not None and job.result)
            assert indices == list(range(job_count))
        finally:
            executor.shutdown(wait=True)


class TestJobStoreFactory:
    """Tests for SPARK_OPTIMA_JOB_STORE-based backend selection."""

    def test_default_is_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env variable the in-memory store is used."""
        monkeypatch.delenv(JOB_STORE_ENV_VAR, raising=False)
        store = create_job_store()
        assert type(store) is JobStore

    def test_sqlite_selected_via_env(self, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPARK_OPTIMA_JOB_STORE=sqlite selects the SQLite store."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "sqlite")
        monkeypatch.setenv(JOB_DB_ENV_VAR, str(db_path))

        store = create_job_store()

        assert isinstance(store, SQLiteJobStore)
        assert store.db_path == db_path

    def test_backend_value_is_case_insensitive(self, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backend names are normalized before matching."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, " SQLite ")
        monkeypatch.setenv(JOB_DB_ENV_VAR, str(db_path))

        assert isinstance(create_job_store(), SQLiteJobStore)

    def test_invalid_backend_falls_back_to_memory_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unknown backend value logs a warning and uses the memory store."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "postgres")

        with caplog.at_level(logging.WARNING, logger="spark_optima.api.jobs"):
            store = create_job_store()

        assert type(store) is JobStore
        assert any("postgres" in record.message and JOB_STORE_ENV_VAR in record.message for record in caplog.records)

    def test_get_job_store_uses_env_at_creation(self, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The global accessor builds the backend selected by the environment."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "sqlite")
        monkeypatch.setenv(JOB_DB_ENV_VAR, str(db_path))
        reset_job_store()
        try:
            store = get_job_store()
            assert isinstance(store, SQLiteJobStore)
            assert store is get_job_store()
        finally:
            reset_job_store()


class TestAsyncEndpointsWithSQLiteStore:
    """End-to-end async endpoint flow running on the SQLite store."""

    @pytest.fixture(autouse=True)
    def sqlite_job_store(self, db_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Select the SQLite backend and run jobs synchronously."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "sqlite")
        monkeypatch.setenv(JOB_DB_ENV_VAR, str(db_path))
        monkeypatch.setattr(jobs_module, "_EXECUTOR", InlineExecutor())
        reset_job_store()
        yield
        reset_job_store()

    @staticmethod
    def _mock_service() -> MagicMock:
        """Build a mocked optimization service for endpoint tests."""
        service = MagicMock()
        service.validate_spark_version.return_value = True
        optimizer = MagicMock()
        optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        service.get_optimizer.return_value = optimizer
        return service

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_submit_poll_and_list_with_sqlite_store(self, mock_get_service: MagicMock, db_path: Path) -> None:
        """Submit, poll, and list jobs through the API with the SQLite backend."""
        from fastapi.testclient import TestClient

        from spark_optima.api.main import app

        mock_get_service.return_value = self._mock_service()
        client = TestClient(app)

        submit = client.post(
            "/api/v1/optimize/async",
            json={
                "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()",
                "platform": "local",
                "spark_version": "3.5.0",
                "resources": {"cpu_cores": 4, "memory_gb": 16},
            },
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        detail = client.get(f"/api/v1/jobs/{job_id}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["status"] == "completed"
        assert payload["result"]["configuration"] == {"spark.executor.memory": "4g"}

        listing = client.get("/api/v1/jobs")
        assert listing.status_code == 200
        assert [job["job_id"] for job in listing.json()["jobs"]] == [job_id]

        # The job state landed in the SQLite file, not in process memory
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "completed"
