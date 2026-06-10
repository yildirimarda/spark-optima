# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the in-memory asynchronous job store."""

from __future__ import annotations

import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

from spark_optima.api import jobs as jobs_module
from spark_optima.api.jobs import JobStore, get_job_store, reset_job_store


class InlineExecutor(Executor):
    """Executor that runs submitted callables synchronously in the caller thread."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Run the callable immediately and return a resolved future."""
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - JobStore handles errors itself
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


class TestJobLifecycle:
    """Tests for job state transitions."""

    def test_completed_job_lifecycle(self) -> None:
        """A successful job ends up completed with its result and timestamps."""
        store = JobStore(executor=InlineExecutor())

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

    def test_job_starts_pending(self) -> None:
        """A job is pending until the executor picks it up."""
        executor = ManualExecutor()
        store = JobStore(executor=executor)

        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "pending"
        assert job.started_at is None
        assert job.finished_at is None

        executor.run_all()
        job = store.get(job_id)
        assert job is not None
        assert job.status == "completed"

    def test_failed_job_records_error(self) -> None:
        """A job whose work raises is marked failed with the error message."""

        def boom() -> dict[str, Any]:
            raise ValueError("kaboom")

        store = JobStore(executor=InlineExecutor())
        job_id = store.submit(boom, platform="databricks", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == "kaboom"
        assert job.result is None
        assert job.finished_at is not None

    def test_job_id_is_uuid4_hex(self) -> None:
        """Job identifiers are 32-character hex strings."""
        store = JobStore(executor=InlineExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        assert len(job_id) == 32
        assert all(c in "0123456789abcdef" for c in job_id)

    def test_job_ids_are_unique(self) -> None:
        """Two submissions never share an identifier."""
        store = JobStore(executor=InlineExecutor())
        id1 = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        id2 = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        assert id1 != id2


class TestJobStoreQueries:
    """Tests for get() and list_jobs()."""

    def test_get_unknown_job_returns_none(self) -> None:
        """Unknown identifiers return None."""
        store = JobStore(executor=InlineExecutor())
        assert store.get("does-not-exist") is None

    def test_list_jobs_newest_first(self) -> None:
        """list_jobs orders by submission time descending."""
        store = JobStore(executor=InlineExecutor())
        ids = [store.submit(lambda: {}, platform="local", spark_version="3.5.0") for _ in range(3)]

        # Force distinct, increasing submission timestamps for deterministic order
        for index, job_id in enumerate(ids):
            store._jobs[job_id].submitted_ts = float(index)

        listed = [job.job_id for job in store.list_jobs()]
        assert listed == list(reversed(ids))

    def test_list_jobs_respects_limit(self) -> None:
        """list_jobs returns at most `limit` entries."""
        store = JobStore(executor=InlineExecutor())
        for _ in range(5):
            store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        assert len(store.list_jobs(limit=2)) == 2
        assert len(store.list_jobs(limit=50)) == 5


class TestJobStoreTTLCleanup:
    """Tests for opportunistic TTL-based eviction of finished jobs."""

    def test_expired_finished_job_is_evicted_on_access(self) -> None:
        """Finished jobs older than the TTL disappear on the next access."""
        store = JobStore(ttl_hours=1.0, executor=InlineExecutor())
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        # Pretend the job finished two hours ago
        store._jobs[job_id].finished_ts = time.time() - 2 * 3600

        assert store.get(job_id) is None
        assert store.list_jobs() == []

    def test_recent_finished_job_is_kept(self) -> None:
        """Finished jobs within the TTL are retained."""
        store = JobStore(ttl_hours=1.0, executor=InlineExecutor())
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        assert store.get(job_id) is not None

    def test_unfinished_jobs_are_never_evicted(self) -> None:
        """Pending/running jobs are not TTL-evicted even if submitted long ago."""
        executor = ManualExecutor()
        store = JobStore(ttl_hours=1.0, executor=executor)
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        store._jobs[job_id].submitted_ts = time.time() - 10 * 3600

        job = store.get(job_id)
        assert job is not None
        assert job.status == "pending"

    def test_zero_ttl_disables_cleanup(self) -> None:
        """A non-positive TTL disables eviction entirely."""
        store = JobStore(ttl_hours=0.0, executor=InlineExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        store._jobs[job_id].finished_ts = time.time() - 100 * 3600

        assert store.get(job_id) is not None


class TestJobStoreThreadSafety:
    """Concurrency smoke tests for the job store."""

    def test_concurrent_submissions_and_reads(self) -> None:
        """Many jobs running on a real thread pool all complete consistently."""
        executor = ThreadPoolExecutor(max_workers=4)
        store = JobStore(executor=executor)
        job_count = 20

        def make_work(index: int):
            def work() -> dict[str, Any]:
                time.sleep(0.001)
                return {"index": index}

            return work

        try:
            ids = [store.submit(make_work(i), platform="local", spark_version="3.5.0") for i in range(job_count)]

            # Poll with reads interleaved to exercise the lock from this thread
            deadline = time.time() + 10.0
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


class TestGlobalJobStore:
    """Tests for the global singleton accessor."""

    def test_get_job_store_is_singleton(self) -> None:
        """get_job_store returns the same instance on repeated calls."""
        reset_job_store()
        try:
            store1 = get_job_store()
            store2 = get_job_store()
            assert store1 is store2
        finally:
            reset_job_store()

    def test_reset_job_store_discards_instance(self) -> None:
        """reset_job_store causes a fresh instance to be created."""
        reset_job_store()
        try:
            store1 = get_job_store()
            reset_job_store()
            store2 = get_job_store()
            assert store1 is not store2
        finally:
            reset_job_store()

    def test_module_executor_is_bounded(self) -> None:
        """The shared executor uses a small fixed worker pool."""
        assert isinstance(jobs_module._EXECUTOR, ThreadPoolExecutor)
        assert jobs_module._EXECUTOR._max_workers == 2
