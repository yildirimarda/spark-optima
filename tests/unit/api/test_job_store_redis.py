# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Redis-backed job store and its factory selection.

The store logic is tested against a minimal in-memory fake Redis client
(no fakeredis dependency, no real network). Only the import-path test
touches the real ``redis`` package, guarded with importorskip.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time
import uuid
from concurrent.futures import Executor, Future
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from spark_optima.api import job_store_redis
from spark_optima.api.job_store_redis import (
    DEFAULT_REDIS_URL,
    REDIS_KEY_PREFIX,
    REDIS_URL_ENV_VAR,
    RedisJobStore,
    _job_to_json,
)
from spark_optima.api.job_store_sqlite import WORKER_LOST_ERROR
from spark_optima.api.jobs import JOB_STORE_ENV_VAR, Job, JobStore, create_job_store


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


class FakeRedis:
    """Minimal in-memory stand-in for a redis client (decode_responses=True).

    Implements exactly the subset of commands the store uses:
    get/set/delete/scan_iter/expire/ping. TTLs are recorded but never
    auto-expire — tests assert on the recorded values instead.
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.ping_calls = 0

    def ping(self) -> bool:
        """Pretend the server is reachable."""
        self.ping_calls += 1
        return True

    def get(self, key: str) -> str | None:
        """Return the stored string value, or None."""
        return self.data.get(key)

    def set(self, key: str, value: str) -> bool:
        """Store a string value, clearing any TTL (real SET semantics)."""
        self.data[key] = value
        self.ttls.pop(key, None)
        return True

    def delete(self, *keys: str) -> int:
        """Delete keys, returning how many existed."""
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                self.ttls.pop(key, None)
                removed += 1
        return removed

    def expire(self, key: str, seconds: int) -> bool:
        """Record a TTL for an existing key."""
        if key not in self.data:
            return False
        self.ttls[key] = seconds
        return True

    def scan_iter(self, match: str | None = None, count: int | None = None):
        """Iterate keys matching the glob pattern."""
        for key in list(self.data):
            if match is None or fnmatch.fnmatch(key, match):
                yield key


def _iso_hours_ago(hours: float) -> str:
    """Return an ISO-8601 UTC timestamp `hours` in the past."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _put_raw_job(fake: FakeRedis, **overrides: Any) -> str:
    """Write a job payload directly into the fake (simulates another process)."""
    job_id = overrides.pop("job_id", uuid.uuid4().hex)
    job = Job(
        job_id=job_id,
        platform="local",
        spark_version="3.5.0",
        submitted_at=_iso_hours_ago(0),
        submitted_ts=time.time(),
    )
    for name, value in overrides.items():
        setattr(job, name, value)
    fake.set(f"{REDIS_KEY_PREFIX}{job_id}", _job_to_json(job))
    return job_id


@pytest.fixture
def fake_redis() -> FakeRedis:
    """Provide a fresh fake redis client."""
    return FakeRedis()


def _make_store(fake: FakeRedis, **kwargs: Any) -> RedisJobStore:
    """Build a RedisJobStore on the fake client with an inline executor."""
    kwargs.setdefault("executor", InlineExecutor())
    return RedisJobStore(client=fake, **kwargs)


class TestRedisJobStoreLifecycle:
    """Tests for submit/get state transitions backed by (fake) Redis."""

    def test_completed_job_lifecycle(self, fake_redis: FakeRedis) -> None:
        """A successful job ends up completed with its result and timestamps."""
        store = _make_store(fake_redis)

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

    def test_job_starts_pending(self, fake_redis: FakeRedis) -> None:
        """A job is pending until the executor picks it up."""
        executor = ManualExecutor()
        store = _make_store(fake_redis, executor=executor)

        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "pending"
        assert job.started_at is None

        executor.run_all()
        job = store.get(job_id)
        assert job is not None
        assert job.status == "completed"

    def test_failed_job_records_error(self, fake_redis: FakeRedis) -> None:
        """A job whose work raises is marked failed with the error message."""

        def boom() -> dict[str, Any]:
            raise ValueError("kaboom")

        store = _make_store(fake_redis)
        job_id = store.submit(boom, platform="databricks", spark_version="3.5.0")
        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == "kaboom"
        assert job.result is None
        assert job.finished_at is not None

    def test_get_unknown_job_returns_none(self, fake_redis: FakeRedis) -> None:
        """Unknown identifiers return None."""
        store = _make_store(fake_redis)
        assert store.get("does-not-exist") is None

    def test_jobs_are_stored_under_the_key_prefix(self, fake_redis: FakeRedis) -> None:
        """Each job lives at spark_optima:job:<job_id> as a JSON string."""
        store = _make_store(fake_redis)
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        key = f"{REDIS_KEY_PREFIX}{job_id}"
        assert key in fake_redis.data
        payload = json.loads(fake_redis.data[key])
        assert payload["job_id"] == job_id
        assert payload["status"] == "completed"

    def test_jobs_visible_to_second_store_instance(self, fake_redis: FakeRedis) -> None:
        """A second store on the same client sees jobs from the first (replicas)."""
        store1 = _make_store(fake_redis)
        job_id = store1.submit(lambda: {"answer": 42}, platform="local", spark_version="3.5.0")

        store2 = _make_store(fake_redis)
        job = store2.get(job_id)

        assert job is not None
        assert job.status == "completed"
        assert job.result == {"answer": 42}

    def test_ping_is_called_at_creation(self, fake_redis: FakeRedis) -> None:
        """The constructor verifies connectivity so the factory can fall back."""
        _make_store(fake_redis)
        assert fake_redis.ping_calls == 1


class TestRedisJobStoreListing:
    """Tests for SCAN-based listing."""

    def test_list_jobs_newest_first_and_limited(self, fake_redis: FakeRedis) -> None:
        """list_jobs orders by submitted_ts descending and honors limit."""
        store = _make_store(fake_redis)
        ids = [
            _put_raw_job(fake_redis, status="completed", submitted_ts=float(index), finished_ts=time.time())
            for index in range(3)
        ]

        listed = [job.job_id for job in store.list_jobs()]
        assert listed == list(reversed(ids))
        assert len(store.list_jobs(limit=2)) == 2

    def test_list_jobs_ignores_foreign_keys(self, fake_redis: FakeRedis) -> None:
        """Keys outside the job prefix are not touched by listing."""
        store = _make_store(fake_redis)
        fake_redis.set("some:other:key", "not-a-job")
        job_id = _put_raw_job(fake_redis)

        assert [job.job_id for job in store.list_jobs()] == [job_id]


class TestRedisJobStoreTTL:
    """Tests for native EXPIRE-based eviction of finished jobs."""

    def test_finished_job_gets_expire(self, fake_redis: FakeRedis) -> None:
        """Completing a job applies an EXPIRE close to the configured TTL."""
        store = _make_store(fake_redis, ttl_hours=1.0)
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        ttl = fake_redis.ttls[f"{REDIS_KEY_PREFIX}{job_id}"]
        assert 3500 <= ttl <= 3600

    def test_failed_job_gets_expire(self, fake_redis: FakeRedis) -> None:
        """Failing a job applies the same EXPIRE as completion."""

        def boom() -> dict[str, Any]:
            raise ValueError("kaboom")

        store = _make_store(fake_redis, ttl_hours=1.0)
        job_id = store.submit(boom, platform="local", spark_version="3.5.0")

        assert f"{REDIS_KEY_PREFIX}{job_id}" in fake_redis.ttls

    def test_unfinished_job_has_no_expire(self, fake_redis: FakeRedis) -> None:
        """Pending jobs never expire — only finished ones do."""
        store = _make_store(fake_redis, ttl_hours=1.0, executor=ManualExecutor())
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        assert f"{REDIS_KEY_PREFIX}{job_id}" not in fake_redis.ttls

    def test_zero_ttl_disables_expiry(self, fake_redis: FakeRedis) -> None:
        """A non-positive TTL never applies EXPIRE."""
        store = _make_store(fake_redis, ttl_hours=0.0)
        job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")

        assert f"{REDIS_KEY_PREFIX}{job_id}" not in fake_redis.ttls
        assert store.get(job_id) is not None

    def test_set_webhook_status_preserves_remaining_ttl(self, fake_redis: FakeRedis) -> None:
        """Rewriting a finished job re-applies the remaining TTL (SET clears it)."""
        store = _make_store(fake_redis, ttl_hours=1.0)
        # Job finished 30 minutes ago — about half the TTL remains
        job_id = _put_raw_job(
            fake_redis,
            status="completed",
            result={"ok": True},
            finished_at=_iso_hours_ago(0.5),
            finished_ts=time.time() - 1800,
        )

        store.set_webhook_status(job_id, "delivered")

        job = store.get(job_id)
        assert job is not None
        assert job.webhook_status == "delivered"
        ttl = fake_redis.ttls[f"{REDIS_KEY_PREFIX}{job_id}"]
        assert 1700 <= ttl <= 1800

    def test_rewrite_of_already_expired_job_deletes_it(self, fake_redis: FakeRedis) -> None:
        """A finished job past its TTL is deleted instead of rewritten."""
        store = _make_store(fake_redis, ttl_hours=1.0)
        job_id = _put_raw_job(
            fake_redis,
            status="completed",
            finished_at=_iso_hours_ago(2.0),
            finished_ts=time.time() - 2 * 3600,
        )

        store.set_webhook_status(job_id, "delivered")

        assert f"{REDIS_KEY_PREFIX}{job_id}" not in fake_redis.data
        assert store.get(job_id) is None


class TestRedisJobStoreStaleness:
    """Tests for the worker-lost staleness rule (consistent with SQLite)."""

    def test_stale_running_job_reported_as_worker_lost(self, fake_redis: FakeRedis) -> None:
        """A running job started 3h ago is reported and persisted as failed."""
        store = _make_store(fake_redis)
        job_id = _put_raw_job(fake_redis, status="running", started_at=_iso_hours_ago(3.0))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == WORKER_LOST_ERROR
        assert job.finished_at is not None
        # The transition is persisted, not just reported
        stored = json.loads(fake_redis.data[f"{REDIS_KEY_PREFIX}{job_id}"])
        assert stored["status"] == "failed"

    def test_stale_pending_job_reported_as_worker_lost(self, fake_redis: FakeRedis) -> None:
        """A pending job submitted 3h ago is reported as failed."""
        store = _make_store(fake_redis)
        job_id = _put_raw_job(fake_redis, status="pending", submitted_ts=time.time() - 3 * 3600)

        job = store.get(job_id)

        assert job is not None
        assert job.status == "failed"
        assert job.error == WORKER_LOST_ERROR

    def test_fresh_running_job_is_not_marked_stale(self, fake_redis: FakeRedis) -> None:
        """A recently started running job is left untouched."""
        store = _make_store(fake_redis)
        job_id = _put_raw_job(fake_redis, status="running", started_at=_iso_hours_ago(0.5))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "running"
        assert job.error is None

    def test_staleness_window_is_configurable(self, fake_redis: FakeRedis) -> None:
        """A custom stale_after_hours moves the cutoff."""
        store = _make_store(fake_redis, stale_after_hours=6.0)
        job_id = _put_raw_job(fake_redis, status="running", started_at=_iso_hours_ago(3.0))

        job = store.get(job_id)

        assert job is not None
        assert job.status == "running"

    def test_stale_jobs_transition_in_listing_too(self, fake_redis: FakeRedis) -> None:
        """list_jobs applies the same staleness rule as get."""
        store = _make_store(fake_redis)
        _put_raw_job(fake_redis, status="running", started_at=_iso_hours_ago(3.0))

        jobs = store.list_jobs()

        assert len(jobs) == 1
        assert jobs[0].status == "failed"
        assert jobs[0].error == WORKER_LOST_ERROR


class TestRedisJobStoreConfiguration:
    """Tests for URL resolution and the guarded optional import."""

    def test_url_from_env_var(self, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPARK_OPTIMA_REDIS_URL overrides the default connection URL."""
        monkeypatch.setenv(REDIS_URL_ENV_VAR, "redis://redis.example.com:6380/2")
        store = _make_store(fake_redis)
        assert store.url == "redis://redis.example.com:6380/2"

    def test_default_url_without_env(self, fake_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env variable the localhost default is used."""
        monkeypatch.delenv(REDIS_URL_ENV_VAR, raising=False)
        store = _make_store(fake_redis)
        assert store.url == DEFAULT_REDIS_URL

    def test_module_imports_without_redis_package(self) -> None:
        """The module is importable regardless of the optional dependency."""
        assert isinstance(job_store_redis.REDIS_AVAILABLE, bool)

    def test_constructor_without_client_requires_package(self) -> None:
        """Without the package and without an injected client, creation fails clearly."""
        if job_store_redis.REDIS_AVAILABLE:
            pytest.skip("redis package is installed in this environment")
        with pytest.raises(RuntimeError, match="redis"):
            RedisJobStore()

    def test_real_redis_client_construction(self) -> None:
        """Import-path check against the real package (no connection is made)."""
        redis = pytest.importorskip("redis")
        client = redis.Redis.from_url(DEFAULT_REDIS_URL, decode_responses=True)
        assert client is not None


class _StubRedisModule:
    """Stand-in for the redis module whose from_url returns a FakeRedis."""

    def __init__(self, client: FakeRedis | None = None, error: Exception | None = None) -> None:
        self._client = client
        self._error = error
        outer = self

        class Redis:
            @staticmethod
            def from_url(url: str, decode_responses: bool = False) -> FakeRedis:
                if outer._error is not None:
                    raise outer._error
                assert outer._client is not None
                return outer._client

        self.Redis = Redis


class TestJobStoreFactoryRedis:
    """Tests for SPARK_OPTIMA_JOB_STORE=redis backend selection."""

    def test_redis_selected_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPARK_OPTIMA_JOB_STORE=redis builds a RedisJobStore."""
        fake = FakeRedis()
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "redis")
        monkeypatch.setattr(job_store_redis, "REDIS_AVAILABLE", True)
        monkeypatch.setattr(job_store_redis, "redis", _StubRedisModule(client=fake))

        store = create_job_store()

        assert isinstance(store, RedisJobStore)
        assert fake.ping_calls == 1

    def test_redis_missing_package_falls_back_to_memory(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Selecting redis without the package warns and uses the memory store."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "redis")
        monkeypatch.setattr(job_store_redis, "REDIS_AVAILABLE", False)

        with caplog.at_level(logging.WARNING, logger="spark_optima.api.jobs"):
            store = create_job_store()

        assert type(store) is JobStore
        assert any("redis" in record.message and "falling back" in record.message for record in caplog.records)

    def test_redis_connection_failure_falls_back_to_memory(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bad URL / unreachable server warns and uses the memory store."""
        monkeypatch.setenv(JOB_STORE_ENV_VAR, "redis")
        monkeypatch.setattr(job_store_redis, "REDIS_AVAILABLE", True)
        monkeypatch.setattr(
            job_store_redis,
            "redis",
            _StubRedisModule(error=ConnectionError("connection refused")),
        )

        with caplog.at_level(logging.WARNING, logger="spark_optima.api.jobs"):
            store = create_job_store()

        assert type(store) is JobStore
        assert any("connection refused" in record.message for record in caplog.records)

    def test_redis_ping_failure_falls_back_to_memory(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A client that connects but cannot ping also falls back to memory."""

        class DeadFakeRedis(FakeRedis):
            def ping(self) -> bool:
                raise ConnectionError("server unreachable")

        monkeypatch.setenv(JOB_STORE_ENV_VAR, "redis")
        monkeypatch.setattr(job_store_redis, "REDIS_AVAILABLE", True)
        monkeypatch.setattr(job_store_redis, "redis", _StubRedisModule(client=DeadFakeRedis()))

        with caplog.at_level(logging.WARNING, logger="spark_optima.api.jobs"):
            store = create_job_store()

        assert type(store) is JobStore
        assert any("server unreachable" in record.message for record in caplog.records)
