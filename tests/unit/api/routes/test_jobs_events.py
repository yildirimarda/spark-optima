# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the SSE job progress endpoint (GET /api/v1/jobs/{job_id}/events).

The TestClient buffers streaming responses until the ASGI app finishes, so
these tests drive job state transitions from timer threads and assert on
the fully buffered event stream once the generator closes.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Executor, Future
from typing import Any

import pytest
from fastapi.testclient import TestClient

from spark_optima.api import jobs as jobs_module
from spark_optima.api.jobs import JobStore, get_job_store, reset_job_store
from spark_optima.api.main import app
from spark_optima.api.routes import jobs as jobs_routes
from spark_optima.api.security import API_KEYS_ENV_VAR

PROGRESS_1 = {"trial_number": 0, "n_trials": 4, "trials_completed": 1, "best_value": 12.0, "state": "COMPLETE"}
PROGRESS_2 = {"trial_number": 3, "n_trials": 4, "trials_completed": 4, "best_value": 9.2, "state": "COMPLETE"}


class ManualExecutor(Executor):
    """Executor that never runs submitted callables (jobs stay pending)."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Swallow the callable and return an unresolved future."""
        return Future()


class InlineExecutor(Executor):
    """Executor that runs submitted callables synchronously."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Run the callable immediately and return a resolved future."""
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - the store handles errors itself
            future.set_exception(exc)
        return future


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def fast_sse(monkeypatch: pytest.MonkeyPatch):
    """Make the SSE poll loop fast and the store fresh for every test."""
    monkeypatch.setattr(jobs_routes, "EVENTS_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(jobs_module, "_EXECUTOR", ManualExecutor())
    reset_job_store()
    yield
    reset_job_store()


def _submit_pending_job(progress: dict[str, Any] | None = None) -> tuple[JobStore, str]:
    """Create a pending job (never-running executor) on the global store."""
    store = get_job_store()
    assert isinstance(store, JobStore)
    job_id = store.submit(lambda: {"ok": True}, platform="local", spark_version="3.5.0")
    if progress is not None:
        store.set_progress(job_id, progress)
    return store, job_id


def _schedule(delay: float, action) -> threading.Timer:
    """Run an action on a timer thread while the stream is being served."""
    timer = threading.Timer(delay, action)
    timer.daemon = True
    timer.start()
    return timer


def _events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse (event, data) pairs out of a raw SSE body."""
    parsed: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif line.startswith("data: ") and current_event is not None:
            parsed.append((current_event, json.loads(line.removeprefix("data: "))))
            current_event = None
    return parsed


class TestJobEventsStream:
    """Tests for the SSE stream contents."""

    def test_unknown_job_returns_404_before_streaming(self, client: TestClient) -> None:
        """An unknown job id is rejected with 404, not an empty stream."""
        response = client.get("/api/v1/jobs/deadbeefdeadbeefdeadbeefdeadbeef/events")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_progress_frame_then_done_frame(self, client: TestClient) -> None:
        """Existing progress is emitted first; the terminal state closes the stream."""
        store, job_id = _submit_pending_job(progress=PROGRESS_1)
        timer = _schedule(0.15, lambda: store._mark_completed(job_id, {"answer": 42}))

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        timer.join()

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        assert events[0] == ("progress", PROGRESS_1)
        assert events[-1][0] == "done"
        assert events[-1][1] == {"job_id": job_id, "status": "completed", "error": None}

    def test_done_frame_for_already_completed_job(self, client: TestClient) -> None:
        """A job that is already terminal yields done immediately and closes."""
        store, job_id = _submit_pending_job()
        store._mark_completed(job_id, {"answer": 42})

        response = client.get(f"/api/v1/jobs/{job_id}/events")

        assert response.status_code == 200
        events = _events(response.text)
        assert [name for name, _ in events] == ["done"]
        assert events[0][1]["status"] == "completed"

    def test_failed_job_done_frame_carries_error(self, client: TestClient) -> None:
        """Failed jobs close the stream with status=failed and the error."""
        store, job_id = _submit_pending_job()
        store._mark_failed(job_id, "optimizer exploded")

        response = client.get(f"/api/v1/jobs/{job_id}/events")

        events = _events(response.text)
        assert events[-1][0] == "done"
        assert events[-1][1]["status"] == "failed"
        assert events[-1][1]["error"] == "optimizer exploded"

    def test_progress_updates_emit_new_frames(self, client: TestClient) -> None:
        """A changed snapshot produces a second progress frame."""
        store, job_id = _submit_pending_job(progress=PROGRESS_1)
        timers = [
            _schedule(0.10, lambda: store.set_progress(job_id, PROGRESS_2)),
            _schedule(0.25, lambda: store._mark_completed(job_id, {})),
        ]

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        for timer in timers:
            timer.join()

        events = _events(response.text)
        progress_events = [data for name, data in events if name == "progress"]
        assert progress_events == [PROGRESS_1, PROGRESS_2]
        assert events[-1][0] == "done"

    def test_unchanged_progress_is_not_repeated(self, client: TestClient) -> None:
        """Many poll cycles over an unchanged snapshot emit a single frame."""
        store, job_id = _submit_pending_job(progress=PROGRESS_1)
        # ~15 poll cycles at the 0.01s test interval before the job finishes
        timer = _schedule(0.15, lambda: store._mark_completed(job_id, {}))

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        timer.join()

        events = _events(response.text)
        progress_events = [data for name, data in events if name == "progress"]
        assert progress_events == [PROGRESS_1]


class TestJobEventsHeartbeat:
    """Tests for the keep-alive heartbeat comments."""

    def test_heartbeat_comment_format(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quiet streams emit ': keep-alive' comment lines."""
        monkeypatch.setattr(jobs_routes, "EVENTS_HEARTBEAT_INTERVAL_SECONDS", 0.0)
        store, job_id = _submit_pending_job()
        timer = _schedule(0.15, lambda: store._mark_completed(job_id, {}))

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        timer.join()

        lines = response.text.splitlines()
        heartbeats = [line for line in lines if line.startswith(":")]
        assert heartbeats
        assert all(line == ": keep-alive" for line in heartbeats)
        assert _events(response.text)[-1][0] == "done"

    def test_no_heartbeat_before_the_interval_elapses(self, client: TestClient) -> None:
        """With the 10s default interval, short streams carry no heartbeat."""
        store, job_id = _submit_pending_job(progress=PROGRESS_1)
        timer = _schedule(0.1, lambda: store._mark_completed(job_id, {}))

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        timer.join()

        assert not [line for line in response.text.splitlines() if line.startswith(":")]


class TestJobEventsSecurity:
    """The SSE endpoint participates in the /api/v1 auth wiring."""

    def test_missing_api_key_returns_401(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """With auth enabled, the events endpoint requires X-API-Key."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.get("/api/v1/jobs/some-job/events")

        assert response.status_code == 401

    def test_valid_api_key_reaches_the_endpoint(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid key passes auth (and an unknown job then yields 404)."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.get("/api/v1/jobs/some-job/events", headers={"X-API-Key": "secret-key"})

        assert response.status_code == 404


class TestAsyncJobProgressWiring:
    """POST /optimize/async wires the optimizer progress into the store."""

    def test_async_job_persists_progress_from_optimizer(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-trial events emitted by the optimizer land on the job record."""
        from unittest.mock import MagicMock, patch

        from spark_optima.api.routes import optimize as optimize_routes

        # Run the job synchronously and persist every progress write
        monkeypatch.setattr(jobs_module, "_EXECUTOR", InlineExecutor())
        monkeypatch.setattr(optimize_routes, "PROGRESS_WRITE_INTERVAL_SECONDS", 0.0)
        reset_job_store()

        def fake_optimize(**kwargs: Any) -> MagicMock:
            """Simulate a run that reports two progress events."""
            progress_callback = kwargs["progress_callback"]
            progress_callback(PROGRESS_1)
            progress_callback(PROGRESS_2)
            return MagicMock(
                configuration={"spark.executor.memory": "4g"},
                estimated_time_minutes=10.0,
                confidence_score=0.95,
                platform_specific={"platform": "local", "spark_version": "3.5.0"},
                code_suggestions=[],
                metadata={"resources": {}, "data_profile": None, "code_analysis": None},
            )

        service = MagicMock()
        service.validate_spark_version.return_value = True
        optimizer = MagicMock()
        optimizer.optimize.side_effect = lambda **kwargs: fake_optimize(**kwargs)
        service.get_optimizer.return_value = optimizer

        with patch("spark_optima.api.routes.optimize.get_optimization_service", return_value=service):
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
        assert payload["progress"] == PROGRESS_2  # Latest snapshot wins

        # The finished stream replays the snapshot and closes with done
        events = _events(client.get(f"/api/v1/jobs/{job_id}/events").text)
        assert ("progress", PROGRESS_2) in events
        assert events[-1][0] == "done"

    def test_throttle_skips_intermediate_writes(self) -> None:
        """With the default interval, rapid events collapse to first + final."""
        from spark_optima.api.routes.optimize import JobProgressRecorder

        store = get_job_store()
        assert isinstance(store, JobStore)
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        recorder = JobProgressRecorder(store)
        recorder.bind(job_id)
        recorder({"trial_number": 0, "n_trials": 3, "trials_completed": 1})
        recorder({"trial_number": 1, "n_trials": 3, "trials_completed": 2})  # Throttled
        job = store.get(job_id)
        assert job is not None
        assert job.progress == {"trial_number": 0, "n_trials": 3, "trials_completed": 1}

        recorder({"trial_number": 2, "n_trials": 3, "trials_completed": 3})  # Final: always written
        job = store.get(job_id)
        assert job is not None
        assert job.progress == {"trial_number": 2, "n_trials": 3, "trials_completed": 3}


class TestJobDetailProgressField:
    """GET /api/v1/jobs/{job_id} exposes the progress snapshot."""

    def test_detail_includes_progress(self, client: TestClient) -> None:
        """The detail payload carries the latest progress dictionary."""
        _, job_id = _submit_pending_job(progress=PROGRESS_1)

        response = client.get(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json()["progress"] == PROGRESS_1

    def test_detail_progress_null_before_first_update(self, client: TestClient) -> None:
        """Jobs without progress report null."""
        _, job_id = _submit_pending_job()

        response = client.get(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json()["progress"] is None


class TestJobProgressRecorderFlush:
    """flush() persists the last throttled event after early-stopped runs."""

    def test_flush_persists_last_throttled_event(self) -> None:
        """Two rapid (throttled) events followed by flush persist the second."""
        from spark_optima.api.routes.optimize import JobProgressRecorder

        store = get_job_store()
        assert isinstance(store, JobStore)
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        recorder = JobProgressRecorder(store)
        recorder.bind(job_id)
        first = {"trial_number": 0, "n_trials": 10, "trials_completed": 1}
        second = {"trial_number": 1, "n_trials": 10, "trials_completed": 2}
        recorder(first)  # Persisted (first write is never throttled)
        recorder(second)  # Throttled away by the 0.5s default interval

        job = store.get(job_id)
        assert job is not None
        assert job.progress == first

        recorder.flush()

        job = store.get(job_id)
        assert job is not None
        assert job.progress == second

    def test_flush_is_noop_without_events(self) -> None:
        """flush() before any progress event leaves the job untouched."""
        from spark_optima.api.routes.optimize import JobProgressRecorder

        store = get_job_store()
        assert isinstance(store, JobStore)
        job_id = store.submit(lambda: {}, platform="local", spark_version="3.5.0")

        recorder = JobProgressRecorder(store)
        recorder.bind(job_id)
        recorder.flush()

        job = store.get(job_id)
        assert job is not None
        assert job.progress is None

    def test_failed_async_job_flushes_throttled_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An early-stopped (failing) run still persists its last progress event."""
        from unittest.mock import MagicMock, patch

        client = TestClient(app)
        monkeypatch.setattr(jobs_module, "_EXECUTOR", InlineExecutor())
        reset_job_store()

        first = {"trial_number": 0, "n_trials": 10, "trials_completed": 1}
        second = {"trial_number": 1, "n_trials": 10, "trials_completed": 2}

        def fake_optimize(**kwargs: Any) -> MagicMock:
            """Emit two rapid (throttled) events, then stop early."""
            progress_callback = kwargs["progress_callback"]
            progress_callback(first)
            progress_callback(second)  # Throttled by the default 0.5s interval
            raise RuntimeError("stopped early: max consecutive failures")

        service = MagicMock()
        service.validate_spark_version.return_value = True
        optimizer = MagicMock()
        optimizer.optimize.side_effect = lambda **kwargs: fake_optimize(**kwargs)
        service.get_optimizer.return_value = optimizer

        with patch("spark_optima.api.routes.optimize.get_optimization_service", return_value=service):
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

        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "failed"
        assert detail["progress"] == second  # Flushed despite the throttle


class TestJobEventsOffLoopPolling:
    """SSE polling must never call the blocking job store on the event loop."""

    def test_store_polling_runs_off_the_event_loop(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every store.get() during streaming runs in a worker thread."""
        import asyncio

        store, job_id = _submit_pending_job(progress=PROGRESS_1)
        timer = _schedule(0.15, lambda: store._mark_completed(job_id, {"answer": 42}))

        real_get = store.get
        on_loop_calls: list[str] = []

        def recording_get(requested_job_id: str):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass  # Worker thread: no running loop, as required
            else:
                on_loop_calls.append(requested_job_id)
            return real_get(requested_job_id)

        monkeypatch.setattr(store, "get", recording_get)

        response = client.get(f"/api/v1/jobs/{job_id}/events")
        timer.join()

        assert response.status_code == 200
        assert _events(response.text)[-1][0] == "done"
        assert on_loop_calls == []  # No store access ever ran on the event loop
