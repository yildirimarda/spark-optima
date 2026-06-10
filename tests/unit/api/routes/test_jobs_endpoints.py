# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the asynchronous optimization endpoints and job routes."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from spark_optima.api import jobs as jobs_module
from spark_optima.api.jobs import reset_job_store
from spark_optima.api.main import app

VALID_REQUEST = {
    "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()",
    "platform": "local",
    "spark_version": "3.5.0",
    "resources": {"cpu_cores": 4, "memory_gb": 16},
}

MAX_POLLS = 50


class InlineExecutor(Executor):
    """Executor that runs submitted callables synchronously for determinism."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        """Run the callable immediately and return a resolved future."""
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - JobStore handles errors itself
            future.set_exception(exc)
        return future


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolated_job_store(monkeypatch: pytest.MonkeyPatch):
    """Run jobs synchronously on a fresh store so polling is deterministic."""
    monkeypatch.setattr(jobs_module, "_EXECUTOR", InlineExecutor())
    reset_job_store()
    yield
    reset_job_store()


def _mock_service(succeed: bool = True) -> MagicMock:
    """Build a mocked optimization service for endpoint tests."""
    service = MagicMock()
    service.validate_spark_version.return_value = True
    optimizer = MagicMock()
    if succeed:
        optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
    else:
        optimizer.optimize.side_effect = RuntimeError("optimizer exploded")
    service.get_optimizer.return_value = optimizer
    return service


def _poll_until_finished(client: TestClient, status_url: str) -> dict[str, Any]:
    """Poll the job status URL until the job leaves pending/running."""
    for _ in range(MAX_POLLS):
        response = client.get(status_url)
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in ("completed", "failed"):
            return payload
    raise AssertionError(f"Job did not finish after {MAX_POLLS} polls")


class TestOptimizeAsyncSubmit:
    """Tests for POST /api/v1/optimize/async."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_submit_returns_202_with_job_info(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Submission is accepted with a job id and polling URL."""
        mock_get_service.return_value = _mock_service()

        response = client.post("/api/v1/optimize/async", json=VALID_REQUEST)

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ("pending", "running", "completed")
        assert data["status_url"] == f"/api/v1/jobs/{data['job_id']}"

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_submit_then_poll_until_completed(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Full happy path: submit, poll, and read the optimization result."""
        mock_get_service.return_value = _mock_service()

        submit = client.post("/api/v1/optimize/async", json=VALID_REQUEST)
        assert submit.status_code == 202

        job = _poll_until_finished(client, submit.json()["status_url"])

        assert job["status"] == "completed"
        assert job["error"] is None
        assert job["platform"] == "local"
        assert job["spark_version"] == "3.5.0"
        assert job["submitted_at"]
        assert job["started_at"] is not None
        assert job["finished_at"] is not None
        assert job["result"]["status"] == "success"
        assert job["result"]["configuration"] == {"spark.executor.memory": "4g"}

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_failed_optimization_marks_job_failed(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """An optimizer error surfaces as a failed job with the error message."""
        mock_get_service.return_value = _mock_service(succeed=False)

        submit = client.post("/api/v1/optimize/async", json=VALID_REQUEST)
        assert submit.status_code == 202

        job = _poll_until_finished(client, submit.json()["status_url"])

        assert job["status"] == "failed"
        assert job["result"] is None
        assert "optimizer exploded" in job["error"]

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_unsupported_spark_version_rejected_upfront(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Invalid Spark versions fail with 400 before any job is queued."""
        service = _mock_service()
        service.validate_spark_version.return_value = False
        service.get_available_spark_versions.return_value = ["3.5.0"]
        mock_get_service.return_value = service

        response = client.post("/api/v1/optimize/async", json=VALID_REQUEST)

        assert response.status_code == 400
        assert "Unsupported Spark version" in response.json()["detail"]
        assert client.get("/api/v1/jobs").json()["jobs"] == []

    def test_submit_invalid_payload_returns_422(self, client: TestClient) -> None:
        """Request validation still applies to the async endpoint."""
        response = client.post("/api/v1/optimize/async", json={"platform": "local"})

        assert response.status_code == 422


class TestGetJobEndpoint:
    """Tests for GET /api/v1/jobs/{job_id}."""

    def test_unknown_job_returns_404(self, client: TestClient) -> None:
        """Polling an unknown job id returns 404."""
        response = client.get("/api/v1/jobs/deadbeefdeadbeefdeadbeefdeadbeef")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestListJobsEndpoint:
    """Tests for GET /api/v1/jobs."""

    def test_empty_store_returns_empty_list(self, client: TestClient) -> None:
        """The list endpoint returns an empty list when no jobs exist."""
        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        assert response.json() == {"jobs": []}

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_list_contains_submitted_jobs(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Submitted jobs appear in the job list with summary fields."""
        mock_get_service.return_value = _mock_service()

        id1 = client.post("/api/v1/optimize/async", json=VALID_REQUEST).json()["job_id"]
        id2 = client.post("/api/v1/optimize/async", json=VALID_REQUEST).json()["job_id"]

        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        jobs = response.json()["jobs"]

        assert len(jobs) == 2
        assert {job["job_id"] for job in jobs} == {id1, id2}
        for job in jobs:
            assert job["platform"] == "local"
            assert job["spark_version"] == "3.5.0"
            assert job["status"] == "completed"
            assert "result" not in job  # summaries omit the payload

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_list_respects_limit(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """The limit query parameter caps the number of returned jobs."""
        mock_get_service.return_value = _mock_service()

        for _ in range(3):
            client.post("/api/v1/optimize/async", json=VALID_REQUEST)

        response = client.get("/api/v1/jobs", params={"limit": 1})
        assert response.status_code == 200
        assert len(response.json()["jobs"]) == 1

    def test_list_rejects_invalid_limit(self, client: TestClient) -> None:
        """Out-of-range limits are rejected by validation."""
        assert client.get("/api/v1/jobs", params={"limit": 0}).status_code == 422
        assert client.get("/api/v1/jobs", params={"limit": 9999}).status_code == 422
