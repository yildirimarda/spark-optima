# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Endpoint tests for webhook callbacks on POST /api/v1/optimize/async.

Webhook delivery is monkeypatched (or routed through httpx.MockTransport)
so no real network traffic occurs.
"""

from __future__ import annotations

from concurrent.futures import Executor, Future
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from spark_optima.api import jobs as jobs_module
from spark_optima.api import webhooks
from spark_optima.api.jobs import JOB_STORE_ENV_VAR, reset_job_store
from spark_optima.api.main import app


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


@pytest.fixture(autouse=True)
def inline_memory_job_store(monkeypatch: pytest.MonkeyPatch):
    """Use a fresh in-memory store and run jobs synchronously."""
    monkeypatch.delenv(JOB_STORE_ENV_VAR, raising=False)
    monkeypatch.setattr(jobs_module, "_EXECUTOR", InlineExecutor())
    reset_job_store()
    yield
    reset_job_store()


@pytest.fixture
def client() -> TestClient:
    """Provide a test client for the API."""
    return TestClient(app)


def _request_body(webhook_url: str | None = None) -> dict[str, Any]:
    """Build a valid async optimization request body."""
    body: dict[str, Any] = {
        "code": "from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()",
        "platform": "local",
        "spark_version": "3.5.0",
        "resources": {"cpu_cores": 4, "memory_gb": 16},
    }
    if webhook_url is not None:
        body["webhook_url"] = webhook_url
    return body


def _mock_service(fail: bool = False) -> MagicMock:
    """Build a mocked optimization service for endpoint tests."""
    service = MagicMock()
    service.validate_spark_version.return_value = True
    if fail:
        service.get_optimizer.side_effect = RuntimeError("optimizer exploded")
        return service
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


class TestWebhookUrlValidation:
    """Request validation for the webhook_url field (422 before any work)."""

    def test_bad_scheme_returns_422(self, client: TestClient) -> None:
        """Non-http(s) schemes are rejected by request validation."""
        response = client.post("/api/v1/optimize/async", json=_request_body("ftp://example.com/hook"))
        assert response.status_code == 422
        assert "http or https" in response.text

    def test_internal_host_returns_422(self, client: TestClient) -> None:
        """Internal targets are rejected by the SSRF guard."""
        for url in (
            "http://localhost:9000/hook",
            "http://127.0.0.1/hook",
            "https://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/x",
        ):
            response = client.post("/api/v1/optimize/async", json=_request_body(url))
            assert response.status_code == 422, url
            assert "not allowed" in response.text

    def test_sync_endpoint_also_validates_webhook_url(self, client: TestClient) -> None:
        """The shared request model rejects bad URLs on /optimize too."""
        response = client.post("/api/v1/optimize", json=_request_body("ftp://example.com/hook"))
        assert response.status_code == 422


class TestWebhookDeliveryFlow:
    """End-to-end webhook behavior through the async endpoints."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_completed_job_delivers_webhook(
        self, mock_get_service: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On completion the webhook gets the result payload and status is recorded."""
        mock_get_service.return_value = _mock_service()
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(webhooks, "deliver_webhook", lambda url, payload: calls.append((url, payload)) or True)

        submit = client.post("/api/v1/optimize/async", json=_request_body("https://example.com/hook"))
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        assert len(calls) == 1
        url, payload = calls[0]
        assert url == "https://example.com/hook"
        assert payload["job_id"] == job_id
        assert payload["status"] == "completed"
        assert payload["submitted_at"]
        assert payload["finished_at"]
        assert payload["result"]["configuration"] == {"spark.executor.memory": "4g"}
        assert "error" not in payload

        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "completed"
        assert detail["webhook_status"] == "delivered"

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_failed_job_delivers_webhook_with_error(
        self, mock_get_service: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Webhooks fire on failure too, carrying the error message."""
        mock_get_service.return_value = _mock_service(fail=True)
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(webhooks, "deliver_webhook", lambda url, payload: calls.append((url, payload)) or True)

        submit = client.post("/api/v1/optimize/async", json=_request_body("https://example.com/hook"))
        job_id = submit.json()["job_id"]

        assert len(calls) == 1
        _, payload = calls[0]
        assert payload["status"] == "failed"
        assert "optimizer exploded" in payload["error"]
        assert "result" not in payload

        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "failed"
        assert detail["webhook_status"] == "delivered"

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_delivery_failure_never_affects_job_state(
        self, mock_get_service: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An undeliverable webhook marks webhook_status failed; the job stays completed."""
        mock_get_service.return_value = _mock_service()
        monkeypatch.setattr(webhooks, "deliver_webhook", lambda url, payload: False)

        submit = client.post("/api/v1/optimize/async", json=_request_body("https://example.com/hook"))
        job_id = submit.json()["job_id"]

        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "completed"
        assert detail["result"]["configuration"] == {"spark.executor.memory": "4g"}
        assert detail["webhook_status"] == "failed"

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_no_webhook_url_means_no_delivery(
        self, mock_get_service: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without webhook_url nothing is delivered and webhook_status stays null."""
        mock_get_service.return_value = _mock_service()
        calls: list[Any] = []
        monkeypatch.setattr(webhooks, "deliver_webhook", lambda url, payload: calls.append(url) or True)

        submit = client.post("/api/v1/optimize/async", json=_request_body())
        job_id = submit.json()["job_id"]

        assert calls == []
        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "completed"
        assert detail["webhook_status"] is None

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_full_delivery_path_with_retry_over_mock_transport(
        self, mock_get_service: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real deliver_webhook retries through httpx and records delivery."""
        mock_get_service.return_value = _mock_service()
        received: list[httpx.Request] = []
        responses = iter([503, 200])

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(request)
            return httpx.Response(next(responses))

        monkeypatch.setattr(webhooks, "_build_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
        monkeypatch.setattr(webhooks, "_sleep", lambda seconds: None)

        submit = client.post("/api/v1/optimize/async", json=_request_body("https://example.com/hook"))
        job_id = submit.json()["job_id"]

        assert len(received) == 2  # 503 then 200
        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["webhook_status"] == "delivered"
