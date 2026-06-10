# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for webhook URL validation, payload shape, and delivery.

Delivery tests run against httpx.MockTransport — no real network.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from spark_optima.api import webhooks
from spark_optima.api.jobs import Job
from spark_optima.api.webhooks import (
    WEBHOOK_MAX_ATTEMPTS,
    build_webhook_payload,
    deliver_webhook,
    validate_webhook_url,
)


class TestValidateWebhookUrl:
    """Tests for scheme validation and the best-effort SSRF guard."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/hook",
            "https://hooks.example.com:8443/spark/done?token=abc",
            "https://203.0.113.10/hook",
        ],
    )
    def test_valid_urls_pass_unchanged(self, url: str) -> None:
        """Public http(s) URLs are accepted as-is."""
        assert validate_webhook_url(url) == url

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/hook",
            "file:///etc/passwd",
            "gopher://example.com",
            "example.com/hook",  # no scheme
        ],
    )
    def test_non_http_schemes_are_rejected(self, url: str) -> None:
        """Only http and https schemes are allowed."""
        with pytest.raises(ValueError, match="http or https"):
            validate_webhook_url(url)

    def test_missing_hostname_is_rejected(self) -> None:
        """A URL without a hostname is rejected."""
        with pytest.raises(ValueError, match="hostname"):
            validate_webhook_url("http:///hook")

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:9000/hook",
            "http://LOCALHOST/hook",
            "http://app.localhost/hook",
            "http://127.0.0.1/hook",
            "http://127.8.9.10/hook",
            "https://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://service.cluster.internal/hook",
            "http://0.0.0.0:8000/hook",
            "http://[::1]:8080/hook",
        ],
    )
    def test_internal_targets_are_rejected(self, url: str) -> None:
        """Obvious internal targets are blocked by the SSRF guard."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_webhook_url(url)


class TestBuildWebhookPayload:
    """Tests for the notification payload shape."""

    def test_completed_job_payload(self) -> None:
        """Completed jobs include the result and omit the error key."""
        job = Job(
            job_id="abc123",
            platform="local",
            spark_version="3.5.0",
            status="completed",
            submitted_at="2026-06-10T10:00:00+00:00",
            finished_at="2026-06-10T10:05:00+00:00",
            result={"configuration": {"spark.executor.memory": "4g"}},
        )

        payload = build_webhook_payload(job)

        assert payload == {
            "job_id": "abc123",
            "status": "completed",
            "submitted_at": "2026-06-10T10:00:00+00:00",
            "finished_at": "2026-06-10T10:05:00+00:00",
            "result": {"configuration": {"spark.executor.memory": "4g"}},
        }

    def test_failed_job_payload(self) -> None:
        """Failed jobs include the error and omit the result key."""
        job = Job(
            job_id="def456",
            platform="local",
            spark_version="3.5.0",
            status="failed",
            submitted_at="2026-06-10T10:00:00+00:00",
            finished_at="2026-06-10T10:01:00+00:00",
            error="kaboom",
        )

        payload = build_webhook_payload(job)

        assert payload["status"] == "failed"
        assert payload["error"] == "kaboom"
        assert "result" not in payload


class _RecordingTransport:
    """Builds mock clients whose responses follow a scripted sequence."""

    def __init__(self, script: list[int | Exception]) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def build_client(self) -> httpx.Client:
        """Build an httpx client backed by the scripted MockTransport."""

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            action = self.script.pop(0)
            if isinstance(action, Exception):
                raise action
            return httpx.Response(action)

        return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture backoff sleeps instead of actually sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(webhooks, "_sleep", recorded.append)
    return recorded


def _use_transport(monkeypatch: pytest.MonkeyPatch, script: list[int | Exception]) -> _RecordingTransport:
    """Point deliver_webhook at a scripted mock transport."""
    transport = _RecordingTransport(script)
    monkeypatch.setattr(webhooks, "_build_client", transport.build_client)
    return transport


class TestDeliverWebhook:
    """Tests for delivery, retries, and backoff."""

    def test_success_on_first_attempt(self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
        """A 2xx response on the first POST means delivered, no retries."""
        transport = _use_transport(monkeypatch, [200])

        delivered = deliver_webhook("https://example.com/hook", {"job_id": "abc"})

        assert delivered is True
        assert len(transport.requests) == 1
        assert sleeps == []

    def test_posted_request_carries_json_payload(self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
        """The notification is POSTed as a JSON body to the exact URL."""
        transport = _use_transport(monkeypatch, [204])
        payload: dict[str, Any] = {"job_id": "abc", "status": "completed"}

        deliver_webhook("https://example.com/hook?token=t", payload)

        request = transport.requests[0]
        assert request.method == "POST"
        assert str(request.url) == "https://example.com/hook?token=t"
        assert json.loads(request.content) == payload

    def test_retries_after_server_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
    ) -> None:
        """A 5xx response is retried with 1s backoff before succeeding."""
        transport = _use_transport(monkeypatch, [500, 200])

        delivered = deliver_webhook("https://example.com/hook", {"job_id": "abc"})

        assert delivered is True
        assert len(transport.requests) == 2
        assert sleeps == [1.0]

    def test_retries_after_connection_error(self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
        """Transport errors are retried just like bad statuses."""
        transport = _use_transport(monkeypatch, [httpx.ConnectError("boom"), 200])

        delivered = deliver_webhook("https://example.com/hook", {"job_id": "abc"})

        assert delivered is True
        assert len(transport.requests) == 2
        assert sleeps == [1.0]

    def test_gives_up_after_max_attempts_with_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
    ) -> None:
        """Three failures exhaust the attempts; backoff doubles between them."""
        transport = _use_transport(monkeypatch, [500, 502, httpx.ConnectError("boom")])

        delivered = deliver_webhook("https://example.com/hook", {"job_id": "abc"})

        assert delivered is False
        assert len(transport.requests) == WEBHOOK_MAX_ATTEMPTS
        assert sleeps == [1.0, 2.0]

    def test_4xx_is_retried_and_reported_failed(self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
        """Non-2xx client errors count as failures too."""
        transport = _use_transport(monkeypatch, [404, 404, 404])

        delivered = deliver_webhook("https://example.com/hook", {"job_id": "abc"})

        assert delivered is False
        assert len(transport.requests) == WEBHOOK_MAX_ATTEMPTS

    def test_never_raises(self, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
        """Delivery failures return False instead of raising."""
        _use_transport(
            monkeypatch,
            [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("boom")],
        )

        assert deliver_webhook("https://example.com/hook", {"job_id": "abc"}) is False
