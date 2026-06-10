# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for opt-in API-key authentication and rate limiting."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app
from spark_optima.api.security import (
    API_KEY_HEADER_NAME,
    API_KEYS_ENV_VAR,
    RATE_LIMIT_ENV_VAR,
    FixedWindowRateLimiter,
    get_configured_api_keys,
    get_configured_rate_limit,
    get_rate_limiter,
)

PROTECTED_URL = "/api/v1/platforms"


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_security_state(monkeypatch: pytest.MonkeyPatch):
    """Ensure auth/rate-limit env vars and limiter state never leak between tests."""
    monkeypatch.delenv(API_KEYS_ENV_VAR, raising=False)
    monkeypatch.delenv(RATE_LIMIT_ENV_VAR, raising=False)
    get_rate_limiter().reset()
    yield
    get_rate_limiter().reset()


class TestConfigAccessors:
    """Tests for the per-request environment accessors."""

    def test_api_keys_unset_means_disabled(self) -> None:
        """No env var means no configured keys."""
        assert get_configured_api_keys() == []

    def test_api_keys_parsing_strips_and_skips_empties(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Comma-separated keys are trimmed and blanks dropped."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, " alpha , beta ,, ")
        assert get_configured_api_keys() == ["alpha", "beta"]

    def test_rate_limit_unset_means_disabled(self) -> None:
        """No env var means a limit of 0 (disabled)."""
        assert get_configured_rate_limit() == 0

    @pytest.mark.parametrize("raw", ["0", "-5", "abc", "  "])
    def test_rate_limit_invalid_values_disable(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """Zero, negative, or non-numeric values disable rate limiting."""
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, raw)
        assert get_configured_rate_limit() == 0

    def test_rate_limit_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A positive integer is returned as-is."""
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, "42")
        assert get_configured_rate_limit() == 42


class TestApiKeyAuth:
    """Tests for the opt-in X-API-Key enforcement."""

    def test_open_by_default(self, client: TestClient) -> None:
        """Without SPARK_OPTIMA_API_KEYS the API stays fully open."""
        assert client.get(PROTECTED_URL).status_code == 200

    def test_missing_key_returns_401(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """When keys are configured, requests without a key are rejected."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.get(PROTECTED_URL)

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API key."
        assert "WWW-Authenticate" in response.headers

    def test_wrong_key_returns_401(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """A non-matching key gets the same 401 as a missing one."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "not-the-key"})

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or missing API key."

    def test_valid_key_returns_200(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """A matching key is accepted."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "secret-key"})

        assert response.status_code == 200

    def test_any_configured_key_is_accepted(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """Every key in the comma-separated list works, including padded ones."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "key-one, key-two ")

        assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "key-one"}).status_code == 200
        assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "key-two"}).status_code == 200

    def test_auth_applies_to_optimize_endpoints(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """POST endpoints under /api/v1 are protected too."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        response = client.post("/api/v1/analyze", json={"code": "print('hello world')"})

        assert response.status_code == 401

    def test_health_endpoints_stay_open(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """Health endpoints never require a key."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        assert client.get("/health").status_code == 200
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200

    def test_root_endpoint_stays_open(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """The root endpoint is outside /api/v1 and stays open."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        assert client.get("/").status_code == 200


class TestRateLimiting:
    """Tests for the opt-in fixed-window rate limiter."""

    def test_disabled_by_default(self, client: TestClient) -> None:
        """Without SPARK_OPTIMA_RATE_LIMIT, any number of requests passes."""
        for _ in range(10):
            assert client.get(PROTECTED_URL).status_code == 200

    def test_over_limit_returns_429_with_retry_after(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """N requests pass, request N+1 gets 429 with a Retry-After header."""
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, "3")

        for _ in range(3):
            assert client.get(PROTECTED_URL).status_code == 200

        response = client.get(PROTECTED_URL)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) >= 1
        assert "Rate limit exceeded" in response.json()["detail"]

    def test_health_open_even_when_rate_limited(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """Health endpoints are exempt from rate limiting."""
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, "1")

        assert client.get(PROTECTED_URL).status_code == 200
        assert client.get(PROTECTED_URL).status_code == 429
        assert client.get("/health/live").status_code == 200

    def test_limit_is_per_api_key_when_auth_enabled(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """With auth on, each API key gets its own request budget."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "alpha,beta")
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, "2")

        for _ in range(2):
            assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "alpha"}).status_code == 200
        assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "alpha"}).status_code == 429

        # A different key has an independent budget
        assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "beta"}).status_code == 200

    def test_unauthenticated_request_does_not_consume_budget(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        """Auth is checked before rate accounting: 401s do not count."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "alpha")
        monkeypatch.setenv(RATE_LIMIT_ENV_VAR, "1")

        assert client.get(PROTECTED_URL).status_code == 401
        assert client.get(PROTECTED_URL, headers={API_KEY_HEADER_NAME: "alpha"}).status_code == 200


class TestFixedWindowRateLimiter:
    """Unit tests for the limiter itself."""

    def test_allows_until_limit(self) -> None:
        """Requests within the limit are allowed with no retry delay."""
        limiter = FixedWindowRateLimiter()

        for _ in range(3):
            allowed, retry_after = limiter.check("client", 3)
            assert allowed is True
            assert retry_after == 0

        allowed, retry_after = limiter.check("client", 3)
        assert allowed is False
        assert retry_after >= 1

    def test_window_expiry_resets_budget(self) -> None:
        """After the window elapses, the same client is allowed again."""
        limiter = FixedWindowRateLimiter(window_seconds=0.05)

        assert limiter.check("client", 1)[0] is True
        assert limiter.check("client", 1)[0] is False

        time.sleep(0.06)
        assert limiter.check("client", 1)[0] is True

    def test_clients_are_independent(self) -> None:
        """One client's traffic never affects another's budget."""
        limiter = FixedWindowRateLimiter()

        assert limiter.check("a", 1)[0] is True
        assert limiter.check("a", 1)[0] is False
        assert limiter.check("b", 1)[0] is True

    def test_reset_clears_state(self) -> None:
        """reset() empties all tracked windows."""
        limiter = FixedWindowRateLimiter()

        assert limiter.check("client", 1)[0] is True
        assert limiter.check("client", 1)[0] is False

        limiter.reset()
        assert limiter.check("client", 1)[0] is True
