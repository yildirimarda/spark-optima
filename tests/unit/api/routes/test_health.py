# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the health check API endpoint.

This module contains tests for the health check endpoint to ensure
the API is functioning correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Test that health check returns HTTP 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_check_response_structure(self, client: TestClient) -> None:
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_includes_timestamp(self, client: TestClient) -> None:
        """Test that health check includes timestamp."""
        response = client.get("/health")
        data = response.json()

        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_health_check_includes_version(self, client: TestClient) -> None:
        """Test that health check includes version info."""
        response = client.get("/health")
        data = response.json()

        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_check_response_time(self, client: TestClient) -> None:
        """Test that health check responds quickly."""
        import time

        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond within 1 second


class TestHealthEndpointEdgeCases:
    """Test edge cases for the health endpoint."""

    def test_health_check_method_not_allowed(self, client: TestClient) -> None:
        """Test that POST to health endpoint is not allowed."""
        response = client.post("/health")

        assert response.status_code == 405  # Method Not Allowed

    def test_health_check_with_query_params(self, client: TestClient) -> None:
        """Test health check with query parameters (should ignore)."""
        response = client.get("/health?test=param")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Test cases for the root endpoint."""

    def test_root_returns_redirect_or_info(self, client: TestClient) -> None:
        """Test root endpoint behavior."""
        response = client.get("/")

        # Should either redirect or return info
        assert response.status_code in [200, 307, 308]

    def test_root_response_when_successful(self, client: TestClient) -> None:
        """Test root endpoint when it returns 200."""
        response = client.get("/")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


class TestReadinessEndpoint:
    """Test cases for the readiness check endpoint."""

    def test_readiness_check_returns_200(self, client: TestClient) -> None:
        """Test that readiness check returns HTTP 200 when ready."""
        response = client.get("/health/ready")

        assert response.status_code == 200

    def test_readiness_check_response_structure(self, client: TestClient) -> None:
        """Test readiness check response structure."""
        response = client.get("/health/ready")
        data = response.json()

        assert "status" in data
        assert isinstance(data["status"], str)

    def test_readiness_check_method_not_allowed(self, client: TestClient) -> None:
        """Test that POST to readiness endpoint is not allowed."""
        response = client.post("/health/ready")

        assert response.status_code == 405  # Method Not Allowed

    @patch("spark_optima.api.routes.health.get_optimization_service")
    def test_readiness_check_not_ready(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test readiness check when service is not ready."""
        # Make get_optimization_service raise an AttributeError directly
        mock_get_service.side_effect = AttributeError("Database not available")

        response = client.get("/health/ready")

        assert response.status_code == 200
        assert "not_ready" in response.json()["status"]


class TestLivenessEndpoint:
    """Test cases for the liveness check endpoint."""

    def test_liveness_check_returns_200(self, client: TestClient) -> None:
        """Test that liveness check returns HTTP 200."""
        response = client.get("/health/live")

        assert response.status_code == 200

    def test_liveness_check_response_structure(self, client: TestClient) -> None:
        """Test liveness check response structure."""
        response = client.get("/health/live")
        data = response.json()

        assert "status" in data
        assert data["status"] == "alive"

    def test_liveness_check_method_not_allowed(self, client: TestClient) -> None:
        """Test that POST to liveness endpoint is not allowed."""
        response = client.post("/health/live")

        assert response.status_code == 405  # Method Not Allowed


class TestHealthEndpointStatus:
    """Test cases for health endpoint status determination."""

    @patch("spark_optima.api.routes.health.get_optimization_service")
    def test_health_status_healthy(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test health check returns healthy when all components healthy."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = ["3.5.0", "3.4.0"]
        mock_get_service.return_value = mock_service

        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["components"]["config_database"] == "healthy"

    @patch("spark_optima.api.routes.health.get_optimization_service")
    def test_health_status_degraded(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test health check returns degraded when components degraded."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.return_value = []  # Empty list = degraded
        mock_get_service.return_value = mock_service

        response = client.get("/health")
        data = response.json()

        assert data["status"] == "degraded"

    @patch("spark_optima.api.routes.health.get_optimization_service")
    def test_health_status_unhealthy(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test health check returns unhealthy when component fails."""
        mock_service = MagicMock()
        mock_service.get_available_spark_versions.side_effect = RuntimeError("Database error")
        mock_get_service.return_value = mock_service

        response = client.get("/health")
        data = response.json()

        assert data["status"] == "unhealthy"
        assert data["components"]["config_database"] == "unhealthy"

    def test_health_check_includes_uptime(self, client: TestClient) -> None:
        """Test that health check includes uptime."""
        response = client.get("/health")
        data = response.json()

        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], int | float)
        assert data["uptime_seconds"] >= 0

    def test_health_check_includes_components(self, client: TestClient) -> None:
        """Test that health check includes components status."""
        response = client.get("/health")
        data = response.json()

        assert "components" in data
        assert isinstance(data["components"], dict)
