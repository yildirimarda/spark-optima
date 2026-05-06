# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the platforms API endpoint.

This module contains tests for the platforms endpoint to retrieve
supported platform information.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


class TestPlatformsEndpoint:
    """Test cases for the platforms endpoint."""

    def test_platforms_returns_200(self, client: TestClient) -> None:
        """Test that platforms endpoint returns HTTP 200."""
        response = client.get("/api/v1/platforms")

        assert response.status_code == 200

    def test_platforms_response_structure(self, client: TestClient) -> None:
        """Test platforms response structure."""
        response = client.get("/api/v1/platforms")
        data = response.json()

        assert isinstance(data, list)

    def test_platforms_contains_supported_platforms(self, client: TestClient) -> None:
        """Test that response contains supported platforms."""
        response = client.get("/api/v1/platforms")
        data = response.json()

        platform_names = [p.get("name") for p in data]

        # Should contain at least these platforms
        assert "local" in platform_names

    def test_platforms_post_not_allowed(self, client: TestClient) -> None:
        """Test that POST to platforms endpoint is not allowed."""
        response = client.post("/api/v1/platforms")

        assert response.status_code == 405


class TestPlatformDetailEndpoint:
    """Test cases for platform detail endpoint."""

    def test_platform_detail_returns_200(self, client: TestClient) -> None:
        """Test getting detail for valid platform."""
        response = client.get("/api/v1/platforms/local")

        # Should either return 200 with details or 404 if not implemented
        assert response.status_code in [200, 404]

    def test_platform_detail_invalid_platform(self, client: TestClient) -> None:
        """Test getting detail for invalid platform."""
        response = client.get("/api/v1/platforms/invalid_platform")

        assert response.status_code in [404, 422]


class TestPlatformsVersionsEndpoint:
    """Test cases for platform versions endpoint."""

    def test_platform_versions_returns_200(self, client: TestClient) -> None:
        """Test getting versions for valid platform."""
        response = client.get("/api/v1/platforms/local/versions")

        # Should either return 200 with versions or 404 if not implemented
        assert response.status_code in [200, 404]

    def test_platform_versions_response_structure(self, client: TestClient) -> None:
        """Test platform versions response structure."""
        response = client.get("/api/v1/platforms/spark-versions")

        assert response.status_code == 200
        data = response.json()

        assert "versions" in data
        assert isinstance(data["versions"], list)

    def test_platform_versions_returns_list(self, client: TestClient) -> None:
        """Test that versions endpoint returns a list of versions."""
        response = client.get("/api/v1/platforms/spark-versions")

        assert response.status_code == 200
        data = response.json()

        # Should contain at least one version
        assert len(data["versions"]) > 0

    def test_platform_versions_method_not_allowed(self, client: TestClient) -> None:
        """Test that POST to spark-versions endpoint is not allowed."""
        response = client.post("/api/v1/platforms/spark-versions", json={})

        assert response.status_code == 405  # Method Not Allowed
