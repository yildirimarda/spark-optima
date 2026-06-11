# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the workload template endpoints (GET /api/v1/templates)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app
from spark_optima.api.routes.templates import get_template_registry
from spark_optima.api.security import API_KEYS_ENV_VAR
from spark_optima.core.templates import TemplateRegistry


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


class TestListTemplates:
    """Tests for GET /api/v1/templates."""

    def test_list_returns_all_bundled_templates(self, client: TestClient) -> None:
        """All bundled template names are listed, sorted by name."""
        response = client.get("/api/v1/templates")

        assert response.status_code == 200
        templates = response.json()["templates"]
        names = [template["name"] for template in templates]
        assert names == sorted(names)
        assert {"etl-batch", "interactive", "ml-training", "streaming"} <= set(names)

    def test_list_entries_carry_summary_fields(self, client: TestClient) -> None:
        """Each summary has name, display_name, description, parameter_count."""
        response = client.get("/api/v1/templates")

        assert response.status_code == 200
        for template in response.json()["templates"]:
            assert set(template.keys()) == {"name", "display_name", "description", "parameter_count"}
            assert template["display_name"]
            assert template["description"]
            assert template["parameter_count"] > 0

    def test_list_matches_core_registry(self, client: TestClient) -> None:
        """The API list mirrors TemplateRegistry (CLI parity)."""
        registry = TemplateRegistry()
        response = client.get("/api/v1/templates")

        listed = {template["name"]: template["parameter_count"] for template in response.json()["templates"]}
        expected = {template.name: len(template.config) for template in registry.list_templates()}
        assert listed == expected


class TestGetTemplate:
    """Tests for GET /api/v1/templates/{name}."""

    def test_detail_includes_config_with_comments(self, client: TestClient) -> None:
        """The detail payload carries values and per-parameter comments."""
        response = client.get("/api/v1/templates/etl-batch")

        assert response.status_code == 200
        payload = response.json()
        assert payload["name"] == "etl-batch"
        assert payload["display_name"]
        assert payload["workload_traits"]
        assert payload["recommended_for"]
        assert payload["config"]
        for parameter in payload["config"].values():
            assert "value" in parameter
            assert "comment" in parameter
        aqe = payload["config"]["spark.sql.adaptive.enabled"]
        assert aqe["value"] == "true"
        assert aqe["comment"]

    def test_detail_matches_core_registry(self, client: TestClient) -> None:
        """The API detail mirrors WorkloadTemplate.to_dict() (CLI parity)."""
        registry = TemplateRegistry()
        template = registry.get_template("streaming")

        response = client.get("/api/v1/templates/streaming")

        assert response.status_code == 200
        payload = response.json()
        assert payload == template.to_dict()

    def test_unknown_template_returns_404(self, client: TestClient) -> None:
        """Unknown template names are rejected with 404 listing alternatives."""
        response = client.get("/api/v1/templates/does-not-exist")

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "does-not-exist" in detail
        assert "etl-batch" in detail  # Available templates are listed


class TestTemplateRegistrySingleton:
    """Tests for the shared registry instance."""

    def test_registry_is_shared(self) -> None:
        """Repeated calls return the same registry instance."""
        assert get_template_registry() is get_template_registry()


class TestTemplatesSecurity:
    """Template endpoints participate in the /api/v1 auth wiring."""

    def test_missing_api_key_returns_401(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """With auth enabled, the templates endpoints require X-API-Key."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")

        assert client.get("/api/v1/templates").status_code == 401
        assert client.get("/api/v1/templates/etl-batch").status_code == 401

    def test_valid_api_key_is_accepted(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid key passes auth on both endpoints."""
        monkeypatch.setenv(API_KEYS_ENV_VAR, "secret-key")
        headers = {"X-API-Key": "secret-key"}

        assert client.get("/api/v1/templates", headers=headers).status_code == 200
        assert client.get("/api/v1/templates/etl-batch", headers=headers).status_code == 200
