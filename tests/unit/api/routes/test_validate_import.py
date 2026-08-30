# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for validate and import API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app

if TYPE_CHECKING:
    pass


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "spark-defaults.conf"
    file_path.write_text("spark.executor.memory=4g\nspark.executor.cores=2\n")
    return file_path


@pytest.fixture
def sample_code_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "test_job.py"
    file_path.write_text("from pyspark.sql import SparkSession\n")
    return file_path


class TestValidateEndpoint:
    def test_validate_missing_file(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/validate",
            json={"config_file": "/nonexistent/config.conf"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "not found" in data.get("detail", "").lower() or "not found" in str(data)

    def test_validate_valid_config(self, client: TestClient, sample_config_file: Path) -> None:
        response = client.post(
            "/api/v1/validate",
            json={
                "config_file": str(sample_config_file),
                "platform": "local",
                "spark_version": "3.5.0",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True or data["valid"] is False
        assert "issues" in data
        assert "parameter_count" in data

    def test_validate_json_format(self, client: TestClient, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"spark.executor.memory": "4g"}))
        response = client.post(
            "/api/v1/validate",
            json={
                "config_file": str(config_path),
                "platform": "",
                "spark_version": "3.5.0",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "issues" in data


class TestImportEndpoint:
    def test_import_valid(self, client: TestClient, sample_config_file: Path, sample_code_file: Path) -> None:
        response = client.post(
            "/api/v1/import",
            json={
                "config_file": str(sample_config_file),
                "code_path": str(sample_code_file),
                "platform": "local",
                "spark_version": "3.5.0",
                "data_size_gb": 10.0,
                "bayesian_trials": 10,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "current" in data
        assert "recommended" in data
        assert "diff" in data
        assert "estimated_time_minutes" in data

    def test_import_missing_config(self, client: TestClient, sample_code_file: Path) -> None:
        response = client.post(
            "/api/v1/import",
            json={
                "config_file": "/nonexistent/config.conf",
                "code_path": str(sample_code_file),
            },
        )
        assert response.status_code == 400
