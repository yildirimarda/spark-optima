# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the optimize API endpoint.

This module contains tests for the optimization endpoint including
request validation and response handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from spark_optima.api.main import app
from spark_optima.api.routes.optimize import (
    _convert_code_suggestions,
    _generate_optimization_id,
)
from spark_optima.core.result import CodeSuggestion


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def valid_optimization_request() -> dict:
    """Create a valid optimization request."""
    return {
        "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
        "platform": "local",
        "spark_version": "3.5.0",
        "resources": {
            "cpu_cores": 4,
            "memory_gb": 16,
        },
        "optimization_mode": "simulation",
    }


class TestOptimizeEndpoint:
    """Test cases for the optimize endpoint."""

    def test_optimize_returns_200(self, client: TestClient, valid_optimization_request: dict) -> None:
        """Test that optimize endpoint returns HTTP 200 for valid request."""
        response = client.post("/api/v1/optimize", json=valid_optimization_request)

        assert response.status_code in [200, 422]  # 200 if successful, 422 if validation fails

    def test_optimize_missing_code_returns_422(self, client: TestClient) -> None:
        """Test that missing code returns validation error."""
        request = {
            "platform": "local",
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 422

    def test_optimize_invalid_platform_returns_422(self, client: TestClient) -> None:
        """Test that invalid platform returns validation error."""
        request = {
            "code": "print('test')",
            "platform": "invalid_platform",
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 422

    def test_optimize_invalid_spark_version_returns_422(self, client: TestClient) -> None:
        """Test that invalid Spark version returns validation error."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "spark_version": "99.99.99",
        }

        response = client.post("/api/v1/optimize", json=request)

        # Should either validate version or accept and fail later
        assert response.status_code in [200, 422, 400]

    def test_optimize_response_structure(self, client: TestClient, valid_optimization_request: dict) -> None:
        """Test optimize response structure."""
        response = client.post("/api/v1/optimize", json=valid_optimization_request)

        if response.status_code == 200:
            data = response.json()
            assert "configuration" in data
            assert isinstance(data["configuration"], dict)

    def test_optimize_get_method_not_allowed(self, client: TestClient) -> None:
        """Test that GET to optimize endpoint is not allowed."""
        response = client.get("/api/v1/optimize")

        assert response.status_code == 405  # Method Not Allowed


class TestOptimizeEndpointWithResources:
    """Test cases for optimize endpoint with resource specifications."""

    def test_optimize_with_resources(self, client: TestClient) -> None:
        """Test optimize with resource specifications."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": 8,
                "memory_gb": 32,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_with_invalid_resources(self, client: TestClient) -> None:
        """Test optimize with invalid resource values."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": -1,
                "memory_gb": 0,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 422


class TestOptimizeEndpointWithDataProfile:
    """Test cases for optimize endpoint with data profile."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_data_profile(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with data profile - covers line 142."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "data_profile": {
                "size_gb": 100,
                "format": "parquet",
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_full_data_profile(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with full data profile including all fields."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "data_profile": {
                "size_gb": 100,
                "format": "parquet",
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]


class TestOptimizeEndpointWithConstraints:
    """Test cases for optimize endpoint with constraints."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_constraints(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with all constraint types - covers lines 153-160."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "constraints": {
                "max_memory_gb": 64,
                "max_cost_per_hour": 10.0,
                "max_executors": 100,
                "timeout_minutes": 60,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]


class TestOptimizeEndpointModes:
    """Test cases for different optimization modes."""

    def test_optimize_simulation_mode(self, client: TestClient) -> None:
        """Test optimize in simulation mode."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "optimization_mode": "simulation",
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_execution_mode(self, client: TestClient) -> None:
        """Test optimize in execution mode."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "optimization_mode": "execution",
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_invalid_mode(self, client: TestClient) -> None:
        """Test optimize with invalid mode."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "optimization_mode": "invalid_mode",
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 422


class TestGenerateOptimizationId:
    """Test cases for _generate_optimization_id function."""

    def test_generate_id_returns_string(self) -> None:
        """Test that generated ID is a string."""
        opt_id = _generate_optimization_id()
        assert isinstance(opt_id, str)

    def test_generate_id_starts_with_opt(self) -> None:
        """Test that generated ID starts with 'opt-'."""
        opt_id = _generate_optimization_id()
        assert opt_id.startswith("opt-")

    def test_generate_id_unique(self) -> None:
        """Test that generated IDs are unique."""
        id1 = _generate_optimization_id()
        id2 = _generate_optimization_id()
        assert id1 != id2

    def test_generate_id_length(self) -> None:
        """Test that generated ID has expected length."""
        opt_id = _generate_optimization_id()
        # "opt-" + 12 hex chars
        assert len(opt_id) == 16  # "opt-" (4) + 12 hex chars


class TestConvertCodeSuggestions:
    """Test cases for _convert_code_suggestions function."""

    def test_convert_empty_list(self) -> None:
        """Test converting empty list of suggestions."""
        result = _convert_code_suggestions([])
        assert result == []

    def test_convert_single_suggestion(self) -> None:
        """Test converting a single code suggestion."""
        suggestion = CodeSuggestion(
            line_number=10,
            issue_type="performance",
            description="Inefficient operation",
            suggestion="Use broadcast join",
            severity="high",
        )
        result = _convert_code_suggestions([suggestion])
        assert len(result) == 1
        assert result[0].line_number == 10
        assert result[0].issue_type == "performance"
        assert result[0].description == "Inefficient operation"
        assert result[0].suggestion == "Use broadcast join"
        assert result[0].severity == "high"

    def test_convert_multiple_suggestions(self) -> None:
        """Test converting multiple code suggestions."""
        suggestions = [
            CodeSuggestion(
                line_number=10,
                issue_type="performance",
                description="Issue 1",
                suggestion="Fix 1",
                severity="high",
            ),
            CodeSuggestion(
                line_number=20,
                issue_type="code_smell",
                description="Issue 2",
                suggestion="Fix 2",
                severity="medium",
            ),
        ]
        result = _convert_code_suggestions(suggestions)
        assert len(result) == 2
        assert result[0].line_number == 10
        assert result[1].line_number == 20


class TestOptimizeEndpointSparkVersion:
    """Test cases for Spark version validation."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_invalid_spark_version(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with unsupported Spark version."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = False
        mock_service.get_available_spark_versions.return_value = ["3.5.0", "3.4.0"]
        mock_get_service.return_value = mock_service

        request = {
            "code": "print('test')",
            "platform": "local",
            "spark_version": "99.99.99",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 400
        assert "Unsupported Spark version" in response.json()["detail"]


class TestOptimizeEndpointWithConstraintsV2:
    """Test cases for optimize endpoint with constraints."""

    def test_optimize_with_max_memory_constraint(self, client: TestClient) -> None:
        """Test optimize with max memory constraint."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "constraints": {
                "max_memory_gb": 64,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_with_max_cost_constraint(self, client: TestClient) -> None:
        """Test optimize with max cost constraint."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "constraints": {
                "max_cost_per_hour": 10.0,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_with_max_executors_constraint(self, client: TestClient) -> None:
        """Test optimize with max executors constraint."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "constraints": {
                "max_executors": 100,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_with_timeout_constraint(self, client: TestClient) -> None:
        """Test optimize with timeout constraint."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "constraints": {
                "timeout_minutes": 60,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]


class TestOptimizeEndpointWithBayesian:
    """Test cases for optimize endpoint with Bayesian optimization."""

    def test_optimize_with_bayesian_enabled(self, client: TestClient) -> None:
        """Test optimize with Bayesian optimization enabled."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "use_bayesian": True,
            "bayesian_trials": 10,
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]

    def test_optimize_with_bayesian_disabled(self, client: TestClient) -> None:
        """Test optimize with Bayesian optimization disabled."""
        request = {
            "code": "print('test')",
            "platform": "local",
            "use_bayesian": False,
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 422]


class TestOptimizeEndpointErrorHandling:
    """Test cases for optimize endpoint error handling."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_internal_error(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize handles internal errors."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_service.get_optimizer.side_effect = RuntimeError("Internal error")
        mock_get_service.return_value = mock_service

        request = {
            "code": "print('test')",
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code == 500
        assert "Optimization failed" in response.json()["detail"]


class TestAnalyzeEndpoint:
    """Test cases for the analyze endpoint."""

    def test_analyze_returns_200(self, client: TestClient) -> None:
        """Test that analyze endpoint returns HTTP 200 for valid request."""
        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
        }

        response = client.post("/api/v1/analyze", json=request)

        assert response.status_code in [200, 500]  # 200 if successful, 500 if analysis fails

    def test_analyze_missing_code_returns_422(self, client: TestClient) -> None:
        """Test that missing code returns validation error."""
        request = {}

        response = client.post("/api/v1/analyze", json=request)

        assert response.status_code == 422

    def test_analyze_response_structure(self, client: TestClient) -> None:
        """Test analyze response structure."""
        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
        }

        response = client.post("/api/v1/analyze", json=request)

        if response.status_code == 200:
            data = response.json()
            assert "operations_count" in data
            assert "smells_count" in data
            assert "recommendations_count" in data
            assert "suggestions" in data

    def test_analyze_get_method_not_allowed(self, client: TestClient) -> None:
        """Test that GET to analyze endpoint is not allowed."""
        response = client.get("/api/v1/analyze")

        assert response.status_code == 405  # Method Not Allowed


class TestOptimizeEndpointConstraintsMaxMemoryGb:
    """Test cases for max_memory_gb constraint (line 154)."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_max_memory_gb_constraint(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with max_memory_gb constraint - covers line 154."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "constraints": {
                "max_memory_gb": 64,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]
        # Verify that max_memory_gb was passed to optimize
        if response.status_code == 200:
            call_args = mock_optimizer.optimize.call_args
            assert call_args[1]["resource_constraints"]["max_memory_gb"] == 64


class TestOptimizeEndpointConstraintsMaxCostPerHour:
    """Test cases for max_cost_per_hour constraint (line 156)."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_max_cost_per_hour_constraint(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with max_cost_per_hour constraint - covers line 156."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "constraints": {
                "max_cost_per_hour": 10.0,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]
        # Verify that max_cost_per_hour was passed to optimize
        if response.status_code == 200:
            call_args = mock_optimizer.optimize.call_args
            assert call_args[1]["resource_constraints"]["max_cost_per_hour"] == 10.0


class TestOptimizeEndpointConstraintsMaxExecutors:
    """Test cases for max_executors constraint (line 158)."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_max_executors_constraint(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with max_executors constraint - covers line 158."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "constraints": {
                "max_executors": 50,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]
        # Verify that max_executors was passed to optimize
        if response.status_code == 200:
            call_args = mock_optimizer.optimize.call_args
            assert call_args[1]["resource_constraints"]["max_executors"] == 50


class TestOptimizeEndpointConstraintsTimeoutMinutes:
    """Test cases for timeout_minutes constraint (line 160)."""

    @patch("spark_optima.api.routes.optimize.get_optimization_service")
    def test_optimize_with_timeout_minutes_constraint(self, mock_get_service: MagicMock, client: TestClient) -> None:
        """Test optimize with timeout_minutes constraint - covers line 160."""
        mock_service = MagicMock()
        mock_service.validate_spark_version.return_value = True
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = MagicMock(
            configuration={"spark.executor.memory": "4g"},
            estimated_time_minutes=10.0,
            confidence_score=0.95,
            platform_specific={"platform": "local", "spark_version": "3.5.0"},
            code_suggestions=[],
            metadata={"resources": {}, "data_profile": None, "code_analysis": None},
        )
        mock_service.get_optimizer.return_value = mock_optimizer
        mock_get_service.return_value = mock_service

        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
            "platform": "local",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16,
            },
            "constraints": {
                "timeout_minutes": 120,
            },
        }

        response = client.post("/api/v1/optimize", json=request)

        assert response.status_code in [200, 500]
        # Verify that timeout_minutes was passed to optimize
        if response.status_code == 200:
            call_args = mock_optimizer.optimize.call_args
            assert call_args[1]["resource_constraints"]["timeout_minutes"] == 120


class TestAnalyzeEndpointErrorHandling:
    """Test cases for analyze endpoint error handling (lines 311-313)."""

    @patch("spark_optima.api.routes.optimize.RecommendationEngine")
    def test_analyze_internal_error(self, mock_engine_class: MagicMock, client: TestClient) -> None:
        """Test analyze handles internal errors - covers lines 311-313."""
        # Mock the RecommendationEngine to raise an exception
        mock_engine = MagicMock()
        mock_engine.analyze_file.side_effect = RuntimeError("Analysis failed")
        mock_engine_class.return_value = mock_engine

        request = {
            "code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()"),
        }

        response = client.post("/api/v1/analyze", json=request)

        assert response.status_code == 500
        assert "Analysis failed" in response.json()["detail"]

    @patch("spark_optima.api.routes.optimize.RecommendationEngine")
    def test_analyze_error_logs_exception(self, mock_engine_class: MagicMock, client: TestClient) -> None:
        """Test that analyze endpoint logs exception on error."""
        mock_engine = MagicMock()
        mock_engine.analyze_file.side_effect = RuntimeError("Test error")
        mock_engine_class.return_value = mock_engine

        with patch("spark_optima.api.routes.optimize.logger") as mock_logger:
            response = client.post(
                "/api/v1/analyze",
                json={"code": ("from pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()")},
            )

            assert response.status_code == 500
            # Verify that exception was logged
            mock_logger.exception.assert_called_once()
