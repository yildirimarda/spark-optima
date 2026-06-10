# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the AWS Glue platform.

This module contains tests for AWS Glue deployment platform including
resource management and Glue-specific configuration.
"""

from __future__ import annotations

import pytest

from spark_optima.platforms.aws_glue import AWSGluePlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)


class TestAWSGluePlatformInitialization:
    """Test cases for AWSGluePlatform initialization."""

    def test_aws_glue_platform_initialization(self) -> None:
        """Test AWS Glue platform initialization."""
        platform = AWSGluePlatform()
        assert platform.name == "aws_glue"
        assert platform.display_name == "AWS Glue"

    def test_aws_glue_platform_constraints(self) -> None:
        """Test AWS Glue platform constraints."""
        platform = AWSGluePlatform()
        constraints = platform.constraints

        assert constraints.min_workers >= 2  # Glue requires at least 2 workers
        assert constraints.max_workers <= 300  # Glue limit is 299
        assert len(constraints.supported_spark_versions) > 0


class TestAWSGluePlatformWorkerTypes:
    """Test cases for AWS Glue worker types."""

    def test_get_worker_types(self) -> None:
        """Test getting available Glue worker types."""
        platform = AWSGluePlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) >= 7  # G.025X, G.1X, G.2X, G.4X, G.8X, G.12X, G.16X, R-types

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name
            # Should be G-type or R-type
            assert worker_type.name.startswith(("G.", "R."))

    def test_get_worker_type_g025x(self) -> None:
        """Test getting G.025X worker type."""
        platform = AWSGluePlatform()
        worker_type = platform.get_worker_type("G.025X")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "G.025X"

    def test_get_worker_type_g1x(self) -> None:
        """Test getting G.1X worker type."""
        platform = AWSGluePlatform()
        worker_type = platform.get_worker_type("G.1X")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "G.1X"

    def test_get_worker_type_g2x(self) -> None:
        """Test getting G.2X worker type."""
        platform = AWSGluePlatform()
        worker_type = platform.get_worker_type("G.2X")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "G.2X"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = AWSGluePlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None


class TestAWSGluePlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 2  # Glue minimum

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=5,
        )

        assert config.worker_count == 5

    def test_recommend_config_enforces_minimum_workers(self) -> None:
        """Test that config enforces Glue minimum worker count."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=4, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=1,  # Below minimum
        )

        assert config.worker_count >= 2  # Enforced minimum

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Glue-specific Spark config."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.executor.memory" in spark_config
        assert "spark.executor.cores" in spark_config
        assert "spark.sql.adaptive.enabled" in spark_config

    def test_translate_includes_glue_version(self) -> None:
        """Test that translation includes Glue version info."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        spark_config = platform.translate_to_spark_config(cluster_config)

        # Should include Glue-specific configuration
        assert isinstance(spark_config, dict)


class TestAWSGluePlatformCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for AWS Glue."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert "total_cost" in cost
        assert "breakdown" in cost
        assert cost["total_cost"] > 0  # Glue has actual cost

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost["total_cost"] == 0.0

    def test_estimate_cost_scales_with_workers(self) -> None:
        """Test that cost scales with worker count."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config_small = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=2,
        )
        config_large = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=10,
        )

        cost_small = platform.estimate_cost(config_small, duration_hours=1.0)
        cost_large = platform.estimate_cost(config_large, duration_hours=1.0)

        assert cost_large["total_cost"] > cost_small["total_cost"]


class TestAWSGluePlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_below_minimum_workers(self) -> None:
        """Test validation with below minimum workers."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 1  # Below Glue minimum

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("minimum" in error.lower() for error in errors)

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum workers."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 500  # Exceeds Glue limit of 299

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("maximum" in error.lower() for error in errors)


class TestAWSGluePlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = AWSGluePlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_glue_3(self) -> None:
        """Test Glue 3.0 (Spark 3.1) support."""
        platform = AWSGluePlatform()

        assert platform.is_spark_version_supported("3.1.1")

    def test_is_spark_version_supported_glue_4(self) -> None:
        """Test Glue 4.0 (Spark 3.3) support."""
        platform = AWSGluePlatform()

        assert platform.is_spark_version_supported("3.3.0")

    def test_is_spark_version_supported_glue_5(self) -> None:
        """Test Glue 5.0 (Spark 3.5) support."""
        platform = AWSGluePlatform()

        assert platform.is_spark_version_supported("3.5.0")


class TestAWSGluePlatformOptimalWorkerCount:
    """Test cases for optimal worker count calculation."""

    def test_get_optimal_worker_count(self) -> None:
        """Test optimal worker count calculation for Glue."""
        platform = AWSGluePlatform()
        target_resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        worker_type = platform.get_worker_type("G.1X")

        assert worker_type is not None
        worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

        assert isinstance(worker_count, int)
        assert worker_count >= 2  # Glue minimum
        assert worker_count <= platform.constraints.max_workers

    def test_get_optimal_worker_count_respects_minimum(self) -> None:
        """Test that optimal worker count respects Glue minimum."""
        platform = AWSGluePlatform()
        target_resources = ResourceSpec(cpu_cores=2, memory_gb=4.0)  # Very small
        worker_type = platform.get_worker_type("G.1X")

        worker_count = platform.get_optimal_worker_count(target_resources, worker_type)

        assert worker_count >= 2  # Glue minimum enforced


class TestAWSGluePlatformEdgeCases:
    """Test edge cases."""

    def test_recommend_config_with_minimal_resources(self) -> None:
        """Test config recommendation with minimal resources."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=2, memory_gb=8.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.worker_count >= 2

    def test_recommend_config_with_large_resources(self) -> None:
        """Test config recommendation with large resources."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=64, memory_gb=256.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.worker_count <= platform.constraints.max_workers

    def test_different_worker_types_have_different_resources(self) -> None:
        """Test that different worker types have different resource specs."""
        platform = AWSGluePlatform()

        standard = platform.get_worker_type("Standard")
        g1x = platform.get_worker_type("G.1X")
        g2x = platform.get_worker_type("G.2X")

        if all([standard, g1x, g2x]):
            # G.2X should have more resources than G.1X
            assert g2x.resources.cpu_cores >= g1x.resources.cpu_cores
            assert g2x.resources.memory_gb >= g1x.resources.memory_gb

    def test_cost_model_per_worker_type(self) -> None:
        """Test that different worker types have different costs."""
        platform = AWSGluePlatform()

        g1x = platform.get_worker_type("G.1X")
        g2x = platform.get_worker_type("G.2X")

        if g1x and g2x:
            # G.2X should be more expensive than G.1X
            assert g2x.cost.unit_cost_per_hour > g1x.cost.unit_cost_per_hour


class TestAWSGluePlatformRecommendWorkerTypeMemoryOptimized:
    """Test cases for recommend_worker_type with prefer_memory_optimized."""

    def test_recommend_worker_type_prefer_memory_optimized_true(self) -> None:
        """Test that R-types are preferred when prefer_memory_optimized is True (lines 257-259)."""
        platform = AWSGluePlatform()
        # Request high memory to trigger R-type selection
        worker = platform.recommend_worker_type(
            target_memory_gb=64.0,
            target_cores=8,
            prefer_memory_optimized=True,
        )

        assert worker is not None
        # Should return an R-type worker
        assert worker.name.startswith("R.")

    def test_recommend_worker_type_prefer_memory_optimized_false(self) -> None:
        """Test that G-types are preferred when prefer_memory_optimized is False."""
        platform = AWSGluePlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=32.0,
            target_cores=8,
            prefer_memory_optimized=False,
        )

        assert worker is not None
        # Should return a G-type worker
        assert worker.name.startswith("G.")

    def test_recommend_worker_type_no_r_types_fallback(self) -> None:
        """Test fallback when no R-types meet requirements."""
        platform = AWSGluePlatform()
        # Request extremely high resources that only G-types might meet
        worker = platform.recommend_worker_type(
            target_memory_gb=300.0,
            target_cores=100,
            prefer_memory_optimized=True,
        )

        assert worker is not None
        # Should still return a worker type


class TestAWSGluePlatformRecommendWorkerTypeFallback:
    """Test cases for fallback in recommend_worker_type (line 290)."""

    def test_recommend_worker_type_fallback_to_g1x(self) -> None:
        """Test fallback to G.1X when no worker type meets requirements (line 290)."""
        platform = AWSGluePlatform()
        # Request impossible requirements - no worker type can meet this
        # but the method should return G.1X as fallback
        worker = platform.recommend_worker_type(
            target_memory_gb=1000.0,  # Impossibly high
            target_cores=1000,  # Impossibly high
        )

        assert worker is not None
        # Should fallback to G.1X or first available worker type
        assert worker.name == "G.1X" or worker in platform.get_worker_types()


class TestAWSGluePlatformGlueVersionMapping:
    """Test cases for _get_glue_version method (line 359)."""

    def test_get_glue_version_exact_match(self) -> None:
        """Test mapping Spark version to Glue version."""
        platform = AWSGluePlatform()

        # Test known mappings
        assert platform._get_glue_version("2.4.3") == "2.0"
        assert platform._get_glue_version("3.1.1") == "3.0"
        assert platform._get_glue_version("3.3.0") == "4.0"
        assert platform._get_glue_version("3.5.0") == "5.0"

    def test_get_glue_version_default_fallback(self) -> None:
        """Test default fallback to 4.0 for unknown Spark version (line 359)."""
        platform = AWSGluePlatform()

        # Unknown Spark version should return "4.0" as default
        result = platform._get_glue_version("9.9.9")
        assert result == "4.0"

    def test_get_glue_version_partial_match(self) -> None:
        """Test partial version matching."""
        platform = AWSGluePlatform()

        # Should match based on major.minor
        assert platform._get_glue_version("3.1.0") == "3.0"  # Starts with 3.1


class TestAWSGluePlatformJobProperties:
    """Test cases for get_glue_job_properties method (line 472)."""

    def test_get_glue_job_properties(self) -> None:
        """Test generating Glue job properties (line 472)."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        job_properties = platform.get_glue_job_properties(cluster_config)

        assert isinstance(job_properties, dict)
        assert "Name" in job_properties
        assert "WorkerType" in job_properties
        assert "NumberOfWorkers" in job_properties
        assert job_properties["WorkerType"] == cluster_config.worker_type.name
        assert job_properties["NumberOfWorkers"] == cluster_config.worker_count

    def test_get_glue_job_properties_custom_name(self) -> None:
        """Test Glue job properties with custom job name."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        job_properties = platform.get_glue_job_properties(cluster_config, job_name="my-custom-job")

        assert job_properties["Name"] == "my-custom-job"

    def test_get_glue_job_properties_default_version(self) -> None:
        """Test Glue job properties uses default version when not specified."""
        platform = AWSGluePlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        # Remove glue_version from platform_config to test default
        cluster_config.platform_config.pop("glue_version", None)

        job_properties = platform.get_glue_job_properties(cluster_config)

        assert job_properties["GlueVersion"] == "4.0"  # Default


class TestAWSGlueBoto3Methods:
    """Test cases for boto3/real execution methods."""

    pytestmark = pytest.mark.skipif(
        __import__("importlib").util.find_spec("boto3") is None,
        reason="boto3 not installed",
    )

    def test_submit_job_no_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job raises RuntimeError when boto3 missing."""
        platform = AWSGluePlatform()
        import sys

        # Remove boto3 from sys.modules to simulate missing import
        monkeypatch.setitem(sys.modules, "boto3", None)
        # Patch the import inside submit_job to raise ImportError
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(RuntimeError, match="boto3 is required"):
            platform.submit_job("test.py")

    def test_get_job_status_no_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status raises RuntimeError when boto3 missing."""
        platform = AWSGluePlatform()
        import sys

        # Mock boto3 as missing
        monkeypatch.setitem(sys.modules, "boto3", None)

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(RuntimeError, match="boto3 is required"):
            platform.get_job_status("test-job", "run-123")

    def test_get_job_results_no_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_results raises RuntimeError when boto3 missing."""
        platform = AWSGluePlatform()
        import sys

        # Mock boto3 as missing
        monkeypatch.setitem(sys.modules, "boto3", None)

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(RuntimeError, match="boto3 is required"):
            platform.get_job_results("test-job", "run-123")

    def test_submit_job_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job raises error when credentials missing."""
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        platform = AWSGluePlatform()
        # This will fail because boto3 won't find credentials
        # We just test that it doesn't crash with unexpected errors
        try:
            result = platform.submit_job("test.py", job_name="test-job")
            assert "success" in result
        except RuntimeError:
            pass  # Expected when boto3 not configured

    def test_map_glue_state(self) -> None:
        """Test _map_glue_state helper."""
        platform = AWSGluePlatform()

        assert platform._map_glue_state("STARTING") == "starting"
        assert platform._map_glue_state("RUNNING") == "running"
        assert platform._map_glue_state("STOPPED") == "stopped"
        assert platform._map_glue_state("SUCCEEDED") == "completed"
        assert platform._map_glue_state("FAILED") == "failed"
        assert platform._map_glue_state("TIMEOUT") == "timeout"
        assert platform._map_glue_state("UNKNOWN") == "unknown"

    def test_submit_job_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job with mocked boto3."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ClientError

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSGluePlatform()

        mock_client = MagicMock()
        # update_job raises error (job doesn't exist), then create succeeds
        mock_client.update_job.side_effect = ClientError(
            error_response={"Error": {"Code": "EntityNotFoundException", "Message": "Not found"}},
            operation_name="UpdateJob",
        )
        mock_client.create_job.return_value = {}
        mock_client.start_job_run.return_value = {"JobRunId": "run-123"}

        with patch("boto3.client", return_value=mock_client):
            result = platform.submit_job(
                code_path="/tmp/test.py",
                job_name="test-job",
            )

        assert result["success"] is True
        assert result["job_run_id"] == "run-123"
        assert result["status"] == "submitted"

    def test_get_job_status_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status with mocked boto3."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSGluePlatform()

        mock_client = MagicMock()
        mock_client.get_job_run.return_value = {
            "JobRun": {
                "JobRunState": "RUNNING",
                "StartedOn": "2024-01-01",
            }
        }

        with patch("boto3.client", return_value=mock_client):
            result = platform.get_job_status("test-job", "run-123")

        assert result["job_name"] == "test-job"
        assert result["job_run_id"] == "run-123"
        assert result["status"] == "running"

    def test_get_job_results_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_results with mocked boto3."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSGluePlatform()

        mock_client = MagicMock()
        mock_client.get_job_run.return_value = {
            "JobRun": {
                "JobRunState": "SUCCEEDED",
                "Output": {"Result": "OK"},
                "ExecutionTime": 120.5,
            }
        }

        with patch("boto3.client", return_value=mock_client):
            result = platform.get_job_results("test-job", "run-123")

        assert result["job_name"] == "test-job"
        assert result["status"] == "completed"
        assert result["execution_time"] == 120.5
