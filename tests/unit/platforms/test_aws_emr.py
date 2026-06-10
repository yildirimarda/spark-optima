# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the AWS EMR platform.

This module contains tests for the Amazon EMR deployment platform including
worker types, cluster recommendation, YARN config translation, cost
estimation, and the run_job_flow cluster definition.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from spark_optima.platforms.aws_emr import AWSEMRPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.pricing import REGION_MULTIPLIERS


class TestAWSEMRPlatformInitialization:
    """Test cases for AWSEMRPlatform initialization."""

    def test_aws_emr_platform_initialization(self) -> None:
        """Test AWS EMR platform initialization."""
        platform = AWSEMRPlatform()
        assert platform.name == "aws_emr"
        assert platform.display_name == "AWS EMR"

    def test_aws_emr_platform_constraints(self) -> None:
        """Test AWS EMR platform constraints."""
        platform = AWSEMRPlatform()
        constraints = platform.constraints

        assert constraints.min_workers == 1
        assert constraints.max_workers == 500
        assert constraints.min_memory_gb == 8  # c5.xlarge
        assert constraints.max_memory_gb == 256  # r5.8xlarge
        assert constraints.min_cores == 4
        assert constraints.max_cores == 36  # c5.9xlarge
        assert len(constraints.supported_spark_versions) > 0

    def test_aws_emr_custom_surcharge_rate(self) -> None:
        """Test that a custom EMR surcharge rate is applied."""
        platform = AWSEMRPlatform(emr_surcharge_rate=0.5)
        worker = platform.get_worker_type("m5.xlarge")

        assert worker is not None
        ec2_price = AWSEMRPlatform.WORKER_SPECS["m5.xlarge"]["ec2_price"]
        assert worker.cost.unit_cost_per_hour == pytest.approx(ec2_price * 1.5)


class TestAWSEMRPlatformWorkerTypes:
    """Test cases for AWS EMR worker types."""

    def test_get_worker_types(self) -> None:
        """Test getting available EMR instance types."""
        platform = AWSEMRPlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) == 12  # 4 x m5, 4 x r5, 4 x c5

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name
            # Should be one of the supported EC2 families
            assert worker_type.name.startswith(("m5.", "r5.", "c5."))

    def test_get_worker_type_m5_xlarge(self) -> None:
        """Test getting m5.xlarge worker type."""
        platform = AWSEMRPlatform()
        worker_type = platform.get_worker_type("m5.xlarge")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "m5.xlarge"
        assert worker_type.resources.cpu_cores == 4
        assert worker_type.resources.memory_gb == 16

    def test_get_worker_type_r5_4xlarge(self) -> None:
        """Test getting r5.4xlarge worker type (memory-optimized)."""
        platform = AWSEMRPlatform()
        worker_type = platform.get_worker_type("r5.4xlarge")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 16
        assert worker_type.resources.memory_gb == 128

    def test_get_worker_type_c5_2xlarge(self) -> None:
        """Test getting c5.2xlarge worker type (compute-optimized)."""
        platform = AWSEMRPlatform()
        worker_type = platform.get_worker_type("c5.2xlarge")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 8
        assert worker_type.resources.memory_gb == 16

    def test_get_worker_type_case_insensitive(self) -> None:
        """Test that instance type lookup is case insensitive."""
        platform = AWSEMRPlatform()
        worker_type = platform.get_worker_type("M5.XLARGE")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "m5.xlarge"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = AWSEMRPlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None

    def test_worker_cost_includes_emr_surcharge(self) -> None:
        """Test that hourly cost is EC2 price plus the EMR surcharge."""
        platform = AWSEMRPlatform()
        worker = platform.get_worker_type("m5.xlarge")

        assert worker is not None
        ec2_price = AWSEMRPlatform.WORKER_SPECS["m5.xlarge"]["ec2_price"]
        expected = ec2_price * (1 + AWSEMRPlatform.EMR_SURCHARGE_RATE)
        assert worker.cost.unit_cost_per_hour == pytest.approx(expected)


class TestAWSEMRPlatformRecommendWorkerType:
    """Test cases for recommend_worker_type family preferences."""

    def test_recommend_worker_type_default_general_purpose(self) -> None:
        """Test that m5 instances are preferred by default."""
        platform = AWSEMRPlatform()
        worker = platform.recommend_worker_type(target_memory_gb=32.0, target_cores=8)

        assert worker is not None
        assert worker.name.startswith("m5.")

    def test_recommend_worker_type_prefer_memory_optimized(self) -> None:
        """Test that r5 instances are preferred for memory-intensive workloads."""
        platform = AWSEMRPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=64.0,
            target_cores=8,
            prefer_memory_optimized=True,
        )

        assert worker is not None
        assert worker.name.startswith("r5.")

    def test_recommend_worker_type_prefer_compute_optimized(self) -> None:
        """Test that c5 instances are preferred for CPU-bound workloads."""
        platform = AWSEMRPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=8.0,
            target_cores=4,
            prefer_compute_optimized=True,
        )

        assert worker is not None
        assert worker.name.startswith("c5.")

    def test_recommend_worker_type_fallback(self) -> None:
        """Test fallback to m5.xlarge when no instance type meets requirements."""
        platform = AWSEMRPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,  # Impossibly high
            target_cores=1000,  # Impossibly high
        )

        assert worker is not None
        assert worker.name == "m5.xlarge"


class TestAWSEMRPlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 1

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=5,
        )

        assert config.worker_count == 5

    def test_recommend_config_enforces_minimum_workers(self) -> None:
        """Test that config enforces the EMR minimum worker count."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=4, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=0,  # Below minimum
        )

        assert config.worker_count >= 1

    def test_recommend_config_includes_master_node(self) -> None:
        """Test that one master node is accounted for."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.driver_type is not None
        assert config.driver_count == 1
        assert config.platform_config["master_instance_type"] == config.driver_type.name

    def test_recommend_config_sets_release_label(self) -> None:
        """Test that the platform config carries the EMR release label."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.platform_config["release_label"] == "emr-7.0.0"
        assert config.platform_config["instance_type"] == config.worker_type.name
        assert config.platform_config["instance_count"] == config.worker_count


class TestAWSEMRPlatformSparkConfigTranslation:
    """Test cases for YARN-style Spark config translation."""

    def _get_cluster_config(self) -> tuple[AWSEMRPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for translation tests."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to EMR-specific Spark config."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert "spark.executor.instances" in spark_config
        assert "spark.executor.memory" in spark_config
        assert "spark.executor.cores" in spark_config
        assert spark_config["spark.sql.adaptive.enabled"] == "true"
        assert spark_config["spark.dynamicAllocation.enabled"] == "true"
        assert spark_config["spark.shuffle.service.enabled"] == "true"

    def test_translate_does_not_set_spark_master(self) -> None:
        """Test that spark.master is not set (EMR configures YARN itself)."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert "spark.master" not in spark_config

    def test_translate_leaves_memory_headroom(self) -> None:
        """Test that executor memory leaves ~10% headroom."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        executor_memory_gb = int(spark_config["spark.executor.memory"].rstrip("g"))
        assert executor_memory_gb < cluster_config.worker_type.resources.memory_gb
        assert executor_memory_gb == int(cluster_config.worker_type.resources.memory_gb * 0.9)

    def test_translate_reserves_core_for_yarn(self) -> None:
        """Test that one core per node is reserved for YARN/OS daemons."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        executor_cores = int(spark_config["spark.executor.cores"])
        assert executor_cores == cluster_config.worker_type.resources.cpu_cores - 1

    def test_translate_dynamic_allocation_bounds(self) -> None:
        """Test dynamic allocation bounds match cluster size."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["spark.dynamicAllocation.minExecutors"] == "1"
        assert spark_config["spark.dynamicAllocation.maxExecutors"] == str(
            cluster_config.worker_count,
        )

    def test_translate_includes_release_label(self) -> None:
        """Test that the EMR release label is carried through."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["emr.releaseLabel"] == "emr-7.0.0"


class TestAWSEMRPlatformCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for AWS EMR."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert "total_cost" in cost
        assert "breakdown" in cost
        assert cost["total_cost"] > 0

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost["total_cost"] == 0.0

    def test_estimate_cost_includes_master(self) -> None:
        """Test that the master node is part of the total cost."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)
        breakdown = cost["breakdown"]

        assert breakdown["master_count"] == 1
        assert breakdown["master_cost"] > 0
        assert cost["total_cost"] == pytest.approx(
            breakdown["master_cost"] + breakdown["worker_cost"],
        )

    def test_estimate_cost_breakdown_surcharge(self) -> None:
        """Test that EC2 and EMR surcharge portions sum to the total."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=3.0)
        breakdown = cost["breakdown"]

        assert breakdown["emr_surcharge_rate"] == AWSEMRPlatform.EMR_SURCHARGE_RATE
        assert cost["total_cost"] == pytest.approx(
            breakdown["ec2_cost"] + breakdown["emr_surcharge"],
        )
        assert breakdown["emr_surcharge"] == pytest.approx(
            breakdown["ec2_cost"] * AWSEMRPlatform.EMR_SURCHARGE_RATE,
        )

    def test_estimate_cost_scales_with_workers(self) -> None:
        """Test that cost scales with worker count."""
        platform = AWSEMRPlatform()
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


class TestAWSEMRPlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum workers."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 501  # Exceeds EMR limit of 500

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("maximum" in error.lower() for error in errors)


class TestAWSEMRPlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = AWSEMRPlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_emr_6_9(self) -> None:
        """Test emr-6.9 (Spark 3.3.0) support."""
        platform = AWSEMRPlatform()

        assert platform.is_spark_version_supported("3.3.0")

    def test_is_spark_version_supported_emr_6_15(self) -> None:
        """Test emr-6.15 (Spark 3.4.1) support."""
        platform = AWSEMRPlatform()

        assert platform.is_spark_version_supported("3.4.1")

    def test_is_spark_version_supported_emr_7(self) -> None:
        """Test emr-7.x (Spark 3.5.x) support."""
        platform = AWSEMRPlatform()

        assert platform.is_spark_version_supported("3.5.0")


class TestAWSEMRPlatformReleaseLabelMapping:
    """Test cases for _get_release_label."""

    def test_get_release_label_exact_match(self) -> None:
        """Test mapping Spark version to EMR release label."""
        platform = AWSEMRPlatform()

        assert platform._get_release_label("3.3.0") == "emr-6.9.0"
        assert platform._get_release_label("3.4.1") == "emr-6.15.0"
        assert platform._get_release_label("3.5.0") == "emr-7.0.0"
        assert platform._get_release_label("3.5.2") == "emr-7.5.0"

    def test_get_release_label_partial_match(self) -> None:
        """Test major.minor fallback matching."""
        platform = AWSEMRPlatform()

        # 3.4.0 has no exact mapping but matches the 3.4 line (emr-6.15.0)
        assert platform._get_release_label("3.4.0") == "emr-6.15.0"

    def test_get_release_label_default_fallback(self) -> None:
        """Test default fallback for unknown Spark version."""
        platform = AWSEMRPlatform()

        assert platform._get_release_label("9.9.9") == AWSEMRPlatform.DEFAULT_RELEASE_LABEL


class TestAWSEMRPlatformClusterDefinition:
    """Test cases for get_emr_cluster_config (run_job_flow payload)."""

    def _get_cluster_config(self) -> tuple[AWSEMRPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for cluster definition tests."""
        platform = AWSEMRPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_get_emr_cluster_config_structure(self) -> None:
        """Test run_job_flow payload structure."""
        platform, cluster_config = self._get_cluster_config()

        emr_config = platform.get_emr_cluster_config(cluster_config)

        assert emr_config["Name"] == "spark-optima-cluster"
        assert emr_config["ReleaseLabel"] == "emr-7.0.0"
        assert emr_config["Applications"] == [{"Name": "Spark"}]
        assert "Instances" in emr_config
        assert "Configurations" in emr_config

    def test_get_emr_cluster_config_instance_groups(self) -> None:
        """Test master and core instance groups."""
        platform, cluster_config = self._get_cluster_config()

        emr_config = platform.get_emr_cluster_config(cluster_config)
        groups = emr_config["Instances"]["InstanceGroups"]

        assert len(groups) == 2
        master = next(g for g in groups if g["InstanceRole"] == "MASTER")
        core = next(g for g in groups if g["InstanceRole"] == "CORE")

        assert master["InstanceCount"] == 1
        assert core["InstanceCount"] == cluster_config.worker_count
        assert core["InstanceType"] == cluster_config.worker_type.name

    def test_get_emr_cluster_config_spark_defaults(self) -> None:
        """Test that the spark-defaults classification carries the optimized config."""
        platform, cluster_config = self._get_cluster_config()

        emr_config = platform.get_emr_cluster_config(cluster_config)
        configurations = emr_config["Configurations"]

        spark_defaults = next(c for c in configurations if c["Classification"] == "spark-defaults")
        properties = spark_defaults["Properties"]

        assert "spark.executor.memory" in properties
        assert "spark.executor.cores" in properties
        # Non spark.* keys must not leak into spark-defaults
        assert all(key.startswith("spark.") for key in properties)

    def test_get_emr_cluster_config_custom_spark_config(self) -> None:
        """Test passing an explicit optimized Spark config."""
        platform, cluster_config = self._get_cluster_config()

        emr_config = platform.get_emr_cluster_config(
            cluster_config,
            cluster_name="my-cluster",
            spark_config={"spark.executor.memory": "8g", "emr.releaseLabel": "emr-7.0.0"},
        )

        assert emr_config["Name"] == "my-cluster"
        properties = emr_config["Configurations"][0]["Properties"]
        assert properties == {"spark.executor.memory": "8g"}

    def test_get_emr_cluster_config_json_serializable(self) -> None:
        """Test that the payload is JSON-serializable (boto3 compatible)."""
        platform, cluster_config = self._get_cluster_config()

        emr_config = platform.get_emr_cluster_config(cluster_config)

        serialized = json.dumps(emr_config)
        assert json.loads(serialized) == emr_config


class TestAWSEMRPlatformStateMapping:
    """Test cases for _map_emr_state helper."""

    def test_map_emr_state(self) -> None:
        """Test EMR step state mapping."""
        platform = AWSEMRPlatform()

        assert platform._map_emr_state("PENDING") == "starting"
        assert platform._map_emr_state("CANCEL_PENDING") == "stopping"
        assert platform._map_emr_state("RUNNING") == "running"
        assert platform._map_emr_state("COMPLETED") == "completed"
        assert platform._map_emr_state("CANCELLED") == "stopped"
        assert platform._map_emr_state("FAILED") == "failed"
        assert platform._map_emr_state("INTERRUPTED") == "failed"
        assert platform._map_emr_state("SOMETHING_ELSE") == "unknown"


class TestAWSEMRBoto3MissingDependency:
    """Test cases for graceful failure when boto3 is missing."""

    def _block_boto3_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force `import boto3` to raise ImportError."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", mock_import)

    def test_submit_job_no_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job raises RuntimeError when boto3 missing."""
        platform = AWSEMRPlatform()
        self._block_boto3_import(monkeypatch)

        with pytest.raises(RuntimeError, match="boto3 is required"):
            platform.submit_job("test.py")

    def test_get_job_status_no_boto3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status raises RuntimeError when boto3 missing."""
        platform = AWSEMRPlatform()
        self._block_boto3_import(monkeypatch)

        with pytest.raises(RuntimeError, match="boto3 is required"):
            platform.get_job_status("j-123", "s-456")


class TestAWSEMRBoto3Methods:
    """Test cases for boto3/real execution methods (require boto3)."""

    pytestmark = pytest.mark.skipif(
        importlib.util.find_spec("boto3") is None,
        reason="boto3 not installed",
    )

    def test_submit_job_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job with mocked boto3."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSEMRPlatform()

        mock_client = MagicMock()
        mock_client.run_job_flow.return_value = {"JobFlowId": "j-123"}
        mock_client.list_steps.return_value = {"Steps": [{"Id": "s-456"}]}

        with patch("boto3.client", return_value=mock_client):
            result = platform.submit_job(
                code_path="s3://bucket/test.py",
                cluster_name="test-cluster",
            )

        assert result["success"] is True
        assert result["cluster_id"] == "j-123"
        assert result["step_id"] == "s-456"
        assert result["status"] == "submitted"

        # The submitted job flow must carry a spark-submit step
        job_flow = mock_client.run_job_flow.call_args.kwargs
        assert job_flow["Name"] == "test-cluster"
        step_args = job_flow["Steps"][0]["HadoopJarStep"]["Args"]
        assert "spark-submit" in step_args
        assert "s3://bucket/test.py" in step_args

    def test_submit_job_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test submit_job returns error dict on ClientError."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ClientError

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSEMRPlatform()

        mock_client = MagicMock()
        mock_client.run_job_flow.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "Bad request"}},
            operation_name="RunJobFlow",
        )

        with patch("boto3.client", return_value=mock_client):
            result = platform.submit_job("s3://bucket/test.py")

        assert result["success"] is False
        assert "ValidationException" in result["error"]

    def test_get_job_status_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status with mocked boto3."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSEMRPlatform()

        mock_client = MagicMock()
        mock_client.describe_step.return_value = {
            "Step": {
                "Status": {
                    "State": "RUNNING",
                    "Timeline": {"StartDateTime": "2024-01-01"},
                },
            },
        }

        with patch("boto3.client", return_value=mock_client):
            result = platform.get_job_status("j-123", "s-456")

        assert result["cluster_id"] == "j-123"
        assert result["step_id"] == "s-456"
        assert result["status"] == "running"

    def test_get_job_status_failed_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_job_status surfaces failure details."""
        from unittest.mock import MagicMock, patch

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

        platform = AWSEMRPlatform()

        mock_client = MagicMock()
        mock_client.describe_step.return_value = {
            "Step": {
                "Status": {
                    "State": "FAILED",
                    "FailureDetails": {"Message": "Out of memory"},
                    "Timeline": {},
                },
            },
        }

        with patch("boto3.client", return_value=mock_client):
            result = platform.get_job_status("j-123", "s-456")

        assert result["status"] == "failed"
        assert result["error_message"] == "Out of memory"


class TestAWSEMRPlatformRegionalPricing:
    """Test cases for regional pricing multipliers in cost estimation."""

    def _get_cluster_config(self) -> ClusterConfig:
        """Build a cluster config for regional pricing tests."""
        platform = AWSEMRPlatform()
        return platform.recommend_config(
            resources=ResourceSpec(cpu_cores=16, memory_gb=64.0),
            spark_version="3.5.0",
        )

    def test_estimate_cost_non_baseline_region_scales_by_multiplier(self) -> None:
        """Test that a non-baseline region scales cost by the table multiplier."""
        cluster_config = self._get_cluster_config()
        multiplier = REGION_MULTIPLIERS["aws_emr"]["sa-east-1"]
        assert multiplier != 1.0  # Sanity: must be a non-baseline region

        baseline_cost = AWSEMRPlatform(region="us-east-1").estimate_cost(cluster_config, duration_hours=2.0)
        regional_cost = AWSEMRPlatform(region="sa-east-1").estimate_cost(cluster_config, duration_hours=2.0)

        assert regional_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"] * multiplier)

    def test_estimate_cost_regional_breakdown_stays_consistent(self) -> None:
        """Test that the master/worker and EC2/surcharge splits still sum to the total."""
        cluster_config = self._get_cluster_config()
        platform = AWSEMRPlatform(region="eu-central-1")

        cost = platform.estimate_cost(cluster_config, duration_hours=3.0)
        breakdown = cost["breakdown"]

        assert cost["total_cost"] == pytest.approx(breakdown["master_cost"] + breakdown["worker_cost"])
        assert cost["total_cost"] == pytest.approx(breakdown["ec2_cost"] + breakdown["emr_surcharge"])

    def test_estimate_cost_breakdown_contains_region_keys(self) -> None:
        """Test that the breakdown includes region and region_multiplier."""
        cluster_config = self._get_cluster_config()
        platform = AWSEMRPlatform(region="eu-central-1")

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "eu-central-1"
        assert cost["breakdown"]["region_multiplier"] == REGION_MULTIPLIERS["aws_emr"]["eu-central-1"]

    def test_estimate_cost_default_region_is_baseline(self) -> None:
        """Test that the default region uses the baseline multiplier 1.0."""
        cluster_config = self._get_cluster_config()
        platform = AWSEMRPlatform()

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)

        assert cost["breakdown"]["region"] == "us-east-1"
        assert cost["breakdown"]["region_multiplier"] == 1.0

    def test_estimate_cost_unknown_region_falls_back_to_baseline(self) -> None:
        """Test that an unknown region falls back to multiplier 1.0 without raising."""
        cluster_config = self._get_cluster_config()

        baseline_cost = AWSEMRPlatform().estimate_cost(cluster_config, duration_hours=1.0)
        unknown_cost = AWSEMRPlatform(region="mars-north-1").estimate_cost(cluster_config, duration_hours=1.0)

        assert unknown_cost["breakdown"]["region_multiplier"] == 1.0
        assert unknown_cost["total_cost"] == pytest.approx(baseline_cost["total_cost"])
