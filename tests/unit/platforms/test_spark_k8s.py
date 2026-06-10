# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Spark-on-Kubernetes platform.

This module contains tests for the self-hosted Spark-on-Kubernetes platform
including pod size presets, cluster recommendation, spark.kubernetes.*
config translation, user-supplied cost estimation, and the SparkApplication
custom resource export.
"""

from __future__ import annotations

import json

import pytest

from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)
from spark_optima.platforms.spark_k8s import SparkOnK8sPlatform


class TestSparkOnK8sPlatformInitialization:
    """Test cases for SparkOnK8sPlatform initialization."""

    def test_spark_k8s_platform_initialization(self) -> None:
        """Test Spark-on-Kubernetes platform initialization."""
        platform = SparkOnK8sPlatform()
        assert platform.name == "kubernetes"
        assert platform.display_name == "Spark on Kubernetes"

    def test_spark_k8s_platform_constraints(self) -> None:
        """Test Spark-on-Kubernetes platform constraints."""
        platform = SparkOnK8sPlatform()
        constraints = platform.constraints

        assert constraints.min_workers == 1
        assert constraints.max_workers == 10000
        assert constraints.min_memory_gb == 8  # small preset
        assert constraints.max_memory_gb == 64  # xlarge preset
        assert constraints.min_cores == 2
        assert constraints.max_cores == 16
        assert len(constraints.supported_spark_versions) > 0

    def test_spark_k8s_default_namespace(self) -> None:
        """Test that the default namespace is 'default'."""
        platform = SparkOnK8sPlatform()
        assert platform.namespace == "default"

    def test_spark_k8s_custom_namespace(self) -> None:
        """Test that a custom namespace is applied."""
        platform = SparkOnK8sPlatform(namespace="data-jobs")
        assert platform.namespace == "data-jobs"

    def test_spark_k8s_default_cost_is_zero(self) -> None:
        """Test that cost is zero by default (self-hosted)."""
        platform = SparkOnK8sPlatform()
        assert platform.cost_per_vcpu_hour == 0.0

        for worker in platform.get_worker_types():
            assert worker.cost.unit_cost_per_hour == 0.0


class TestSparkOnK8sPlatformWorkerTypes:
    """Test cases for pod size presets."""

    def test_get_worker_types(self) -> None:
        """Test getting available pod presets."""
        platform = SparkOnK8sPlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) == 4  # small, medium, large, xlarge

        names = [w.name for w in worker_types]
        assert names == ["small", "medium", "large", "xlarge"]
        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)

    def test_get_worker_type_small(self) -> None:
        """Test getting the small pod preset."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("small")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "small"
        assert worker_type.resources.cpu_cores == 2
        assert worker_type.resources.memory_gb == 8

    def test_get_worker_type_medium(self) -> None:
        """Test getting the medium pod preset."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("medium")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 4
        assert worker_type.resources.memory_gb == 16

    def test_get_worker_type_large(self) -> None:
        """Test getting the large pod preset."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("large")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 8
        assert worker_type.resources.memory_gb == 32

    def test_get_worker_type_xlarge(self) -> None:
        """Test getting the xlarge pod preset."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("xlarge")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 16
        assert worker_type.resources.memory_gb == 64

    def test_get_worker_type_case_insensitive(self) -> None:
        """Test that preset lookup is case insensitive."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("MEDIUM")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "medium"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid preset returns None."""
        platform = SparkOnK8sPlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None

    def test_worker_cost_uses_user_supplied_pricing(self) -> None:
        """Test that hourly cost is cost_per_vcpu_hour x CPUs."""
        platform = SparkOnK8sPlatform(cost_per_vcpu_hour=0.05)
        worker = platform.get_worker_type("large")

        assert worker is not None
        assert worker.cost.unit_cost_per_hour == pytest.approx(0.05 * 8)


class TestSparkOnK8sPlatformRecommendWorkerType:
    """Test cases for recommend_worker_type."""

    def test_recommend_worker_type_exact_fit(self) -> None:
        """Test that the smallest preset meeting the targets is picked."""
        platform = SparkOnK8sPlatform()
        worker = platform.recommend_worker_type(target_memory_gb=16.0, target_cores=4)

        assert worker is not None
        assert worker.name == "medium"

    def test_recommend_worker_type_small_targets(self) -> None:
        """Test that small targets map to the small preset."""
        platform = SparkOnK8sPlatform()
        worker = platform.recommend_worker_type(target_memory_gb=4.0, target_cores=1)

        assert worker is not None
        assert worker.name == "small"

    def test_recommend_worker_type_fallback_to_largest(self) -> None:
        """Test fallback to xlarge when targets exceed all presets."""
        platform = SparkOnK8sPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,  # Impossibly high
            target_cores=1000,  # Impossibly high
        )

        assert worker is not None
        assert worker.name == "xlarge"


class TestSparkOnK8sPlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 1

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific executor count."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=5,
        )

        assert config.worker_count == 5

    def test_recommend_config_includes_driver_pod(self) -> None:
        """Test that one driver pod is accounted for."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.driver_type is not None
        assert config.driver_type.name == SparkOnK8sPlatform.DEFAULT_DRIVER_PRESET
        assert config.driver_count == 1

    def test_recommend_config_platform_config(self) -> None:
        """Test that the platform config carries namespace, preset, and image."""
        platform = SparkOnK8sPlatform(namespace="data-jobs")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.platform_config["namespace"] == "data-jobs"
        assert config.platform_config["pod_preset"] == config.worker_type.name
        assert config.platform_config["num_executors"] == config.worker_count
        assert config.platform_config["container_image"] == "apache/spark:3.5.0"


class TestSparkOnK8sPlatformSparkConfigTranslation:
    """Test cases for spark.kubernetes.* config translation."""

    def _get_cluster_config(self) -> tuple[SparkOnK8sPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for translation tests."""
        platform = SparkOnK8sPlatform(namespace="data-jobs")
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Kubernetes-specific Spark config."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert isinstance(spark_config, dict)
        assert spark_config["spark.kubernetes.container.image"] == "apache/spark:3.5.0"
        assert spark_config["spark.kubernetes.namespace"] == "data-jobs"
        assert "spark.executor.instances" in spark_config
        assert spark_config["spark.sql.adaptive.enabled"] == "true"

    def test_translate_does_not_set_spark_master(self) -> None:
        """Test that spark.master is not set (depends on the cluster API server)."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert "spark.master" not in spark_config

    def test_translate_sets_pod_cpu_request_and_limit(self) -> None:
        """Test that executor pod CPU request and limit match the preset."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        cores = str(cluster_config.worker_type.resources.cpu_cores)
        assert spark_config["spark.kubernetes.executor.request.cores"] == cores
        assert spark_config["spark.kubernetes.executor.limit.cores"] == cores
        assert spark_config["spark.executor.cores"] == cores

    def test_translate_leaves_memory_overhead_headroom(self) -> None:
        """Test that executor memory leaves ~10% headroom for overhead."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        executor_memory_gb = int(spark_config["spark.executor.memory"].rstrip("g"))
        assert executor_memory_gb < cluster_config.worker_type.resources.memory_gb
        assert executor_memory_gb == int(cluster_config.worker_type.resources.memory_gb * 0.9)
        assert spark_config["spark.executor.memoryOverheadFactor"] == "0.1"

    def test_translate_enables_shuffle_tracking(self) -> None:
        """Test dynamic allocation uses shuffle tracking (required on K8s)."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["spark.dynamicAllocation.enabled"] == "true"
        assert spark_config["spark.dynamicAllocation.shuffleTracking.enabled"] == "true"

    def test_translate_does_not_enable_external_shuffle_service(self) -> None:
        """Test that the external shuffle service is never enabled (none on K8s)."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert "spark.shuffle.service.enabled" not in spark_config

    def test_translate_dynamic_allocation_bounds(self) -> None:
        """Test dynamic allocation bounds match cluster size."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["spark.dynamicAllocation.minExecutors"] == "1"
        assert spark_config["spark.dynamicAllocation.maxExecutors"] == str(
            cluster_config.worker_count,
        )


class TestSparkOnK8sPlatformCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost_zero_by_default(self) -> None:
        """Test that the default (self-hosted) estimate is zero."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert isinstance(cost, dict)
        assert cost["total_cost"] == 0.0
        assert cost["breakdown"]["cost_per_vcpu_hour"] == 0.0

    def test_estimate_cost_with_user_supplied_pricing(self) -> None:
        """Test cost estimation with a user-supplied cost_per_vcpu_hour."""
        platform = SparkOnK8sPlatform(cost_per_vcpu_hour=0.04)
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert cost["total_cost"] > 0
        driver = cluster_config.driver_type
        assert driver is not None
        total_vcpus = (
            cluster_config.worker_type.resources.cpu_cores * cluster_config.worker_count + driver.resources.cpu_cores
        )
        assert cost["breakdown"]["total_vcpus"] == total_vcpus
        assert cost["total_cost"] == pytest.approx(0.04 * total_vcpus * 2.0)

    def test_estimate_cost_zero_duration(self) -> None:
        """Test cost estimation with zero duration."""
        platform = SparkOnK8sPlatform(cost_per_vcpu_hour=0.04)
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost["total_cost"] == 0.0

    def test_estimate_cost_includes_driver(self) -> None:
        """Test that the driver pod is part of the total cost."""
        platform = SparkOnK8sPlatform(cost_per_vcpu_hour=0.04)
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=1.0)
        breakdown = cost["breakdown"]

        assert breakdown["driver_count"] == 1
        assert breakdown["driver_cost"] > 0
        assert cost["total_cost"] == pytest.approx(
            breakdown["driver_cost"] + breakdown["worker_cost"],
        )

    def test_estimate_cost_scales_with_workers(self) -> None:
        """Test that cost scales with executor count."""
        platform = SparkOnK8sPlatform(cost_per_vcpu_hour=0.04)
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


class TestSparkOnK8sPlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum executors."""
        platform = SparkOnK8sPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 10001  # Exceeds the limit of 10000

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("maximum" in error.lower() for error in errors)


class TestSparkOnK8sPlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = SparkOnK8sPlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_3x(self) -> None:
        """Test Spark 3.x support."""
        platform = SparkOnK8sPlatform()

        assert platform.is_spark_version_supported("3.0.0")
        assert platform.is_spark_version_supported("3.5.0")

    def test_is_spark_version_supported_4x(self) -> None:
        """Test Spark 4.x support."""
        platform = SparkOnK8sPlatform()

        assert platform.is_spark_version_supported("4.0.0")
        assert platform.is_spark_version_supported("4.1.0")


class TestSparkOnK8sPlatformSparkApplicationCRD:
    """Test cases for get_spark_application_crd (Spark Operator manifest)."""

    def _get_cluster_config(self) -> tuple[SparkOnK8sPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for CRD tests."""
        platform = SparkOnK8sPlatform(namespace="data-jobs")
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_get_spark_application_crd_structure(self) -> None:
        """Test SparkApplication manifest structure."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)

        assert manifest["apiVersion"] == "sparkoperator.k8s.io/v1beta2"
        assert manifest["kind"] == "SparkApplication"
        assert manifest["metadata"]["name"] == "spark-optima-app"
        assert manifest["metadata"]["namespace"] == "data-jobs"

    def test_get_spark_application_crd_spec(self) -> None:
        """Test SparkApplication spec carries type, version, and image."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)
        spec = manifest["spec"]

        assert spec["type"] == "Python"
        assert spec["mode"] == "cluster"
        assert spec["sparkVersion"] == "3.5.0"
        assert spec["image"] == "apache/spark:3.5.0"
        assert spec["mainApplicationFile"] == "local:///opt/spark/app/main.py"

    def test_get_spark_application_crd_driver_executor_blocks(self) -> None:
        """Test driver and executor blocks carry cores, memory, and instances."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)
        spec = manifest["spec"]

        driver = cluster_config.driver_type
        assert driver is not None
        assert spec["driver"]["cores"] == driver.resources.cpu_cores
        assert spec["driver"]["memory"].endswith("g")

        executor = spec["executor"]
        assert executor["cores"] == cluster_config.worker_type.resources.cpu_cores
        assert executor["instances"] == cluster_config.worker_count
        assert executor["memory"] == f"{int(cluster_config.worker_type.resources.memory_gb * 0.9)}g"

    def test_get_spark_application_crd_spark_conf(self) -> None:
        """Test sparkConf carries the optimized config as strings."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)
        spark_conf = manifest["spec"]["sparkConf"]

        assert spark_conf["spark.dynamicAllocation.shuffleTracking.enabled"] == "true"
        assert spark_conf["spark.sql.adaptive.enabled"] == "true"
        assert all(isinstance(v, str) for v in spark_conf.values())

    def test_get_spark_application_crd_excludes_spec_managed_keys(self) -> None:
        """Test that keys derived from the CR spec are not duplicated in sparkConf."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)
        spark_conf = manifest["spec"]["sparkConf"]

        assert "spark.kubernetes.container.image" not in spark_conf
        assert "spark.kubernetes.namespace" not in spark_conf
        assert "spark.executor.instances" not in spark_conf
        assert "spark.executor.memory" not in spark_conf
        assert "spark.driver.memory" not in spark_conf

    def test_get_spark_application_crd_custom_args(self) -> None:
        """Test passing a custom app name, entrypoint, and Spark config."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(
            cluster_config,
            app_name="nightly-etl",
            main_application_file="local:///opt/spark/app/etl.py",
            spark_config={"spark.sql.shuffle.partitions": 400, "not.a.spark.key": "x"},
        )

        assert manifest["metadata"]["name"] == "nightly-etl"
        assert manifest["spec"]["mainApplicationFile"] == "local:///opt/spark/app/etl.py"
        assert manifest["spec"]["sparkConf"] == {"spark.sql.shuffle.partitions": "400"}

    def test_get_spark_application_crd_json_serializable(self) -> None:
        """Test that the manifest is JSON-serializable (kubectl/YAML compatible)."""
        platform, cluster_config = self._get_cluster_config()

        manifest = platform.get_spark_application_crd(cluster_config)

        serialized = json.dumps(manifest)
        assert json.loads(serialized) == manifest
