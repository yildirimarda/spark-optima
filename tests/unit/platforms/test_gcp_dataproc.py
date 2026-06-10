# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the GCP Dataproc platform.

This module contains tests for the Google Cloud Dataproc deployment platform
including worker machine types, cluster recommendation, YARN config
translation, Compute Engine + Dataproc fee cost estimation, preemptible
worker discounting, and the clusters.create request body.
"""

from __future__ import annotations

import json

import pytest

from spark_optima.platforms.gcp_dataproc import GCPDataprocPlatform
from spark_optima.platforms.models import (
    ClusterConfig,
    ResourceSpec,
    WorkerType,
)


class TestGCPDataprocPlatformInitialization:
    """Test cases for GCPDataprocPlatform initialization."""

    def test_gcp_dataproc_platform_initialization(self) -> None:
        """Test GCP Dataproc platform initialization."""
        platform = GCPDataprocPlatform()
        assert platform.name == "gcp_dataproc"
        assert platform.display_name == "GCP Dataproc"

    def test_gcp_dataproc_platform_constraints(self) -> None:
        """Test GCP Dataproc platform constraints."""
        platform = GCPDataprocPlatform()
        constraints = platform.constraints

        assert constraints.min_workers == 2  # Dataproc requires >= 2 primary workers
        assert constraints.max_workers == 1000
        assert constraints.min_memory_gb == 16  # n2-standard-4
        assert constraints.max_memory_gb == 512  # n2-highmem-64
        assert constraints.min_cores == 4
        assert constraints.max_cores == 64  # n2-standard-64 / n2-highmem-64
        assert len(constraints.supported_spark_versions) > 0

    def test_gcp_dataproc_default_region(self) -> None:
        """Test that the default region is us-central1."""
        platform = GCPDataprocPlatform()
        assert platform.region == "us-central1"

    def test_gcp_dataproc_preemptible_flag_default_off(self) -> None:
        """Test that preemptible workers are off by default."""
        platform = GCPDataprocPlatform()
        assert platform.use_preemptible_workers is False


class TestGCPDataprocPlatformWorkerTypes:
    """Test cases for GCP Dataproc worker machine types."""

    def test_get_worker_types(self) -> None:
        """Test getting available Dataproc machine types."""
        platform = GCPDataprocPlatform()
        worker_types = platform.get_worker_types()

        assert isinstance(worker_types, list)
        assert len(worker_types) == 10  # 5 x n2-standard, 5 x n2-highmem

        for worker_type in worker_types:
            assert isinstance(worker_type, WorkerType)
            assert worker_type.name
            # Should be one of the supported N2 families
            assert worker_type.name.startswith(("n2-standard-", "n2-highmem-"))

    def test_get_worker_type_n2_standard_4(self) -> None:
        """Test getting n2-standard-4 worker type."""
        platform = GCPDataprocPlatform()
        worker_type = platform.get_worker_type("n2-standard-4")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "n2-standard-4"
        assert worker_type.resources.cpu_cores == 4
        assert worker_type.resources.memory_gb == 16

    def test_get_worker_type_n2_highmem_16(self) -> None:
        """Test getting n2-highmem-16 worker type (memory-optimized)."""
        platform = GCPDataprocPlatform()
        worker_type = platform.get_worker_type("n2-highmem-16")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 16
        assert worker_type.resources.memory_gb == 128

    def test_get_worker_type_n2_standard_64(self) -> None:
        """Test getting n2-standard-64 worker type (largest general purpose)."""
        platform = GCPDataprocPlatform()
        worker_type = platform.get_worker_type("n2-standard-64")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.resources.cpu_cores == 64
        assert worker_type.resources.memory_gb == 256

    def test_get_worker_type_case_insensitive(self) -> None:
        """Test that machine type lookup is case insensitive."""
        platform = GCPDataprocPlatform()
        worker_type = platform.get_worker_type("N2-STANDARD-8")

        assert isinstance(worker_type, WorkerType)
        assert worker_type.name == "n2-standard-8"

    def test_get_worker_type_invalid(self) -> None:
        """Test getting invalid worker type returns None."""
        platform = GCPDataprocPlatform()
        worker_type = platform.get_worker_type("InvalidType")

        assert worker_type is None

    def test_worker_cost_includes_dataproc_fee(self) -> None:
        """Test that hourly cost is GCE price plus the Dataproc per-vCPU fee."""
        platform = GCPDataprocPlatform()
        worker = platform.get_worker_type("n2-standard-8")

        assert worker is not None
        spec = GCPDataprocPlatform.WORKER_SPECS["n2-standard-8"]
        expected = spec["compute_price"] + GCPDataprocPlatform.DATAPROC_FEE_PER_VCPU_HOUR * spec["cpu_cores"]
        assert worker.cost.unit_cost_per_hour == pytest.approx(expected)


class TestGCPDataprocPlatformRecommendWorkerType:
    """Test cases for recommend_worker_type family preferences."""

    def test_recommend_worker_type_default_general_purpose(self) -> None:
        """Test that n2-standard machines are preferred by default."""
        platform = GCPDataprocPlatform()
        worker = platform.recommend_worker_type(target_memory_gb=32.0, target_cores=8)

        assert worker is not None
        assert worker.name.startswith("n2-standard-")

    def test_recommend_worker_type_prefer_memory_optimized(self) -> None:
        """Test that n2-highmem machines are preferred for memory-intensive workloads."""
        platform = GCPDataprocPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=64.0,
            target_cores=8,
            prefer_memory_optimized=True,
        )

        assert worker is not None
        assert worker.name.startswith("n2-highmem-")

    def test_recommend_worker_type_fallback(self) -> None:
        """Test fallback to n2-standard-4 when no machine type meets requirements."""
        platform = GCPDataprocPlatform()
        worker = platform.recommend_worker_type(
            target_memory_gb=10000.0,  # Impossibly high
            target_cores=1000,  # Impossibly high
        )

        assert worker is not None
        assert worker.name == "n2-standard-4"


class TestGCPDataprocPlatformClusterConfig:
    """Test cases for cluster configuration."""

    def test_recommend_config_basic(self) -> None:
        """Test basic cluster config recommendation."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert isinstance(config, ClusterConfig)
        assert config.spark_version == "3.5.0"
        assert config.worker_count >= 2

    def test_recommend_config_with_worker_count(self) -> None:
        """Test config recommendation with specific worker count."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=5,
        )

        assert config.worker_count == 5

    def test_recommend_config_enforces_minimum_workers(self) -> None:
        """Test that config enforces the Dataproc 2-primary-worker minimum."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=4, memory_gb=16.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
            worker_count=1,  # Below the Dataproc minimum
        )

        assert config.worker_count >= 2

    def test_recommend_config_includes_master_node(self) -> None:
        """Test that one master node is accounted for."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.driver_type is not None
        assert config.driver_count == 1
        assert config.platform_config["master_machine_type"] == config.driver_type.name

    def test_recommend_config_sets_image_version(self) -> None:
        """Test that the platform config carries the Dataproc image version."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        assert config.platform_config["image_version"] == "2.2"
        assert config.platform_config["machine_type"] == config.worker_type.name
        assert config.platform_config["num_workers"] == config.worker_count


class TestGCPDataprocPlatformSparkConfigTranslation:
    """Test cases for YARN-style Spark config translation."""

    def _get_cluster_config(self) -> tuple[GCPDataprocPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for translation tests."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_translate_to_spark_config(self) -> None:
        """Test translating cluster config to Dataproc-specific Spark config."""
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
        """Test that spark.master is not set (Dataproc configures YARN itself)."""
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

        assert spark_config["spark.dynamicAllocation.minExecutors"] == "2"
        assert spark_config["spark.dynamicAllocation.maxExecutors"] == str(
            cluster_config.worker_count,
        )

    def test_translate_includes_image_version(self) -> None:
        """Test that the Dataproc image version is carried through."""
        platform, cluster_config = self._get_cluster_config()

        spark_config = platform.translate_to_spark_config(cluster_config)

        assert spark_config["dataproc.imageVersion"] == "2.2"


class TestGCPDataprocPlatformCostEstimation:
    """Test cases for cost estimation."""

    def test_estimate_cost(self) -> None:
        """Test cost estimation for GCP Dataproc."""
        platform = GCPDataprocPlatform()
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
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=0.0)

        assert cost["total_cost"] == 0.0

    def test_estimate_cost_includes_master(self) -> None:
        """Test that the master node is part of the total cost."""
        platform = GCPDataprocPlatform()
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

    def test_estimate_cost_breakdown_dataproc_fee(self) -> None:
        """Test that compute and Dataproc fee portions sum to the total."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=3.0)
        breakdown = cost["breakdown"]

        assert breakdown["dataproc_fee_per_vcpu_hour"] == GCPDataprocPlatform.DATAPROC_FEE_PER_VCPU_HOUR
        assert cost["total_cost"] == pytest.approx(
            breakdown["compute_cost"] + breakdown["dataproc_fee"],
        )

        # The fee is $0.01 per vCPU per hour across all nodes
        master = cluster_config.driver_type
        assert master is not None
        total_vcpus = (
            cluster_config.worker_type.resources.cpu_cores * cluster_config.worker_count
            + master.resources.cpu_cores * cluster_config.driver_count
        )
        expected_fee = GCPDataprocPlatform.DATAPROC_FEE_PER_VCPU_HOUR * total_vcpus * 3.0
        assert breakdown["dataproc_fee"] == pytest.approx(expected_fee)

    def test_estimate_cost_preemptible_discount(self) -> None:
        """Test that preemptible workers discount only the worker compute portion."""
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        on_demand = GCPDataprocPlatform()
        preemptible = GCPDataprocPlatform(use_preemptible_workers=True)

        cluster_config = on_demand.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        cost_on_demand = on_demand.estimate_cost(cluster_config, duration_hours=2.0)
        cost_preemptible = preemptible.estimate_cost(cluster_config, duration_hours=2.0)

        assert cost_preemptible["total_cost"] < cost_on_demand["total_cost"]
        # Master cost and Dataproc fee are unaffected by the discount
        assert cost_preemptible["breakdown"]["master_cost"] == pytest.approx(
            cost_on_demand["breakdown"]["master_cost"],
        )
        assert cost_preemptible["breakdown"]["dataproc_fee"] == pytest.approx(
            cost_on_demand["breakdown"]["dataproc_fee"],
        )
        assert cost_preemptible["breakdown"]["preemptible_workers"] is True
        assert cost_preemptible["breakdown"]["preemptible_discount"] == GCPDataprocPlatform.PREEMPTIBLE_DISCOUNT

    def test_estimate_cost_preemptible_discount_amount(self) -> None:
        """Test the preemptible discount is ~65% of the worker compute portion."""
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)

        on_demand = GCPDataprocPlatform()
        preemptible = GCPDataprocPlatform(use_preemptible_workers=True)

        cluster_config = on_demand.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        breakdown_od = on_demand.estimate_cost(cluster_config, duration_hours=1.0)["breakdown"]
        breakdown_pe = preemptible.estimate_cost(cluster_config, duration_hours=1.0)["breakdown"]

        worker_fee = (
            GCPDataprocPlatform.DATAPROC_FEE_PER_VCPU_HOUR
            * cluster_config.worker_type.resources.cpu_cores
            * cluster_config.worker_count
        )
        worker_compute_od = breakdown_od["worker_cost"] - worker_fee
        worker_compute_pe = breakdown_pe["worker_cost"] - worker_fee

        assert worker_compute_pe == pytest.approx(
            worker_compute_od * (1.0 - GCPDataprocPlatform.PREEMPTIBLE_DISCOUNT),
        )

    def test_estimate_cost_scales_with_workers(self) -> None:
        """Test that cost scales with worker count."""
        platform = GCPDataprocPlatform()
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


class TestGCPDataprocPlatformValidation:
    """Test cases for configuration validation."""

    def test_validate_config_valid(self) -> None:
        """Test validation with valid config."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )

        errors = platform.validate_config(cluster_config)

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_config_below_min_workers(self) -> None:
        """Test validation below the Dataproc 2-primary-worker minimum."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 1  # Below the Dataproc minimum of 2

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("minimum" in error.lower() for error in errors)

    def test_validate_config_exceeds_max_workers(self) -> None:
        """Test validation exceeding maximum workers."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)

        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        cluster_config.worker_count = 1001  # Exceeds the limit of 1000

        errors = platform.validate_config(cluster_config)

        assert len(errors) > 0
        assert any("maximum" in error.lower() for error in errors)


class TestGCPDataprocPlatformSparkVersions:
    """Test cases for Spark version support."""

    def test_get_supported_spark_versions(self) -> None:
        """Test getting supported Spark versions."""
        platform = GCPDataprocPlatform()
        versions = platform.get_supported_spark_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_is_spark_version_supported_image_2_0(self) -> None:
        """Test Dataproc 2.0 (Spark 3.1.3) support."""
        platform = GCPDataprocPlatform()

        assert platform.is_spark_version_supported("3.1.3")

    def test_is_spark_version_supported_image_2_1(self) -> None:
        """Test Dataproc 2.1 (Spark 3.3.x) support."""
        platform = GCPDataprocPlatform()

        assert platform.is_spark_version_supported("3.3.2")

    def test_is_spark_version_supported_image_2_2(self) -> None:
        """Test Dataproc 2.2 (Spark 3.5.x) support."""
        platform = GCPDataprocPlatform()

        assert platform.is_spark_version_supported("3.5.0")


class TestGCPDataprocPlatformImageVersionMapping:
    """Test cases for match_image_version."""

    def test_match_image_version_exact(self) -> None:
        """Test mapping Spark version to a Dataproc image version."""
        assert GCPDataprocPlatform.match_image_version("3.1.3") == "2.0"
        assert GCPDataprocPlatform.match_image_version("3.3.2") == "2.1"
        assert GCPDataprocPlatform.match_image_version("3.5.3") == "2.2"

    def test_match_image_version_partial_match(self) -> None:
        """Test major.minor fallback matching."""
        # 3.5.0 has no exact mapping but matches the 3.5 line (image 2.2)
        assert GCPDataprocPlatform.match_image_version("3.5.0") == "2.2"
        assert GCPDataprocPlatform.match_image_version("3.1.0") == "2.0"
        assert GCPDataprocPlatform.match_image_version("3.3.0") == "2.1"

    def test_match_image_version_default_fallback(self) -> None:
        """Test default fallback for unknown Spark version."""
        assert GCPDataprocPlatform.match_image_version("9.9.9") == GCPDataprocPlatform.DEFAULT_IMAGE_VERSION

    def test_get_image_version_instance_method(self) -> None:
        """Test the instance-level wrapper delegates to the class mapping."""
        platform = GCPDataprocPlatform()

        assert platform._get_image_version("3.5.0") == "2.2"


class TestGCPDataprocPlatformClusterDefinition:
    """Test cases for get_dataproc_cluster_config (clusters.create body)."""

    def _get_cluster_config(self) -> tuple[GCPDataprocPlatform, ClusterConfig]:
        """Build a platform and cluster config pair for cluster definition tests."""
        platform = GCPDataprocPlatform()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64.0)
        cluster_config = platform.recommend_config(
            resources=resources,
            spark_version="3.5.0",
        )
        return platform, cluster_config

    def test_get_dataproc_cluster_config_structure(self) -> None:
        """Test clusters.create body structure."""
        platform, cluster_config = self._get_cluster_config()

        body = platform.get_dataproc_cluster_config(cluster_config)

        assert body["clusterName"] == "spark-optima-cluster"
        assert "gceClusterConfig" in body["config"]
        assert "masterConfig" in body["config"]
        assert "workerConfig" in body["config"]
        assert "softwareConfig" in body["config"]

    def test_get_dataproc_cluster_config_node_groups(self) -> None:
        """Test master and worker node configuration."""
        platform, cluster_config = self._get_cluster_config()

        body = platform.get_dataproc_cluster_config(cluster_config)

        master = body["config"]["masterConfig"]
        worker = body["config"]["workerConfig"]

        assert master["numInstances"] == 1
        assert master["machineTypeUri"] == GCPDataprocPlatform.DEFAULT_MASTER_MACHINE_TYPE
        assert worker["numInstances"] == cluster_config.worker_count
        assert worker["machineTypeUri"] == cluster_config.worker_type.name

    def test_get_dataproc_cluster_config_software_config(self) -> None:
        """Test that softwareConfig carries the image version and spark: properties."""
        platform, cluster_config = self._get_cluster_config()

        body = platform.get_dataproc_cluster_config(cluster_config)
        software = body["config"]["softwareConfig"]

        assert software["imageVersion"] == "2.2"
        properties = software["properties"]
        assert "spark:spark.executor.memory" in properties
        assert "spark:spark.executor.cores" in properties
        # All properties must be spark: prefixed spark.* keys (non spark.* keys are filtered)
        assert all(key.startswith("spark:spark.") for key in properties)

    def test_get_dataproc_cluster_config_custom_spark_config(self) -> None:
        """Test passing an explicit optimized Spark config."""
        platform, cluster_config = self._get_cluster_config()

        body = platform.get_dataproc_cluster_config(
            cluster_config,
            cluster_name="my-cluster",
            spark_config={"spark.executor.memory": "8g", "dataproc.imageVersion": "2.2"},
        )

        assert body["clusterName"] == "my-cluster"
        properties = body["config"]["softwareConfig"]["properties"]
        assert properties == {"spark:spark.executor.memory": "8g"}

    def test_get_dataproc_cluster_config_json_serializable(self) -> None:
        """Test that the request body is JSON-serializable (REST compatible)."""
        platform, cluster_config = self._get_cluster_config()

        body = platform.get_dataproc_cluster_config(cluster_config)

        serialized = json.dumps(body)
        assert json.loads(serialized) == body


class TestGCPDataprocRegionalPricing:
    """Tests for regional pricing in Dataproc cost estimation."""

    def _cost_for_region(self, region: str, duration_hours: float = 2.0) -> dict:
        platform = GCPDataprocPlatform(region=region)
        resources = ResourceSpec(cpu_cores=8, memory_gb=32.0)
        cluster_config = platform.recommend_config(resources=resources, spark_version="3.5.0")
        return platform.estimate_cost(cluster_config, duration_hours=duration_hours)

    def test_baseline_region_multiplier_is_one(self) -> None:
        """Test us-central1 (baseline) reports multiplier 1.0."""
        cost = self._cost_for_region("us-central1")
        assert cost["breakdown"]["region_multiplier"] == 1.0
        assert cost["breakdown"]["region"] == "us-central1"

    def test_non_baseline_region_scales_compute_only(self) -> None:
        """Test a non-baseline region scales the compute portion by the table multiplier."""
        baseline = self._cost_for_region("us-central1")
        tokyo = self._cost_for_region("asia-northeast1")

        assert tokyo["breakdown"]["region_multiplier"] == 1.2
        # Compute portion scales by the multiplier; the Dataproc fee does not.
        assert tokyo["breakdown"]["compute_cost"] == pytest.approx(baseline["breakdown"]["compute_cost"] * 1.2)
        assert tokyo["breakdown"]["dataproc_fee"] == pytest.approx(baseline["breakdown"]["dataproc_fee"])
        assert tokyo["total_cost"] > baseline["total_cost"]

    def test_unknown_region_falls_back_to_baseline(self) -> None:
        """Test an unknown region falls back to multiplier 1.0 without raising."""
        cost = self._cost_for_region("mars-north-1")
        assert cost["breakdown"]["region_multiplier"] == 1.0


class TestGCPDataprocPlatformLivePricing:
    """Test cases for live pricing behavior (always static for Dataproc)."""

    def test_pricing_source_is_always_static(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Dataproc reports static pricing even when opted in.

        The GCP Cloud Billing Catalog API requires an API key, so live
        pricing for Dataproc is deferred (see PLAN.md backlog).
        """
        monkeypatch.setenv("SPARK_OPTIMA_LIVE_PRICING", "1")
        platform = GCPDataprocPlatform()
        cluster_config = platform.recommend_config(
            resources=ResourceSpec(cpu_cores=16, memory_gb=64.0),
            spark_version="3.5.0",
        )

        cost = platform.estimate_cost(cluster_config, duration_hours=2.0)

        assert cost["pricing_source"] == "static"
