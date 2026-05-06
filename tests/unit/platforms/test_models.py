# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for platform models."""

import pytest

from spark_optima.platforms.models import (
    ClusterConfig,
    CostModel,
    InstanceSize,
    PlatformConstraints,
    ResourceSpec,
    WorkerType,
)


class TestResourceSpec:
    """Test cases for ResourceSpec class."""

    def test_creation(self):
        """Test creating a ResourceSpec."""
        spec = ResourceSpec(cpu_cores=4, memory_gb=16, disk_gb=100)
        assert spec.cpu_cores == 4
        assert spec.memory_gb == 16
        assert spec.disk_gb == 100
        assert spec.gpu_count == 0
        assert spec.network_gbps == 10.0

    def test_validation(self):
        """Test ResourceSpec validation."""
        with pytest.raises(ValueError, match="CPU cores must be at least 1"):
            ResourceSpec(cpu_cores=0, memory_gb=16)

        with pytest.raises(ValueError, match="Memory must be positive"):
            ResourceSpec(cpu_cores=4, memory_gb=0)

        with pytest.raises(ValueError, match="Disk cannot be negative"):
            ResourceSpec(cpu_cores=4, memory_gb=16, disk_gb=-1)

    def test_to_dict(self):
        """Test ResourceSpec serialization."""
        spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data = spec.to_dict()
        assert data["cpu_cores"] == 8
        assert data["memory_gb"] == 32

    def test_from_dict(self):
        """Test ResourceSpec deserialization."""
        data = {"cpu_cores": 4, "memory_gb": 16, "disk_gb": 50}
        spec = ResourceSpec.from_dict(data)
        assert spec.cpu_cores == 4
        assert spec.memory_gb == 16
        assert spec.disk_gb == 50

    def test_scale(self):
        """Test scaling resources."""
        spec = ResourceSpec(cpu_cores=4, memory_gb=16)
        scaled = spec.scale(2.0)
        assert scaled.cpu_cores == 8
        assert scaled.memory_gb == 32

    def test_add(self):
        """Test adding ResourceSpecs."""
        spec1 = ResourceSpec(cpu_cores=4, memory_gb=16)
        spec2 = ResourceSpec(cpu_cores=2, memory_gb=8)
        total = spec1 + spec2
        assert total.cpu_cores == 6
        assert total.memory_gb == 24


class TestCostModel:
    """Test cases for CostModel class."""

    def test_creation(self):
        """Test creating a CostModel."""
        cost = CostModel(currency="USD", unit_cost_per_hour=0.44)
        assert cost.currency == "USD"
        assert cost.unit_cost_per_hour == 0.44

    def test_calculate(self):
        """Test cost calculation."""
        cost = CostModel(unit_cost_per_hour=1.0)
        assert cost.calculate(2.0) == 2.0
        assert cost.calculate(0) == 0.0

    def test_estimate_monthly(self):
        """Test monthly cost estimation."""
        cost = CostModel(unit_cost_per_hour=1.0)
        monthly = cost.estimate_monthly(hours_per_day=8)
        assert monthly == 240.0  # 8 hours * 30 days


class TestPlatformConstraints:
    """Test cases for PlatformConstraints class."""

    def test_creation(self):
        """Test creating PlatformConstraints."""
        constraints = PlatformConstraints(min_workers=2, max_workers=100)
        assert constraints.min_workers == 2
        assert constraints.max_workers == 100

    def test_validation(self):
        """Test PlatformConstraints validation."""
        with pytest.raises(ValueError, match="max_workers must be >= min_workers"):
            PlatformConstraints(min_workers=10, max_workers=5)

    def test_validate_resources(self):
        """Test resource validation."""
        constraints = PlatformConstraints(min_workers=2, max_workers=10)
        spec = ResourceSpec(cpu_cores=4, memory_gb=16)
        errors = constraints.validate_resources(spec, worker_count=5)
        assert len(errors) == 0

        errors = constraints.validate_resources(spec, worker_count=1)
        assert len(errors) == 1
        assert "below minimum" in errors[0]


class TestWorkerType:
    """Test cases for WorkerType class."""

    def test_creation(self):
        """Test creating a WorkerType."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=0.44)
        worker = WorkerType(
            name="test-worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        assert worker.name == "test-worker"
        assert worker.size == InstanceSize.SMALL

    def test_estimate_job_cost(self):
        """Test job cost estimation."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=1.0)
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        assert worker.estimate_job_cost(duration_hours=2.0, worker_count=2) == 4.0


class TestClusterConfig:
    """Test cases for ClusterConfig class."""

    def test_creation(self):
        """Test creating a ClusterConfig."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=0.44)
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        config = ClusterConfig(worker_type=worker, worker_count=3)
        assert config.worker_count == 3
        assert config.spark_version == "3.5.0"  # default

    def test_total_resources(self):
        """Test total resources calculation."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=0.44)
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        config = ClusterConfig(worker_type=worker, worker_count=2)
        total = config.total_resources
        # 2 workers * 4 cores + 1 driver * 4 cores = 12 cores
        assert total.cpu_cores == 12
        assert total.memory_gb == 48  # 2 workers * 16 GB + 1 driver * 16 GB

    def test_estimate_cost(self):
        """Test cost estimation."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=1.0)
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        config = ClusterConfig(worker_type=worker, worker_count=2)
        estimated = config.estimate_cost(duration_hours=2.0)
        # 2 workers * $1/hour * 2 hours + 1 driver * $1/hour * 2 hours = $6
        assert estimated == 6.0


class TestResourceSpecGPUValidation:
    """Test cases for GPU count validation (line 62)."""

    def test_gpu_count_negative_raises_error(self):
        """Test that negative GPU count raises ValueError (line 62)."""
        with pytest.raises(ValueError, match="GPU count cannot be negative"):
            ResourceSpec(cpu_cores=4, memory_gb=16, gpu_count=-1)

    def test_gpu_count_zero_valid(self):
        """Test that zero GPU count is valid."""
        spec = ResourceSpec(cpu_cores=4, memory_gb=16, gpu_count=0)
        assert spec.gpu_count == 0

    def test_gpu_count_positive_valid(self):
        """Test that positive GPU count is valid."""
        spec = ResourceSpec(cpu_cores=4, memory_gb=16, gpu_count=2)
        assert spec.gpu_count == 2


class TestPlatformConstraintsValidation:
    """Test cases for PlatformConstraints validation (lines 142, 146, 148, 150, 152)."""

    def test_min_workers_negative_raises_error(self):
        """Test that negative min_workers raises ValueError (line 142)."""
        with pytest.raises(ValueError, match="min_workers cannot be negative"):
            PlatformConstraints(min_workers=-1)

    def test_max_workers_less_than_min_raises_error(self):
        """Test that max_workers < min_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers must be >= min_workers"):
            PlatformConstraints(min_workers=10, max_workers=5)

    def test_min_memory_gb_zero_raises_error(self):
        """Test that zero min_memory_gb raises ValueError (line 146)."""
        with pytest.raises(ValueError, match="min_memory_gb must be positive"):
            PlatformConstraints(min_memory_gb=0)

    def test_min_memory_gb_negative_raises_error(self):
        """Test that negative min_memory_gb raises ValueError."""
        with pytest.raises(ValueError, match="min_memory_gb must be positive"):
            PlatformConstraints(min_memory_gb=-1.0)

    def test_max_memory_less_than_min_raises_error(self):
        """Test that max_memory_gb < min_memory_gb raises ValueError (line 148)."""
        with pytest.raises(ValueError, match="max_memory_gb must be >= min_memory_gb"):
            PlatformConstraints(min_memory_gb=32.0, max_memory_gb=16.0)

    def test_min_cores_zero_raises_error(self):
        """Test that min_cores=0 raises ValueError (line 150)."""
        with pytest.raises(ValueError, match="min_cores must be at least 1"):
            PlatformConstraints(min_cores=0)

    def test_min_cores_negative_raises_error(self):
        """Test that negative min_cores raises ValueError."""
        with pytest.raises(ValueError, match="min_cores must be at least 1"):
            PlatformConstraints(min_cores=-1)

    def test_max_cores_less_than_min_raises_error(self):
        """Test that max_cores < min_cores raises ValueError (line 152)."""
        with pytest.raises(ValueError, match="max_cores must be >= min_cores"):
            PlatformConstraints(min_cores=8, max_cores=4)


class TestPlatformConstraintsToDict:
    """Test cases for PlatformConstraints.to_dict (line 156)."""

    def test_to_dict_basic(self):
        """Test to_dict method (line 156)."""
        constraints = PlatformConstraints(
            min_workers=2,
            max_workers=100,
            min_memory_gb=8.0,
            max_memory_gb=256.0,
            min_cores=2,
            max_cores=32,
        )

        data = constraints.to_dict()

        assert data["min_workers"] == 2
        assert data["max_workers"] == 100
        assert data["min_memory_gb"] == 8.0
        assert data["max_memory_gb"] == 256.0
        assert data["min_cores"] == 2
        assert data["max_cores"] == 32

    def test_to_dict_supported_versions(self):
        """Test to_dict includes supported Spark versions."""
        constraints = PlatformConstraints()

        data = constraints.to_dict()

        assert "supported_spark_versions" in data
        assert isinstance(data["supported_spark_versions"], list)

    def test_to_dict_custom_config_keys(self):
        """Test to_dict includes custom config keys."""
        constraints = PlatformConstraints(custom_config_keys={"key1": "value1"})

        data = constraints.to_dict()

        assert data["custom_config_keys"] == {"key1": "value1"}


class TestPlatformConstraintsFromDict:
    """Test cases for PlatformConstraints.from_dict (line 170)."""

    def test_from_dict_basic(self):
        """Test from_dict classmethod (line 170)."""
        data = {
            "min_workers": 5,
            "max_workers": 200,
            "min_memory_gb": 16.0,
            "max_memory_gb": 512.0,
            "min_cores": 4,
            "max_cores": 64,
        }

        constraints = PlatformConstraints.from_dict(data)

        assert constraints.min_workers == 5
        assert constraints.max_workers == 200
        assert constraints.min_memory_gb == 16.0
        assert constraints.max_memory_gb == 512.0

    def test_from_dict_defaults(self):
        """Test from_dict with missing keys uses defaults."""
        data = {}  # Empty dict

        constraints = PlatformConstraints.from_dict(data)

        assert constraints.min_workers == 1
        assert constraints.max_workers == 1000

    def test_from_dict_partial(self):
        """Test from_dict with partial data."""
        data = {
            "min_workers": 3,
            "max_workers": 50,
        }

        constraints = PlatformConstraints.from_dict(data)

        assert constraints.min_workers == 3
        assert constraints.max_workers == 50
        # Defaults for others
        assert constraints.min_memory_gb == 1.0

    def test_from_dict_custom_versions(self):
        """Test from_dict with custom Spark versions."""
        data = {
            "supported_spark_versions": ["3.0.0", "3.1.0", "3.2.0"],
        }

        constraints = PlatformConstraints.from_dict(data)

        assert constraints.supported_spark_versions == ["3.0.0", "3.1.0", "3.2.0"]


class TestPlatformConstraintsValidateResources:
    """Additional test cases for validate_resources."""

    def test_validate_resources_exceeds_max_memory(self):
        """Test validation when memory exceeds maximum."""
        constraints = PlatformConstraints(min_memory_gb=8.0, max_memory_gb=32.0)
        spec = ResourceSpec(cpu_cores=4, memory_gb=64.0)

        errors = constraints.validate_resources(spec, worker_count=5)

        assert len(errors) > 0
        assert any("exceeds maximum" in error for error in errors)

    def test_validate_resources_below_min_memory(self):
        """Test validation when memory is below minimum."""
        constraints = PlatformConstraints(min_memory_gb=8.0, max_memory_gb=32.0)
        spec = ResourceSpec(cpu_cores=4, memory_gb=4.0)

        errors = constraints.validate_resources(spec, worker_count=5)

        assert len(errors) > 0
        assert any("below minimum" in error for error in errors)

    def test_validate_resources_below_min_cores(self):
        """Test validation when cores are below minimum."""
        constraints = PlatformConstraints(min_cores=4, max_cores=32)
        spec = ResourceSpec(cpu_cores=2, memory_gb=16.0)

        errors = constraints.validate_resources(spec, worker_count=5)

        assert len(errors) > 0
        assert any("below minimum" in error for error in errors)

    def test_validate_resources_exceeds_max_cores(self):
        """Test validation when cores exceed maximum."""
        constraints = PlatformConstraints(min_cores=4, max_cores=32)
        spec = ResourceSpec(cpu_cores=64, memory_gb=16.0)

        errors = constraints.validate_resources(spec, worker_count=5)

        assert len(errors) > 0
        assert any("exceeds maximum" in error for error in errors)


class TestCostModelToDict:
    """Test cases for CostModel.to_dict."""

    def test_to_dict_basic(self):
        """Test CostModel.to_dict method."""
        cost = CostModel(
            currency="EUR",
            unit_cost_per_hour=0.50,
            unit_name="DPU",
            granularity_minutes=1,
            minimum_charge_minutes=1,
        )

        data = cost.to_dict()

        assert data["currency"] == "EUR"
        assert data["unit_cost_per_hour"] == 0.50
        assert data["unit_name"] == "DPU"
        assert data["granularity_minutes"] == 1
        assert data["minimum_charge_minutes"] == 1

    def test_to_dict_defaults(self):
        """Test to_dict with default values."""
        cost = CostModel()

        data = cost.to_dict()

        assert data["currency"] == "USD"
        assert data["unit_cost_per_hour"] == 0.0
        assert data["unit_name"] == "instance"


class TestCostModelFromDict:
    """Test cases for CostModel.from_dict."""

    def test_from_dict_basic(self):
        """Test CostModel.from_dict classmethod."""
        data = {
            "currency": "EUR",
            "unit_cost_per_hour": 0.75,
            "unit_name": "DBU",
            "granularity_minutes": 1,
            "minimum_charge_minutes": 1,
        }

        cost = CostModel.from_dict(data)

        assert cost.currency == "EUR"
        assert cost.unit_cost_per_hour == 0.75
        assert cost.unit_name == "DBU"

    def test_from_dict_defaults(self):
        """Test from_dict with missing keys uses defaults."""
        data = {}

        cost = CostModel.from_dict(data)

        assert cost.currency == "USD"
        assert cost.unit_cost_per_hour == 0.0
        assert cost.unit_name == "instance"

    def test_from_dict_partial(self):
        """Test from_dict with partial data."""
        data = {"unit_cost_per_hour": 1.0}

        cost = CostModel.from_dict(data)

        assert cost.unit_cost_per_hour == 1.0
        assert cost.currency == "USD"  # Default


class TestCostModelCalculate:
    """Additional test cases for CostModel.calculate."""

    def test_calculate_zero_duration(self):
        """Test cost calculation with zero duration."""
        cost = CostModel(unit_cost_per_hour=1.0)

        result = cost.calculate(0.0)

        assert result == 0.0

    def test_calculate_negative_duration(self):
        """Test cost calculation with negative duration."""
        cost = CostModel(unit_cost_per_hour=1.0)

        result = cost.calculate(-1.0)

        assert result == 0.0

    def test_calculate_with_multiple_units(self):
        """Test cost calculation with multiple units."""
        cost = CostModel(unit_cost_per_hour=1.0)

        result = cost.calculate(1.0, units=10)

        assert result == 10.0  # 1 hour * $1 * 10 units

    def test_calculate_with_minimum_charge(self):
        """Test cost calculation with minimum charge."""
        cost = CostModel(
            unit_cost_per_hour=1.0,
            granularity_minutes=60,  # 1 hour granularity
            minimum_charge_minutes=60,  # 1 hour minimum
        )

        # 30 minutes should be billed as 1 hour due to minimum charge
        result = cost.calculate(0.5, units=1)

        assert result == 1.0  # 1 hour minimum


class TestWorkerTypeToDict:
    """Test cases for WorkerType.to_dict."""

    def test_to_dict_basic(self):
        """Test WorkerType.to_dict method."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=0.44)
        worker = WorkerType(
            name="test-worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
            description="Test worker",
            is_spot=True,
            availability_zones=["us-east-1a", "us-east-1b"],
        )

        data = worker.to_dict()

        assert data["name"] == "test-worker"
        assert data["size"] == "small"
        assert data["description"] == "Test worker"
        assert data["is_spot"] is True
        assert len(data["availability_zones"]) == 2
        assert "resources" in data
        assert "cost" in data

    def test_to_dict_defaults(self):
        """Test to_dict with default values."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="test",
            size=InstanceSize.MEDIUM,
            resources=resources,
            cost=cost,
        )

        data = worker.to_dict()

        assert data["description"] == ""
        assert data["is_spot"] is False
        assert data["availability_zones"] == []


class TestWorkerTypeFromDict:
    """Test cases for WorkerType.from_dict."""

    def test_from_dict_basic(self):
        """Test WorkerType.from_dict classmethod."""
        data = {
            "name": "test-worker",
            "size": "small",
            "resources": {
                "cpu_cores": 4,
                "memory_gb": 16.0,
                "disk_gb": 64.0,
            },
            "cost": {
                "currency": "USD",
                "unit_cost_per_hour": 0.44,
                "unit_name": "DPU",
            },
            "description": "Test worker",
            "is_spot": True,
            "availability_zones": ["us-east-1a"],
        }

        worker = WorkerType.from_dict(data)

        assert worker.name == "test-worker"
        assert worker.size == InstanceSize.SMALL
        assert worker.resources.cpu_cores == 4
        assert worker.cost.unit_cost_per_hour == 0.44
        assert worker.is_spot is True

    def test_from_dict_without_optional(self):
        """Test from_dict without optional fields."""
        data = {
            "name": "test",
            "size": "medium",
            "resources": {
                "cpu_cores": 8,
                "memory_gb": 32.0,
            },
            "cost": {
                "unit_cost_per_hour": 1.0,
            },
        }

        worker = WorkerType.from_dict(data)

        assert worker.description == ""
        assert worker.is_spot is False
        assert worker.availability_zones == []


class TestClusterConfigTotalResources:
    """Test cases for ClusterConfig.total_resources property."""

    def test_total_resources_with_driver(self):
        """Test total_resources when driver_type is specified."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )
        driver = WorkerType(
            name="driver",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            worker_count=2,
            driver_type=driver,
            driver_count=1,
        )

        total = config.total_resources

        # 2 workers * 4 cores + 1 driver * 4 cores = 12 cores
        assert total.cpu_cores == 12
        assert total.memory_gb == 48.0  # 2 * 16 + 16

    def test_total_resources_without_driver(self):
        """Test total_resources when driver_type is None."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            worker_count=3,
        )

        total = config.total_resources

        # driver_type defaults to worker_type
        # 3 workers * 4 cores + 1 driver * 4 cores = 16 cores
        assert total.cpu_cores == 16


class TestClusterConfigTotalCostPerHour:
    """Test cases for ClusterConfig.total_cost_per_hour property."""

    def test_total_cost_per_hour_basic(self):
        """Test total_cost_per_hour calculation."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        worker_cost = CostModel(unit_cost_per_hour=1.0)
        driver_cost = CostModel(unit_cost_per_hour=2.0)

        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=worker_cost,
        )
        driver = WorkerType(
            name="driver",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=driver_cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            worker_count=2,
            driver_type=driver,
            driver_count=1,
        )

        # 2 workers * $1 + 1 driver * $2 = $4/hour
        assert config.total_cost_per_hour == 4.0

    def test_total_cost_per_hour_no_driver_type(self):
        """Test when driver_type is None."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel(unit_cost_per_hour=1.0)

        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            worker_count=3,
        )

        # 3 workers * $1 + 1 driver (same as worker) * $1 = $4/hour
        assert config.total_cost_per_hour == 4.0


class TestClusterConfigToDict:
    """Test cases for ClusterConfig.to_dict."""

    def test_to_dict_basic(self):
        """Test ClusterConfig.to_dict method."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            worker_count=2,
            spark_version="3.5.0",
            platform_config={"key": "value"},
        )

        data = config.to_dict()

        assert data["worker_count"] == 2
        assert data["spark_version"] == "3.5.0"
        assert data["platform_config"] == {"key": "value"}
        assert "worker_type" in data

    def test_to_dict_with_driver(self):
        """Test to_dict with driver_type."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="worker",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        config = ClusterConfig(
            worker_type=worker,
            driver_type=worker,
        )

        data = config.to_dict()

        assert data["driver_type"] is not None
        assert "resources" in data["worker_type"]


class TestClusterConfigFromDict:
    """Test cases for ClusterConfig.from_dict."""

    def test_from_dict_basic(self):
        """Test ClusterConfig.from_dict classmethod."""
        data = {
            "worker_type": {
                "name": "test",
                "size": "small",
                "resources": {"cpu_cores": 4, "memory_gb": 16.0},
                "cost": {"unit_cost_per_hour": 0.44},
            },
            "worker_count": 3,
            "spark_version": "3.5.0",
            "platform_config": {},
        }

        config = ClusterConfig.from_dict(data)

        assert config.worker_count == 3
        assert config.spark_version == "3.5.0"
        assert config.worker_type.name == "test"

    def test_from_dict_with_driver(self):
        """Test from_dict with driver_type."""
        data = {
            "worker_type": {
                "name": "worker",
                "size": "medium",
                "resources": {"cpu_cores": 8, "memory_gb": 32.0},
                "cost": {"unit_cost_per_hour": 1.0},
            },
            "driver_type": {
                "name": "driver",
                "size": "large",
                "resources": {"cpu_cores": 16, "memory_gb": 64.0},
                "cost": {"unit_cost_per_hour": 2.0},
            },
            "worker_count": 2,
        }

        config = ClusterConfig.from_dict(data)

        assert config.driver_type is not None
        assert config.driver_type.name == "driver"

    def test_from_dict_defaults(self):
        """Test from_dict with minimal data."""
        data = {
            "worker_type": {
                "name": "test",
                "size": "small",
                "resources": {"cpu_cores": 4, "memory_gb": 16.0},
                "cost": {"unit_cost_per_hour": 0.44},
            },
        }

        config = ClusterConfig.from_dict(data)

        assert config.worker_count == 2  # Default
        assert config.spark_version == "3.5.0"  # Default
        assert config.platform_config == {}  # Default


class TestClusterConfigValidation:
    """Test cases for ClusterConfig validation in __post_init__."""

    def test_worker_count_negative_raises_error(self):
        """Test that negative worker_count raises ValueError."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        with pytest.raises(ValueError, match="worker_count cannot be negative"):
            ClusterConfig(worker_type=worker, worker_count=-1)

    def test_driver_count_zero_raises_error(self):
        """Test that driver_count=0 raises ValueError."""
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        cost = CostModel()
        worker = WorkerType(
            name="test",
            size=InstanceSize.SMALL,
            resources=resources,
            cost=cost,
        )

        with pytest.raises(ValueError, match="driver_count must be at least 1"):
            ClusterConfig(worker_type=worker, driver_count=0)


class TestResourceSpecToDict:
    """Additional test cases for ResourceSpec.to_dict."""

    def test_to_dict_with_gpu_and_network(self):
        """Test to_dict with GPU and network specified."""
        spec = ResourceSpec(
            cpu_cores=8,
            memory_gb=32.0,
            disk_gb=500.0,
            gpu_count=2,
            network_gbps=25.0,
        )

        data = spec.to_dict()

        assert data["gpu_count"] == 2
        assert data["network_gbps"] == 25.0

    def test_to_dict_defaults(self):
        """Test to_dict with default values."""
        spec = ResourceSpec(cpu_cores=4, memory_gb=16)

        data = spec.to_dict()

        assert data["disk_gb"] == 0.0
        assert data["gpu_count"] == 0
        assert data["network_gbps"] == 10.0


class TestResourceSpecFromDict:
    """Additional test cases for ResourceSpec.from_dict."""

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields specified."""
        data = {
            "cpu_cores": 16,
            "memory_gb": 64.0,
            "disk_gb": 1000.0,
            "gpu_count": 4,
            "network_gbps": 50.0,
        }

        spec = ResourceSpec.from_dict(data)

        assert spec.cpu_cores == 16
        assert spec.memory_gb == 64.0
        assert spec.disk_gb == 1000.0
        assert spec.gpu_count == 4
        assert spec.network_gbps == 50.0

    def test_from_dict_partial(self):
        """Test from_dict with partial data."""
        data = {
            "cpu_cores": 8,
            "memory_gb": 32.0,
        }

        spec = ResourceSpec.from_dict(data)

        assert spec.cpu_cores == 8
        assert spec.memory_gb == 32.0
        assert spec.disk_gb == 0.0  # Default


class TestCostModelValidation:
    """Test cases for CostModel validation (lines 251, 253)."""

    def test_unit_cost_negative_raises_error(self):
        """Test that negative unit_cost_per_hour raises ValueError (line 251)."""
        with pytest.raises(ValueError, match="Unit cost cannot be negative"):
            CostModel(unit_cost_per_hour=-1.0)

    def test_granularity_less_than_one_raises_error(self):
        """Test that granularity < 1 raises ValueError (line 253)."""
        with pytest.raises(ValueError, match="Granularity must be at least 1 minute"):
            CostModel(granularity_minutes=0)

    def test_granularity_zero_raises_error(self):
        """Test that granularity = 0 raises ValueError."""
        with pytest.raises(ValueError, match="Granularity must be at least 1 minute"):
            CostModel(granularity_minutes=0)

    def test_valid_cost_model(self):
        """Test that valid CostModel doesn't raise errors."""
        # These should not raise
        cost = CostModel(unit_cost_per_hour=0.0)
        assert cost.unit_cost_per_hour == 0.0

        cost = CostModel(granularity_minutes=1)
        assert cost.granularity_minutes == 1
