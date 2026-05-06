# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the heuristic engine."""

from __future__ import annotations

from spark_optima.core.config_engine.database import ConfigDatabase
from spark_optima.core.config_engine.models import ParameterCategory
from spark_optima.core.heuristics import DataProfile, HeuristicEngine
from spark_optima.platforms.models import ResourceSpec


class TestHeuristicEngine:
    """Test cases for HeuristicEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        db = ConfigDatabase()
        config_set = db.get_config_set("3.5.0")
        self.engine = HeuristicEngine(config_set)
        self.resources = ResourceSpec(cpu_cores=16, memory_gb=64)

    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.config_set is not None
        assert self.engine.config_set.version == "3.5.0"
        assert self.engine.rule_registry is not None

    def test_engine_initialization_without_config(self):
        """Test engine initialization without config set."""
        engine = HeuristicEngine()
        assert engine.config_set is not None
        assert engine.config_set.version == "3.5.0"

    def test_evaluate_local_platform(self):
        """Test evaluation for local platform."""
        config = self.engine.evaluate(
            resources=self.resources,
            platform="local",
        )

        assert len(config) > 0
        # Check for expected keys
        assert "spark.driver.memory" in config
        assert "spark.executor.memory" in config
        assert "spark.executor.cores" in config

    def test_evaluate_aws_glue_platform(self):
        """Test evaluation for AWS Glue platform."""
        config = self.engine.evaluate(
            resources=self.resources,
            platform="aws_glue",
        )

        assert len(config) > 0
        # AWS Glue should have dynamic allocation enabled
        assert "spark.dynamicAllocation.enabled" in config

    def test_evaluate_with_data_profile(self):
        """Test evaluation with data profile."""
        data_profile = DataProfile(
            format="parquet",
            size_gb=100,
        )

        config = self.engine.evaluate(
            resources=self.resources,
            platform="local",
            data_profile=data_profile,
        )

        assert len(config) > 0

    def test_get_config_by_category(self):
        """Test getting config by category."""
        # First evaluate to populate config
        self.engine.evaluate(resources=self.resources, platform="local")

        memory_config = self.engine.get_config_by_category(ParameterCategory.MEMORY)
        assert len(memory_config) > 0
        assert all(k.startswith("spark.") for k in memory_config)

    def test_get_memory_config(self):
        """Test getting memory config."""
        self.engine.evaluate(resources=self.resources, platform="local")

        memory_config = self.engine.get_memory_config()
        assert len(memory_config) > 0
        assert "spark.driver.memory" in memory_config

    def test_get_cpu_config(self):
        """Test getting CPU config."""
        self.engine.evaluate(resources=self.resources, platform="local")

        cpu_config = self.engine.get_cpu_config()
        assert len(cpu_config) > 0
        assert "spark.executor.cores" in cpu_config

    def test_get_sql_config(self):
        """Test getting SQL config."""
        self.engine.evaluate(resources=self.resources, platform="local")

        sql_config = self.engine.get_sql_config()
        assert len(sql_config) > 0
        assert "spark.sql.adaptive.enabled" in sql_config

    def test_validate_config(self):
        """Test config validation."""
        self.engine.evaluate(resources=self.resources, platform="local")

        errors = self.engine.validate_config()
        # Should return list (may be empty if all valid)
        assert isinstance(errors, list)

    def test_base_resource_calculation(self):
        """Test base resource calculation."""
        from spark_optima.core.heuristics.context import EvaluationContext

        context = EvaluationContext(
            resources=self.resources,
            platform="local",
        )

        self.engine._calculate_base_resources(context, "local")

        assert context.num_executors > 0
        assert context.executor_cores > 0
        assert context.executor_memory_gb > 0
        assert context.driver_memory_gb > 0

    def test_format_value(self):
        """Test value formatting."""
        assert self.engine._format_value(1024**3, "bytes") == "1g"
        assert self.engine._format_value(3600, "duration") == "1h"
        assert self.engine._format_value(42, "integer") == 42
        assert self.engine._format_value(3.14, "float") == 3.14
        assert self.engine._format_value("true", "boolean") is True
        assert self.engine._format_value("false", "boolean") is False


class TestConfigDatabaseIntegration:
    """Test cases for ConfigDatabase integration with heuristics."""

    def test_get_heuristic_config(self):
        """Test getting heuristic config from database."""
        db = ConfigDatabase()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64)

        config = db.get_heuristic_config(
            version="3.5.0",
            resources=resources,
            platform="local",
        )

        assert len(config) > 0
        assert "spark.driver.memory" in config
        assert "spark.executor.memory" in config

    def test_get_heuristic_config_invalid_version(self):
        """Test getting heuristic config for invalid version."""
        db = ConfigDatabase()
        resources = ResourceSpec(cpu_cores=16, memory_gb=64)

        config = db.get_heuristic_config(
            version="invalid.version",
            resources=resources,
            platform="local",
        )

        assert config == {}
