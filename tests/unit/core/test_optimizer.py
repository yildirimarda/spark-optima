# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the core Optimizer class.

This module contains comprehensive tests for the main Optimizer functionality
including initialization, optimization workflow, and result handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.core.optimizer import Optimizer
from spark_optima.core.result import OptimizationResult
from spark_optima.platforms.models import ResourceSpec

if TYPE_CHECKING:
    from pathlib import Path


class TestOptimizerInitialization:
    """Test cases for Optimizer initialization."""

    def test_optimizer_initialization_valid_platform(self) -> None:
        """Test optimizer initialization with valid platform."""
        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        assert optimizer.platform == "local"
        assert optimizer.spark_version == "3.5.0"
        assert optimizer.optimization_mode == "simulation"

    def test_optimizer_initialization_all_valid_platforms(self) -> None:
        """Test optimizer initialization with all valid platforms."""
        valid_platforms = ["local", "databricks", "aws_glue", "azure_synapse"]
        for platform in valid_platforms:
            optimizer = Optimizer(platform=platform)
            assert optimizer.platform == platform

    def test_optimizer_initialization_invalid_platform(self) -> None:
        """Test optimizer initialization with invalid platform raises error."""
        with pytest.raises(ValueError, match="Invalid platform"):
            Optimizer(platform="invalid_platform")

    def test_optimizer_initialization_invalid_mode(self) -> None:
        """Test optimizer initialization with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Optimizer(platform="local", optimization_mode="invalid_mode")

    def test_optimizer_initialization_valid_modes(self) -> None:
        """Test optimizer initialization with valid modes."""
        valid_modes = ["simulation", "execution"]
        for mode in valid_modes:
            optimizer = Optimizer(platform="local", optimization_mode=mode)
            assert optimizer.optimization_mode == mode

    def test_optimizer_initialization_default_version(self) -> None:
        """Test optimizer uses default Spark version."""
        optimizer = Optimizer(platform="local")
        assert optimizer.spark_version == "3.5.0"

    def test_optimizer_initialization_unsupported_version(self) -> None:
        """Test optimizer with unsupported Spark version."""
        with pytest.raises(ValueError, match="Spark version .* not available"):
            Optimizer(platform="local", spark_version="9.9.9")

    def test_optimizer_repr(self) -> None:
        """Test optimizer string representation."""
        optimizer = Optimizer(platform="databricks", spark_version="3.5.0")
        repr_str = repr(optimizer)
        assert "Optimizer" in repr_str
        assert "databricks" in repr_str
        assert "3.5.0" in repr_str
        assert "simulation" in repr_str


class TestOptimizerOptimize:
    """Test cases for the optimize method."""

    @pytest.fixture
    def sample_spark_code(self, tmp_path: Path) -> Path:
        """Create a sample Spark code file."""
        code_file = tmp_path / "test_job.py"
        code_content = """
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()
df = spark.read.parquet("data.parquet")
df.filter(df.col1 > 10).write.parquet("output.parquet")
"""
        code_file.write_text(code_content)
        return code_file

    def test_optimize_nonexistent_file(self, tmp_path: Path) -> None:
        """Test optimize with non-existent file raises error."""
        optimizer = Optimizer(platform="local")
        nonexistent_file = tmp_path / "nonexistent.py"

        with pytest.raises(FileNotFoundError, match="Code file not found"):
            optimizer.optimize(code_path=nonexistent_file)

    def test_optimize_with_existing_file(self, tmp_path: Path) -> None:
        """Test optimize with existing file returns result."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test_spark_job.py"
        test_file.write_text("# Test Spark job\n")

        result = optimizer.optimize(code_path=test_file)

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.configuration, dict)
        assert isinstance(result.code_suggestions, list)

    def test_optimize_without_code_path(self) -> None:
        """Test optimize without code path (heuristics only)."""
        optimizer = Optimizer(platform="local")

        result = optimizer.optimize()

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.configuration, dict)
        assert len(result.configuration) > 0

    def test_optimize_with_resources(self, sample_spark_code: Path) -> None:
        """Test optimize with resource specifications."""
        optimizer = Optimizer(platform="local")
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = optimizer.optimize(
            code_path=sample_spark_code,
            resources=resources,
        )

        assert isinstance(result, OptimizationResult)
        assert result.metadata["resources"]["cpu_cores"] == 8
        assert result.metadata["resources"]["memory_gb"] == 32

    def test_optimize_with_data_profile(self, sample_spark_code: Path) -> None:
        """Test optimize with data profile."""
        optimizer = Optimizer(platform="local")
        data_profile = {"size_gb": 100, "format": "parquet"}

        result = optimizer.optimize(
            code_path=sample_spark_code,
            data_profile=data_profile,
        )

        assert isinstance(result, OptimizationResult)
        assert result.metadata["data_profile"]["size_gb"] == 100

    def test_optimize_without_bayesian(self, sample_spark_code: Path) -> None:
        """Test optimize with Bayesian optimization disabled."""
        optimizer = Optimizer(platform="local")

        result = optimizer.optimize(
            code_path=sample_spark_code,
            use_bayesian=False,
        )

        assert isinstance(result, OptimizationResult)
        assert not result.metadata.get("bayesian_used", True)

    @patch("spark_optima.core.optimizer.BayesianOptimizer")
    def test_optimize_with_bayesian_failure_fallback(
        self,
        mock_bayesian: MagicMock,
        sample_spark_code: Path,
    ) -> None:
        """Test fallback to heuristic when Bayesian fails."""
        mock_optimizer = MagicMock()
        # The except block catches (RuntimeError, ValueError, KeyError, AttributeError, TypeError)
        mock_optimizer.optimize.side_effect = RuntimeError("Bayesian error")
        mock_bayesian.return_value = mock_optimizer

        optimizer = Optimizer(platform="local")

        result = optimizer.optimize(
            code_path=sample_spark_code,
            use_bayesian=True,
        )

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.configuration, dict)


class TestOptimizerResultHandling:
    """Test cases for result handling methods."""

    @pytest.fixture
    def optimizer_with_result(self, tmp_path: Path) -> Optimizer:
        """Create optimizer with a completed optimization result."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")
        optimizer.optimize(code_path=test_file)
        return optimizer

    def test_get_heuristic_config_after_optimize(
        self,
        optimizer_with_result: Optimizer,
    ) -> None:
        """Test getting heuristic config after optimization."""
        config = optimizer_with_result.get_heuristic_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_get_heuristic_config_before_optimize(self) -> None:
        """Test getting heuristic config before optimization returns None."""
        optimizer = Optimizer(platform="local")
        config = optimizer.get_heuristic_config()
        assert config is None

    def test_get_last_result_after_optimize(
        self,
        optimizer_with_result: Optimizer,
    ) -> None:
        """Test getting last result after optimization."""
        result = optimizer_with_result.get_last_result()
        assert result is not None
        assert isinstance(result, OptimizationResult)

    def test_get_last_result_before_optimize(self) -> None:
        """Test getting last result before optimization returns None."""
        optimizer = Optimizer(platform="local")
        result = optimizer.get_last_result()
        assert result is None

    def test_get_bayesian_result_without_bayesian(self, tmp_path: Path) -> None:
        """Test getting Bayesian result when not used."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")
        optimizer.optimize(code_path=test_file, use_bayesian=False)

        result = optimizer.get_bayesian_result()
        assert result is None


class TestOptimizerConfiguration:
    """Test cases for configuration-related functionality."""

    def test_optimizer_has_config_database(self) -> None:
        """Test optimizer initializes config database."""
        optimizer = Optimizer(platform="local")
        assert optimizer.config_database is not None

    def test_optimizer_has_config_set(self) -> None:
        """Test optimizer loads config set for version."""
        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        assert optimizer.config_set is not None
        assert optimizer.config_set.version == "3.5.0"

    def test_optimizer_has_heuristic_engine(self) -> None:
        """Test optimizer initializes heuristic engine."""
        optimizer = Optimizer(platform="local")
        assert optimizer.heuristic_engine is not None


class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_optimize_with_empty_file(self, tmp_path: Path) -> None:
        """Test optimize with empty file."""
        optimizer = Optimizer(platform="local")
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        result = optimizer.optimize(code_path=empty_file)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_invalid_code(self, tmp_path: Path) -> None:
        """Test optimize with invalid Python code."""
        optimizer = Optimizer(platform="local")
        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text("this is not valid python {{{")

        # Should not raise, but may have empty suggestions
        result = optimizer.optimize(code_path=invalid_file)
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_zero_resources(self, tmp_path: Path) -> None:
        """Test optimize with minimal resources."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")
        resources = ResourceSpec(cpu_cores=1, memory_gb=1)

        result = optimizer.optimize(
            code_path=test_file,
            resources=resources,
        )
        assert isinstance(result, OptimizationResult)

    def test_optimize_with_large_resources(self, tmp_path: Path) -> None:
        """Test optimize with large resources."""
        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")
        resources = ResourceSpec(cpu_cores=64, memory_gb=512)

        result = optimizer.optimize(
            code_path=test_file,
            resources=resources,
        )
        assert isinstance(result, OptimizationResult)

    def test_multiple_optimizations_same_instance(self, tmp_path: Path) -> None:
        """Test running multiple optimizations with same optimizer."""
        optimizer = Optimizer(platform="local")

        file1 = tmp_path / "test1.py"
        file1.write_text("# Test 1")
        result1 = optimizer.optimize(code_path=file1)

        file2 = tmp_path / "test2.py"
        file2.write_text("# Test 2")
        result2 = optimizer.optimize(code_path=file2)

        assert isinstance(result1, OptimizationResult)
        assert isinstance(result2, OptimizationResult)
        # Last result should be from second optimization
        assert optimizer.get_last_result() == result2


class TestOptimizerPlatformSpecific:
    """Test platform-specific behavior."""

    @pytest.mark.parametrize("platform", ["local", "databricks", "aws_glue", "azure_synapse"])
    def test_optimize_on_all_platforms(self, platform: str, tmp_path: Path) -> None:
        """Test optimization works on all platforms."""
        optimizer = Optimizer(platform=platform)
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")

        result = optimizer.optimize(code_path=test_file)

        assert isinstance(result, OptimizationResult)
        assert result.platform_specific["platform"] == platform

    def test_platform_specific_config_structure(self, tmp_path: Path) -> None:
        """Test platform-specific config has correct structure."""
        optimizer = Optimizer(platform="databricks")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")

        result = optimizer.optimize(code_path=test_file)

        assert "spark_version" in result.platform_specific
        assert "spark_config" in result.platform_specific

        # Databricks-specific fields
        if optimizer.platform == "databricks":
            assert "cluster_config" in result.platform_specific


class TestOptimizerValidationErrors:
    """Test validation error handling (covers lines 179-180)."""

    @patch("spark_optima.core.optimizer.HeuristicEngine")
    def test_optimize_with_validation_errors(
        self,
        mock_heuristic_engine_class: MagicMock,
        tmp_path: Path,
        caplog,
    ) -> None:
        """Test that validation errors are logged as warnings."""
        # Setup mock heuristic engine
        mock_engine = MagicMock()
        mock_heuristic_engine_class.return_value = mock_engine
        mock_engine.evaluate.return_value = {"spark.executor.memory": "4g"}
        mock_engine.validate_config.return_value = [
            "Invalid memory format",
            "Cores too high for resources",
        ]

        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")

        result = optimizer.optimize(code_path=test_file)

        # Verify result is still returned despite validation errors
        assert isinstance(result, OptimizationResult)

        # Check that warnings were logged
        assert "Invalid memory format" in caplog.text
        assert "Cores too high for resources" in caplog.text

    @patch("spark_optima.core.optimizer.HeuristicEngine")
    def test_optimize_with_empty_validation_errors(
        self,
        mock_heuristic_engine_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test optimization with no validation errors."""
        # Setup mock heuristic engine
        mock_engine = MagicMock()
        mock_heuristic_engine_class.return_value = mock_engine
        mock_engine.evaluate.return_value = {"spark.executor.memory": "4g"}
        mock_engine.validate_config.return_value = []  # No errors

        optimizer = Optimizer(platform="local")
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test")

        result = optimizer.optimize(code_path=test_file)

        assert isinstance(result, OptimizationResult)


class TestOptimizerBayesianConfigSetNone:
    """Test edge case when config_set is None (covers line 256)."""

    def test_run_bayesian_optimization_config_set_none(self) -> None:
        """Test that ValueError is raised when config_set is None (line 256)."""
        optimizer = Optimizer(platform="local")

        # Manually set config_set to None to trigger line 256
        optimizer.config_set = None

        # Mock the heuristic config and resources
        heuristic_config = {"spark.executor.memory": "4g"}
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)

        # Call _run_bayesian_optimization and expect ValueError
        with pytest.raises(ValueError, match="config_set is None"):
            optimizer._run_bayesian_optimization(
                heuristic_config=heuristic_config,
                resources=resources,
                data_profile=None,
                n_trials=10,
                timeout_minutes=None,
                objectives=None,
            )

    @patch("spark_optima.core.optimizer.ConfigDatabase")
    def test_optimize_bayesian_config_set_none_after_init(
        self,
        mock_config_db_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test optimize when config_set becomes None unexpectedly."""
        # Setup mock to return None for get_config_set
        mock_db = MagicMock()
        mock_db.get_config_set.return_value = None
        mock_config_db_class.return_value = mock_db

        # Initializing should fail since config_set will be None
        with pytest.raises(ValueError, match="not available"):
            Optimizer(platform="local", spark_version="3.5.0")


class TestOptimizerMultiObjective:
    """Multi-objective optimization surfaced end-to-end (Workstream N)."""

    @pytest.fixture
    def sample_code(self, tmp_path: Path) -> Path:
        """Create a small Spark code file for optimization runs."""
        code_file = tmp_path / "multi_obj_job.py"
        code_file.write_text(
            "from pyspark.sql import SparkSession\n"
            "spark = SparkSession.builder.getOrCreate()\n"
            "df = spark.read.parquet('/data/input')\n"
            "df.groupBy('key').count().write.parquet('/data/output')\n",
        )
        return code_file

    def test_multi_objective_run_populates_pareto_metadata(self, sample_code: Path) -> None:
        """A 2-objective run completes and persists the Pareto frontier in metadata."""
        optimizer = Optimizer(platform="local", spark_version="3.5.0")

        result = optimizer.optimize(
            code_path=sample_code,
            data_profile={"size_gb": 1, "format": "parquet"},
            use_bayesian=True,
            bayesian_trials=5,
            objectives=["minimize_time", "minimize_cost"],
        )

        # Bayesian must have actually run (no silent heuristic fallback)
        assert optimizer.get_bayesian_result() is not None
        assert result.metadata["bayesian_used"] is True

        assert result.metadata["objectives"] == ["minimize_time", "minimize_cost"]
        frontier = result.metadata["pareto_frontier"]
        assert isinstance(frontier, list)
        assert len(frontier) >= 1
        for point in frontier:
            assert set(point.keys()) == {"trial_number", "objective_values", "configuration"}
            assert isinstance(point["trial_number"], int)
            assert set(point["objective_values"].keys()) == {"minimize_time", "minimize_cost"}
            assert isinstance(point["configuration"], dict)
            assert point["configuration"]

    def test_multi_objective_metadata_survives_json_round_trip(self, sample_code: Path) -> None:
        """Pareto metadata survives result.to_dict() -> JSON -> dict (history/compare path)."""
        import json

        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        result = optimizer.optimize(
            code_path=sample_code,
            data_profile={"size_gb": 1, "format": "parquet"},
            use_bayesian=True,
            bayesian_trials=4,
            objectives=["minimize_time", "minimize_memory"],
        )

        round_trip = json.loads(json.dumps(result.to_dict()))
        assert round_trip["metadata"]["pareto_frontier"] == result.metadata["pareto_frontier"]
        assert round_trip["metadata"]["objectives"] == ["minimize_time", "minimize_memory"]

    def test_single_objective_run_has_no_pareto_key(self, sample_code: Path) -> None:
        """Single-objective runs leave metadata unchanged (no pareto_frontier key)."""
        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        result = optimizer.optimize(
            code_path=sample_code,
            data_profile={"size_gb": 1, "format": "parquet"},
            use_bayesian=True,
            bayesian_trials=3,
        )

        assert "pareto_frontier" not in result.metadata

    def test_pareto_frontier_capped_at_50_points(self) -> None:
        """The persisted Pareto frontier is capped at MAX_PARETO_POINTS points."""
        from spark_optima.core.bayesian.models import BayesianOptimizationResult, ParetoPoint
        from spark_optima.core.optimizer import MAX_PARETO_POINTS

        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        frontier = [
            ParetoPoint(
                trial_number=i,
                objective_values={"minimize_time": float(i), "minimize_cost": float(100 - i)},
                configuration={"spark.executor.memory": "4g"},
            )
            for i in range(MAX_PARETO_POINTS + 10)
        ]
        optimizer._bayesian_result = BayesianOptimizationResult(
            best_config={"spark.executor.memory": "4g"},
            pareto_frontier=frontier,
            metadata={"objectives": ["minimize_time", "minimize_cost"]},
        )

        result = optimizer._build_result(
            final_config={"spark.executor.memory": "4g"},
            heuristic_config={"spark.executor.memory": "4g"},
            resources=ResourceSpec(cpu_cores=4, memory_gb=16),
            data_profile=None,
        )

        persisted = result.metadata["pareto_frontier"]
        assert len(persisted) == MAX_PARETO_POINTS
        # First MAX_PARETO_POINTS points are kept, in order
        assert persisted[0]["trial_number"] == 0
        assert persisted[-1]["trial_number"] == MAX_PARETO_POINTS - 1


class TestOptimizerProgressCallbackPassthrough:
    """Tests that Optimizer.optimize forwards progress_callback (Workstream Z)."""

    @patch("spark_optima.core.optimizer.BayesianOptimizer")
    def test_progress_callback_forwarded_to_bayesian(self, mock_bayesian: MagicMock, tmp_path: Path) -> None:
        """The progress_callback kwarg reaches BayesianOptimizer.optimize."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            best_config={"spark.executor.memory": "4g"},
            all_trials=[],
            pareto_frontier=[],
            metadata={"objectives": ["minimize_time"]},
        )
        mock_bayesian.return_value = mock_instance
        callback = MagicMock()
        code_file = tmp_path / "job.py"
        code_file.write_text("# Spark job")

        optimizer = Optimizer(platform="local")
        optimizer.optimize(code_path=code_file, use_bayesian=True, progress_callback=callback)

        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["progress_callback"] is callback

    @patch("spark_optima.core.optimizer.BayesianOptimizer")
    def test_progress_callback_defaults_to_none(self, mock_bayesian: MagicMock, tmp_path: Path) -> None:
        """Without the kwarg, None is forwarded (no behavior change)."""
        mock_instance = MagicMock()
        mock_instance.optimize.return_value = MagicMock(
            best_config={"spark.executor.memory": "4g"},
            all_trials=[],
            pareto_frontier=[],
            metadata={"objectives": ["minimize_time"]},
        )
        mock_bayesian.return_value = mock_instance
        code_file = tmp_path / "job.py"
        code_file.write_text("# Spark job")

        optimizer = Optimizer(platform="local")
        optimizer.optimize(code_path=code_file, use_bayesian=True)

        _, kwargs = mock_instance.optimize.call_args
        assert kwargs["progress_callback"] is None

    def test_progress_callback_receives_events_end_to_end(self, tmp_path: Path) -> None:
        """A real (small) run delivers per-trial events through the Optimizer."""
        events: list[dict] = []
        code_file = tmp_path / "job.py"
        code_file.write_text(
            "from pyspark.sql import SparkSession\n"
            "spark = SparkSession.builder.getOrCreate()\n"
            "df = spark.read.parquet('/data/input')\n",
        )

        optimizer = Optimizer(platform="local", spark_version="3.5.0")
        optimizer.optimize(
            code_path=code_file,
            data_profile={"size_gb": 1, "format": "parquet"},
            use_bayesian=True,
            bayesian_trials=2,
            progress_callback=events.append,
        )

        assert len(events) == 2
        assert all(event["n_trials"] == 2 for event in events)
        assert {"trial_number", "trials_completed", "state", "best_value"} <= set(events[0].keys())
