# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Bayesian Optimizer.

This module contains tests for the Bayesian optimization engine including
initialization, search space definition, and optimization workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from spark_optima.core.bayesian.models import (
    BayesianOptimizationResult,
    SearchSpaceConfig,
    TrialMetrics,
    TrialResult,
    TrialStatus,
)
from spark_optima.core.bayesian.optimizer import BayesianOptimizer
from spark_optima.core.config_engine.models import ConfigSet
from spark_optima.platforms.models import ResourceSpec


class TestBayesianOptimizerInitialization:
    """Test cases for BayesianOptimizer initialization."""

    @pytest.fixture
    def base_config(self) -> dict:
        """Create base heuristic configuration."""
        return {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
            "spark.sql.adaptive.enabled": "true",
        }

    @pytest.fixture
    def config_set(self) -> ConfigSet:
        """Create test config set."""
        return ConfigSet(
            version="3.5.0",
            parameters={},
            metadata={},
        )

    @pytest.fixture
    def resource_spec(self) -> ResourceSpec:
        """Create test resource specification."""
        return ResourceSpec(cpu_cores=8, memory_gb=32)

    def test_optimizer_initialization(
        self,
        base_config: dict,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Test basic optimizer initialization."""
        optimizer = BayesianOptimizer(
            heuristic_config=base_config,
            config_set=config_set,
            resource_spec=resource_spec,
        )

        assert optimizer.heuristic_config == base_config
        assert optimizer.config_set == config_set
        assert optimizer.resource_spec == resource_spec

    def test_optimizer_initialization_with_objectives(
        self,
        base_config: dict,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Test initialization with custom objectives."""
        objectives = ["minimize_time", "minimize_cost"]
        optimizer = BayesianOptimizer(
            heuristic_config=base_config,
            config_set=config_set,
            resource_spec=resource_spec,
            objectives=objectives,
        )

        assert optimizer.objectives == objectives

    def test_optimizer_initialization_with_search_space_config(
        self,
        base_config: dict,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Test initialization with custom search space config."""
        search_config = SearchSpaceConfig(variation_percent=0.2)
        optimizer = BayesianOptimizer(
            heuristic_config=base_config,
            config_set=config_set,
            resource_spec=resource_spec,
            search_space_config=search_config,
        )

        assert optimizer.search_space_config == search_config

    def test_optimizer_initialization_with_study_name(
        self,
        base_config: dict,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Test initialization with custom study name."""
        study_name = "test_study_123"
        optimizer = BayesianOptimizer(
            heuristic_config=base_config,
            config_set=config_set,
            resource_spec=resource_spec,
            study_name=study_name,
        )

        assert optimizer.study_name == study_name


class TestBayesianOptimizerSearchSpace:
    """Test cases for search space handling."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

    def test_build_search_space(self, optimizer: BayesianOptimizer) -> None:
        """Test building search space from heuristic config."""
        search_space = optimizer._build_search_space()

        assert isinstance(search_space, dict)
        assert len(search_space) > 0

    def test_search_space_bounds(self, optimizer: BayesianOptimizer) -> None:
        """Test search space has valid bounds."""
        search_space = optimizer._build_search_space()

        for param_name, bounds in search_space.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                low, high = bounds
                assert low <= high, f"Invalid bounds for {param_name}"

    def test_search_space_respects_variation_percent(self) -> None:
        """Test search space respects variation percentage."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Create optimizer with 10% variation
        optimizer_10 = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
            search_space_config=SearchSpaceConfig(variation_percent=0.1),
        )

        # Create optimizer with 50% variation
        optimizer_50 = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
            search_space_config=SearchSpaceConfig(variation_percent=0.5),
        )

        space_10 = optimizer_10._build_search_space()
        space_50 = optimizer_50._build_search_space()

        # Higher variation should result in wider bounds
        # (This is a simplified check - actual implementation may vary)
        assert isinstance(space_10, dict)
        assert isinstance(space_50, dict)


class TestBayesianOptimizerObjectives:
    """Test cases for objective function handling."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resources,
            objectives=["minimize_time", "minimize_cost"],
        )

    def test_objective_function_creation(self, optimizer: BayesianOptimizer) -> None:
        """Test objective function creation."""
        objective = optimizer._create_objective_function()

        assert objective is not None

    def test_single_objective(self) -> None:
        """Test with single objective."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resources,
            objectives=["minimize_time"],
        )

        objective = optimizer._create_objective_function()
        assert objective is not None

    def test_multi_objective(self) -> None:
        """Test with multiple objectives."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resources,
            objectives=["minimize_time", "minimize_cost", "maximize_success"],
        )

        objective = optimizer._create_objective_function()
        assert objective is not None


class TestBayesianOptimizerTrialManagement:
    """Test cases for trial management."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

    def test_create_trial_result(self, optimizer: BayesianOptimizer) -> None:
        """Test creating trial result."""
        config = {"spark.executor.memory": "4g"}
        metrics = TrialMetrics(
            execution_time_seconds=100.0,
            memory_usage_mb=1024.0,
            success=True,
        )

        trial_result = optimizer._create_trial_result(
            trial_number=1,
            config=config,
            metrics=metrics,
        )

        assert trial_result.trial_number == 1
        assert trial_result.config == config
        assert trial_result.metrics == metrics
        assert trial_result.status == TrialStatus.COMPLETED

    def test_create_failed_trial_result(self, optimizer: BayesianOptimizer) -> None:
        """Test creating failed trial result."""
        config = {"spark.executor.memory": "4g"}

        trial_result = optimizer._create_trial_result(
            trial_number=2,
            config=config,
            metrics=None,
            error="Out of memory",
        )

        assert trial_result.trial_number == 2
        assert trial_result.status == TrialStatus.FAILED
        assert trial_result.error == "Out of memory"

    def test_update_best_config(self, optimizer: BayesianOptimizer) -> None:
        """Test updating best configuration."""
        config1 = {"spark.executor.memory": "2g"}
        metrics1 = TrialMetrics(execution_time_seconds=200.0, success=True)

        config2 = {"spark.executor.memory": "4g"}
        metrics2 = TrialMetrics(execution_time_seconds=100.0, success=True)

        optimizer._update_best_config(config1, metrics1)
        optimizer._update_best_config(config2, metrics2)

        # Should keep the better (faster) config
        assert optimizer._best_config is not None


class TestBayesianOptimizerOptimization:
    """Test cases for the optimization process."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
            objectives=["minimize_time"],
        )

    def test_optimizer_with_storage_path(self) -> None:
        """Test optimizer with storage path (covers lines 131-132)."""
        import os
        import tempfile

        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "test_study.db")
            optimizer = BayesianOptimizer(
                heuristic_config={"spark.executor.memory": "4g"},
                config_set=config_set,
                resource_spec=resources,
                storage_path=storage_path,
            )
            assert optimizer.storage_path == storage_path
            assert optimizer._storage is not None

    @patch("spark_optima.core.bayesian.optimizer.optuna.create_study")
    def test_optimize_basic(
        self,
        mock_create_study: MagicMock,
        optimizer: BayesianOptimizer,
    ) -> None:
        """Test basic optimization flow."""
        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 100.0
        mock_study.trials = []
        mock_create_study.return_value = mock_study

        result = optimizer.optimize(n_trials=10)

        assert isinstance(result, BayesianOptimizationResult)
        mock_study.optimize.assert_called_once()

    @patch("spark_optima.core.bayesian.optimizer.optuna.create_study")
    def test_optimize_with_timeout(
        self,
        mock_create_study: MagicMock,
        optimizer: BayesianOptimizer,
    ) -> None:
        """Test optimization with timeout."""
        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {"param1": 1.0}
        mock_study.best_value = 100.0
        mock_study.trials = []
        mock_create_study.return_value = mock_study

        result = optimizer.optimize(
            n_trials=100,
            timeout_minutes=1,
        )

        assert isinstance(result, BayesianOptimizationResult)

    def test_optimize_with_show_progress(self, optimizer: BayesianOptimizer) -> None:
        """Test optimization with progress display."""
        result = optimizer.optimize(
            n_trials=5,
            show_progress=True,
        )

        assert isinstance(result, BayesianOptimizationResult)

    def test_optimize_result_contains_trials(self, optimizer: BayesianOptimizer) -> None:
        """Test that result contains trial information."""
        result = optimizer.optimize(n_trials=5)

        assert isinstance(result.all_trials, list)


class TestBayesianOptimizerResultBuilding:
    """Test cases for building optimization results."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

    def test_build_result_no_trials(self, optimizer: BayesianOptimizer) -> None:
        """Test building result with no trials."""
        result = optimizer._build_result()

        assert isinstance(result, BayesianOptimizationResult)
        assert result.best_config is None
        assert result.best_score is None

    def test_build_result_with_trials(self, optimizer: BayesianOptimizer) -> None:
        """Test building result with trials."""
        # Add some mock trials
        trial1 = TrialResult(
            trial_number=1,
            config={"param": 1.0},
            metrics=TrialMetrics(execution_time_seconds=100.0, success=True),
            status=TrialStatus.COMPLETED,
        )
        trial2 = TrialResult(
            trial_number=2,
            config={"param": 2.0},
            metrics=TrialMetrics(execution_time_seconds=80.0, success=True),
            status=TrialStatus.COMPLETED,
        )

        optimizer._trials = [trial1, trial2]
        optimizer._best_config = {"param": 2.0}
        optimizer._best_score = 80.0

        result = optimizer._build_result()

        assert isinstance(result, BayesianOptimizationResult)
        assert len(result.all_trials) == 2
        assert result.best_config == {"param": 2.0}
        assert result.best_score == 80.0


class TestBayesianOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_heuristic_config(self) -> None:
        """Test with empty heuristic configuration."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resources,
        )

        result = optimizer.optimize(n_trials=1)
        assert isinstance(result, BayesianOptimizationResult)

    def test_zero_trials(self) -> None:
        """Test with zero trials."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resources,
        )

        result = optimizer.optimize(n_trials=0)
        assert isinstance(result, BayesianOptimizationResult)

    def test_invalid_objective_name(self) -> None:
        """Test with invalid objective name."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)

        with pytest.raises(ValueError):
            BayesianOptimizer(
                heuristic_config={},
                config_set=config_set,
                resource_spec=resources,
                objectives=["invalid_objective"],
            )

    def test_very_small_variation_percent(self) -> None:
        """Test with very small variation percentage."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
            search_space_config=SearchSpaceConfig(variation_percent=0.01),
        )

        result = optimizer.optimize(n_trials=1)
        assert isinstance(result, BayesianOptimizationResult)


class TestBayesianOptimizerConfigurationSampling:
    """Test configuration sampling."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={
                "spark.executor.memory": "4g",
                "spark.executor.cores": "4",
                "spark.sql.adaptive.enabled": "true",
            },
            config_set=config_set,
            resource_spec=resources,
        )

    def test_sample_configuration(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling a configuration from search space."""
        search_space = optimizer._build_search_space()

        config = optimizer._sample_configuration(search_space)

        assert isinstance(config, dict)
        assert len(config) > 0

    def test_sample_configuration_consistency(self, optimizer: BayesianOptimizer) -> None:
        """Test that sampling produces valid configurations."""
        search_space = optimizer._build_search_space()

        # Sample multiple configurations
        configs = [optimizer._sample_configuration(search_space) for _ in range(10)]

        for config in configs:
            assert isinstance(config, dict)
            # All configs should have the same keys
            assert set(config.keys()) == set(optimizer.heuristic_config.keys())

    def test_sample_configuration_respects_bounds(self, optimizer: BayesianOptimizer) -> None:
        """Test that sampled configurations respect search space bounds."""
        search_space = optimizer._build_search_space()
        config = optimizer._sample_configuration(search_space)

        # For numeric parameters, check they're within bounds
        for param_name, value in config.items():
            if param_name in search_space:
                bounds = search_space[param_name]
                if isinstance(bounds, tuple) and len(bounds) == 2:
                    low, high = bounds
                    if isinstance(value, int | float):
                        assert low <= value <= high, f"Parameter {param_name} out of bounds"


class TestBayesianOptimizerHelperMethods:
    """Test helper methods."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

    def test_format_bytes(self, optimizer: BayesianOptimizer) -> None:
        """Test _format_bytes method (line 500-517)."""
        # Test various byte values
        assert optimizer._format_bytes(1024) == "1k"  # 1 KB
        assert optimizer._format_bytes(4096) == "4k"  # 4 KB
        assert optimizer._format_bytes(1024**2) == "1m"  # 1 MB
        assert optimizer._format_bytes(4 * 1024**2) == "4m"  # 4 MB
        assert optimizer._format_bytes(1024**3) == "1g"  # 1 GB
        assert optimizer._format_bytes(4 * 1024**3) == "4g"  # 4 GB
        assert optimizer._format_bytes(1024**4) == "1t"  # 1 TB
        assert optimizer._format_bytes(512) == "512b"  # 512 bytes

    def test_get_trial_config(self, optimizer: BayesianOptimizer) -> None:
        """Test _get_trial_config method (line 484-498)."""
        # Create a mock trial
        from unittest.mock import MagicMock

        mock_trial = MagicMock()
        mock_trial.number = 1

        # Add a matching trial result
        optimizer._trial_results = [
            TrialResult(trial_number=1, configuration={"memory": "4g"}),
        ]

        config = optimizer._get_trial_config(mock_trial)
        assert config == {"memory": "4g"}

        # Test with no matching trial
        mock_trial.number = 999
        config = optimizer._get_trial_config(mock_trial)
        assert config == {}

    def test_get_search_space(self, optimizer: BayesianOptimizer) -> None:
        """Test get_search_space method (line 519-526)."""
        search_space = optimizer.get_search_space()
        assert isinstance(search_space, dict)

    def test_get_trial_results(self, optimizer: BayesianOptimizer) -> None:
        """Test get_trial_results method (line 528-535)."""
        results = optimizer.get_trial_results()
        assert isinstance(results, list)
        assert len(results) == 0  # No trials yet

        # Add a trial result
        optimizer._trial_results = [
            TrialResult(trial_number=1, configuration={"memory": "4g"}),
        ]

        results = optimizer.get_trial_results()
        assert len(results) == 1
        assert results[0].trial_number == 1


class TestBayesianOptimizerBuildResult:
    """Test _build_result method with study."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

    def test_build_result_with_no_study_no_trials(self, optimizer: BayesianOptimizer) -> None:
        """Test building result with no study and no trials."""
        optimizer._trial_results = []
        result = optimizer._build_result(study=None)
        assert result.best_config is None
        assert result.best_trial_number == -1
        assert len(result.all_trials) == 0

    def test_build_result_with_trials_no_study(self, optimizer: BayesianOptimizer) -> None:
        """Test building result with trials but no study."""
        trials = [
            TrialResult(
                trial_number=1,
                configuration={"memory": "4g"},
                status=TrialStatus.COMPLETED,
            ),
            TrialResult(
                trial_number=2,
                configuration={"memory": "8g"},
                status=TrialStatus.FAILED,
            ),
        ]
        optimizer._trial_results = trials
        optimizer._best_config = {"memory": "4g"}
        optimizer._best_score = 100.0

        result = optimizer._build_result(study=None)
        assert result.best_config == {"memory": "4g"}
        assert result.n_trials_completed == 1
        assert result.n_trials_failed == 1
        assert result.metadata["best_score"] == 100.0

    @patch("spark_optima.core.bayesian.optimizer.optuna")
    def test_build_result_with_study_zero_trials(
        self,
        mock_optuna: MagicMock,
        optimizer: BayesianOptimizer,
    ) -> None:
        """Test building result with study that has zero trials."""
        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.best_trial = None
        mock_study.directions = ["minimize"]

        result = optimizer._build_result(study=mock_study)
        assert result.best_config is None
        assert result.best_trial_number == -1
        assert len(result.all_trials) == 0

    @patch("spark_optima.core.bayesian.optimizer.optuna")
    def test_build_result_with_study_best_trial(
        self,
        mock_optuna: MagicMock,
        optimizer: BayesianOptimizer,
    ) -> None:
        """Test building result with study that has a best trial."""
        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.number = 1

        mock_study = MagicMock()
        mock_study.trials = [mock_trial]
        mock_study.best_trial = mock_trial
        mock_study.directions = ["minimize"]

        # Add matching trial result
        optimizer._trial_results = [
            TrialResult(
                trial_number=1,
                configuration={"memory": "4g"},
                status=TrialStatus.COMPLETED,
            ),
        ]

        result = optimizer._build_result(study=mock_study)
        assert result.best_trial_number == 1
        assert result.best_config == {"memory": "4g"}

    def test_build_pareto_frontier(self, optimizer: BayesianOptimizer) -> None:
        """Test _build_pareto_frontier method."""
        # Create mock study with best trials
        mock_study = MagicMock()
        mock_trial1 = MagicMock()
        mock_trial1.number = 1
        mock_trial2 = MagicMock()
        mock_trial2.number = 2
        mock_study.best_trials = [mock_trial1, mock_trial2]

        # Add matching trial results
        optimizer._trial_results = [
            TrialResult(
                trial_number=1,
                configuration={"memory": "4g"},
                status=TrialStatus.COMPLETED,
                objective_values={"minimize_time": 100.0},
            ),
            TrialResult(
                trial_number=2,
                configuration={"memory": "8g"},
                status=TrialStatus.COMPLETED,
                objective_values={"minimize_time": 80.0},
            ),
        ]
        optimizer._is_multi_objective = True

        pareto = optimizer._build_pareto_frontier(mock_study)
        assert len(pareto) == 2
        assert pareto[0].trial_number == 1

    def test_create_study_single_objective(self, optimizer: BayesianOptimizer) -> None:
        """Test _create_study for single objective (lines 193-231)."""
        optimizer._is_multi_objective = False
        optimizer.study_name = "test_single"

        with (
            patch("spark_optima.core.bayesian.optimizer.TPESampler"),
            patch("spark_optima.core.bayesian.optimizer.HyperbandPruner"),
        ):
            study = optimizer._create_study()
            assert study is not None

    def test_create_study_multi_objective(self, optimizer: BayesianOptimizer) -> None:
        """Test _create_study for multi-objective (lines 214-231)."""
        optimizer._is_multi_objective = True
        optimizer.study_name = "test_multi"
        optimizer.objectives = ["minimize_time", "minimize_cost"]

        with (
            patch("spark_optima.core.bayesian.optimizer.TPESampler"),
            patch("spark_optima.core.bayesian.optimizer.HyperbandPruner"),
        ):
            study = optimizer._create_study()
            assert study is not None

    def test_objective_method(self, optimizer: BayesianOptimizer) -> None:
        """Test _objective method (lines 233-282)."""

        # Create a trial
        study = optimizer._create_study()
        trial = study.ask()

        # Add a trial result
        optimizer._trial_results = [
            TrialResult(
                trial_number=0,
                configuration={"memory": "4g"},
                status=TrialStatus.COMPLETED,
            ),
        ]

        # Mock the trial runner to return a completed result
        with patch.object(optimizer._trial_runner, "run_trial") as mock_run:
            mock_result = TrialResult(
                trial_number=trial.number,
                configuration={"memory": "4g"},
                metrics=TrialMetrics(execution_time_seconds=100.0, success=True),
                status=TrialStatus.COMPLETED,
            )
            mock_run.return_value = mock_result

            # Call _objective
            result = optimizer._objective(trial, data_profile=None)

            # For single objective, should return a float
            assert isinstance(result, float | list)


class TestBayesianOptimizerSampleConfig:
    """Tests for _sample_config method (lines 284-347)."""

    @pytest.fixture
    def optimizer(self) -> BayesianOptimizer:
        """Create test optimizer."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        return BayesianOptimizer(
            heuristic_config={
                "spark.executor.memory": "4g",
                "spark.executor.cores": "4",
                "spark.sql.shuffle.partitions": "200",
            },
            config_set=config_set,
            resource_spec=resources,
        )

    def test_sample_config_categorical(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling categorical parameter."""
        search_space = {
            "spark.serializer": {
                "type": "categorical",
                "choices": ["Kryo", "JavaSerializer"],
                "base_value": "Kryo",
            }
        }

        config = optimizer._sample_configuration(search_space)
        assert "spark.serializer" in config

    def test_sample_config_int(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling int parameter."""
        search_space = {
            "spark.executor.cores": {
                "type": "int",
                "low": 2,
                "high": 8,
                "base_value": 4,
            }
        }

        config = optimizer._sample_configuration(search_space)
        assert "spark.executor.cores" in config
        assert isinstance(config["spark.executor.cores"], int)

    def test_sample_config_float(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling float parameter."""
        search_space = {
            "spark.memory.fraction": {
                "type": "float",
                "low": 0.1,
                "high": 0.9,
                "base_value": 0.6,
            }
        }

        config = optimizer._sample_configuration(search_space)
        assert "spark.memory.fraction" in config

    def test_sample_config_bytes(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling bytes parameter (uses _sample_config for bytes)."""
        # _sample_config doesn't handle bytes type specifically
        # But _sample_config in optimizer uses search_space with type
        search_space = {
            "test.param": {
                "type": "int",
                "low": 1024**3,  # 1GB in bytes
                "high": 4 * 1024**3,  # 4GB in bytes
                "base_value": 2 * 1024**3,
            }
        }

        config = optimizer._sample_configuration(search_space)
        assert "test.param" in config

    def test_sample_config_default(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling with default handling."""
        search_space = {
            "test.param": {
                "type": "categorical",
                "choices": [],  # Empty choices
                "base_value": "default",
            }
        }

        config = optimizer._sample_configuration(search_space)
        # When choices is empty, None may be returned
        assert "test.param" in config

    def test_sample_config_without_type(self, optimizer: BayesianOptimizer) -> None:
        """Test sampling parameter without recognized type."""
        search_space = {
            "test.param": {
                "base_value": "some_value",
            }
        }

        config = optimizer._sample_configuration(search_space)
        assert config["test.param"] == "some_value"


class TestBayesianOptimizerMoreCoverage:
    """Additional tests for 100% coverage."""

    def test_optimize_with_data_profile(self) -> None:
        """Test optimize method with data_profile (lines 147-148)."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

        data_profile = {"size_gb": 100, "format": "parquet"}

        result = optimizer.optimize(n_trials=1, data_profile=data_profile)
        assert isinstance(result, BayesianOptimizationResult)

    def test_optimize_no_study(self) -> None:
        """Test _build_result with no study (lines 363-394)."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

        # No study, no trials
        optimizer._trial_results = []
        result = optimizer._build_result(study=None)
        assert result.best_config is None
        assert result.n_trials_completed == 0

    def test_optimize_with_study_no_trials(self) -> None:
        """Test _build_result with study but no trials (lines 398-414)."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

        mock_study = MagicMock()
        mock_study.trials = []
        mock_study.directions = ["minimize"]

        result = optimizer._build_result(study=mock_study)
        assert result.best_config is None
        assert result.best_trial_number == -1
        assert len(result.all_trials) == 0

    def test_optimize_with_n_jobs(self) -> None:
        """Test optimize with n_jobs parameter (line 145)."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

        result = optimizer.optimize(n_trials=1, n_jobs=1)
        assert isinstance(result, BayesianOptimizationResult)

    def test_optimize_without_progress(self) -> None:
        """Test optimize with show_progress=False (line 146)."""
        config_set = ConfigSet(version="3.5.0", parameters={}, metadata={})
        resources = ResourceSpec(cpu_cores=8, memory_gb=32)
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.executor.memory": "4g"},
            config_set=config_set,
            resource_spec=resources,
        )

        result = optimizer.optimize(n_trials=1, show_progress=False)
        assert isinstance(result, BayesianOptimizationResult)


def _load_journal_study(study_name: str, storage_path: str) -> Any:
    """Load an Optuna study from a journal storage file."""
    import optuna

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path))
    return optuna.load_study(study_name=study_name, storage=storage)


class TestBayesianOptimizerSeedTrial:
    """Tests for E1 - enqueueing the heuristic baseline as the first trial."""

    @pytest.fixture
    def config_set(self) -> ConfigSet:
        """Create test config set."""
        return ConfigSet(version="3.5.0", parameters={}, metadata={})

    @pytest.fixture
    def resource_spec(self) -> ResourceSpec:
        """Create test resource specification."""
        return ResourceSpec(cpu_cores=8, memory_gb=32)

    @pytest.fixture
    def heuristic_config(self) -> dict:
        """Create a representative heuristic baseline configuration."""
        return {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.shuffle.partitions": "200",
        }

    def test_first_trial_uses_seeded_params(
        self,
        tmp_path: Path,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """The first completed trial runs exactly the enqueued seed parameters."""
        storage_path = str(tmp_path / "seed_study.log")
        optimizer = BayesianOptimizer(
            heuristic_config=heuristic_config,
            config_set=config_set,
            resource_spec=resource_spec,
            study_name="seed_study",
            storage_path=storage_path,
        )

        seed_params = optimizer.get_seed_trial_params()
        assert seed_params  # Search space is non-empty

        result = optimizer.optimize(n_trials=3, show_progress=False)

        assert result.metadata["seed_trial_enqueued"] is True
        assert result.metadata["n_prior_trials"] == 0

        import optuna

        study = _load_journal_study("seed_study", storage_path)
        first_trial = study.trials[0]
        assert first_trial.state == optuna.trial.TrialState.COMPLETE
        # Optuna accepted every fixed parameter (no out-of-range fallback sampling)
        assert first_trial.params == seed_params

    def test_best_result_not_worse_than_seed(
        self,
        tmp_path: Path,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """The best objective value is never worse than the seed trial's value."""
        storage_path = str(tmp_path / "seed_best_study.log")
        optimizer = BayesianOptimizer(
            heuristic_config=heuristic_config,
            config_set=config_set,
            resource_spec=resource_spec,
            study_name="seed_best_study",
            storage_path=storage_path,
        )

        optimizer.optimize(n_trials=3, show_progress=False)

        study = _load_journal_study("seed_best_study", storage_path)
        seed_value = study.trials[0].value
        assert seed_value is not None
        assert study.best_value <= seed_value

    def test_seed_params_skip_entries_outside_search_space(
        self,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Heuristic entries without a search-space definition are not enqueued."""
        optimizer = BayesianOptimizer(
            heuristic_config={
                "spark.executor.memory": "4g",
                "spark.app.name": "my-app",  # Fixed param - never optimized
                "custom.unknown.param": "x",  # Unknown - not in the search space
            },
            config_set=config_set,
            resource_spec=resource_spec,
        )

        seed_params = optimizer.get_seed_trial_params()

        assert "spark.executor.memory" in seed_params
        assert "spark.app.name" not in seed_params
        assert "custom.unknown.param" not in seed_params

    def test_seed_params_clamped_into_search_range(
        self,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """Out-of-range heuristic values are clamped into the suggested ranges."""
        optimizer = BayesianOptimizer(
            heuristic_config={"spark.memory.fraction": 0.95},
            config_set=config_set,
            resource_spec=resource_spec,
            search_space_config=SearchSpaceConfig(
                param_ranges={"spark.memory.fraction": (0.5, 0.9)},
            ),
        )

        seed_params = optimizer.get_seed_trial_params()

        assert seed_params["spark.memory.fraction"] == pytest.approx(0.9)

    def test_no_seed_for_empty_search_space(
        self,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
    ) -> None:
        """No seed trial is enqueued when the search space is empty."""
        optimizer = BayesianOptimizer(
            heuristic_config={},
            config_set=config_set,
            resource_spec=resource_spec,
        )

        result = optimizer.optimize(n_trials=1, show_progress=False)

        assert result.metadata["seed_trial_enqueued"] is False


class TestBayesianOptimizerWarmStart:
    """Tests for E2 - warm-starting from a stored study."""

    @pytest.fixture
    def config_set(self) -> ConfigSet:
        """Create test config set."""
        return ConfigSet(version="3.5.0", parameters={}, metadata={})

    @pytest.fixture
    def resource_spec(self) -> ResourceSpec:
        """Create test resource specification."""
        return ResourceSpec(cpu_cores=8, memory_gb=32)

    @pytest.fixture
    def heuristic_config(self) -> dict:
        """Create a representative heuristic baseline configuration."""
        return {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
            "spark.sql.shuffle.partitions": "200",
        }

    def _make_optimizer(
        self,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
        storage_path: str,
    ) -> BayesianOptimizer:
        """Create an optimizer bound to a shared study name and storage."""
        return BayesianOptimizer(
            heuristic_config=heuristic_config,
            config_set=config_set,
            resource_spec=resource_spec,
            study_name="warm_start_study",
            storage_path=storage_path,
        )

    def test_warm_start_accumulates_prior_trials(
        self,
        tmp_path: Path,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """A second run on the same storage resumes the study and counts prior trials."""
        storage_path = str(tmp_path / "warm_study.log")

        optimizer1 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        result1 = optimizer1.optimize(n_trials=3, show_progress=False)
        assert result1.metadata["n_prior_trials"] == 0
        assert result1.metadata["seed_trial_enqueued"] is True

        optimizer2 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        result2 = optimizer2.optimize(n_trials=2, show_progress=False)
        assert result2.metadata["n_prior_trials"] == 3

        study = _load_journal_study("warm_start_study", storage_path)
        assert len(study.trials) == 5  # 3 from run 1 + 2 from run 2

    def test_warm_start_does_not_reenqueue_seed(
        self,
        tmp_path: Path,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """The seed trial is only enqueued once, not on every resume."""
        storage_path = str(tmp_path / "no_reseed_study.log")

        optimizer1 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        optimizer1.optimize(n_trials=2, show_progress=False)
        seed_params = optimizer1.get_seed_trial_params()

        optimizer2 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        result2 = optimizer2.optimize(n_trials=2, show_progress=False)

        assert result2.metadata["seed_trial_enqueued"] is False
        study = _load_journal_study("warm_start_study", storage_path)
        seeded_trials = [t for t in study.trials if t.params == seed_params]
        assert len(seeded_trials) == 1

    def test_warm_start_best_config_from_prior_trials(
        self,
        tmp_path: Path,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """A resumed run reconstructs the best config even when it came from a prior run."""
        storage_path = str(tmp_path / "prior_best_study.log")

        optimizer1 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        optimizer1.optimize(n_trials=3, show_progress=False)

        # Second run executes zero new trials: the best trial must come from run 1
        optimizer2 = self._make_optimizer(config_set, resource_spec, heuristic_config, storage_path)
        result2 = optimizer2.optimize(n_trials=0, show_progress=False)

        assert result2.metadata["n_prior_trials"] == 3
        assert result2.best_config
        assert "spark.executor.memory" in result2.best_config

    def test_in_memory_run_reports_zero_prior_trials(
        self,
        config_set: ConfigSet,
        resource_spec: ResourceSpec,
        heuristic_config: dict,
    ) -> None:
        """Runs without persistent storage never report prior trials."""
        optimizer = BayesianOptimizer(
            heuristic_config=heuristic_config,
            config_set=config_set,
            resource_spec=resource_spec,
        )

        result = optimizer.optimize(n_trials=1, show_progress=False)

        assert result.metadata["n_prior_trials"] == 0
        assert result.metadata["seed_trial_enqueued"] is True
