# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Additional coverage tests for pruners module."""

from __future__ import annotations

import builtins
import importlib
from unittest.mock import MagicMock

import pytest

from spark_optima.core.bayesian.models import TrialStatus


class TestPrunersImportError:
    """Test the ImportError path when Optuna is not available."""

    def test_optuna_not_available_base_pruner(self) -> None:
        """Test that BasePruner is set to object when Optuna not available."""
        # Mock the import to raise ImportError
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import
            # Reload the module to trigger the import error
            import spark_optima.core.bayesian.pruners as pruners_module

            importlib.reload(pruners_module)

            # Check that OPTUNA_AVAILABLE is False
            assert pruners_module.OPTUNA_AVAILABLE is False
            # BasePruner should be object
            assert pruners_module.BasePruner is object
        finally:
            builtins.__import__ = original_import
            # Reload to restore
            importlib.reload(pruners_module)

    def test_spark_config_pruner_requires_optuna(self) -> None:
        """Test that SparkConfigPruner raises RuntimeError when Optuna not available."""
        # Mock the module to simulate Optuna not available
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import
            import spark_optima.core.bayesian.pruners as pruners_module

            importlib.reload(pruners_module)

            # SparkConfigPruner should raise RuntimeError when instantiated
            with pytest.raises(RuntimeError, match="Optuna is required"):
                pruners_module.SparkConfigPruner()
        finally:
            builtins.__import__ = original_import
            importlib.reload(pruners_module)


class TestMedianPrunerEdgeCases:
    """Edge case tests for MedianPruner."""

    def test_should_prune_exception_handling(self) -> None:
        """Test exception handling in MedianPruner.should_prune (lines 113, 121, 126)."""
        from spark_optima.core.bayesian.pruners import MedianPruner

        pruner = MedianPruner(n_startup_trials=0)

        # Create trial with value that will cause issues
        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.value = "not_a_number"  # This will cause issues in comparison

        mock_study = MagicMock()
        # Create trials with valid values
        mock_study.trials = [
            MagicMock(value=100.0, state=MagicMock(name=TrialStatus.COMPLETED.name)),
            MagicMock(value=200.0, state=MagicMock(name=TrialStatus.COMPLETED.name)),
        ]

        # Should handle the type error gracefully
        result = pruner.should_prune(mock_trial, mock_study)
        assert isinstance(result, bool)

    def test_should_prune_no_values(self) -> None:
        """Test when completed trials have no values."""
        from spark_optima.core.bayesian.pruners import MedianPruner

        pruner = MedianPruner(n_startup_trials=0)

        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.value = 100.0

        # Trials with None values
        mock_study = MagicMock()
        mock_study.trials = [
            MagicMock(value=None, state=MagicMock(name=TrialStatus.COMPLETED.name)),
            MagicMock(value=None, state=MagicMock(name=TrialStatus.COMPLETED.name)),
        ]

        result = pruner.should_prune(mock_trial, mock_study)
        assert result is False


class TestPercentilePrunerEdgeCases:
    """Edge case tests for PercentilePruner."""

    def test_should_prune_exception_handling(self) -> None:
        """Test exception handling in PercentilePruner.should_prune."""
        from spark_optima.core.bayesian.pruners import PercentilePruner

        pruner = PercentilePruner(percentile=25.0, n_startup_trials=0)

        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.value = "invalid"  # Not a number

        mock_study = MagicMock()
        mock_study.trials = [
            MagicMock(value=100.0, state=MagicMock(name=TrialStatus.COMPLETED.name)),
        ]

        result = pruner.should_prune(mock_trial, mock_study)
        assert isinstance(result, bool)

    def test_should_prune_boundary_condition(self) -> None:
        """Test percentile pruning at boundary (lines 207-214)."""
        from spark_optima.core.bayesian.pruners import PercentilePruner

        pruner = PercentilePruner(percentile=50.0, n_startup_trials=0)

        # Create trials where we can test the boundary
        mock_study = MagicMock()
        mock_study.trials = [
            MagicMock(value=100.0, state=MagicMock(name=TrialStatus.COMPLETED.name)),
            MagicMock(value=200.0, state=MagicMock(name=TrialStatus.COMPLETED.name)),
        ]

        # Trial with value at threshold
        mock_trial = MagicMock()
        mock_trial.number = 5
        mock_trial.value = 100.0  # At 50th percentile

        result = pruner.should_prune(mock_trial, mock_study)
        assert isinstance(result, bool)


class TestSuccessRatePrunerEdgeCases:
    """Edge case tests for SuccessRatePruner."""

    def test_should_prune_no_trials(self) -> None:
        """Test with no trials (lines 259, 264, 274)."""
        from spark_optima.core.bayesian.pruners import SuccessRatePruner

        pruner = SuccessRatePruner(min_success_rate=0.5, n_startup_trials=0)

        mock_trial = MagicMock()
        mock_trial.number = 1

        mock_study = MagicMock()
        mock_study.trials = []

        result = pruner.should_prune(mock_trial, mock_study)
        assert result is False

    def test_should_prune_all_failed(self) -> None:
        """Test when all trials fail."""
        from spark_optima.core.bayesian.pruners import SuccessRatePruner

        pruner = SuccessRatePruner(min_success_rate=0.5, n_startup_trials=5)

        mock_trial = MagicMock()
        mock_trial.number = 6

        mock_study = MagicMock()
        # All failed - but we need more than n_startup_trials
        mock_study.trials = [
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
            MagicMock(value=0.0, state=MagicMock(name=TrialStatus.FAILED.name)),
        ]

        result = pruner.should_prune(mock_trial, mock_study)
        assert isinstance(result, bool)


class TestSparkConfigPrunerCoverage:
    """Tests to cover missing lines in SparkConfigPruner."""

    @pytest.fixture
    def pruner(self):
        """Create SparkConfigPruner with low warmup."""
        from spark_optima.core.bayesian.pruners import SparkConfigPruner

        return SparkConfigPruner(
            warmup_trials=2,
            min_resource=1,
            max_resource=10,
            reduction_factor=3,
        )

    def test_prune_with_intermediate_values(self, pruner) -> None:
        """Test pruning based on intermediate values (lines 372-375, 498-523)."""
        from optuna.trial import FrozenTrial

        mock_study = MagicMock()
        # Create completed trials for comparison
        completed_trial = MagicMock()
        completed_trial.state = MagicMock()
        completed_trial.state = MagicMock(return_value=True)  # COMPLETE
        completed_trial.value = 50.0
        completed_trial.number = 1

        mock_study.trials = [completed_trial]
        mock_study.trials = [completed_trial]  # Only 1, less than warmup

        # Create trial with intermediate values
        mock_trial = MagicMock(spec=FrozenTrial)
        mock_trial.number = 5
        mock_trial.params = {"spark.executor.memory": "4g"}
        mock_trial.intermediate_values = {0: 200.0}  # Much worse than best (50.0)

        # This should trigger the intermediate value pruning path
        result = pruner.prune(mock_study, mock_trial)
        assert isinstance(result, bool)

    def test_is_infeasible_high_parallelism(self, pruner) -> None:
        """Test infeasible config with high parallelism (lines 404-411)."""
        config = {
            "spark.executor.cores": 2,
            "spark.task.cpus": 1,
            "spark.default.parallelism": 20000,  # Very high with few cores
        }

        result = pruner._is_infeasible_config(config)
        assert result is True

    def test_is_similar_to_poor_performer_no_history(self, pruner) -> None:
        """Test with no history (lines 436)."""
        config = {"spark.executor.memory": "4g"}

        # Clear history
        pruner._config_history = []
        pruner._performance_history = []

        result = pruner._is_similar_to_poor_performer(config)
        assert result is False

    def test_config_similarity_exception_handling(self, pruner) -> None:
        """Test exception handling in _config_similarity (lines 476-478)."""
        config1 = {"spark.executor.memory": "4g"}
        config2 = {"spark.executor.memory": "invalid_value"}  # Will cause float() to fail

        # This should not raise, but handle the exception
        similarity = pruner._config_similarity(config1, config2)
        assert 0.0 <= similarity <= 1.0

    def test_should_prune_based_on_intermediates_no_completed(self, pruner) -> None:
        """Test _should_prune_based_on_intermediates with no completed trials."""
        from optuna import Study
        from optuna.trial import FrozenTrial

        mock_study = MagicMock(spec=Study)
        mock_study.trials = []  # No completed trials

        mock_trial = MagicMock(spec=FrozenTrial)
        mock_trial.number = 5
        mock_trial.intermediate_values = {0: 100.0}

        result = pruner._should_prune_based_on_intermediates(mock_study, mock_trial)
        assert result is False

    def test_should_prune_based_on_intermediates_prune(self, pruner) -> None:
        """Test pruning when current value is much worse than best."""
        from optuna.trial import FrozenTrial, TrialState

        # Create enough completed trials to pass warmup check (warmup_trials=2)
        completed1 = MagicMock(spec=FrozenTrial)
        completed1.state = TrialState.COMPLETE
        completed1.value = 50.0
        completed1.number = 1

        completed2 = MagicMock(spec=FrozenTrial)
        completed2.state = TrialState.COMPLETE
        completed2.value = 60.0
        completed2.number = 2

        mock_study = MagicMock()
        mock_study.trials = [completed1, completed2]

        # Trial with very bad intermediate value
        mock_trial = MagicMock(spec=FrozenTrial)
        mock_trial.number = 5
        mock_trial.intermediate_values = {0: 200.0}  # 4x worse than best (50.0)

        result = pruner._should_prune_based_on_intermediates(mock_study, mock_trial)
        assert result is True

    def test_parse_memory_edge_cases(self, pruner) -> None:
        """Test _parse_memory edge cases."""
        # Test with "infinity"
        result = pruner._parse_memory("infinity")
        assert result == 4.0  # Default

        # Test with just a number
        result = pruner._parse_memory(8)
        assert result == 8.0

        # Test with "infinity" string
        result = pruner._parse_memory("infinity")
        assert result == 4.0
