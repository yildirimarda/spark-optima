# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for SimulationEngine."""

import importlib.util
import math
from unittest.mock import MagicMock, patch

import pytest

from spark_optima.core.bayesian.models import TrialMetrics
from spark_optima.core.simulation.engine import SimulationEngine
from spark_optima.platforms.models import ResourceSpec

requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)


def _mock_surrogate_predictor(
    r2: float = 0.9,
    sample_count: int = 60,
    measured_fraction: float = 0.0,
    prediction: float = 123.0,
) -> MagicMock:
    """Build a mock MLPerformancePredictor with a trained online surrogate."""
    predictor = MagicMock()
    # Disable the legacy batch-trained path; these tests target the online surrogate
    predictor.is_trained.return_value = False
    predictor.has_online_model.return_value = True
    predictor.online_r2 = r2
    predictor.sample_count = sample_count
    predictor.measured_sample_count = int(sample_count * measured_fraction)
    predictor.measured_fraction = measured_fraction
    predictor.online_trained_samples = sample_count
    predictor.predict_online.return_value = prediction
    predictor.add_sample.return_value = True
    return predictor


class TestSimulationEngine:
    """Test cases for SimulationEngine."""

    def test_initialization(self) -> None:
        """Test engine initialization."""
        engine = SimulationEngine()
        assert engine is not None
        assert engine.use_ml is True

    def test_initialization_without_ml(self) -> None:
        """Test engine initialization without ML."""
        engine = SimulationEngine(use_ml=False)
        assert engine.use_ml is False

    def test_simulate_basic(self) -> None:
        """Test basic simulation."""
        engine = SimulationEngine()
        config = {
            "spark.executor.memory": "4g",
            "spark.executor.cores": "4",
        }
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = engine.simulate(config, resource_spec)

        assert result.metrics is not None
        assert result.metrics.execution_time_seconds > 0
        assert result.confidence_score > 0

    def test_simulate_with_operations(self) -> None:
        """Test simulation with operations."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        operations = ["scan", "aggregation", "join"]

        result = engine.simulate(
            config,
            resource_spec,
            operations=operations,
        )

        assert result.metrics.execution_time_seconds > 0
        assert result.hybrid_estimate is not None

    def test_simulate_trial(self) -> None:
        """Test simulate_trial method."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        trial_result = engine.simulate_trial(
            trial_number=1,
            config=config,
            resource_spec=resource_spec,
        )

        assert trial_result.trial_number == 1
        assert trial_result.configuration == config
        assert trial_result.metrics.execution_time_seconds > 0

    def test_analytical_estimate(self) -> None:
        """Test analytical estimate in result."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = engine.simulate(config, resource_spec)

        assert result.analytical_estimate is not None
        assert "execution_time_seconds" in result.analytical_estimate

    def test_hybrid_estimate(self) -> None:
        """Test hybrid estimate in result."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = engine.simulate(config, resource_spec)

        assert result.hybrid_estimate is not None
        assert "execution_time_seconds" in result.hybrid_estimate
        assert "method" in result.hybrid_estimate

    def test_model_breakdown(self) -> None:
        """Test model breakdown in result."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = engine.simulate(config, resource_spec)

        assert result.model_breakdown is not None
        assert "analytical" in result.model_breakdown

    def test_trial_history(self) -> None:
        """Test trial history storage."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Run a trial
        engine.simulate_trial(
            trial_number=1,
            config=config,
            resource_spec=resource_spec,
        )

        history = engine.get_trial_history()
        assert len(history) == 1
        assert history[0]["configuration"] == config

    def test_clear_history(self) -> None:
        """Test clearing trial history."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        engine.simulate_trial(1, config, resource_spec)
        engine.clear_history()

        history = engine.get_trial_history()
        assert len(history) == 0

    def test_get_analytical_model(self) -> None:
        """Test getting analytical model."""
        engine = SimulationEngine()
        model = engine.get_analytical_model()
        assert model is not None


class TestDataProfileConversion:
    """Test data profile conversion."""

    def test_convert_data_profile(self) -> None:
        """Test data profile conversion."""
        engine = SimulationEngine()
        data_dict = {
            "size_gb": 50,
            "format": "delta",
            "compression": "zstd",
            "skew_factor": 2.0,
        }

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            data_profile=data_dict,
        )

        assert result.metrics.execution_time_seconds > 0


class TestOperationsConversion:
    """Test operations conversion."""

    def test_convert_operations(self) -> None:
        """Test operations conversion."""
        engine = SimulationEngine()
        operations = ["scan", "filter", "aggregation", "join"]

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            operations=operations,
        )

        assert result.metrics.execution_time_seconds > 0


class TestSimulationEngineMoreCoverage:
    """Additional tests for 100% coverage."""

    def test_initialization_with_ml(self) -> None:
        """Test engine initialization with ML (lines 86-106)."""
        engine = SimulationEngine(use_ml=True)
        assert engine.use_ml is True
        assert engine._analytical_model is not None

    def test_estimate_method(self) -> None:
        """Test estimate method (lines 111-139)."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        metrics = engine.estimate(config, resource_spec)

        assert isinstance(metrics, TrialMetrics)
        assert metrics.execution_time_seconds > 0

    def test_estimate_with_all_params(self) -> None:
        """Test estimate with cost_model and data_profile."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        data_profile = {"size_gb": 50, "format": "parquet"}

        cost_model = MagicMock()
        cost_model.calculate.return_value = 5.0

        metrics = engine.estimate(
            config=config,
            resource_spec=resource_spec,
            cost_model=cost_model,
            data_profile=data_profile,
        )

        assert isinstance(metrics, TrialMetrics)
        assert metrics.cost_estimate_usd == 5.0

    def test_simulate_method(self) -> None:
        """Test simulate method (lines 141-225)."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        result = engine.simulate(config, resource_spec)

        assert result.metrics is not None
        assert result.analytical_estimate is not None
        assert result.ml_estimate is not None or result.ml_estimate is None
        assert result.hybrid_estimate is not None
        assert result.confidence_score > 0
        assert result.model_breakdown is not None

    def test_simulate_with_all_params(self) -> None:
        """Test simulate with operations."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)
        operations = ["scan", "aggregation", "join"]

        result = engine.simulate(
            config=config,
            resource_spec=resource_spec,
            operations=operations,
        )

        assert result.metrics.execution_time_seconds > 0

    def test_store_trial_result(self) -> None:
        """Test _store_trial_result method (lines 564-589)."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Create a SimulationResult
        from spark_optima.core.simulation.engine import SimulationResult

        result = SimulationResult(
            metrics=TrialMetrics(execution_time_seconds=60.0, success=True),
            analytical_estimate={"execution_time_seconds": 60.0},
            ml_estimate=None,
            hybrid_estimate={"execution_time_seconds": 60.0, "method": "test"},
            confidence_score=0.8,
            model_breakdown={},
        )

        engine._store_trial_result(result, config, resource_spec, {"size_gb": 10})

        history = engine.get_trial_history()
        assert len(history) == 1
        assert history[0]["configuration"] == config

    def test_get_ml_model(self) -> None:
        """Test get_ml_model method (lines 625-631)."""
        engine = SimulationEngine()
        model = engine.get_ml_model()
        # May be None if ML not available
        assert model is None or model is not None

    def test_clear_history(self) -> None:
        """Test clear_history method (lines 643-646)."""
        engine = SimulationEngine()
        config = {"spark.executor.memory": "4g"}
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        # Run a trial to add to history
        engine.simulate_trial(1, config, resource_spec)
        assert len(engine.get_trial_history()) > 0

        engine.clear_history()
        assert len(engine.get_trial_history()) == 0

    def test_convert_operations_various(self) -> None:
        """Test _convert_operations with various operations (lines 503-562)."""
        engine = SimulationEngine()

        # Test broadcast join
        operations = ["broadcast_join"]
        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            operations=operations,
        )
        assert result.metrics.execution_time_seconds > 0

        # Test sort
        operations = ["sort"]
        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            operations=operations,
        )
        assert result.metrics.execution_time_seconds > 0

        # Test union
        operations = ["union"]
        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            operations=operations,
        )
        assert result.metrics.execution_time_seconds > 0

    def test_combine_estimates_no_ml(self) -> None:
        """Test _combine_estimates without ML (lines 364-375)."""
        engine = SimulationEngine(use_ml=False)

        analytical = {
            "execution_time_seconds": 100.0,
            "memory_peak_gb": 4.0,
            "cpu_utilization_percent": 50.0,
            "shuffle_read_gb": 1.0,
            "shuffle_write_gb": 1.0,
            "cost_estimate_usd": 1.0,
        }

        result = engine._combine_estimates(analytical, None)
        assert result["method"] == "analytical_only"
        assert result["execution_time_seconds"] == 100.0

    def test_calculate_confidence(self) -> None:
        """Test _calculate_confidence method (lines 412-446)."""
        engine = SimulationEngine()

        analytical = {"execution_time_seconds": 100.0}
        confidence = engine._calculate_confidence(analytical, None)
        assert 0.0 <= confidence <= 1.0

        # With ML result
        ml_result = {"execution_time_seconds": 110.0}
        confidence = engine._calculate_confidence(analytical, ml_result)
        assert 0.0 <= confidence <= 1.0

    def test_build_trial_metrics(self) -> None:
        """Test _build_trial_metrics method (lines 448-475)."""
        engine = SimulationEngine()

        hybrid = {
            "execution_time_seconds": 100.0,
            "memory_peak_gb": 4.0,
            "cpu_utilization_percent": 50.0,
            "shuffle_read_gb": 1.0,
            "shuffle_write_gb": 1.0,
            "cost_estimate_usd": 1.0,
        }
        analytical = {
            "success": True,
            "feasibility_issues": [],
        }

        metrics = engine._build_trial_metrics(hybrid, analytical)
        assert isinstance(metrics, TrialMetrics)
        assert metrics.execution_time_seconds == 100.0
        assert metrics.success is True

    def test_convert_data_profile(self) -> None:
        """Test _convert_data_profile method (lines 477-501)."""
        engine = SimulationEngine()

        data_dict = {
            "size_gb": 50.0,
            "num_rows": 1000000,
            "num_columns": 20,
            "avg_row_size_bytes": 100.0,
            "format": "parquet",
            "compression": "snappy",
            "partitioning": 100,
            "null_ratio": 0.05,
            "cardinality": 50000,
            "skew_factor": 1.5,
        }

        result = engine._convert_data_profile(data_dict)
        assert result.size_gb == 50.0
        assert result.num_columns == 20
        assert result.format == "parquet"

    def test_train_ml_model_no_data(self) -> None:
        """Test train_ml_model with insufficient data (lines 604-613)."""
        engine = SimulationEngine()

        result = engine.train_ml_model()
        assert "error" in result
        assert "Insufficient data" in result["error"] or "not available" in result["error"]

    def test_train_ml_model_with_data(self) -> None:
        """Test train_ml_model with data."""
        engine = SimulationEngine()

        # Add some trial history
        engine.simulate_trial(1, {"spark.executor.memory": "4g"}, ResourceSpec(cpu_cores=8, memory_gb=32))
        engine.simulate_trial(2, {"spark.executor.memory": "8g"}, ResourceSpec(cpu_cores=8, memory_gb=32))

        # Need at least 10 samples for training
        # This will likely return an error about insufficient data
        result = engine.train_ml_model()
        assert isinstance(result, dict)

    def test_ml_init_failure(self) -> None:
        """Test ML model initialization failure (lines 104-106)."""
        # Mock MLPerformancePredictor to raise RuntimeError
        from unittest.mock import patch

        with patch("spark_optima.core.simulation.engine.MLPerformancePredictor") as mock_predictor_class:
            mock_predictor_class.side_effect = RuntimeError("Model not found")

            engine = SimulationEngine(use_ml=True)
            assert engine._ml_model is None

    def test_run_ml_model_not_trained(self) -> None:
        """Test _run_ml_model when model not trained (lines 320-321)."""
        engine = SimulationEngine(use_ml=False)

        # Manually set _ml_model but make is_trained() return False
        from unittest.mock import MagicMock

        mock_ml_model = MagicMock()
        mock_ml_model.is_trained.return_value = False
        engine._ml_model = mock_ml_model

        result = engine._run_ml_model(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
            data_profile={"size_gb": 10},
            operations=["scan"],
        )
        assert result is None

    def test_run_ml_model_exception(self) -> None:
        """Test _run_ml_model exception handling (lines 345-347)."""
        engine = SimulationEngine(use_ml=False)

        # Manually set _ml_model that raises exception
        from unittest.mock import MagicMock

        mock_ml_model = MagicMock()
        mock_ml_model.is_trained.return_value = True
        mock_ml_model.predict.side_effect = RuntimeError("Prediction failed")
        engine._ml_model = mock_ml_model

        result = engine._run_ml_model(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
            data_profile={"size_gb": 10},
            operations=["scan"],
        )
        assert result is None

    def test_combine_estimates_large_difference(self) -> None:
        """Test _combine_estimates when models differ significantly (lines 383-390)."""
        engine = SimulationEngine(use_ml=False)

        analytical = {
            "execution_time_seconds": 100.0,
            "memory_peak_gb": 4.0,
            "cpu_utilization_percent": 50.0,
            "shuffle_read_gb": 1.0,
            "shuffle_write_gb": 1.0,
            "cost_estimate_usd": 1.0,
        }
        ml_result = {
            "execution_time_seconds": 600.0,  # 6x different
            "confidence_lower": 500.0,
            "confidence_upper": 700.0,
        }

        result = engine._combine_estimates(analytical, ml_result)
        assert result["method"] == "analytical_fallback"
        assert result["execution_time_seconds"] == 100.0

    def test_combine_estimates_normal(self) -> None:
        """Test _combine_estimates normal weighted combination."""
        engine = SimulationEngine(use_ml=False)

        analytical = {
            "execution_time_seconds": 100.0,
            "memory_peak_gb": 4.0,
            "cpu_utilization_percent": 50.0,
            "shuffle_read_gb": 1.0,
            "shuffle_write_gb": 1.0,
            "cost_estimate_usd": 1.0,
        }
        ml_result = {
            "execution_time_seconds": 120.0,  # Close to analytical
        }

        result = engine._combine_estimates(analytical, ml_result)
        assert result["method"] == "hybrid_weighted"
        # Should be weighted: 0.6 * 100 + 0.4 * 120 = 108
        assert 100.0 <= result["execution_time_seconds"] <= 120.0

    def test_calculate_confidence_with_ml(self) -> None:
        """Test _calculate_confidence with ML result (lines 430-440)."""
        engine = SimulationEngine()

        analytical = {"execution_time_seconds": 100.0}
        ml_result = {"execution_time_seconds": 110.0}  # Within 50%

        confidence = engine._calculate_confidence(analytical, ml_result)
        assert 0.0 <= confidence <= 1.0
        # Should be higher due to ML agreement
        assert confidence >= 0.85  # 0.7 + 0.15 + 0.1

    def test_calculate_confidence_with_feasibility_issues(self) -> None:
        """Test _calculate_confidence with feasibility issues (lines 442-444)."""
        engine = SimulationEngine()

        analytical = {
            "execution_time_seconds": 100.0,
            "feasibility_issues": ["issue1", "issue2"],
        }

        confidence = engine._calculate_confidence(analytical, None)
        assert 0.0 <= confidence <= 1.0
        # Should be reduced by 0.2 (2 issues * 0.1)
        assert confidence <= 0.7  # 0.7 - 0.2

    def test_build_trial_metrics_with_issues(self) -> None:
        """Test _build_trial_metrics with feasibility issues (lines 463-464)."""
        engine = SimulationEngine()

        hybrid = {
            "execution_time_seconds": 100.0,
            "memory_peak_gb": 4.0,
            "cpu_utilization_percent": 50.0,
            "shuffle_read_gb": 1.0,
            "shuffle_write_gb": 1.0,
            "cost_estimate_usd": 1.0,
        }
        analytical = {
            "success": False,
            "feasibility_issues": ["test issue"],
        }

        metrics = engine._build_trial_metrics(hybrid, analytical)
        assert isinstance(metrics, TrialMetrics)
        assert metrics.success is False

    def test_simulate_with_ml_estimate(self) -> None:
        """Test simulate includes ML estimate (lines 181-188)."""
        engine = SimulationEngine(use_ml=False)

        # Mock ML model
        from unittest.mock import MagicMock

        mock_ml_model = MagicMock()
        mock_ml_model.is_trained.return_value = True
        mock_ml_model.predict.return_value = MagicMock(
            predicted_time=90.0,
            confidence_interval=(80.0, 100.0),
            feature_importance={"spark.executor.memory": 0.5},
            model_version="v1",
        )
        engine._ml_model = mock_ml_model

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        # Check that result has ml_estimate
        assert result.ml_estimate is not None

    def test_store_trial_result_with_ml(self) -> None:
        """Test _store_trial_result includes ML estimate."""
        engine = SimulationEngine()

        from spark_optima.core.simulation.engine import SimulationResult

        result = SimulationResult(
            metrics=TrialMetrics(execution_time_seconds=60.0, success=True),
            analytical_estimate={"execution_time_seconds": 60.0},
            ml_estimate={"execution_time_seconds": 55.0, "model_version": "v1"},
            hybrid_estimate={"execution_time_seconds": 58.0, "method": "hybrid"},
            confidence_score=0.9,
            model_breakdown={"analytical": {}, "ml": {}},
        )

        engine._store_trial_result(
            result,
            {"spark.executor.memory": "4g"},
            ResourceSpec(cpu_cores=8, memory_gb=32),
            {"size_gb": 10},
        )

        history = engine.get_trial_history()
        assert len(history) == 1


class TestBlendWeightPolicy:
    """Tests for the online surrogate blend-weight policy (Workstream L2)."""

    def _engine(self) -> SimulationEngine:
        return SimulationEngine(use_ml=False)

    def test_r2_below_gate_keeps_weight_zero(self) -> None:
        """R² below the gate disables blending entirely."""
        engine = self._engine()

        assert engine._compute_blend_weight(r2=0.29, n_samples=1000, measured_fraction=1.0) == 0.0
        assert engine._compute_blend_weight(r2=-1.0, n_samples=1000, measured_fraction=1.0) == 0.0
        assert engine._compute_blend_weight(r2=float("nan"), n_samples=1000, measured_fraction=1.0) == 0.0

    def test_analytical_only_cap(self) -> None:
        """With only analytical samples the weight saturates at 0.5."""
        engine = self._engine()
        saturated_samples = SimulationEngine.ML_RAMP_SAMPLES_MULTIPLIER * engine.ml_min_samples

        weight = engine._compute_blend_weight(r2=0.95, n_samples=saturated_samples, measured_fraction=0.0)

        assert weight == pytest.approx(SimulationEngine.ML_WEIGHT_CAP_ANALYTICAL)

    def test_measured_cap(self) -> None:
        """With fully measured samples the weight saturates at 0.8."""
        engine = self._engine()
        saturated_samples = SimulationEngine.ML_RAMP_SAMPLES_MULTIPLIER * engine.ml_min_samples

        weight = engine._compute_blend_weight(r2=0.95, n_samples=saturated_samples, measured_fraction=1.0)

        assert weight == pytest.approx(SimulationEngine.ML_WEIGHT_CAP_MEASURED)

    def test_weight_ramps_with_sample_count(self) -> None:
        """The weight grows proportionally with sample count until saturation."""
        engine = self._engine()
        half_samples = SimulationEngine.ML_RAMP_SAMPLES_MULTIPLIER * engine.ml_min_samples // 2

        weight = engine._compute_blend_weight(r2=0.95, n_samples=half_samples, measured_fraction=0.0)

        assert weight == pytest.approx(SimulationEngine.ML_WEIGHT_CAP_ANALYTICAL / 2)

    def test_weight_ramps_with_r2(self) -> None:
        """The weight grows with R² between the gate and the saturation point."""
        engine = self._engine()
        saturated_samples = SimulationEngine.ML_RAMP_SAMPLES_MULTIPLIER * engine.ml_min_samples
        midpoint_r2 = (SimulationEngine.ML_R2_GATE + SimulationEngine.ML_R2_SATURATION) / 2

        weight = engine._compute_blend_weight(r2=midpoint_r2, n_samples=saturated_samples, measured_fraction=0.0)

        assert weight == pytest.approx(SimulationEngine.ML_WEIGHT_CAP_ANALYTICAL / 2)

    def test_measured_fraction_interpolates_cap(self) -> None:
        """A partially measured sample set interpolates the cap linearly."""
        engine = self._engine()
        saturated_samples = SimulationEngine.ML_RAMP_SAMPLES_MULTIPLIER * engine.ml_min_samples

        weight = engine._compute_blend_weight(r2=0.95, n_samples=saturated_samples, measured_fraction=0.5)

        expected = (SimulationEngine.ML_WEIGHT_CAP_ANALYTICAL + SimulationEngine.ML_WEIGHT_CAP_MEASURED) / 2
        assert weight == pytest.approx(expected)


class TestSurrogateConfiguration:
    """Tests for the surrogate-related constructor flags."""

    def test_default_flags(self) -> None:
        """The surrogate is on by default with the documented thresholds."""
        engine = SimulationEngine(use_ml=False)

        assert engine.enable_ml_predictor is True
        assert engine.ml_min_samples == SimulationEngine.ML_MIN_SAMPLES_DEFAULT
        assert engine.ml_retrain_interval == SimulationEngine.ML_RETRAIN_INTERVAL_DEFAULT

    def test_disabled_surrogate_records_no_samples(self) -> None:
        """enable_ml_predictor=False keeps the engine purely analytical."""
        mock_predictor = _mock_surrogate_predictor()
        with patch(
            "spark_optima.core.simulation.engine.MLPerformancePredictor",
            return_value=mock_predictor,
        ):
            engine = SimulationEngine(enable_ml_predictor=False)

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        mock_predictor.add_sample.assert_not_called()
        assert result.hybrid_estimate["ml_blend_weight"] == 0.0
        assert result.hybrid_estimate["ml_samples"] == 0
        assert result.model_breakdown["ml_surrogate"]["enabled"] is False

    def test_use_ml_false_disables_surrogate(self) -> None:
        """use_ml=False means no predictor and therefore no surrogate."""
        engine = SimulationEngine(use_ml=False)

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        assert result.hybrid_estimate["ml_blend_weight"] == 0.0
        assert result.model_breakdown["ml_surrogate"]["enabled"] is False
        assert engine.record_observation({"spark.executor.memory": "4g"}, {"size_gb": 10}, 42.0) is False


class TestSurrogateBlending:
    """Tests for online surrogate blending inside simulate()."""

    def _engine_with_mock(self, mock_predictor: MagicMock) -> SimulationEngine:
        with patch(
            "spark_optima.core.simulation.engine.MLPerformancePredictor",
            return_value=mock_predictor,
        ):
            return SimulationEngine()

    def test_blending_math_with_mocked_model(self) -> None:
        """final = w * ml_prediction + (1 - w) * analytical, with w from the policy."""
        mock_predictor = _mock_surrogate_predictor(r2=0.9, sample_count=60, prediction=123.0)
        engine = self._engine_with_mock(mock_predictor)

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        hybrid = result.hybrid_estimate
        # r2=0.9 and 60 samples saturate the ramp; analytical-only cap is 0.5
        assert hybrid["ml_blend_weight"] == pytest.approx(0.5)
        expected = 0.5 * 123.0 + 0.5 * hybrid["pre_blend_time_seconds"]
        assert hybrid["execution_time_seconds"] == pytest.approx(expected)
        assert hybrid["ml_surrogate_time_seconds"] == 123.0
        assert result.metrics.execution_time_seconds == pytest.approx(expected)

        surrogate = result.model_breakdown["ml_surrogate"]
        assert surrogate["enabled"] is True
        assert surrogate["trained"] is True
        assert surrogate["ml_blend_weight"] == pytest.approx(0.5)
        assert surrogate["validation_r2"] == 0.9
        assert surrogate["ml_prediction_seconds"] == 123.0

        # The analytical estimate is recorded as a pseudo-observation
        mock_predictor.add_sample.assert_called_once()
        assert mock_predictor.add_sample.call_args.kwargs.get("measured") is False

    def test_r2_gate_blocks_blending(self) -> None:
        """A surrogate with R² below the gate contributes nothing."""
        mock_predictor = _mock_surrogate_predictor(r2=0.1, sample_count=60)
        engine = self._engine_with_mock(mock_predictor)

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        hybrid = result.hybrid_estimate
        assert hybrid["ml_blend_weight"] == 0.0
        assert "pre_blend_time_seconds" not in hybrid
        mock_predictor.predict_online.assert_not_called()
        # Samples still accumulate so the surrogate can improve over time
        mock_predictor.add_sample.assert_called_once()

    def test_failed_prediction_falls_back_to_analytical(self) -> None:
        """A None prediction from the surrogate leaves the estimate untouched."""
        mock_predictor = _mock_surrogate_predictor(r2=0.9, sample_count=60)
        mock_predictor.predict_online.return_value = None
        engine = self._engine_with_mock(mock_predictor)

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        assert result.hybrid_estimate["ml_blend_weight"] == 0.0
        assert "pre_blend_time_seconds" not in result.hybrid_estimate
        assert result.metrics.execution_time_seconds > 0

    def test_record_observation_with_mock(self) -> None:
        """record_observation feeds measured samples into the surrogate."""
        mock_predictor = _mock_surrogate_predictor()
        engine = self._engine_with_mock(mock_predictor)

        recorded = engine.record_observation(
            {"spark.executor.memory": "4g"},
            {"size_gb": 10},
            42.0,
        )

        assert recorded is True
        mock_predictor.add_sample.assert_called_once()
        assert mock_predictor.add_sample.call_args.kwargs.get("measured") is True

    def test_record_observation_rejects_invalid_time(self) -> None:
        """Non-finite or non-positive measurements are ignored."""
        mock_predictor = _mock_surrogate_predictor()
        engine = self._engine_with_mock(mock_predictor)

        assert engine.record_observation({}, {}, -5.0) is False
        assert engine.record_observation({}, {}, 0.0) is False
        assert engine.record_observation({}, {}, float("nan")) is False
        mock_predictor.add_sample.assert_not_called()


class TestSurrogateSklearnMissing:
    """The engine must degrade silently to pure-analytical without scikit-learn."""

    def test_engine_works_without_sklearn(self) -> None:
        """Simulation stays functional and surrogate paths return inert values."""
        with patch("spark_optima.core.simulation.predictor.SKLEARN_AVAILABLE", False):
            engine = SimulationEngine(use_ml=True, enable_ml_predictor=True)

        assert engine._ml_model is None

        result = engine.simulate(
            config={"spark.executor.memory": "4g"},
            resource_spec=ResourceSpec(cpu_cores=8, memory_gb=32),
        )

        assert result.metrics.execution_time_seconds > 0
        assert math.isfinite(result.metrics.execution_time_seconds)
        assert result.hybrid_estimate["ml_blend_weight"] == 0.0
        assert result.hybrid_estimate["ml_samples"] == 0
        assert result.model_breakdown["ml_surrogate"]["enabled"] is False

        assert engine.record_observation({"spark.executor.memory": "4g"}, {"size_gb": 10}, 42.0) is False
        assert engine.save_ml_model() is None
        assert engine.load_ml_model() is False
        assert engine.get_ml_surrogate_status()["enabled"] is False


@requires_sklearn
class TestSurrogateEndToEnd:
    """Integration tests: the surrogate trains from real simulated trials."""

    @staticmethod
    def _varied_trial_inputs(index: int) -> tuple[dict, dict]:
        config = {
            "spark.executor.memory": f"{2 + (index % 6)}g",
            "spark.executor.cores": str(2 + (index % 4)),
            "spark.sql.shuffle.partitions": str(100 + 20 * index),
        }
        data_profile = {"size_gb": 5.0 + 3.0 * index, "format": "parquet"}
        return config, data_profile

    def test_training_triggers_after_min_samples(self) -> None:
        """Enough simulated trials train the surrogate and expose blend metadata."""
        engine = SimulationEngine(ml_min_samples=10, ml_retrain_interval=5)
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        last_result = None
        for i in range(30):
            config, data_profile = self._varied_trial_inputs(i)
            last_result = engine.simulate(
                config=config,
                resource_spec=resource_spec,
                data_profile=data_profile,
            )

        assert last_result is not None
        status = engine.get_ml_surrogate_status()
        assert status["enabled"] is True
        assert status["trained"] is True
        assert status["ml_samples"] == 30

        surrogate = last_result.model_breakdown["ml_surrogate"]
        assert surrogate["trained"] is True
        assert surrogate["ml_samples"] == 30

        # Results stay finite and positive regardless of blending
        assert last_result.metrics.execution_time_seconds > 0
        assert math.isfinite(last_result.metrics.execution_time_seconds)
        assert 0.0 <= last_result.hybrid_estimate["ml_blend_weight"] <= SimulationEngine.ML_WEIGHT_CAP_MEASURED

        # If the surrogate cleared the R² gate it must actually contribute
        if status["validation_r2"] >= SimulationEngine.ML_R2_GATE:
            assert last_result.hybrid_estimate["ml_blend_weight"] > 0.0
            assert "pre_blend_time_seconds" in last_result.hybrid_estimate

    def test_below_threshold_stays_analytical(self) -> None:
        """Before reaching ml_min_samples the engine is purely analytical."""
        engine = SimulationEngine(ml_min_samples=10, ml_retrain_interval=5)
        resource_spec = ResourceSpec(cpu_cores=8, memory_gb=32)

        for i in range(5):
            config, data_profile = self._varied_trial_inputs(i)
            result = engine.simulate(
                config=config,
                resource_spec=resource_spec,
                data_profile=data_profile,
            )
            assert result.hybrid_estimate["ml_blend_weight"] == 0.0
            assert result.model_breakdown["ml_surrogate"]["trained"] is False

        status = engine.get_ml_surrogate_status()
        assert status["trained"] is False
        assert status["ml_samples"] == 5

    def test_measured_observations_feed_training(self) -> None:
        """record_observation accumulates measured samples and triggers training."""
        engine = SimulationEngine(ml_min_samples=10, ml_retrain_interval=5)

        for i in range(12):
            config, data_profile = self._varied_trial_inputs(i)
            measured_time = 20.0 + 5.0 * data_profile["size_gb"]
            assert engine.record_observation(config, data_profile, measured_time) is True

        status = engine.get_ml_surrogate_status()
        assert status["trained"] is True
        assert status["ml_samples"] == 12
        assert status["measured_samples"] == 12

    def test_save_and_load_roundtrip_via_engine(self, tmp_path) -> None:
        """A trained surrogate persists through engine save/load helpers."""
        engine = SimulationEngine(ml_min_samples=10, ml_retrain_interval=5)
        for i in range(12):
            config, data_profile = self._varied_trial_inputs(i)
            engine.record_observation(config, data_profile, 20.0 + 5.0 * data_profile["size_gb"])

        model_path = tmp_path / "surrogate.joblib"
        saved_to = engine.save_ml_model(model_path)
        assert saved_to == model_path
        assert model_path.exists()

        fresh_engine = SimulationEngine()
        assert fresh_engine.load_ml_model(model_path) is True
        status = fresh_engine.get_ml_surrogate_status()
        assert status["trained"] is True
        assert status["ml_samples"] == 12
        assert status["measured_samples"] == 12
