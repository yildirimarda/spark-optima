# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for ML performance predictor module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        from spark_optima.core.simulation.predictor import PredictionResult

        result = PredictionResult(
            predicted_time=100.0,
            confidence_interval=(80.0, 120.0),
            feature_importance={"memory": 0.5, "cores": 0.3},
            model_version="1.0",
        )

        assert result.predicted_time == 100.0
        assert result.confidence_interval == (80.0, 120.0)
        assert result.feature_importance["memory"] == 0.5
        assert result.model_version == "1.0"

    def test_prediction_result_defaults(self):
        """Test PredictionResult with default values."""
        from spark_optima.core.simulation.predictor import PredictionResult

        result = PredictionResult(
            predicted_time=50.0,
            confidence_interval=(40.0, 60.0),
            feature_importance={},
        )

        assert result.model_version == "1.0"


class TestMLPredictorWithMock:
    """Tests for ML predictor using mocked sklearn."""

    def test_predictor_initialization_no_sklearn(self):
        """Test predictor initialization when sklearn unavailable."""
        # This test just imports - if sklearn unavailable, we get import error
        # which is expected behavior
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()
            assert predictor.use_ensemble is True
            assert predictor._is_trained is False
        except RuntimeError:
            # sklearn not available - that's OK for this test
            pass

    def test_feature_names_constant(self):
        """Test FEATURE_NAMES is defined."""
        from spark_optima.core.simulation.predictor import MLPerformancePredictor

        expected_features = [
            "executor_memory_gb",
            "executor_cores",
            "num_executors",
            "parallelism",
            "shuffle_partitions",
            "memory_fraction",
            "data_size_gb",
            "num_columns",
            "has_aggregation",
            "has_join",
            "has_shuffle",
            "skew_factor",
            "format_encoded",
            "compression_encoded",
        ]

        assert expected_features == MLPerformancePredictor.FEATURE_NAMES

    def test_predictor_get_model_info(self):
        """Test get_model_info method."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            info = predictor.get_model_info()

            assert info["is_trained"] is False
            assert info["model_version"] == "1.0"
            assert info["training_samples"] == 0
            assert info["use_ensemble"] is False
            assert info["feature_count"] == 14
            assert "feature_names" in info
        except RuntimeError:
            # sklearn not available - skip
            pytest.skip("scikit-learn not available")

    def test_predictor_encode_format(self):
        """Test format encoding."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            assert predictor._encode_format("parquet") == 0.8
            assert predictor._encode_format("csv") == 0.0
            assert predictor._encode_format("delta") == 1.0
            assert predictor._encode_format("unknown") == 0.5
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_encode_compression(self):
        """Test compression encoding."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            assert predictor._encode_compression("snappy") == 0.3
            assert predictor._encode_compression("none") == 0.0
            assert predictor._encode_compression("zstd") == 0.9
            assert predictor._encode_compression("unknown") == 0.3
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_parse_memory_value(self):
        """Test memory value parsing."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            # Test various formats
            assert predictor._parse_memory_value("4g") == 4.0
            assert predictor._parse_memory_value("4G") == 4.0
            assert predictor._parse_memory_value("1024m") == 1.0
            assert predictor._parse_memory_value("1t") == 1024.0
            assert predictor._parse_memory_value(4) == 4.0
            assert predictor._parse_memory_value(4.5) == 4.5

            # Invalid format returns default
            assert predictor._parse_memory_value("invalid") == 4.0
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_extract_single_features(self):
        """Test single feature extraction."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            features = predictor._extract_single_features(
                config={
                    "spark.executor.memory": "4g",
                    "spark.executor.cores": "2",
                    "spark.default.parallelism": "200",
                    "spark.sql.shuffle.partitions": "200",
                    "spark.memory.fraction": "0.6",
                },
                data_profile={
                    "size_gb": 10,
                    "num_columns": 20,
                    "format": "parquet",
                    "compression": "snappy",
                    "skew_factor": 1.5,
                },
                resource_spec={
                    "cpu_cores": 8,
                },
                operations={
                    "has_aggregation": True,
                    "has_join": True,
                    "has_shuffle": True,
                },
            )

            assert len(features) == 14
            assert features[0] == 4.0  # executor_memory_gb
            assert features[1] == 2.0  # executor_cores
            assert features[9] == 1.0  # has_join
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_extract_features_batch(self):
        """Test batch feature extraction."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            trials = [
                {
                    "configuration": {"spark.executor.memory": "4g"},
                    "data_profile": {"size_gb": 10},
                    "resource_spec": {"cpu_cores": 8},
                    "operations": {},
                }
                for _ in range(5)
            ]

            features = predictor._extract_features(trials)

            assert features.shape == (5, 14)
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_is_trained(self):
        """Test is_trained method."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            assert predictor.is_trained() is False
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_get_feature_importance(self):
        """Test get_feature_importance when not trained."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            importance = predictor.get_feature_importance()

            assert importance == {}
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_train_insufficient_data(self):
        """Test train with insufficient data."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()
            trials = [
                {"configuration": {}, "data_profile": {}, "resource_spec": {}} for _ in range(5)
            ]

            result = predictor.train(trials)
            assert result["r2_score"] == 0.0
            assert result["mae"] == float("inf")
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_train_sufficient_data(self):
        """Test train with sufficient data."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Create 20 trials with proper structure
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {
                            "spark.executor.memory": "4g",
                            "spark.executor.cores": "2",
                        },
                        "data_profile": {
                            "size_gb": 10,
                            "num_columns": 10,
                            "format": "parquet",
                            "compression": "snappy",
                        },
                        "resource_spec": {"cpu_cores": 8},
                        "operations": {
                            "has_aggregation": True,
                            "has_join": False,
                            "has_shuffle": True,
                        },
                        "execution_time_seconds": 100.0 + i,
                        "memory_peak_gb": 4.0,
                    }
                )

            result = predictor.train(trials)
            assert predictor.is_trained() is True
            assert result["training_samples"] == 20
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_predict_without_training(self):
        """Test predict without training."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()
            with pytest.raises(RuntimeError, match="Model not trained"):
                predictor.predict({}, {}, {})
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_predict_after_training(self):
        """Test predict after training."""
        try:
            from spark_optima.core.simulation.predictor import (
                MLPerformancePredictor,
                PredictionResult,
            )

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Train first
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10, "format": "parquet"},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # Now predict
            result = predictor.predict(
                config={"spark.executor.memory": "4g"},
                data_profile={"size_gb": 10, "format": "parquet"},
                resource_spec={"cpu_cores": 8},
            )

            assert isinstance(result, PredictionResult)
            assert result.predicted_time > 0
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_predict_batch(self):
        """Test batch prediction."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Train first
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # Batch predict
            configs = [{"spark.executor.memory": "4g"} for _ in range(5)]
            results = predictor.predict_batch(
                configs=configs,
                data_profile={"size_gb": 10},
                resource_spec={"cpu_cores": 8},
            )

            assert len(results) == 5
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_save_load_model(self, tmp_path):
        """Test save and load model."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(
                use_ensemble=False, model_path=str(tmp_path / "model.pkl")
            )

            # Train first
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # Save model
            predictor._save_model()

            # Create new predictor and load
            new_predictor = MLPerformancePredictor(model_path=str(tmp_path / "model.pkl"))
            assert new_predictor.is_trained() is True

        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_update_with_trial(self):
        """Test update_with_trial method (not yet fully implemented)."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Train first
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)
            initial_samples = predictor._training_samples

            # Update with new trial (currently just logs debug message)
            new_trial = {
                "execution_time_seconds": 120.0,
                "memory_peak_gb": 5.0,
            }
            predictor.update_with_trial(new_trial)
            # Online learning not yet implemented, so count stays the same
            assert predictor._training_samples == initial_samples
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_get_model_info_v2(self):
        """Test get_model_info method."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Before training
            info = predictor.get_model_info()
            assert info["is_trained"] is False
            assert info["training_samples"] == 0

            # Train
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # After training
            info = predictor.get_model_info()
            assert info["is_trained"] is True
            assert info["training_samples"] == 20
            assert "feature_names" in info
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_is_trained_method(self):
        """Test is_trained method."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()
            assert predictor.is_trained() is False
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_different_formats_encoding(self):
        """Test encoding for different formats."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            # Test all format types
            assert predictor._encode_format("parquet") == 0.8
            assert predictor._encode_format("delta") == 1.0
            assert predictor._encode_format("json") == 0.2
            assert predictor._encode_format("csv") == 0.0
            assert predictor._encode_format("avro") == 0.4
            assert predictor._encode_format("orc") == 0.6
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_different_compression_encoding(self):
        """Test encoding for different compression codecs."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor()

            # Test all compression types
            assert predictor._encode_compression("snappy") == 0.3
            assert predictor._encode_compression("none") == 0.0
            assert predictor._encode_compression("gzip") == 0.7
            assert predictor._encode_compression("lz4") == 0.5
            assert predictor._encode_compression("zstd") == 0.9
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_init_without_sklearn(self):
        """Test predictor initialization when sklearn is not available.

        Note: Lines 29-31 are in the except ImportError block at module level.
        These lines execute only when sklearn cannot be imported during module loading.
        Since sklearn is installed in the test environment, we cannot directly cover
        those lines. Instead, we verify that RuntimeError is raised when
        SKLEARN_AVAILABLE is False (which tests line 98).
        """
        # Mock SKLEARN_AVAILABLE to be False to test line 98
        with patch("spark_optima.core.simulation.predictor.SKLEARN_AVAILABLE", False):
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            with pytest.raises(RuntimeError, match="scikit-learn is required"):
                MLPerformancePredictor()

    def test_predictor_train_too_few_valid_trials_after_filtering(self):
        """Test train method when too few valid trials remain after filtering."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Create trials with invalid execution times (0 or too high)
            trials = []
            for _i in range(15):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 0,  # Invalid: 0 time
                        "memory_peak_gb": 4.0,
                    }
                )

            with patch.object(predictor, "_extract_features") as mock_extract:
                mock_extract.return_value = np.array([[1.0] * 14] * 15)
                result = predictor.train(trials)

            assert result["r2_score"] == 0.0
            assert result["mae"] == float("inf")
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_train_with_ensemble_true(self):
        """Test train method with use_ensemble=True (GradientBoostingRegressor)."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=True)

            # Create 20 trials with proper structure
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {
                            "spark.executor.memory": "4g",
                            "spark.executor.cores": "2",
                        },
                        "data_profile": {
                            "size_gb": 10,
                            "num_columns": 10,
                            "format": "parquet",
                            "compression": "snappy",
                        },
                        "resource_spec": {"cpu_cores": 8},
                        "operations": {
                            "has_aggregation": True,
                            "has_join": False,
                            "has_shuffle": True,
                        },
                        "execution_time_seconds": 100.0 + i,
                        "memory_peak_gb": 4.0,
                    }
                )

            result = predictor.train(trials)
            assert predictor.is_trained() is True
            assert result["training_samples"] == 20
            # Verify GradientBoostingRegressor was used
            from sklearn.ensemble import GradientBoostingRegressor

            assert isinstance(predictor._time_model, GradientBoostingRegressor)
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_predict_with_gradient_boosting_staged_predict(self):
        """Test predict with GradientBoostingRegressor using staged_predict."""
        try:
            from spark_optima.core.simulation.predictor import (
                MLPerformancePredictor,
                PredictionResult,
            )

            predictor = MLPerformancePredictor(use_ensemble=True)

            # Train first with enough data
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10, "format": "parquet"},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # Now predict - should use staged_predict for GradientBoosting
            result = predictor.predict(
                config={"spark.executor.memory": "4g"},
                data_profile={"size_gb": 10, "format": "parquet"},
                resource_spec={"cpu_cores": 8},
            )

            assert isinstance(result, PredictionResult)
            assert result.predicted_time > 0
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_predict_batch_with_exception(self):
        """Test predict_batch when individual prediction fails."""
        try:
            from spark_optima.core.simulation.predictor import (
                MLPerformancePredictor,
                PredictionResult,
            )

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Train first
            trials = []
            for i in range(20):
                trials.append(
                    {
                        "configuration": {"spark.executor.memory": "4g"},
                        "data_profile": {"size_gb": 10},
                        "resource_spec": {"cpu_cores": 8},
                        "execution_time_seconds": 100.0 + i,
                    }
                )

            predictor.train(trials)

            # Mock predict to raise an exception for one config
            # The except block catches (RuntimeError, ValueError, AttributeError, TypeError, KeyError)
            original_predict = predictor.predict
            call_count = [0]

            def mock_predict(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise ValueError("Prediction failed")
                return original_predict(*args, **kwargs)

            with patch.object(predictor, "predict", side_effect=mock_predict):
                configs = [{"spark.executor.memory": "4g"} for _ in range(3)]
                results = predictor.predict_batch(
                    configs=configs,
                    data_profile={"size_gb": 10},
                    resource_spec={"cpu_cores": 8},
                )

            assert len(results) == 3
            # Second result should be a fallback result
            assert isinstance(results[1], PredictionResult)
            assert results[1].predicted_time == 0
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_calculate_feature_importance_model_none(self):
        """Test _calculate_feature_importance when model is None."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Ensure model is None
            predictor._time_model = None

            # Call should return early
            predictor._calculate_feature_importance()

            # No feature_importance should be set
            assert (
                not hasattr(predictor, "_feature_importance") or predictor._feature_importance == {}
            )
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_calculate_feature_importance_no_importances(self):
        """Test _calculate_feature_importance when model lacks feature_importances_."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Mock a model that doesn't have feature_importances_
            mock_model = Mock(spec=[])  # Empty spec means no attributes
            predictor._time_model = mock_model
            predictor._is_trained = True

            predictor._calculate_feature_importance()

            # Should set empty feature_importance
            assert predictor._feature_importance == {}
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_save_model_no_path(self):
        """Test _save_model when model_path is None."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)
            predictor.model_path = None

            # Should return early without error
            predictor._save_model()

            # No error means test passed
            assert True
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_load_model_no_path(self):
        """Test _load_model when model_path is None."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)
            predictor.model_path = None

            # Should return early without error
            predictor._load_model()

            # is_trained should still be False
            assert predictor._is_trained is False
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_load_model_path_not_exists(self):
        """Test _load_model when model_path doesn't exist."""
        try:
            from pathlib import Path

            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)
            predictor.model_path = Path("/nonexistent/path/model.pkl")

            # Should return early without error
            predictor._load_model()

            # is_trained should still be False
            assert predictor._is_trained is False
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_load_model_exception_handling(self, tmp_path):
        """Test _load_model exception handling."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            model_path = tmp_path / "corrupt_model.pkl"

            # Create a corrupt pickle file
            with open(model_path, "wb") as f:
                f.write(b"corrupt data")

            predictor = MLPerformancePredictor(use_ensemble=False)
            predictor.model_path = model_path

            with patch.object(predictor, "_is_trained", False):
                predictor._load_model()

            # After exception, is_trained should be False
            assert predictor._is_trained is False
        except RuntimeError:
            pytest.skip("scikit-learn not available")

    def test_predictor_update_with_trial_not_trained(self):
        """Test update_with_trial when model is not trained."""
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            predictor = MLPerformancePredictor(use_ensemble=False)

            # Ensure model is not trained
            predictor._is_trained = False

            with patch("spark_optima.core.simulation.predictor.logger") as mock_logger:
                predictor.update_with_trial({"execution_time_seconds": 100.0})

                # Verify warning was logged
                mock_logger.warning.assert_called_once_with("Cannot update: model not trained yet")
        except RuntimeError:
            pytest.skip("scikit-learn not available")
