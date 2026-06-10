# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""ML-based performance prediction for Spark jobs.

This module provides machine learning-based performance prediction
that learns from historical trial data to improve estimation accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Optional ML imports with graceful fallback
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML prediction disabled.")


@dataclass
class PredictionResult:
    """Result from ML-based prediction.

    Attributes:
        predicted_time: Predicted execution time in seconds.
        confidence_interval: Tuple of (lower, upper) bounds.
        feature_importance: Dict of feature importances.
        model_version: Version identifier for the model.

    """

    predicted_time: float
    confidence_interval: tuple[float, float]
    feature_importance: dict[str, float]
    model_version: str = "1.0"


class MLPerformancePredictor:
    """Machine learning-based performance predictor for Spark jobs.

    This predictor uses historical trial data to train models that can
    predict job execution time based on configuration and data characteristics.
    It combines multiple ML models for robust predictions.

    Example:
        >>> predictor = MLPerformancePredictor()
        >>> predictor.train(historical_trials)
        >>> result = predictor.predict(config, data_profile, resource_spec)
        >>> print(f"Predicted time: {result.predicted_time:.1f}s")

    """

    # Feature names for model input
    FEATURE_NAMES = [
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

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_ensemble: bool = True,
    ) -> None:
        """Initialize the ML predictor.

        Args:
            model_path: Path to saved model file (optional).
            use_ensemble: Whether to use ensemble of models.

        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required for ML prediction. Install with: pip install scikit-learn",
            )

        self.model_path = Path(model_path) if model_path else None
        self.use_ensemble = use_ensemble

        # Initialize models
        self._time_model: Any = None
        self._memory_model: Any = None
        self._scaler: Any = None

        # Model metadata
        self._is_trained = False
        self._training_samples = 0
        self._model_version = "1.0"

        # Feature statistics for normalization
        self._feature_stats: dict[str, dict[str, float]] = {}

        # Load existing model if available
        if self.model_path and self.model_path.exists():
            self._load_model()

    def train(
        self,
        trials: list[dict[str, Any]],
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the ML model on historical trial data.

        Args:
            trials: List of trial results with features and metrics.
            validation_split: Fraction of data for validation.

        Returns:
            Dictionary with training metrics.

        """
        if len(trials) < 10:
            logger.warning(
                f"Insufficient training data ({len(trials)} samples). Need at least 10 trials for reliable training.",
            )
            return {"r2_score": 0.0, "mae": float("inf")}

        # Extract features and targets
        x = self._extract_features(trials)
        y_time = np.array([t.get("execution_time_seconds", 0) for t in trials])
        y_memory = np.array([t.get("memory_peak_gb", 0) for t in trials])

        # Filter out failed trials (time = 0 or very high)
        valid_mask = (y_time > 0) & (y_time < 3600 * 24)  # Max 24 hours
        x = x[valid_mask]
        y_time = y_time[valid_mask]
        y_memory = y_memory[valid_mask]

        if len(x) < 10:
            logger.warning("Too few valid trials after filtering")
            return {"r2_score": 0.0, "mae": float("inf")}

        # Split data
        x_train, x_val, y_time_train, y_time_val = train_test_split(
            x,
            y_time,
            test_size=validation_split,
            random_state=42,
        )

        # Scale features
        self._scaler = StandardScaler()
        x_train_scaled = self._scaler.fit_transform(x_train)
        x_val_scaled = self._scaler.transform(x_val)

        # Train time prediction model
        if self.use_ensemble:
            self._time_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            self._time_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

        self._time_model.fit(x_train_scaled, y_time_train)

        # Train memory prediction model
        self._memory_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
        )
        _, _, y_mem_train, y_mem_val = train_test_split(
            x,
            y_memory,
            test_size=validation_split,
            random_state=42,
        )
        self._memory_model.fit(x_train_scaled, y_mem_train)

        # Evaluate
        y_time_pred = self._time_model.predict(x_val_scaled)
        r2 = r2_score(y_time_val, y_time_pred)
        mae = mean_absolute_error(y_time_val, y_time_pred)

        # Calculate feature importance
        self._calculate_feature_importance()

        self._is_trained = True
        self._training_samples = len(trials)

        logger.info(
            f"Model trained on {len(trials)} trials. Validation R²={r2:.3f}, MAE={mae:.1f}s",
        )

        # Save model
        if self.model_path:
            self._save_model()

        return {
            "r2_score": r2,
            "mae": mae,
            "training_samples": len(trials),
            "valid_samples": len(x),
        }

    def predict(
        self,
        config: dict[str, Any],
        data_profile: dict[str, Any],
        resource_spec: dict[str, Any],
        operations: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict performance for a configuration.

        Args:
            config: Spark configuration dictionary.
            data_profile: Data characteristics.
            resource_spec: Resource specifications.
            operations: Operation profile (optional).

        Returns:
            PredictionResult with predicted metrics.

        Raises:
            RuntimeError: If model is not trained.

        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        # Extract features
        features = self._extract_single_features(config, data_profile, resource_spec, operations)
        x = np.array([features])

        # Scale features
        x_scaled = self._scaler.transform(x)

        # Predict time
        predicted_time = self._time_model.predict(x_scaled)[0]

        # Calculate confidence interval using prediction intervals
        # For gradient boosting, we can use staged predictions
        if hasattr(self._time_model, "staged_predict"):
            staged_predictions = list(self._time_model.staged_predict(x_scaled))
            # Use variance across stages as confidence measure
            predictions_array = np.array([p[0] for p in staged_predictions[-20:]])
            std_dev = np.std(predictions_array)
        else:
            std_dev = predicted_time * 0.2  # Default 20% uncertainty

        lower_bound = max(0, predicted_time - 1.96 * std_dev)
        upper_bound = predicted_time + 1.96 * std_dev

        # Get feature importance
        importance = self._feature_importance if hasattr(self, "_feature_importance") else {}

        return PredictionResult(
            predicted_time=max(0, predicted_time),
            confidence_interval=(lower_bound, upper_bound),
            feature_importance=importance,
            model_version=self._model_version,
        )

    def predict_batch(
        self,
        configs: list[dict[str, Any]],
        data_profile: dict[str, Any],
        resource_spec: dict[str, Any],
    ) -> list[PredictionResult]:
        """Predict performance for multiple configurations.

        Args:
            configs: List of configuration dictionaries.
            data_profile: Data characteristics.
            resource_spec: Resource specifications.

        Returns:
            List of PredictionResults.

        """
        results = []
        for config in configs:
            try:
                result = self.predict(config, data_profile, resource_spec)
                results.append(result)
            except (RuntimeError, ValueError, AttributeError, TypeError, KeyError) as e:
                logger.warning(f"Prediction failed for config: {e}")
                # Return fallback result
                results.append(
                    PredictionResult(
                        predicted_time=0,
                        confidence_interval=(0, 0),
                        feature_importance={},
                    ),
                )
        return results

    def _extract_features(self, trials: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        """Extract feature matrix from trial data.

        Args:
            trials: List of trial results.

        Returns:
            Feature matrix as numpy array.

        """
        features_list = []

        for trial in trials:
            config = trial.get("configuration", {})
            data_profile = trial.get("data_profile", {})
            resource_spec = trial.get("resource_spec", {})
            operations = trial.get("operations", {})

            features = self._extract_single_features(
                config,
                data_profile,
                resource_spec,
                operations,
            )
            features_list.append(features)

        return np.array(features_list)

    def _extract_single_features(
        self,
        config: dict[str, Any],
        data_profile: dict[str, Any],
        resource_spec: dict[str, Any],
        operations: dict[str, Any] | None = None,
    ) -> list[float]:
        """Extract features from a single trial.

        Args:
            config: Spark configuration.
            data_profile: Data characteristics.
            resource_spec: Resource specifications.
            operations: Operation profile.

        Returns:
            List of feature values.

        """
        operations = operations or {}

        # Parse memory values
        executor_memory = self._parse_memory_value(config.get("spark.executor.memory", "4g"))

        # Extract features
        features = [
            # Executor configuration
            executor_memory,
            float(config.get("spark.executor.cores", 4)),
            float(resource_spec.get("cpu_cores", 4))
            / max(float(config.get("spark.executor.cores", 4)), 1),  # Estimated num_executors
            # Parallelism settings
            float(config.get("spark.default.parallelism", 200)),
            float(config.get("spark.sql.shuffle.partitions", 200)),
            float(config.get("spark.memory.fraction", 0.6)),
            # Data characteristics
            float(data_profile.get("size_gb", 10)),
            float(data_profile.get("num_columns", 10)),
            # Operation flags
            1.0 if operations.get("has_aggregation", False) else 0.0,
            1.0 if operations.get("has_join", False) else 0.0,
            1.0 if operations.get("has_shuffle", False) else 0.0,
            float(data_profile.get("skew_factor", 1.0)),
            # Encoded categorical features
            self._encode_format(data_profile.get("format", "parquet")),
            self._encode_compression(data_profile.get("compression", "snappy")),
        ]

        return features

    def _encode_format(self, format_str: str) -> float:
        """Encode data format as numeric value.

        Args:
            format_str: Format name.

        Returns:
            Encoded value.

        """
        format_map = {
            "csv": 0.0,
            "json": 0.2,
            "avro": 0.4,
            "orc": 0.6,
            "parquet": 0.8,
            "delta": 1.0,
        }
        return format_map.get(format_str.lower(), 0.5)

    def _encode_compression(self, compression: str) -> float:
        """Encode compression codec as numeric value.

        Args:
            compression: Compression codec name.

        Returns:
            Encoded value.

        """
        compression_map = {
            "none": 0.0,
            "snappy": 0.3,
            "lz4": 0.5,
            "gzip": 0.7,
            "zstd": 0.9,
        }
        return compression_map.get(compression.lower(), 0.3)

    def _parse_memory_value(self, value: str | int | float) -> float:
        """Parse memory value to GB.

        Args:
            value: Memory value string or number.

        Returns:
            Memory in GB.

        """
        if isinstance(value, int | float):
            return float(value)

        import re

        value = str(value).strip().lower()
        match = re.match(r"^([\d.]+)\s*([kmgt]?)\s*b?$", value)

        if not match:
            return 4.0

        number = float(match.group(1))
        unit = match.group(2)

        multipliers = {
            "k": 1 / (1024 * 1024),
            "m": 1 / 1024,
            "g": 1,
            "t": 1024,
        }

        return number * multipliers.get(unit, 1)

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance from trained model."""
        if self._time_model is None:
            return

        if hasattr(self._time_model, "feature_importances_"):
            importance = self._time_model.feature_importances_
            self._feature_importance = {
                name: float(imp) for name, imp in zip(self.FEATURE_NAMES, importance, strict=False)
            }
        else:
            self._feature_importance = {}

    def _save_model(self) -> None:
        """Save trained model to disk."""
        if self.model_path is None:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "time_model": self._time_model,
            "memory_model": self._memory_model,
            "scaler": self._scaler,
            "feature_importance": getattr(self, "_feature_importance", {}),
            "model_version": self._model_version,
            "training_samples": self._training_samples,
        }

        joblib.dump(model_data, self.model_path)

        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if self.model_path is None or not self.model_path.exists():
            return

        try:
            model_data = joblib.load(self.model_path)

            self._time_model = model_data["time_model"]
            self._memory_model = model_data["memory_model"]
            self._scaler = model_data["scaler"]
            self._feature_importance = model_data.get("feature_importance", {})
            self._model_version = model_data.get("model_version", "1.0")
            self._training_samples = model_data.get("training_samples", 0)
            self._is_trained = True

            logger.info(
                f"Model loaded from {self.model_path} "
                f"(version {self._model_version}, {self._training_samples} samples)",
            )
        except (FileNotFoundError, ValueError, OSError, KeyError, AttributeError, ImportError) as e:
            logger.error(f"Failed to load model: {e}")
            self._is_trained = False

    def is_trained(self) -> bool:
        """Check if model is trained and ready for prediction.

        Returns:
            True if model is trained.

        """
        return self._is_trained

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model.

        Returns:
            Dictionary mapping feature names to importance scores.

        """
        return getattr(self, "_feature_importance", {}).copy()

    def update_with_trial(self, _trial_result: dict[str, Any]) -> None:
        """Update model with a new trial result (online learning).

        Args:
            trial_result: New trial result dictionary.

        """
        if not self._is_trained:
            logger.warning("Cannot update: model not trained yet")
            return

        # Incremental update not implemented for tree-based models
        # Would require more sophisticated approach like online gradient descent
        logger.debug("Online learning not yet implemented for tree models")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary with model metadata.

        """
        return {
            "is_trained": self._is_trained,
            "model_version": self._model_version,
            "training_samples": self._training_samples,
            "use_ensemble": self.use_ensemble,
            "feature_count": len(self.FEATURE_NAMES),
            "feature_names": self.FEATURE_NAMES,
        }
