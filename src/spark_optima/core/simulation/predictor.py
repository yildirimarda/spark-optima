# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""ML-based performance prediction for Spark jobs.

This module provides machine learning-based performance prediction
that learns from historical trial data to improve estimation accuracy.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

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

#: Environment variable that overrides the default model persistence directory.
MODEL_DIR_ENV_VAR = "SPARK_OPTIMA_MODEL_DIR"

#: Fixed feature order for the online surrogate model.
#: Training and prediction always use this exact order so they can never misalign.
SURROGATE_FEATURE_NAMES: tuple[str, ...] = (
    "executor_memory_gb",
    "driver_memory_gb",
    "executor_memory_overhead_gb",
    "executor_cores",
    "shuffle_partitions",
    "default_parallelism",
    "aqe_enabled",
    "dynamic_allocation_enabled",
    "data_size_gb",
)

#: Stable defaults used when a parameter is missing from the configuration or data profile.
SURROGATE_FEATURE_DEFAULTS: dict[str, float] = {
    "executor_memory_gb": 4.0,
    "driver_memory_gb": 2.0,
    "executor_memory_overhead_gb": 0.4,
    "executor_cores": 4.0,
    "shuffle_partitions": 200.0,
    "default_parallelism": 200.0,
    "aqe_enabled": 1.0,
    "dynamic_allocation_enabled": 0.0,
    "data_size_gb": 10.0,
}

_MEMORY_PATTERN = re.compile(r"^([\d.]+)\s*([kmgt]?)\s*b?$")
_MEMORY_MULTIPLIERS = {"k": 1 / (1024 * 1024), "m": 1 / 1024, "g": 1.0, "t": 1024.0}


def parse_memory_gb(value: str | int | float, default: float = 4.0) -> float:
    """Parse a Spark memory value (e.g. ``4g``, ``512m``, ``1t``) to gigabytes.

    Args:
        value: Memory value as a Spark size string or a plain number (already GB).
        default: Value returned when the string cannot be parsed.

    Returns:
        Memory in gigabytes.

    """
    if isinstance(value, int | float):
        return float(value)

    match = _MEMORY_PATTERN.match(str(value).strip().lower())
    if not match:
        return default

    number = float(match.group(1))
    unit = match.group(2)
    return number * _MEMORY_MULTIPLIERS.get(unit, 1.0)


def _parse_bool_flag(value: Any, default: bool) -> float:
    """Parse a Spark boolean config value to 0.0/1.0 with a stable default.

    Args:
        value: Raw config value (bool, string, or None).
        default: Default used when the value is missing or unrecognized.

    Returns:
        1.0 for true, 0.0 for false.

    """
    if value is None:
        return 1.0 if default else 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0

    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return 1.0
    if text in ("false", "0", "no"):
        return 0.0
    return 1.0 if default else 0.0


def _parse_numeric(value: Any, default: float) -> float:
    """Parse a numeric config value with a stable default for missing or invalid input.

    Args:
        value: Raw config value.
        default: Default used when the value is missing or invalid.

    Returns:
        Parsed finite float or the default.

    """
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def extract_features(
    config: dict[str, Any],
    data_profile: dict[str, Any] | None = None,
) -> list[float]:
    """Map a Spark configuration and data profile onto a fixed-order numeric feature vector.

    The output order is exactly ``SURROGATE_FEATURE_NAMES`` so that training and
    prediction can never misalign. Missing parameters fall back to the stable
    defaults in ``SURROGATE_FEATURE_DEFAULTS``. This function is deterministic and
    has no scikit-learn dependency.

    Args:
        config: Spark configuration dictionary (``spark.*`` keys).
        data_profile: Optional data characteristics (uses ``size_gb``).

    Returns:
        Feature vector as a list of floats, one entry per ``SURROGATE_FEATURE_NAMES`` name.

    """
    profile = data_profile or {}
    defaults = SURROGATE_FEATURE_DEFAULTS

    values: dict[str, float] = {
        "executor_memory_gb": parse_memory_gb(
            config.get("spark.executor.memory", defaults["executor_memory_gb"]),
            default=defaults["executor_memory_gb"],
        ),
        "driver_memory_gb": parse_memory_gb(
            config.get("spark.driver.memory", defaults["driver_memory_gb"]),
            default=defaults["driver_memory_gb"],
        ),
        "executor_memory_overhead_gb": parse_memory_gb(
            config.get("spark.executor.memoryOverhead", defaults["executor_memory_overhead_gb"]),
            default=defaults["executor_memory_overhead_gb"],
        ),
        "executor_cores": _parse_numeric(config.get("spark.executor.cores"), defaults["executor_cores"]),
        "shuffle_partitions": _parse_numeric(
            config.get("spark.sql.shuffle.partitions"),
            defaults["shuffle_partitions"],
        ),
        "default_parallelism": _parse_numeric(
            config.get("spark.default.parallelism"),
            defaults["default_parallelism"],
        ),
        "aqe_enabled": _parse_bool_flag(config.get("spark.sql.adaptive.enabled"), default=True),
        "dynamic_allocation_enabled": _parse_bool_flag(
            config.get("spark.dynamicAllocation.enabled"),
            default=False,
        ),
        "data_size_gb": _parse_numeric(profile.get("size_gb"), defaults["data_size_gb"]),
    }

    return [values[name] for name in SURROGATE_FEATURE_NAMES]


def _sanitize_token(token: str) -> str:
    """Sanitize a platform or version token for use in a file name.

    Args:
        token: Raw token string.

    Returns:
        Token with unsafe characters replaced by hyphens (never empty).

    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", token.strip())
    return cleaned or "default"


def default_model_path(platform: str = "local", spark_version: str = "default") -> Path:
    """Build the default persistence path for a surrogate model.

    The directory defaults to ``~/.spark_optima/models`` and can be overridden via
    the ``SPARK_OPTIMA_MODEL_DIR`` environment variable. The file name is derived
    from the platform and Spark version so models trained for different targets
    never collide.

    Args:
        platform: Platform identifier (e.g. ``local``, ``aws_emr``).
        spark_version: Spark version string (e.g. ``3.5``).

    Returns:
        Path to the model file (the file may not exist yet).

    """
    env_dir = os.environ.get(MODEL_DIR_ENV_VAR)
    base_dir = Path(env_dir).expanduser() if env_dir else Path.home() / ".spark_optima" / "models"
    return base_dir / f"surrogate_{_sanitize_token(platform)}_{_sanitize_token(spark_version)}.joblib"


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

    # Maximum number of online samples kept in memory (oldest are dropped first)
    MAX_ONLINE_SAMPLES = 5000

    # Persistence payload format version
    PERSISTENCE_FORMAT_VERSION = 1

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_ensemble: bool = True,
        platform: str = "local",
        spark_version: str = "default",
    ) -> None:
        """Initialize the ML predictor.

        Args:
            model_path: Path to saved model file (optional).
            use_ensemble: Whether to use ensemble of models.
            platform: Platform identifier used for default persistence naming.
            spark_version: Spark version used for default persistence naming.

        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required for ML prediction. Install with: pip install scikit-learn",
            )

        self.model_path = Path(model_path) if model_path else None
        self.use_ensemble = use_ensemble
        self.platform = platform
        self.spark_version = spark_version

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

        # Online surrogate state: samples accumulated across trials within a run
        self._online_model: Any = None
        self._online_r2: float = 0.0
        self._online_trained_samples: int = 0
        self._online_features: list[list[float]] = []
        self._online_targets: list[float] = []
        self._online_measured: list[bool] = []

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
        return parse_memory_gb(value, default=4.0)

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

    # ------------------------------------------------------------------
    # Online surrogate API (v1.3, Workstream L)
    # ------------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        """Number of online samples accumulated so far."""
        return len(self._online_targets)

    @property
    def measured_sample_count(self) -> int:
        """Number of online samples that come from real measured runs."""
        return sum(1 for measured in self._online_measured if measured)

    @property
    def measured_fraction(self) -> float:
        """Fraction of online samples backed by real measurements (0.0-1.0)."""
        total = self.sample_count
        return self.measured_sample_count / total if total else 0.0

    @property
    def online_r2(self) -> float:
        """Hold-out validation R² from the last online training (0.0 when untrained)."""
        return self._online_r2

    @property
    def online_trained_samples(self) -> int:
        """Sample count at the time of the last online training (0 when untrained)."""
        return self._online_trained_samples

    def has_online_model(self) -> bool:
        """Check whether the online surrogate model has been trained.

        Returns:
            True if an online model is available for prediction.

        """
        return self._online_model is not None

    def add_sample(
        self,
        features: Sequence[float],
        target_time: float,
        measured: bool = False,
    ) -> bool:
        """Add an online training sample.

        Args:
            features: Feature vector in ``SURROGATE_FEATURE_NAMES`` order.
            target_time: Execution time in seconds (analytical or measured).
            measured: True when the target comes from a real measured run.

        Returns:
            True if the sample was accepted, False if it was rejected as invalid.

        """
        try:
            feature_list = [float(value) for value in features]
        except (TypeError, ValueError):
            logger.debug("Rejected online sample: features are not numeric")
            return False

        if len(feature_list) != len(SURROGATE_FEATURE_NAMES):
            logger.debug(
                f"Rejected online sample: expected {len(SURROGATE_FEATURE_NAMES)} features, got {len(feature_list)}",
            )
            return False
        if not all(math.isfinite(value) for value in feature_list):
            logger.debug("Rejected online sample: non-finite feature value")
            return False
        if not isinstance(target_time, int | float) or not math.isfinite(target_time) or target_time <= 0:
            logger.debug("Rejected online sample: invalid target time")
            return False

        self._online_features.append(feature_list)
        self._online_targets.append(float(target_time))
        self._online_measured.append(measured)

        # Bound memory usage by dropping the oldest samples
        while len(self._online_targets) > self.MAX_ONLINE_SAMPLES:
            self._online_features.pop(0)
            self._online_targets.pop(0)
            self._online_measured.pop(0)

        return True

    def train_online(
        self,
        min_samples: int = 20,
        validation_split: float = 0.25,
    ) -> dict[str, Any]:
        """Train the online surrogate model on accumulated samples.

        A hold-out split is used to compute a validation R² that callers can use
        to gate how much weight the surrogate prediction receives. The final
        model is refit on all samples after validation.

        Args:
            min_samples: Minimum number of samples required for training.
            validation_split: Fraction of samples held out for validation.

        Returns:
            Dictionary with ``trained``, ``r2``, ``n_samples`` and (when trained)
            ``measured_fraction`` keys.

        """
        n_samples = self.sample_count
        if n_samples < max(min_samples, 4):
            logger.debug(f"Online training skipped: {n_samples} samples < {min_samples} required")
            return {"trained": False, "r2": 0.0, "n_samples": n_samples, "reason": "insufficient_samples"}

        x = np.asarray(self._online_features, dtype=float)
        y = np.asarray(self._online_targets, dtype=float)

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=validation_split,
            random_state=42,
        )

        model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=1)
        model.fit(x_train, y_train)

        if len(y_val) >= 2 and float(np.std(y_val)) > 0.0:
            r2 = float(r2_score(y_val, model.predict(x_val)))
            if not math.isfinite(r2):
                r2 = 0.0
        else:
            # R² is undefined for constant or single-point validation targets
            r2 = 0.0

        # Refit on the full sample set so the deployed model uses all data
        model.fit(x, y)

        self._online_model = model
        self._online_r2 = r2
        self._online_trained_samples = n_samples

        logger.debug(
            f"Online surrogate trained on {n_samples} samples "
            f"({self.measured_sample_count} measured), validation R²={r2:.3f}",
        )

        return {
            "trained": True,
            "r2": r2,
            "n_samples": n_samples,
            "measured_fraction": self.measured_fraction,
        }

    def predict_online(self, features: Sequence[float]) -> float | None:
        """Predict execution time with the online surrogate model.

        Args:
            features: Feature vector in ``SURROGATE_FEATURE_NAMES`` order.

        Returns:
            Predicted execution time in seconds, or None when no model is
            available or the prediction is invalid.

        """
        if self._online_model is None:
            return None

        try:
            x = np.asarray([list(features)], dtype=float)
            prediction = float(self._online_model.predict(x)[0])
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning(f"Online surrogate prediction failed: {exc}")
            return None

        if not math.isfinite(prediction) or prediction <= 0:
            return None
        return prediction

    # ------------------------------------------------------------------
    # Persistence (v1.3, Workstream L)
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Save the predictor state (online surrogate + legacy models) via joblib.

        Args:
            path: Target file path. Defaults to ``default_model_path()`` derived
                from this predictor's platform and Spark version (overridable via
                the ``SPARK_OPTIMA_MODEL_DIR`` environment variable).

        Returns:
            The path the model was saved to.

        """
        target = Path(path) if path is not None else default_model_path(self.platform, self.spark_version)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": self.PERSISTENCE_FORMAT_VERSION,
            "feature_names": list(SURROGATE_FEATURE_NAMES),
            # Online surrogate state
            "online_model": self._online_model,
            "online_r2": self._online_r2,
            "online_trained_samples": self._online_trained_samples,
            "online_features": self._online_features,
            "online_targets": self._online_targets,
            "online_measured": self._online_measured,
            # Legacy batch-trained state
            "time_model": self._time_model,
            "memory_model": self._memory_model,
            "scaler": self._scaler,
            "feature_importance": getattr(self, "_feature_importance", {}),
            "model_version": self._model_version,
            "training_samples": self._training_samples,
        }

        joblib.dump(payload, target)
        logger.info(f"Predictor state saved to {target}")
        return target

    def load(self, path: str | Path | None = None) -> bool:
        """Load predictor state previously written by ``save()``.

        Args:
            path: Source file path. Defaults to ``default_model_path()`` derived
                from this predictor's platform and Spark version.

        Returns:
            True if the state was loaded, False when the file is missing,
            corrupt, or was saved with an incompatible feature schema.

        """
        source = Path(path) if path is not None else default_model_path(self.platform, self.spark_version)
        if not source.exists():
            logger.debug(f"No saved predictor state at {source}")
            return False

        try:
            payload = joblib.load(source)
        except (FileNotFoundError, ValueError, OSError, KeyError, AttributeError, ImportError, EOFError) as exc:
            logger.error(f"Failed to load predictor state from {source}: {exc}")
            return False

        if not isinstance(payload, dict):
            logger.warning(f"Ignoring saved predictor state at {source}: unexpected payload type")
            return False

        if payload.get("feature_names") != list(SURROGATE_FEATURE_NAMES):
            logger.warning(f"Ignoring saved predictor state at {source}: feature schema mismatch")
            return False

        self._online_model = payload.get("online_model")
        self._online_r2 = float(payload.get("online_r2", 0.0))
        self._online_trained_samples = int(payload.get("online_trained_samples", 0))
        self._online_features = list(payload.get("online_features", []))
        self._online_targets = list(payload.get("online_targets", []))
        self._online_measured = list(payload.get("online_measured", []))

        # Restore legacy batch-trained state when present
        if payload.get("time_model") is not None:
            self._time_model = payload["time_model"]
            self._memory_model = payload.get("memory_model")
            self._scaler = payload.get("scaler")
            self._feature_importance = payload.get("feature_importance", {})
            self._model_version = payload.get("model_version", "1.0")
            self._training_samples = payload.get("training_samples", 0)
            self._is_trained = True

        logger.info(
            f"Predictor state loaded from {source} ({self.sample_count} online samples, R²={self._online_r2:.3f})",
        )
        return True
