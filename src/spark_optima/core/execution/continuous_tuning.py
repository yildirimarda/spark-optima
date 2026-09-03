# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Continuous retuning from production history.

Polls a Spark History Server (``execution/history_server.py``) on a
schedule, detects workload drift versus the config's original profile,
and produces a "re-tune recommended" report when the surrogate model
predicts >10% improvement.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from spark_optima.core.execution.event_log import EventLogSummary
from spark_optima.core.execution.history_server import HistoryServerClient, HistoryServerError
from spark_optima.core.heuristics.context import DataProfile
from spark_optima.core.heuristics.engine import HeuristicEngine
from spark_optima.core.history import OptimizationHistory
from spark_optima.core.simulation.predictor import (
    extract_features,
    parse_memory_gb,
)
from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)

IMPROVEMENT_THRESHOLD = 0.10  # 10%


@dataclass
class RetuneReport:
    """ "Re-tune recommended" report produced from production history."""

    recommended: bool = False
    improvement_percent: float = 0.0
    current_summary: dict[str, Any] = field(default_factory=dict)
    original_config: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    predicted_time_current: float = 0.0
    predicted_time_recommended: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "improvement_percent": self.improvement_percent,
            "message": self.message,
            "predicted_time_current": self.predicted_time_current,
            "predicted_time_recommended": self.predicted_time_recommended,
            "current_summary": self.current_summary,
            "original_config": self.original_config,
        }


class ContinuousRetuner:
    """Polls a Spark History Server and evaluates retuning recommendations.

    Example:
        >>> retuner = ContinuousRetuner(
        ...     history_server_url="http://history:18080",
        ...     original_config={"spark.executor.memory": "4g"},
        ...     app_id="app-20260101-0001",
        ... )
        >>> retuner.schedule_poll(interval_seconds=300)
        >>> # Later, after a run completes:
        >>> report = retuner.generate_report(retuner.poll())
    """

    def __init__(
        self,
        history_server_url: str,
        original_config: dict[str, Any],
        app_id: str,
        predictor: Any | None = None,
    ) -> None:
        self.client = HistoryServerClient(history_server_url)
        self.original_config = original_config.copy()
        self.app_id = app_id
        # Lazy predictor import to keep dependency optional
        self._predictor_factory = predictor
        self._timer: threading.Timer | None = None
        self._running = False
        self._last_report: RetuneReport | None = None

    def _get_predictor(self) -> Any:
        if self._predictor_factory is not None:
            return self._predictor_factory
        try:
            from spark_optima.core.simulation.predictor import MLPerformancePredictor

            return MLPerformancePredictor(use_ensemble=False)
        except Exception as exc:
            logger.debug("Predictor unavailable: %s", exc)
            return None

    def poll(self) -> EventLogSummary:
        """Fetch the latest summary from the History Server."""
        try:
            return self.client.fetch_summary(self.app_id)
        except HistoryServerError:
            # Fall back to empty summary so callers don't crash on missing apps
            logger.warning("History Server fetch failed for %s; returning empty summary", self.app_id)
            return EventLogSummary()

    def detect_drift(self, summary: EventLogSummary) -> bool:
        """Detect workload drift vs the original profile."""
        hints = summary.to_tuning_hints()
        original_profile = self.original_config.get("_data_profile", {})
        original_size = original_profile.get("size_gb", hints.get("data_size_gb", 0.0))
        current_size = hints.get("data_size_gb", 0.0)
        # Drift when data size changed >20% or GC/shuffle patterns changed
        gc_shift = (
            abs(float(hints.get("gc_time_fraction", 0.0)) - float(original_profile.get("gc_time_fraction", 0.0))) > 0.05
        )
        size_ratio = 0.0
        if original_size > 0 and current_size > 0:
            size_ratio = abs(current_size - original_size) / original_size
        # Only consider size shift when we have production data on both sides
        significant_shift = (size_ratio > 0.20) and (current_size > 0)
        return significant_shift or gc_shift

    def _recommended_config_from_hints(self, hints: dict[str, Any], base_config: dict[str, Any]) -> dict[str, Any]:
        """Derive a retuned config from tuning hints via HeuristicEngine."""
        # Derive resource spec from base_config when available; fall back to defaults
        cpu_cores = 4
        try:
            cpu_cores = int(str(base_config.get("spark.executor.cores", "4")).replace(" ", ""))
        except (ValueError, TypeError):
            cpu_cores = 4

        mem_str = base_config.get("spark.executor.memory", "4g")
        try:
            mem_gb = parse_memory_gb(mem_str, default=4.0)
        except Exception:
            mem_gb = 4.0

        # Use a rough total memory estimate (executor memory * 4) for heuristics
        resources = ResourceSpec(cpu_cores=cpu_cores, memory_gb=max(mem_gb * 4, 8.0))

        # Derive platform and version from base_config or defaults
        platform = base_config.get("platform", "local")
        version = base_config.get("spark_version", "3.5.0")

        data_profile = DataProfile(size_gb=hints.get("data_size_gb", 10.0))
        custom_vars = dict(hints)

        engine = HeuristicEngine()
        retuned = engine.evaluate(
            resources=resources,
            platform=platform,
            spark_version=version,
            data_profile=data_profile,
            custom_vars=custom_vars,
        )

        # Merge engine recommendations over the base config for consistent recommendations
        result = base_config.copy()
        result.update(retuned)
        return result

    def generate_report(
        self,
        summary: EventLogSummary,
        retuned_config: dict[str, Any] | None = None,
    ) -> RetuneReport:
        """Generate a retune recommendation based on production metrics."""
        predictor = self._get_predictor()
        hints = summary.to_tuning_hints()
        base_config = self.original_config.copy()
        # Derive retuned config from hints
        recommended_config = retuned_config or self._recommended_config_from_hints(hints, base_config)

        # Build feature vectors
        current_features = extract_features(
            base_config,
            {"size_gb": hints.get("data_size_gb", 10.0)},
        )
        retuned_features = extract_features(
            recommended_config,
            {"size_gb": hints.get("data_size_gb", 10.0)},
        )

        predicted_current = 0.0
        predicted_recommended = 0.0

        if predictor is not None and hasattr(predictor, "add_sample"):
            # Train the online surrogate on real measured trials from SQLite
            # OptimizationHistory rather than synthetic samples, so predictions
            # reflect actual production workload drift.
            try:
                with OptimizationHistory() as history:
                    entries = history.list_entries(limit=500)
                    for entry in entries:
                        # Only use execution-mode entries as real measured trials
                        if entry.mode != "execution":
                            continue
                        config = entry.configuration or {}
                        result = entry.result or {}
                        # Derive a basic data profile from result metadata or defaults
                        data_profile = result.get("metadata", {}) or {}
                        if not isinstance(data_profile, dict) or not data_profile.get("size_gb"):
                            data_profile = {"size_gb": hints.get("data_size_gb", 10.0)}
                        features = extract_features(config, data_profile)
                        # Convert estimated minutes to seconds for training target
                        target_seconds = float(entry.estimated_time_minutes or 0.0) * 60.0
                        if target_seconds > 0:
                            predictor.add_sample(features, target_time=target_seconds, measured=True)
                # Train if enough real samples accumulated
                result = predictor.train_online(min_samples=4)
                if result.get("trained"):
                    pred_current = predictor.predict_online(current_features)
                    pred_recommended = predictor.predict_online(retuned_features)
                    if pred_current is not None and pred_recommended is not None:
                        predicted_current = float(pred_current)
                        predicted_recommended = float(pred_recommended)
            except Exception as exc:
                logger.debug("Surrogate training/prediction failed: %s", exc)

        # Fallback when predictor unavailable: estimate improvement from hints
        if predicted_current <= 0:
            # Heuristic estimate: larger shuffle/gc = longer time
            shuffle_total = hints.get("shuffle_total_gb", 0.0)
            gc_frac = hints.get("gc_time_fraction", 0.0)
            predicted_current = 300.0 + shuffle_total * 10.0 + gc_frac * 500.0
            improvement_factor = 0.0
            if hints.get("large_shuffles"):
                improvement_factor += 0.15
            if hints.get("memory_intensive"):
                improvement_factor += 0.10
            if hints.get("skew_factor", 1.0) > 1.5:
                improvement_factor += 0.08
            predicted_recommended = predicted_current * (1.0 - improvement_factor)

        improvement_pct = 0.0
        if predicted_current > 0:
            improvement_pct = (predicted_current - predicted_recommended) / predicted_current

        recommended = improvement_pct > IMPROVEMENT_THRESHOLD
        message = (
            f"Re-tune recommended: predicted {improvement_pct:.1%} improvement "
            f"({predicted_current:.0f}s -> {predicted_recommended:.0f}s)"
            if recommended
            else f"No significant improvement predicted ({improvement_pct:.1%})"
        )

        report = RetuneReport(
            recommended=recommended,
            improvement_percent=improvement_pct,
            current_summary=summary.to_dict(),
            original_config=base_config,
            message=message,
            predicted_time_current=predicted_current,
            predicted_time_recommended=predicted_recommended,
        )
        self._last_report = report
        return report

    def schedule_poll(self, interval_seconds: float = 300.0) -> None:
        """Schedule periodic polling of the History Server."""
        if self._running:
            return
        self._running = True

        def _poll() -> None:
            try:
                summary = self.poll()
                report = self.generate_report(summary)
                if report.recommended:
                    logger.info("%s", report.message)
            except Exception as exc:
                logger.debug("Scheduled poll failed: %s", exc)
            finally:
                if self._running:
                    self._timer = threading.Timer(interval_seconds, _poll)
                    self._timer.start()

        self._timer = threading.Timer(interval_seconds, _poll)
        self._timer.start()
        logger.info("Scheduled retune poll for %s every %.0fs", self.app_id, interval_seconds)

    def stop_schedule(self) -> None:
        """Stop the scheduled polling loop."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def get_last_report(self) -> RetuneReport | None:
        return self._last_report
