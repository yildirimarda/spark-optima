# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Dynamic allocation advisor: reconstruct executor timeline and recommend settings.

This module reads a Spark event log, reconstructs the executor timeline
(from ``SparkListenerExecutorAdded`` / ``Removed`` events), compares observed
peak concurrency with the current ``spark.dynamicAllocation.*`` config, and
recommends ``minExecutors``, ``maxExecutors``, ``initialExecutors``, plus
shuffle-tracking settings.  Idle-executor waste (configured capacity that
was never utilized) is computed as evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3

# Dynamic-allocation parameter names we care about.
_DYN_PARAMS = {
    "spark.dynamicAllocation.enabled",
    "spark.dynamicAllocation.minExecutors",
    "spark.dynamicAllocation.maxExecutors",
    "spark.dynamicAllocation.initialExecutors",
    "spark.dynamicAllocation.executorIdleTimeout",
    "spark.dynamicAllocation.cachedExecutorIdleTimeout",
    "spark.dynamicAllocation.schedulerBacklogTimeout",
    "spark.dynamicAllocation.shuffleTracking.enabled",
    "spark.shuffle.service.enabled",
}


@dataclass
class ExecutorTimelineEvent:
    """A single point in the reconstructed executor timeline."""

    timestamp_ms: int = 0
    executor_id: str = ""
    action: str = ""  # "added" or "removed"


@dataclass
class DynamicAllocationRecommendation:
    """Recommendation produced by the advisor."""

    min_executors: int = 2
    max_executors: int = 20
    initial_executors: int = 2
    shuffle_tracking_enabled: bool = True
    shuffle_service_enabled: bool = True
    idle_executor_waste_seconds: float = 0.0
    idle_executor_waste_percent: float = 0.0
    peak_executors_observed: int = 0
    configured_max_executors: int | None = None
    configured_initial_executors: int | None = None
    configured_min_executors: int | None = None
    evidence_summary: str = ""
    current_config: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_executors": self.min_executors,
            "max_executors": self.max_executors,
            "initial_executors": self.initial_executors,
            "shuffle_tracking_enabled": self.shuffle_tracking_enabled,
            "shuffle_service_enabled": self.shuffle_service_enabled,
            "idle_executor_waste_seconds": self.idle_executor_waste_seconds,
            "idle_executor_waste_percent": self.idle_executor_waste_percent,
            "peak_executors_observed": self.peak_executors_observed,
            "configured_max_executors": self.configured_max_executors,
            "configured_initial_executors": self.configured_initial_executors,
            "configured_min_executors": self.configured_min_executors,
            "evidence_summary": self.evidence_summary,
            "current_config": self.current_config,
        }


class DynamicAllocationAdvisor:
    """Reconstruct executor timeline and recommend dynamic-allocation settings.

    Example:
        >>> advisor = DynamicAllocationAdvisor("/path/to/eventlog")
        >>> rec = advisor.analyze()
        >>> print(rec.evidence_summary)

    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        if not self.log_path.is_file():
            raise FileNotFoundError(f"Event log not found: {self.log_path}")

    def _read_raw_events(self) -> list[dict[str, Any]]:
        """Read every line from the event log file as JSON objects."""
        open_kwargs: dict[str, Any] = {"encoding": "utf-8"}
        # Simple gzip detection by suffix
        if self.log_path.suffix == ".gz" or self.log_path.name.endswith(".gz"):
            import gzip

            with gzip.open(self.log_path, "rt", **open_kwargs) as handle:
                return self._read_lines(handle)
        else:
            with self.log_path.open("r", **open_kwargs) as handle:
                return self._read_lines(handle)

    def _read_lines(self, handle: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                if isinstance(event, dict):
                    events.append(event)
            except json.JSONDecodeError:
                continue
        return events

    def _reconstruct_timeline(self, events: list[dict[str, Any]]) -> list[ExecutorTimelineEvent]:
        timeline: list[ExecutorTimelineEvent] = []
        for event in events:
            event_type = event.get("Event", "")
            if event_type == "SparkListenerExecutorAdded":
                timeline.append(
                    ExecutorTimelineEvent(
                        timestamp_ms=int(event.get("Timestamp", 0) or 0),
                        executor_id=str(event.get("Executor ID", "")),
                        action="added",
                    ),
                )
            elif event_type == "SparkListenerExecutorRemoved":
                timeline.append(
                    ExecutorTimelineEvent(
                        timestamp_ms=int(event.get("Timestamp", 0) or 0),
                        executor_id=str(event.get("Executor ID", "")),
                        action="removed",
                    ),
                )
        # Sort by timestamp to make timeline deterministic
        timeline.sort(key=lambda e: e.timestamp_ms)
        return timeline

    @staticmethod
    def _extract_current_config(events: list[dict[str, Any]]) -> dict[str, str]:
        config: dict[str, str] = {}
        for event in events:
            if event.get("Event") == "SparkListenerEnvironmentUpdate":
                props = event.get("Spark Properties")
                if isinstance(props, dict):
                    for k, v in props.items():
                        if (
                            k in _DYN_PARAMS
                            or k.startswith("spark.dynamicAllocation")
                            or k.startswith("spark.shuffle.service")
                        ):
                            config[str(k)] = str(v)
        return config

    @staticmethod
    def _parse_int(value: str | int | float | None, default: int) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))  # handles "20" and 20.0
        except (TypeError, ValueError):
            return None

    def analyze(self) -> DynamicAllocationRecommendation:
        """Analyze the event log and return recommendations."""
        raw_events = self._read_raw_events()
        timeline = self._reconstruct_timeline(raw_events)
        current_config = self._extract_current_config(raw_events)

        # Compute peak executors observed from timeline
        active_executors = 0
        peak_executors = 0
        for evt in timeline:
            if evt.action == "added":
                active_executors += 1
            elif evt.action == "removed":
                active_executors = max(0, active_executors - 1)
            peak_executors = max(peak_executors, active_executors)

        # Determine current config values
        enabled_str = current_config.get("spark.dynamicAllocation.enabled", "false")
        enabled = enabled_str.lower() in ("true", "1", "yes")

        configured_max = self._parse_int(current_config.get("spark.dynamicAllocation.maxExecutors"), 20)
        configured_min = self._parse_int(current_config.get("spark.dynamicAllocation.minExecutors"), 2)
        configured_initial = self._parse_int(current_config.get("spark.dynamicAllocation.initialExecutors"), 2)

        # If config is missing, fall back to parsing from the raw events
        # or use sensible defaults based on observed workload.
        observed_max = max(peak_executors, 1)

        # Recommend max: either scale with data/shuffle, or cap at observed peak + headroom
        # Use a simple heuristic: max = max(observed_peak * 1.5, 10) bounded to 128
        recommended_max = max(min(int(observed_max * 1.5), 128), 10)
        # If the workload is small, don't over-provision
        if observed_max <= 2:
            recommended_max = max(min(observed_max + 4, 20), 10)

        recommended_min = max(min(observed_max // 2, 5), 2)
        recommended_initial = recommended_min

        # Shuffle tracking settings
        # If dynamic allocation is enabled and there's shuffle volume, recommend tracking
        # We don't have per-event shuffle totals here; we recommend tracking
        # when any completed stage implies shuffle potential.
        has_shuffle_potential = any(evt.get("Event") == "SparkListenerStageCompleted" for evt in raw_events)
        recommended_shuffle_tracking = enabled and has_shuffle_potential
        recommended_shuffle_service = enabled and not recommended_shuffle_tracking

        # Compute idle-executor waste evidence
        # If current max is set, waste = (configured_max - observed_peak) * app_duration / max
        # We approximate app duration from first to last event timestamp
        if raw_events:
            first_ts = min(int(evt.get("Timestamp", 0) or 0) for evt in raw_events if evt.get("Timestamp") is not None)
            last_ts = max(int(evt.get("Timestamp", 0) or 0) for evt in raw_events if evt.get("Timestamp") is not None)
            app_duration_seconds = max((last_ts - first_ts) / 1000.0, 1.0)
        else:
            app_duration_seconds = 1.0

        # Waste calculation: assume each idle executor slot costs full duration
        if configured_max is not None and configured_max > observed_max:
            idle_slots = max(configured_max - observed_max, 0)
            idle_executor_waste_seconds = idle_slots * app_duration_seconds
        else:
            idle_executor_waste_seconds = 0.0

        if configured_max is not None and configured_max > 0:
            idle_executor_waste_percent = (
                idle_executor_waste_seconds / (configured_max * app_duration_seconds)
            ) * 100.0
        else:
            idle_executor_waste_percent = 0.0

        # Evidence summary string
        idle_slots = 0
        if configured_max is not None and configured_max > observed_max:
            idle_slots = max(configured_max - observed_max, 0)
        evidence_parts: list[str] = []
        evidence_parts.append(
            f"Peak executors observed: {observed_max} (timeline from {len(timeline)} add/remove events)"
        )
        if configured_max is not None:
            evidence_parts.append(f"Configured maxExecutors: {configured_max}")
        else:
            evidence_parts.append("No maxExecutors configured (defaults to 20)")
        if idle_executor_waste_seconds > 0:
            evidence_parts.append(
                f"Idle-executor waste: {idle_executor_waste_seconds:.1f}s ({idle_executor_waste_percent:.1f}%) — "
                f"{idle_slots} unused executor slots over "
                f"{app_duration_seconds:.0f}s app duration"
            )
        else:
            evidence_parts.append("No idle-executor waste detected (observed peak matches or exceeds configured max)")

        # Build recommendation
        recommendation = DynamicAllocationRecommendation(
            min_executors=recommended_min,
            max_executors=recommended_max,
            initial_executors=recommended_initial,
            shuffle_tracking_enabled=recommended_shuffle_tracking,
            shuffle_service_enabled=recommended_shuffle_service,
            idle_executor_waste_seconds=idle_executor_waste_seconds,
            idle_executor_waste_percent=idle_executor_waste_percent,
            peak_executors_observed=observed_max,
            configured_max_executors=configured_max,
            configured_initial_executors=configured_initial,
            configured_min_executors=configured_min,
            evidence_summary="; ".join(evidence_parts),
            current_config=current_config,
        )
        return recommendation
