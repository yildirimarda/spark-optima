# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spark event log parsing for post-run analysis.

This module provides EventLogParser, which reads Spark event log files
(JSON-lines, optionally gzip-compressed) and condenses them into an
EventLogSummary with stage, GC, shuffle, spill, and skew metrics that can
be fed back into the optimization pipeline as tuning hints.
"""

from __future__ import annotations

import gzip
import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
GZIP_MAGIC = b"\x1f\x8b"

# Thresholds used when deriving tuning hints from a summary.
LARGE_SHUFFLE_THRESHOLD_GB = 10.0
GC_PRESSURE_THRESHOLD = 0.1
SKEW_MODERATE_THRESHOLD = 1.5
SKEW_SEVERE_THRESHOLD = 5.0


@dataclass
class StageSummary:
    """Aggregated metrics for a single completed stage.

    Attributes:
        stage_id: Stage identifier.
        name: Stage name from "Stage Info".
        duration_seconds: Stage wall-clock duration (completion - submission).
        num_tasks: Number of tasks in the stage.
        shuffle_read_gb: Shuffle bytes read (remote + local) in GB.
        shuffle_write_gb: Shuffle bytes written in GB.
        spill_gb: Memory + disk bytes spilled in GB.
        skew_ratio: Max task duration / median task duration (1.0 if unknown).

    """

    stage_id: int
    name: str = ""
    duration_seconds: float = 0.0
    num_tasks: int = 0
    shuffle_read_gb: float = 0.0
    shuffle_write_gb: float = 0.0
    spill_gb: float = 0.0
    skew_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "num_tasks": self.num_tasks,
            "shuffle_read_gb": self.shuffle_read_gb,
            "shuffle_write_gb": self.shuffle_write_gb,
            "spill_gb": self.spill_gb,
            "skew_ratio": self.skew_ratio,
        }


@dataclass
class EventLogSummary:
    """Condensed view of a Spark application run extracted from an event log.

    Attributes:
        app_name: Application name from SparkListenerApplicationStart.
        app_duration_seconds: Application wall-clock duration.
        total_tasks: Number of task-end events observed.
        failed_tasks: Number of failed tasks.
        total_gc_time_seconds: Cumulative JVM GC time across tasks.
        gc_time_fraction: GC time as a fraction of total executor run time.
        total_shuffle_read_gb: Cumulative shuffle bytes read in GB.
        total_shuffle_write_gb: Cumulative shuffle bytes written in GB.
        total_spill_gb: Cumulative memory + disk spill in GB.
        peak_execution_memory_gb: Max per-task peak execution memory in GB.
        executor_count_max: Maximum number of concurrently running executors.
        input_data_gb: Cumulative input bytes read in GB.
        max_skew_ratio: Maximum per-stage task skew ratio across all stages.
        stages: Per-stage summaries, ordered by stage id.
        spark_conf: Spark properties from SparkListenerEnvironmentUpdate.
        skipped_lines: Number of unparseable lines that were ignored.

    """

    app_name: str = ""
    app_duration_seconds: float = 0.0
    total_tasks: int = 0
    failed_tasks: int = 0
    total_gc_time_seconds: float = 0.0
    gc_time_fraction: float = 0.0
    total_shuffle_read_gb: float = 0.0
    total_shuffle_write_gb: float = 0.0
    total_spill_gb: float = 0.0
    peak_execution_memory_gb: float = 0.0
    executor_count_max: int = 0
    input_data_gb: float = 0.0
    max_skew_ratio: float = 1.0
    stages: list[StageSummary] = field(default_factory=list)
    spark_conf: dict[str, str] = field(default_factory=dict)
    skipped_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "app_name": self.app_name,
            "app_duration_seconds": self.app_duration_seconds,
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "total_gc_time_seconds": self.total_gc_time_seconds,
            "gc_time_fraction": self.gc_time_fraction,
            "total_shuffle_read_gb": self.total_shuffle_read_gb,
            "total_shuffle_write_gb": self.total_shuffle_write_gb,
            "total_spill_gb": self.total_spill_gb,
            "peak_execution_memory_gb": self.peak_execution_memory_gb,
            "executor_count_max": self.executor_count_max,
            "input_data_gb": self.input_data_gb,
            "max_skew_ratio": self.max_skew_ratio,
            "stages": [stage.to_dict() for stage in self.stages],
            "spark_conf": self.spark_conf,
            "skipped_lines": self.skipped_lines,
        }

    def to_tuning_hints(self) -> dict[str, Any]:
        """Derive optimization-relevant hints from this summary.

        The emitted keys align with the variable names consumed by the
        heuristic EvaluationContext (see core/heuristics/context.py) and
        rule conditions (see core/heuristics/rules.py):

        - ``data_size_gb`` (float): input bytes read; maps to
          ``DataProfile.size_gb``. Only emitted when greater than zero.
        - ``skew_factor`` (float): max stage skew ratio; consumed by the
          ``skew_factor > 1.5`` rule condition.
        - ``large_shuffles`` (bool): total shuffle volume exceeds
          ``LARGE_SHUFFLE_THRESHOLD_GB``; consumed by ``large_shuffles``
          rule conditions and ``EvaluationContext.has_large_shuffles()``.
        - ``gc_pressure`` (bool): ``gc_time_fraction`` exceeds
          ``GC_PRESSURE_THRESHOLD``.
        - ``gc_time_fraction`` (float): GC share of executor run time.
        - ``spill_detected`` (bool): any memory/disk spill was recorded.
        - ``spill_gb`` (float): total spilled volume in GB.
        - ``shuffle_total_gb`` (float): shuffle read + write volume in GB.
        - ``memory_intensive`` (bool): spill or GC pressure observed;
          consumed by ``EvaluationContext.is_memory_intensive()``.

        Returns:
            Dictionary of tuning hints suitable for merging into heuristic
            custom variables and data profiles.

        """
        shuffle_total_gb = self.total_shuffle_read_gb + self.total_shuffle_write_gb
        gc_pressure = self.gc_time_fraction > GC_PRESSURE_THRESHOLD
        spill_detected = self.total_spill_gb > 0.0

        hints: dict[str, Any] = {
            "skew_factor": self.max_skew_ratio,
            "large_shuffles": shuffle_total_gb > LARGE_SHUFFLE_THRESHOLD_GB,
            "gc_pressure": gc_pressure,
            "gc_time_fraction": self.gc_time_fraction,
            "spill_detected": spill_detected,
            "spill_gb": self.total_spill_gb,
            "shuffle_total_gb": shuffle_total_gb,
            "memory_intensive": spill_detected or gc_pressure,
        }
        if self.input_data_gb > 0.0:
            hints["data_size_gb"] = self.input_data_gb
        return hints


@dataclass
class _StageAccumulator:
    """Internal mutable accumulator for per-stage metrics."""

    stage_id: int
    name: str = ""
    duration_seconds: float = 0.0
    declared_tasks: int = 0
    task_durations_ms: list[float] = field(default_factory=list)
    shuffle_read_bytes: int = 0
    shuffle_write_bytes: int = 0
    spill_bytes: int = 0
    has_task_metrics: bool = False


def _as_int(value: Any) -> int:
    """Coerce an accumulable/metric value to int, returning 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class EventLogParser:
    """Parser for Spark event log files.

    Reads a Spark event log (JSON-lines format, one SparkListener event per
    line; plain text or gzip-compressed) and produces an EventLogSummary.
    Unparseable lines are skipped and counted instead of failing the parse.

    Example:
        >>> parser = EventLogParser("/path/to/eventlog")
        >>> summary = parser.parse()
        >>> print(f"GC fraction: {summary.gc_time_fraction:.1%}")

    """

    def __init__(self, log_path: str | Path) -> None:
        """Initialize the parser.

        Args:
            log_path: Path to the event log file (plain or ``.gz``).

        Raises:
            FileNotFoundError: If the log file does not exist.

        """
        self.log_path = Path(log_path)
        if not self.log_path.is_file():
            raise FileNotFoundError(f"Event log not found: {self.log_path}")

    def parse(self) -> EventLogSummary:
        """Parse the event log into a summary.

        Returns:
            EventLogSummary with aggregated application, stage, and task metrics.

        """
        self._reset_state()

        with self._open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    self._skipped_lines += 1
                    continue
                if not isinstance(event, dict):
                    self._skipped_lines += 1
                    continue
                self._handle_event(event)

        return self._build_summary()

    def _open(self) -> IO[str]:
        """Open the log file, transparently handling gzip compression.

        Detection uses the ``.gz`` suffix or the gzip magic bytes.

        Returns:
            Text-mode file handle.

        """
        with self.log_path.open("rb") as raw:
            magic = raw.read(2)
        if self.log_path.suffix == ".gz" or magic == GZIP_MAGIC:
            return gzip.open(self.log_path, "rt", encoding="utf-8")
        return self.log_path.open("r", encoding="utf-8")

    def _reset_state(self) -> None:
        """Reset all parse-state accumulators."""
        self._app_name = ""
        self._app_start_ms: int | None = None
        self._app_end_ms: int | None = None
        self._spark_conf: dict[str, str] = {}
        self._stages: dict[int, _StageAccumulator] = {}
        self._total_tasks = 0
        self._failed_tasks = 0
        self._total_gc_ms = 0
        self._total_run_ms = 0
        self._input_bytes = 0
        self._peak_execution_memory_bytes = 0
        self._executor_count = 0
        self._executor_count_max = 0
        self._skipped_lines = 0

    def _handle_event(self, event: dict[str, Any]) -> None:
        """Dispatch a single event to the matching handler."""
        event_type = event.get("Event", "")
        if event_type == "SparkListenerApplicationStart":
            self._app_name = str(event.get("App Name", ""))
            self._app_start_ms = _as_int(event.get("Timestamp")) or None
        elif event_type == "SparkListenerApplicationEnd":
            self._app_end_ms = _as_int(event.get("Timestamp")) or None
        elif event_type == "SparkListenerEnvironmentUpdate":
            properties = event.get("Spark Properties")
            if isinstance(properties, dict):
                self._spark_conf = {str(key): str(value) for key, value in properties.items()}
        elif event_type == "SparkListenerExecutorAdded":
            self._executor_count += 1
            self._executor_count_max = max(self._executor_count_max, self._executor_count)
        elif event_type == "SparkListenerExecutorRemoved":
            self._executor_count = max(0, self._executor_count - 1)
        elif event_type == "SparkListenerTaskEnd":
            self._handle_task_end(event)
        elif event_type == "SparkListenerStageCompleted":
            self._handle_stage_completed(event)
        # SparkListenerJobStart/JobEnd carry no metrics we aggregate today.

    def _get_stage(self, stage_id: int) -> _StageAccumulator:
        """Get or create the accumulator for a stage."""
        if stage_id not in self._stages:
            self._stages[stage_id] = _StageAccumulator(stage_id=stage_id)
        return self._stages[stage_id]

    def _handle_task_end(self, event: dict[str, Any]) -> None:
        """Aggregate metrics from a SparkListenerTaskEnd event."""
        stage = self._get_stage(_as_int(event.get("Stage ID")))
        task_info = event.get("Task Info") or {}
        task_metrics = event.get("Task Metrics") or {}

        self._total_tasks += 1
        reason = (event.get("Task End Reason") or {}).get("Reason", "Success")
        failed = bool(task_info.get("Failed", False)) or reason != "Success"
        if failed:
            self._failed_tasks += 1

        if isinstance(task_metrics, dict) and task_metrics:
            stage.has_task_metrics = True
            run_time_ms = _as_int(task_metrics.get("Executor Run Time"))
            self._total_run_ms += run_time_ms
            self._total_gc_ms += _as_int(task_metrics.get("JVM GC Time"))

            shuffle_read = task_metrics.get("Shuffle Read Metrics") or {}
            stage.shuffle_read_bytes += _as_int(shuffle_read.get("Remote Bytes Read")) + _as_int(
                shuffle_read.get("Local Bytes Read"),
            )
            shuffle_write = task_metrics.get("Shuffle Write Metrics") or {}
            stage.shuffle_write_bytes += _as_int(shuffle_write.get("Shuffle Bytes Written"))

            stage.spill_bytes += _as_int(task_metrics.get("Memory Bytes Spilled")) + _as_int(
                task_metrics.get("Disk Bytes Spilled"),
            )
            self._input_bytes += _as_int((task_metrics.get("Input Metrics") or {}).get("Bytes Read"))
            self._peak_execution_memory_bytes = max(
                self._peak_execution_memory_bytes,
                _as_int(task_metrics.get("Peak Execution Memory")),
            )

        # Failed tasks are excluded from skew statistics to avoid distortion.
        if not failed:
            launch_ms = _as_int(task_info.get("Launch Time"))
            finish_ms = _as_int(task_info.get("Finish Time"))
            if finish_ms > launch_ms > 0:
                stage.task_durations_ms.append(float(finish_ms - launch_ms))
            elif isinstance(task_metrics, dict) and _as_int(task_metrics.get("Executor Run Time")) > 0:
                stage.task_durations_ms.append(float(_as_int(task_metrics.get("Executor Run Time"))))

    def _handle_stage_completed(self, event: dict[str, Any]) -> None:
        """Aggregate metrics from a SparkListenerStageCompleted event."""
        stage_info = event.get("Stage Info") or {}
        stage = self._get_stage(_as_int(stage_info.get("Stage ID")))
        stage.name = str(stage_info.get("Stage Name", stage.name))
        stage.declared_tasks = _as_int(stage_info.get("Number of Tasks"))

        submission_ms = _as_int(stage_info.get("Submission Time"))
        completion_ms = _as_int(stage_info.get("Completion Time"))
        if completion_ms > submission_ms > 0:
            stage.duration_seconds = (completion_ms - submission_ms) / 1000.0

        # Fall back to "Stage Info" accumulables only when no per-task metric
        # rollups were observed for this stage (avoids double counting).
        if not stage.has_task_metrics:
            self._apply_accumulables(stage, stage_info.get("Accumulables") or [])

    def _apply_accumulables(self, stage: _StageAccumulator, accumulables: list[Any]) -> None:
        """Populate stage and global metrics from internal metric accumulables."""
        values: dict[str, int] = {}
        for accumulable in accumulables:
            if not isinstance(accumulable, dict):
                continue
            name = str(accumulable.get("Name", ""))
            if name.startswith("internal.metrics."):
                values[name.removeprefix("internal.metrics.")] = _as_int(accumulable.get("Value"))

        if not values:
            return

        stage.shuffle_read_bytes += values.get("shuffle.read.remoteBytesRead", 0) + values.get(
            "shuffle.read.localBytesRead",
            0,
        )
        stage.shuffle_write_bytes += values.get("shuffle.write.bytesWritten", 0)
        stage.spill_bytes += values.get("memoryBytesSpilled", 0) + values.get("diskBytesSpilled", 0)
        self._total_run_ms += values.get("executorRunTime", 0)
        self._total_gc_ms += values.get("jvmGCTime", 0)
        self._input_bytes += values.get("input.bytesRead", 0)
        self._peak_execution_memory_bytes = max(
            self._peak_execution_memory_bytes,
            values.get("peakExecutionMemory", 0),
        )

    @staticmethod
    def _skew_ratio(durations_ms: list[float]) -> float:
        """Compute max/median task duration ratio (1.0 when not computable)."""
        if len(durations_ms) < 2:
            return 1.0
        median = statistics.median(durations_ms)
        if median <= 0:
            return 1.0
        return max(durations_ms) / median

    def _build_summary(self) -> EventLogSummary:
        """Assemble the final EventLogSummary from accumulated state."""
        stages: list[StageSummary] = []
        total_shuffle_read = 0
        total_shuffle_write = 0
        total_spill = 0
        max_skew = 1.0

        for accumulator in sorted(self._stages.values(), key=lambda item: item.stage_id):
            skew = self._skew_ratio(accumulator.task_durations_ms)
            max_skew = max(max_skew, skew)
            total_shuffle_read += accumulator.shuffle_read_bytes
            total_shuffle_write += accumulator.shuffle_write_bytes
            total_spill += accumulator.spill_bytes
            stages.append(
                StageSummary(
                    stage_id=accumulator.stage_id,
                    name=accumulator.name,
                    duration_seconds=accumulator.duration_seconds,
                    num_tasks=accumulator.declared_tasks or len(accumulator.task_durations_ms),
                    shuffle_read_gb=accumulator.shuffle_read_bytes / BYTES_PER_GB,
                    shuffle_write_gb=accumulator.shuffle_write_bytes / BYTES_PER_GB,
                    spill_gb=accumulator.spill_bytes / BYTES_PER_GB,
                    skew_ratio=skew,
                ),
            )

        app_duration = 0.0
        if self._app_start_ms is not None and self._app_end_ms is not None and self._app_end_ms > self._app_start_ms:
            app_duration = (self._app_end_ms - self._app_start_ms) / 1000.0

        gc_fraction = (self._total_gc_ms / self._total_run_ms) if self._total_run_ms > 0 else 0.0

        return EventLogSummary(
            app_name=self._app_name,
            app_duration_seconds=app_duration,
            total_tasks=self._total_tasks,
            failed_tasks=self._failed_tasks,
            total_gc_time_seconds=self._total_gc_ms / 1000.0,
            gc_time_fraction=gc_fraction,
            total_shuffle_read_gb=total_shuffle_read / BYTES_PER_GB,
            total_shuffle_write_gb=total_shuffle_write / BYTES_PER_GB,
            total_spill_gb=total_spill / BYTES_PER_GB,
            peak_execution_memory_gb=self._peak_execution_memory_bytes / BYTES_PER_GB,
            executor_count_max=self._executor_count_max,
            input_data_gb=self._input_bytes / BYTES_PER_GB,
            max_skew_ratio=max_skew,
            stages=stages,
            spark_conf=self._spark_conf,
            skipped_lines=self._skipped_lines,
        )
