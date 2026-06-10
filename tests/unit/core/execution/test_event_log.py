# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Spark event log parser."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any

import pytest

from spark_optima.core.execution.event_log import (
    EventLogParser,
    EventLogSummary,
    StageSummary,
)

if TYPE_CHECKING:
    from pathlib import Path

GB = 1024**3


def _task_end(
    stage_id: int,
    launch: int,
    finish: int,
    run_time: int,
    gc_time: int = 0,
    shuffle_remote: int = 0,
    shuffle_local: int = 0,
    shuffle_write: int = 0,
    memory_spill: int = 0,
    disk_spill: int = 0,
    peak_memory: int = 0,
    input_bytes: int = 0,
    failed: bool = False,
    include_metrics: bool = True,
) -> dict[str, Any]:
    """Build a SparkListenerTaskEnd event."""
    event: dict[str, Any] = {
        "Event": "SparkListenerTaskEnd",
        "Stage ID": stage_id,
        "Task End Reason": {"Reason": "ExceptionFailure" if failed else "Success"},
        "Task Info": {"Launch Time": launch, "Finish Time": finish, "Failed": failed},
    }
    if include_metrics:
        event["Task Metrics"] = {
            "Executor Run Time": run_time,
            "JVM GC Time": gc_time,
            "Memory Bytes Spilled": memory_spill,
            "Disk Bytes Spilled": disk_spill,
            "Peak Execution Memory": peak_memory,
            "Shuffle Read Metrics": {"Remote Bytes Read": shuffle_remote, "Local Bytes Read": shuffle_local},
            "Shuffle Write Metrics": {"Shuffle Bytes Written": shuffle_write},
            "Input Metrics": {"Bytes Read": input_bytes},
        }
    return event


def _stage_completed(
    stage_id: int,
    name: str,
    num_tasks: int,
    submission: int,
    completion: int,
    accumulables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a SparkListenerStageCompleted event."""
    return {
        "Event": "SparkListenerStageCompleted",
        "Stage Info": {
            "Stage ID": stage_id,
            "Stage Name": name,
            "Number of Tasks": num_tasks,
            "Submission Time": submission,
            "Completion Time": completion,
            "Accumulables": accumulables or [],
        },
    }


def _standard_events() -> list[dict[str, Any]]:
    """Build a small but complete synthetic event log.

    Expected aggregates:
        app duration 60 s; 6 tasks (1 failed); GC 1.0 s of 21 s run time;
        shuffle read 2 GB / write 1 GB; spill 0.5 GB; peak memory 2 GB;
        input 10 GB; max 2 executors; stage skews 1.6 and 10.0.
    """
    return [
        {"Event": "SparkListenerApplicationStart", "App Name": "test-app", "Timestamp": 1_000_000},
        {
            "Event": "SparkListenerEnvironmentUpdate",
            "Spark Properties": {"spark.executor.memory": "4g", "spark.sql.shuffle.partitions": "200"},
        },
        {"Event": "SparkListenerExecutorAdded", "Executor ID": "1", "Timestamp": 1_000_000},
        {"Event": "SparkListenerExecutorAdded", "Executor ID": "2", "Timestamp": 1_000_100},
        {"Event": "SparkListenerExecutorRemoved", "Executor ID": "1", "Timestamp": 1_005_000},
        {"Event": "SparkListenerExecutorAdded", "Executor ID": "3", "Timestamp": 1_006_000},
        {"Event": "SparkListenerJobStart", "Job ID": 0, "Submission Time": 1_000_000},
        _task_end(
            stage_id=0,
            launch=1_000_000,
            finish=1_002_000,
            run_time=1500,
            gc_time=200,
            shuffle_remote=1 * GB,
            shuffle_write=GB // 2,
            memory_spill=GB // 4,
            disk_spill=GB // 4,
            peak_memory=2 * GB,
            input_bytes=5 * GB,
        ),
        _task_end(
            stage_id=0,
            launch=1_000_000,
            finish=1_008_000,
            run_time=7500,
            gc_time=800,
            shuffle_local=1 * GB,
            shuffle_write=GB // 2,
            peak_memory=1 * GB,
            input_bytes=5 * GB,
        ),
        _stage_completed(0, "stage zero", 2, 1_000_000, 1_010_000),
        _task_end(stage_id=1, launch=1_010_000, finish=1_011_000, run_time=1000),
        _task_end(stage_id=1, launch=1_010_000, finish=1_011_000, run_time=1000),
        _task_end(stage_id=1, launch=1_010_000, finish=1_020_000, run_time=10_000),
        _task_end(
            stage_id=1,
            launch=1_010_000,
            finish=1_011_000,
            run_time=0,
            failed=True,
            include_metrics=False,
        ),
        _stage_completed(1, "stage one", 4, 1_010_000, 1_030_000),
        {"Event": "SparkListenerJobEnd", "Job ID": 0, "Completion Time": 1_030_000},
        {"Event": "SparkListenerApplicationEnd", "Timestamp": 1_060_000},
    ]


def _write_log(path: Path, events: list[dict[str, Any]], extra_lines: list[str] | None = None) -> Path:
    """Write events (and optional raw lines) as a JSON-lines event log."""
    lines = [json.dumps(event) for event in events]
    if extra_lines:
        lines = lines[:2] + extra_lines + lines[2:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def standard_log(tmp_path: Path) -> Path:
    """Write the standard synthetic event log to a temporary file."""
    return _write_log(tmp_path / "eventlog", _standard_events())


class TestEventLogParser:
    """Test cases for EventLogParser on plain-text logs."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test that a missing log file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            EventLogParser(tmp_path / "does-not-exist")

    def test_application_summary(self, standard_log: Path) -> None:
        """Test application-level aggregates are exact."""
        summary = EventLogParser(standard_log).parse()

        assert summary.app_name == "test-app"
        assert summary.app_duration_seconds == pytest.approx(60.0)
        assert summary.total_tasks == 6
        assert summary.failed_tasks == 1
        assert summary.executor_count_max == 2
        assert summary.skipped_lines == 0

    def test_gc_metrics(self, standard_log: Path) -> None:
        """Test GC time and GC fraction are computed from task metrics."""
        summary = EventLogParser(standard_log).parse()

        assert summary.total_gc_time_seconds == pytest.approx(1.0)
        # 1000 ms GC over 21000 ms total executor run time
        assert summary.gc_time_fraction == pytest.approx(1000 / 21000)

    def test_shuffle_spill_memory_and_input(self, standard_log: Path) -> None:
        """Test shuffle, spill, peak memory, and input aggregates are exact."""
        summary = EventLogParser(standard_log).parse()

        assert summary.total_shuffle_read_gb == pytest.approx(2.0)
        assert summary.total_shuffle_write_gb == pytest.approx(1.0)
        assert summary.total_spill_gb == pytest.approx(0.5)
        assert summary.peak_execution_memory_gb == pytest.approx(2.0)
        assert summary.input_data_gb == pytest.approx(10.0)

    def test_stage_summaries(self, standard_log: Path) -> None:
        """Test per-stage aggregates including skew ratios."""
        summary = EventLogParser(standard_log).parse()

        assert len(summary.stages) == 2
        stage0, stage1 = summary.stages

        assert stage0.stage_id == 0
        assert stage0.name == "stage zero"
        assert stage0.duration_seconds == pytest.approx(10.0)
        assert stage0.num_tasks == 2
        assert stage0.shuffle_read_gb == pytest.approx(2.0)
        assert stage0.shuffle_write_gb == pytest.approx(1.0)
        assert stage0.spill_gb == pytest.approx(0.5)
        # max 8000 ms / median 5000 ms
        assert stage0.skew_ratio == pytest.approx(1.6)

        assert stage1.stage_id == 1
        assert stage1.name == "stage one"
        assert stage1.duration_seconds == pytest.approx(20.0)
        assert stage1.num_tasks == 4
        # max 10000 ms / median 1000 ms — failed task excluded
        assert stage1.skew_ratio == pytest.approx(10.0)

        assert summary.max_skew_ratio == pytest.approx(10.0)

    def test_spark_conf_captured(self, standard_log: Path) -> None:
        """Test the Spark properties from EnvironmentUpdate are captured."""
        summary = EventLogParser(standard_log).parse()

        assert summary.spark_conf == {
            "spark.executor.memory": "4g",
            "spark.sql.shuffle.partitions": "200",
        }

    def test_corrupt_lines_are_skipped_and_counted(self, tmp_path: Path) -> None:
        """Test unparseable lines are tolerated and counted."""
        log = _write_log(
            tmp_path / "eventlog",
            _standard_events(),
            extra_lines=["this is not json {", "[1, 2, 3]"],
        )

        summary = EventLogParser(log).parse()

        assert summary.skipped_lines == 2
        # Remaining events still parsed correctly
        assert summary.app_name == "test-app"
        assert summary.total_tasks == 6

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test an empty log produces a default summary."""
        log = tmp_path / "empty"
        log.write_text("", encoding="utf-8")

        summary = EventLogParser(log).parse()

        assert summary.app_name == ""
        assert summary.app_duration_seconds == 0.0
        assert summary.total_tasks == 0
        assert summary.gc_time_fraction == 0.0
        assert summary.max_skew_ratio == 1.0
        assert summary.stages == []

    def test_to_dict(self, standard_log: Path) -> None:
        """Test dictionary serialization includes all summary fields."""
        summary = EventLogParser(standard_log).parse()
        data = summary.to_dict()

        assert data["app_name"] == "test-app"
        assert data["app_duration_seconds"] == pytest.approx(60.0)
        assert data["total_tasks"] == 6
        assert data["failed_tasks"] == 1
        assert data["input_data_gb"] == pytest.approx(10.0)
        assert data["executor_count_max"] == 2
        assert len(data["stages"]) == 2
        assert data["stages"][1]["skew_ratio"] == pytest.approx(10.0)
        assert data["spark_conf"]["spark.executor.memory"] == "4g"


class TestEventLogParserGzip:
    """Test cases for gzip-compressed event logs."""

    def test_gzip_detected_by_suffix(self, tmp_path: Path) -> None:
        """Test a .gz log is decompressed transparently."""
        log = tmp_path / "eventlog.gz"
        content = "\n".join(json.dumps(event) for event in _standard_events())
        with gzip.open(log, "wt", encoding="utf-8") as handle:
            handle.write(content)

        summary = EventLogParser(log).parse()

        assert summary.app_name == "test-app"
        assert summary.total_tasks == 6

    def test_gzip_detected_by_magic_bytes(self, tmp_path: Path) -> None:
        """Test gzip content without a .gz suffix is detected by magic bytes."""
        log = tmp_path / "eventlog-compressed"
        content = "\n".join(json.dumps(event) for event in _standard_events())
        with gzip.open(log, "wt", encoding="utf-8") as handle:
            handle.write(content)

        summary = EventLogParser(log).parse()

        assert summary.app_name == "test-app"
        assert summary.input_data_gb == pytest.approx(10.0)


class TestEventLogAccumulablesFallback:
    """Test stage accumulables used when no task-level metrics exist."""

    def test_accumulables_populate_stage_and_totals(self, tmp_path: Path) -> None:
        """Test internal.metrics accumulables drive the summary."""
        events = [
            {"Event": "SparkListenerApplicationStart", "App Name": "acc-app", "Timestamp": 1_000_000},
            _stage_completed(
                0,
                "acc stage",
                8,
                1_000_000,
                1_005_000,
                accumulables=[
                    {"Name": "internal.metrics.executorRunTime", "Value": 10_000},
                    {"Name": "internal.metrics.jvmGCTime", "Value": 2_000},
                    {"Name": "internal.metrics.shuffle.read.localBytesRead", "Value": GB},
                    {"Name": "internal.metrics.shuffle.write.bytesWritten", "Value": GB},
                    {"Name": "internal.metrics.memoryBytesSpilled", "Value": GB // 2},
                    {"Name": "internal.metrics.input.bytesRead", "Value": 4 * GB},
                    {"Name": "internal.metrics.peakExecutionMemory", "Value": 3 * GB},
                    {"Name": "some.user.accumulator", "Value": 42},
                ],
            ),
            {"Event": "SparkListenerApplicationEnd", "Timestamp": 1_010_000},
        ]
        log = _write_log(tmp_path / "eventlog", events)

        summary = EventLogParser(log).parse()

        assert summary.total_gc_time_seconds == pytest.approx(2.0)
        assert summary.gc_time_fraction == pytest.approx(0.2)
        assert summary.total_shuffle_read_gb == pytest.approx(1.0)
        assert summary.total_shuffle_write_gb == pytest.approx(1.0)
        assert summary.total_spill_gb == pytest.approx(0.5)
        assert summary.input_data_gb == pytest.approx(4.0)
        assert summary.peak_execution_memory_gb == pytest.approx(3.0)
        assert summary.stages[0].num_tasks == 8


class TestEventLogTuningHints:
    """Test cases for EventLogSummary.to_tuning_hints()."""

    def test_hint_keys_and_values(self, standard_log: Path) -> None:
        """Test the documented hint keys and their values."""
        summary = EventLogParser(standard_log).parse()
        hints = summary.to_tuning_hints()

        assert set(hints) == {
            "data_size_gb",
            "skew_factor",
            "large_shuffles",
            "gc_pressure",
            "gc_time_fraction",
            "spill_detected",
            "spill_gb",
            "shuffle_total_gb",
            "memory_intensive",
        }
        assert hints["data_size_gb"] == pytest.approx(10.0)
        assert hints["skew_factor"] == pytest.approx(10.0)
        assert hints["large_shuffles"] is False  # 3 GB total shuffle < 10 GB threshold
        assert hints["gc_pressure"] is False  # ~4.8% < 10% threshold
        assert hints["gc_time_fraction"] == pytest.approx(1000 / 21000)
        assert hints["spill_detected"] is True
        assert hints["spill_gb"] == pytest.approx(0.5)
        assert hints["shuffle_total_gb"] == pytest.approx(3.0)
        assert hints["memory_intensive"] is True  # spill observed

    def test_data_size_omitted_when_no_input_metrics(self) -> None:
        """Test data_size_gb is not emitted for a run without input bytes."""
        hints = EventLogSummary().to_tuning_hints()

        assert "data_size_gb" not in hints
        assert hints["skew_factor"] == 1.0
        assert hints["memory_intensive"] is False

    def test_large_shuffles_and_gc_pressure_flags(self) -> None:
        """Test threshold-based hint flags trigger correctly."""
        summary = EventLogSummary(
            total_shuffle_read_gb=8.0,
            total_shuffle_write_gb=4.0,
            gc_time_fraction=0.25,
        )
        hints = summary.to_tuning_hints()

        assert hints["large_shuffles"] is True  # 12 GB > 10 GB threshold
        assert hints["gc_pressure"] is True
        assert hints["memory_intensive"] is True  # GC pressure without spill


class TestStageSummary:
    """Test cases for StageSummary."""

    def test_to_dict(self) -> None:
        """Test StageSummary dictionary serialization."""
        stage = StageSummary(
            stage_id=3,
            name="map at job.py:10",
            duration_seconds=12.5,
            num_tasks=16,
            shuffle_read_gb=1.25,
            shuffle_write_gb=0.75,
            spill_gb=0.5,
            skew_ratio=2.0,
        )
        data = stage.to_dict()

        assert data == {
            "stage_id": 3,
            "name": "map at job.py:10",
            "duration_seconds": 12.5,
            "num_tasks": 16,
            "shuffle_read_gb": 1.25,
            "shuffle_write_gb": 0.75,
            "spill_gb": 0.5,
            "skew_ratio": 2.0,
        }
