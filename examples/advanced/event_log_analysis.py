#!/usr/bin/env python3
"""Example: Post-run analysis of a Spark event log.

This example builds a tiny synthetic Spark event log (the JSON-lines format
Spark writes when ``spark.eventLog.enabled=true``), parses it with
EventLogParser, and prints:

1. The EventLogSummary — application, stage, GC, shuffle, spill, and skew
   metrics condensed from the raw listener events.
2. The tuning hints derived via ``summary.to_tuning_hints()`` — the same
   hints the optimizer consumes to warm-start a re-optimization.

No real Spark cluster is needed: the synthetic log mimics the events of a
small two-stage job with GC pressure, shuffle spill, and one straggler task.

CLI equivalent:
    spark-optima analyze-log -l ./eventlog.json
    spark-optima analyze-log --history-server http://localhost:18080
"""

import json
import tempfile
from pathlib import Path

from spark_optima.core.execution.event_log import EventLogParser

GB = 1024**3
MB = 1024**2


def _task_end(
    stage_id: int,
    launch_ms: int,
    finish_ms: int,
    gc_ms: int = 0,
    shuffle_read_mb: int = 0,
    shuffle_write_mb: int = 0,
    spill_mb: int = 0,
    input_mb: int = 0,
) -> dict:
    """Build a SparkListenerTaskEnd event with task metrics."""
    return {
        "Event": "SparkListenerTaskEnd",
        "Stage ID": stage_id,
        "Task End Reason": {"Reason": "Success"},
        "Task Info": {"Failed": False, "Launch Time": launch_ms, "Finish Time": finish_ms},
        "Task Metrics": {
            "Executor Run Time": finish_ms - launch_ms,
            "JVM GC Time": gc_ms,
            "Shuffle Read Metrics": {"Remote Bytes Read": shuffle_read_mb * MB, "Local Bytes Read": 0},
            "Shuffle Write Metrics": {"Shuffle Bytes Written": shuffle_write_mb * MB},
            "Memory Bytes Spilled": spill_mb * MB,
            "Disk Bytes Spilled": 0,
            "Input Metrics": {"Bytes Read": input_mb * MB},
            "Peak Execution Memory": 2 * GB,
        },
    }


def create_synthetic_event_log(log_path: Path) -> None:
    """Write a small synthetic Spark event log (JSON lines) to log_path.

    The log models a two-stage job: stage 0 reads ~6GB of input and writes
    shuffle data (with spill and GC pressure); stage 1 reads the shuffle and
    contains one straggler task (~6x the median duration) to trigger the
    skew detection.
    """
    events: list[dict] = [
        {"Event": "SparkListenerApplicationStart", "App Name": "sales-etl-nightly", "Timestamp": 1_700_000_000_000},
        {
            "Event": "SparkListenerEnvironmentUpdate",
            "Spark Properties": {
                "spark.executor.memory": "4g",
                "spark.executor.cores": "4",
                "spark.sql.shuffle.partitions": "200",
            },
        },
        {"Event": "SparkListenerExecutorAdded"},
        {"Event": "SparkListenerExecutorAdded"},
    ]

    # Stage 0: input read + shuffle write, heavy GC and some spill
    start = 1_700_000_001_000
    for _ in range(4):
        events.append(
            _task_end(
                stage_id=0,
                launch_ms=start,
                finish_ms=start + 20_000,
                gc_ms=4_000,  # 20% of run time -> GC pressure
                shuffle_write_mb=4_096,  # 4GB written per task
                spill_mb=512,
                input_mb=1_536,  # ~1.5GB read per task
            ),
        )
        start += 1_000
    events.append(
        {
            "Event": "SparkListenerStageCompleted",
            "Stage Info": {
                "Stage ID": 0,
                "Stage Name": "Exchange hashpartitioning",
                "Number of Tasks": 4,
                "Submission Time": 1_700_000_001_000,
                "Completion Time": 1_700_000_026_000,
            },
        },
    )

    # Stage 1: shuffle read with one straggler task (~6x median duration)
    start = 1_700_000_030_000
    durations_ms = [10_000, 10_000, 11_000, 60_000]  # last task is the straggler
    for duration in durations_ms:
        events.append(
            _task_end(
                stage_id=1,
                launch_ms=start,
                finish_ms=start + duration,
                gc_ms=500,
                shuffle_read_mb=4_096,
            ),
        )
        start += 500
    events.append(
        {
            "Event": "SparkListenerStageCompleted",
            "Stage Info": {
                "Stage ID": 1,
                "Stage Name": "ResultStage groupBy",
                "Number of Tasks": 4,
                "Submission Time": 1_700_000_030_000,
                "Completion Time": 1_700_000_095_000,
            },
        },
    )

    events.append({"Event": "SparkListenerApplicationEnd", "Timestamp": 1_700_000_100_000})

    with log_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def main() -> None:
    """Build a synthetic event log, parse it, and print summary + hints."""
    print("=" * 70)
    print("📜 Spark Optima - Event Log Analysis Example")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "eventlog.json"
        print(f"\nWriting synthetic event log: {log_path}")
        create_synthetic_event_log(log_path)

        # Parse the event log (plain JSON lines or .gz both work)
        parser = EventLogParser(log_path)
        summary = parser.parse()

    print("\n📊 APPLICATION SUMMARY")
    print("-" * 70)
    print(f"  App name:               {summary.app_name}")
    print(f"  Duration:               {summary.app_duration_seconds:.0f} s")
    print(f"  Tasks (failed):         {summary.total_tasks} ({summary.failed_tasks})")
    print(f"  Max executors:          {summary.executor_count_max}")
    print(f"  Input read:             {summary.input_data_gb:.1f} GB")
    print(f"  Shuffle read / write:   {summary.total_shuffle_read_gb:.1f} / {summary.total_shuffle_write_gb:.1f} GB")
    print(f"  Spill:                  {summary.total_spill_gb:.1f} GB")
    print(f"  GC time fraction:       {summary.gc_time_fraction:.1%}")
    print(f"  Peak execution memory:  {summary.peak_execution_memory_gb:.1f} GB")
    print(f"  Max task skew ratio:    {summary.max_skew_ratio:.1f}x")

    print("\n📋 PER-STAGE BREAKDOWN")
    print("-" * 70)
    for stage in summary.stages:
        print(
            f"  Stage {stage.stage_id} ({stage.name}): "
            f"{stage.duration_seconds:.0f}s, {stage.num_tasks} tasks, "
            f"shuffle r/w {stage.shuffle_read_gb:.1f}/{stage.shuffle_write_gb:.1f} GB, "
            f"spill {stage.spill_gb:.1f} GB, skew {stage.skew_ratio:.1f}x",
        )

    print("\n⚙️  CAPTURED SPARK CONF (from the run)")
    print("-" * 70)
    for key, value in summary.spark_conf.items():
        print(f"  {key:45s} = {value}")

    # Derive tuning hints: these keys align with the heuristic engine's
    # EvaluationContext variables and can warm-start a re-optimization.
    hints = summary.to_tuning_hints()
    print("\n💡 TUNING HINTS (summary.to_tuning_hints())")
    print("-" * 70)
    for key, value in hints.items():
        print(f"  {key:25s} = {value}")

    print("\nNext step: feed these hints into a re-optimization, e.g.")
    print('  optimizer.optimize(resource_constraints={**hints, "max_memory_gb": 64})')


if __name__ == "__main__":
    main()
