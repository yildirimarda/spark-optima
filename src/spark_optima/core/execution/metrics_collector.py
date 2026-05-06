# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Metrics collector for Spark job execution.

This module provides MetricsCollector class that collects detailed
performance metrics from Spark execution for optimization analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from spark_optima.core.bayesian.models import TrialMetrics

logger = logging.getLogger(__name__)

# Optional PySpark import
try:
    import importlib.util

    pyspark_spec = importlib.util.find_spec("pyspark")
    PYSPARK_AVAILABLE = pyspark_spec is not None
except ImportError:
    PYSPARK_AVAILABLE = False


@dataclass
class StageMetrics:
    """Metrics for a single Spark stage.

    Attributes:
        stage_id: Stage identifier.
        stage_name: Stage name.
        num_tasks: Number of tasks in stage.
        executor_run_time: Total executor run time.
        executor_cpu_time: CPU time used.
        input_bytes: Input data size.
        output_bytes: Output data size.
        shuffle_read_bytes: Shuffle data read.
        shuffle_write_bytes: Shuffle data written.
        peak_memory: Peak memory usage.

    """

    stage_id: int
    stage_name: str = ""
    num_tasks: int = 0
    executor_run_time: int = 0
    executor_cpu_time: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    shuffle_read_bytes: int = 0
    shuffle_write_bytes: int = 0
    peak_memory: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "num_tasks": self.num_tasks,
            "executor_run_time_ms": self.executor_run_time,
            "executor_cpu_time_ms": self.executor_cpu_time,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "shuffle_read_bytes": self.shuffle_read_bytes,
            "shuffle_write_bytes": self.shuffle_write_bytes,
            "peak_memory_bytes": self.peak_memory,
        }


@dataclass
class JobMetrics:
    """Metrics for a complete Spark job.

    Attributes:
        job_id: Job identifier.
        job_name: Job name/description.
        submission_time: Job submission timestamp.
        completion_time: Job completion timestamp.
        stage_metrics: List of stage metrics.
        total_tasks: Total number of tasks.
        failed_tasks: Number of failed tasks.

    """

    job_id: int
    job_name: str = ""
    submission_time: float = 0.0
    completion_time: float = 0.0
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    total_tasks: int = 0
    failed_tasks: int = 0

    @property
    def duration_seconds(self) -> float:
        """Calculate job duration."""
        if self.completion_time > self.submission_time:
            return self.completion_time - self.submission_time
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "duration_seconds": self.duration_seconds,
            "total_tasks": self.total_tasks,
            "failed_tasks": self.failed_tasks,
            "num_stages": len(self.stage_metrics),
            "stages": [s.to_dict() for s in self.stage_metrics],
        }


@dataclass
class ExecutionMetrics:
    """Complete execution metrics.

    Attributes:
        execution_time_seconds: Total execution time.
        memory_peak_gb: Peak memory in GB.
        memory_average_gb: Average memory in GB.
        cpu_utilization_percent: Average CPU utilization.
        shuffle_read_gb: Shuffle read in GB.
        shuffle_write_gb: Shuffle write in GB.
        jobs: List of job metrics.
        gc_time_seconds: Garbage collection time.
        success: Whether execution was successful.
        error_message: Error message if failed.

    """

    execution_time_seconds: float = 0.0
    memory_peak_gb: float = 0.0
    memory_average_gb: float = 0.0
    cpu_utilization_percent: float = 0.0
    shuffle_read_gb: float = 0.0
    shuffle_write_gb: float = 0.0
    jobs: list[JobMetrics] = field(default_factory=list)
    gc_time_seconds: float = 0.0
    success: bool = True
    error_message: str = ""

    def to_trial_metrics(self) -> TrialMetrics:
        """Convert to TrialMetrics for Bayesian optimization."""
        return TrialMetrics(
            execution_time_seconds=self.execution_time_seconds,
            memory_peak_gb=self.memory_peak_gb,
            cpu_utilization_percent=self.cpu_utilization_percent,
            shuffle_read_gb=self.shuffle_read_gb,
            shuffle_write_gb=self.shuffle_write_gb,
            success=self.success,
            error_message=self.error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_time_seconds": self.execution_time_seconds,
            "memory_peak_gb": self.memory_peak_gb,
            "memory_average_gb": self.memory_average_gb,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "shuffle_read_gb": self.shuffle_read_gb,
            "shuffle_write_gb": self.shuffle_write_gb,
            "num_jobs": len(self.jobs),
            "jobs": [j.to_dict() for j in self.jobs],
            "gc_time_seconds": self.gc_time_seconds,
            "success": self.success,
            "error_message": self.error_message,
        }


class MetricsCollector:
    """Collects detailed metrics from Spark execution.

    This class interfaces with Spark's status API and Spark UI to collect
    comprehensive performance metrics during job execution.

    Example:
        >>> collector = MetricsCollector(spark)
        >>> collector.start_collection()
        >>> # Run Spark job
        >>> metrics = collector.collect_metrics()
        >>> print(f"Execution time: {metrics.execution_time_seconds:.1f}s")

    """

    def __init__(self, spark: Any | None = None) -> None:
        """Initialize metrics collector.

        Args:
            spark: SparkSession instance (optional).

        """
        self.spark = spark
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._job_metrics: list[JobMetrics] = []

    def start_collection(self) -> None:
        """Start metrics collection."""
        self._start_time = time.time()
        self._job_metrics = []
        logger.debug("Metrics collection started")

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._end_time = time.time()
        logger.debug("Metrics collection stopped")

    def collect_metrics(self) -> ExecutionMetrics:
        """Collect metrics from Spark.

        Returns:
            ExecutionMetrics with collected data.

        """
        if self.spark is None:
            logger.warning("No Spark session available for metrics collection")
            return ExecutionMetrics()

        try:
            # Collect job metrics
            job_metrics = self._collect_job_metrics()

            # Calculate aggregate metrics
            execution_time = self._calculate_execution_time()
            memory_metrics = self._collect_memory_metrics()
            shuffle_metrics = self._collect_shuffle_metrics()
            gc_time = self._collect_gc_metrics()

            return ExecutionMetrics(
                execution_time_seconds=execution_time,
                memory_peak_gb=memory_metrics.get("peak_gb", 0.0),
                memory_average_gb=memory_metrics.get("average_gb", 0.0),
                cpu_utilization_percent=self._estimate_cpu_utilization(),
                shuffle_read_gb=shuffle_metrics.get("read_gb", 0.0),
                shuffle_write_gb=shuffle_metrics.get("write_gb", 0.0),
                jobs=job_metrics,
                gc_time_seconds=gc_time,
                success=True,
            )

        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ExecutionMetrics(
                success=False,
                error_message=str(e),
            )

    def _collect_job_metrics(self) -> list[JobMetrics]:
        """Collect metrics for all jobs.

        Returns:
            List of JobMetrics.

        Raises:
            Exception: If Spark access fails (propagates to collect_metrics).

        """
        jobs: list[JobMetrics] = []

        if not PYSPARK_AVAILABLE or self.spark is None:
            return jobs

        # Access Spark status tracker
        sc = self.spark.sparkContext
        status_tracker = sc.statusTracker()

        # Get active jobs
        for _job_info in status_tracker.getActiveJobsIds():
            # Note: Detailed job info requires Spark UI API
            # This is a simplified version
            pass

        # For more detailed metrics, we'd need to query Spark UI REST API
        # or use SparkListener callbacks

        return jobs

    def _calculate_execution_time(self) -> float:
        """Calculate total execution time.

        Returns:
            Execution time in seconds.

        """
        if self._start_time is None:
            return 0.0

        end = self._end_time or time.time()
        return end - self._start_time

    def _collect_memory_metrics(self) -> dict[str, float]:
        """Collect memory usage metrics.

        Returns:
            Dictionary with memory metrics in GB.

        Raises:
            Exception: If Spark access fails (propagates to collect_metrics).

        """
        if not PYSPARK_AVAILABLE or self.spark is None:
            return {"peak_gb": 0.0, "average_gb": 0.0}

        sc = self.spark.sparkContext

        # Get executor memory status (not available in Spark 4.x)
        # This is simplified - full implementation would track over time
        try:
            if hasattr(sc, "getExecutorMemoryStatus"):
                executor_memory_status = sc.getExecutorMemoryStatus()

                if executor_memory_status:
                    total_used = sum(
                        (total - free) for (_, (total, free)) in executor_memory_status.items()
                    )
                    total_available = sum(total for (_, (total, _)) in executor_memory_status.items())

                    avg_gb = total_used / (1024**3) if total_used > 0 else 0.0
                    peak_gb = total_available / (1024**3) * 0.8  # Estimate

                    return {"peak_gb": peak_gb, "average_gb": avg_gb}
        except (AttributeError, TypeError, KeyError):
            pass

        return {"peak_gb": 0.0, "average_gb": 0.0}

    def _collect_shuffle_metrics(self) -> dict[str, float]:
        """Collect shuffle metrics.

        Returns:
            Dictionary with shuffle metrics in GB.

        Raises:
            Exception: If Spark access fails (propagates to collect_metrics).

        """
        if not PYSPARK_AVAILABLE or self.spark is None:
            return {"read_gb": 0.0, "write_gb": 0.0}

        # Get shuffle metrics from status tracker
        # This requires accessing internal Spark metrics
        # Simplified implementation

        return {"read_gb": 0.0, "write_gb": 0.0}

    def _collect_gc_metrics(self) -> float:
        """Collect garbage collection metrics.

        Returns:
            GC time in seconds.

        Raises:
            Exception: If access fails (propagates to collect_metrics).

        """
        # This is a placeholder - real GC metrics require JVM access
        return 0.0

    def _estimate_cpu_utilization(self) -> float:
        """Estimate CPU utilization.

        Returns:
            Estimated CPU utilization percentage.

        """
        # This would require system-level monitoring
        # Placeholder implementation
        return 0.0

    def get_stage_summary(self) -> dict[int, dict[str, Any]]:
        """Get summary of all stages.

        Returns:
            Dictionary mapping stage IDs to stage info.

        """
        summary = {}

        for job in self._job_metrics:
            for stage in job.stage_metrics:
                summary[stage.stage_id] = stage.to_dict()

        return summary

    def get_shuffle_summary(self) -> dict[str, float]:
        """Get shuffle operation summary.

        Returns:
            Dictionary with shuffle statistics.

        """
        total_read = 0.0
        total_write = 0.0

        for job in self._job_metrics:
            for stage in job.stage_metrics:
                total_read += stage.shuffle_read_bytes / (1024**3)
                total_write += stage.shuffle_write_bytes / (1024**3)

        return {
            "total_read_gb": total_read,
            "total_write_gb": total_write,
            "total_gb": total_read + total_write,
        }

    def set_spark_session(self, spark: Any) -> None:
        """Set or update Spark session.

        Args:
            spark: SparkSession instance.

        """
        self.spark = spark

    def reset(self) -> None:
        """Reset collector state."""
        self._start_time = None
        self._end_time = None
        self._job_metrics = []
