# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Execution monitor for real-time Spark job monitoring.

This module provides ExecutionMonitor class that monitors running Spark jobs
and provides real-time progress updates and alerts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class MonitoringEvent:
    """Event during job execution monitoring.

    Attributes:
        timestamp: Event timestamp.
        event_type: Type of event (progress, warning, error, complete).
        message: Human-readable message.
        metrics: Associated metrics snapshot.

    """

    timestamp: float
    event_type: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for execution monitoring.

    Attributes:
        check_interval_seconds: How often to check status.
        progress_threshold: Progress percentage to trigger update.
        memory_warning_threshold: Memory usage % to trigger warning.
        timeout_seconds: Maximum monitoring time.
        enable_alerts: Whether to enable alerts.

    """

    check_interval_seconds: float = 5.0
    progress_threshold: float = 10.0  # Report every 10% progress
    memory_warning_threshold: float = 80.0
    timeout_seconds: float = 3600.0
    enable_alerts: bool = True


class ExecutionMonitor:
    """Monitors Spark job execution in real-time.

    This class provides live monitoring of Spark job execution with
    progress tracking, resource usage alerts, and event callbacks.

    Example:
        >>> monitor = ExecutionMonitor()
        >>> monitor.start_monitoring(spark)
        >>> monitor.add_callback(lambda e: print(e.message))
        >>> # Run job
        >>> monitor.stop_monitoring()

    """

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        """Initialize execution monitor.

        Args:
            config: Monitoring configuration.

        """
        self.config = config or MonitoringConfig()
        self._spark: Any | None = None
        self._monitoring_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable[[MonitoringEvent], None]] = []
        self._events: list[MonitoringEvent] = []
        self._last_progress: float = 0.0
        self._lock = threading.Lock()

    def start_monitoring(self, spark: Any) -> None:
        """Start monitoring a Spark session.

        Args:
            spark: SparkSession to monitor.

        """
        with self._lock:
            self._spark = spark
            self._stop_event.clear()
            self._events = []
            self._last_progress = 0.0

            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
            )
            self._monitoring_thread.start()

        logger.info("Execution monitoring started")
        self._emit_event("start", "Monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()

        with self._lock:
            monitoring_thread = self._monitoring_thread

        if monitoring_thread:
            monitoring_thread.join(timeout=5.0)

        self._emit_event("stop", "Monitoring stopped")
        logger.info("Execution monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout_seconds:
                    self._emit_event(
                        "warning",
                        f"Monitoring timeout after {elapsed:.0f}s",
                        {"elapsed_seconds": elapsed},
                    )
                    break

                # Collect current metrics
                metrics = self._collect_current_metrics()

                # Check progress (protected by lock)
                progress = metrics.get("progress_percent", 0.0)
                with self._lock:
                    last_progress = self._last_progress
                    if progress - last_progress >= self.config.progress_threshold:
                        should_emit = True
                        self._last_progress = progress
                    else:
                        should_emit = False

                if should_emit:
                    self._emit_event("progress", f"Job progress: {progress:.1f}%", metrics)

                # Check memory
                memory_pct = metrics.get("memory_usage_percent", 0.0)
                if memory_pct > self.config.memory_warning_threshold:
                    self._emit_event("warning", f"High memory usage: {memory_pct:.1f}%", metrics)

                # Sleep until next check
                time.sleep(self.config.check_interval_seconds)

            except (RuntimeError, AttributeError, KeyError, TypeError, ValueError) as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.check_interval_seconds)

    def _collect_current_metrics(self) -> dict[str, Any]:
        """Collect current execution metrics.

        Returns:
            Dictionary with current metrics.

        """
        metrics = {
            "timestamp": time.time(),
            "progress_percent": 0.0,
            "memory_usage_percent": 0.0,
        }

        with self._lock:
            spark = self._spark

        if spark is None:
            return metrics

        try:
            sc = spark.sparkContext
            status_tracker = sc.statusTracker()

            # Get active jobs
            active_jobs = status_tracker.getActiveJobsIds()
            metrics["active_jobs"] = len(active_jobs)

            # Estimate progress (simplified)
            # Real implementation would track task completion
            if active_jobs:
                metrics["progress_percent"] = 50.0  # Placeholder

            # Memory usage
            memory_status = sc.getExecutorMemoryStatus()
            if memory_status:
                total_used = sum((total - free) for (_, (total, free)) in memory_status.items())
                total_available = sum(total for (_, (total, _)) in memory_status.items())
                if total_available > 0:
                    metrics["memory_usage_percent"] = total_used / total_available * 100

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Could not collect metrics: {e}")

        return metrics

    def _emit_event(
        self,
        event_type: str,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Emit a monitoring event.

        Args:
            event_type: Type of event.
            message: Event message.
            metrics: Associated metrics.

        """
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            metrics=metrics or {},
        )

        # Protect shared state with lock
        with self._lock:
            self._events.append(event)
            # Copy callbacks to avoid holding lock during callback execution
            callbacks = list(self._callbacks)

        # Call registered callbacks outside lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(event)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Callback error: {e}")

    def add_callback(self, callback: Callable[[MonitoringEvent], None]) -> None:
        """Add a callback for monitoring events.

        Args:
            callback: Function to call on events.

        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[MonitoringEvent], None]) -> None:
        """Remove a callback.

        Args:
            callback: Callback to remove.

        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_events(self, event_type: str | None = None) -> list[MonitoringEvent]:
        """Get recorded events.

        Args:
            event_type: Filter by event type (optional).

        Returns:
            List of events.

        """
        with self._lock:
            events_copy = list(self._events)

        if event_type is None:
            return events_copy

        return [e for e in events_copy if e.event_type == event_type]

    def get_latest_metrics(self) -> dict[str, Any]:
        """Get the most recent metrics snapshot.

        Returns:
            Latest metrics dictionary.

        """
        return self._collect_current_metrics()

    def is_monitoring(self) -> bool:
        """Check if monitoring is active.

        Returns:
            True if monitoring is running.

        """
        with self._lock:
            monitoring_thread = self._monitoring_thread

        return monitoring_thread is not None and monitoring_thread.is_alive()

    def reset(self) -> None:
        """Reset monitor state."""
        self.stop_monitoring()
        with self._lock:
            self._events = []
            self._callbacks = []
            self._last_progress = 0.0
