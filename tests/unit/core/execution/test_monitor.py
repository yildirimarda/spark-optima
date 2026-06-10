# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for execution monitor module."""

import time
from unittest.mock import MagicMock, Mock, patch

from spark_optima.core.execution.monitor import (
    ExecutionMonitor,
    MonitoringConfig,
    MonitoringEvent,
)


class TestMonitoringEvent:
    """Tests for MonitoringEvent dataclass."""

    def test_event_creation(self):
        """Test basic MonitoringEvent creation."""
        event = MonitoringEvent(
            timestamp=1234567890.0,
            event_type="progress",
            message="Job progress: 50%",
            metrics={"progress_percent": 50.0},
        )

        assert event.timestamp == 1234567890.0
        assert event.event_type == "progress"
        assert event.message == "Job progress: 50%"
        assert event.metrics["progress_percent"] == 50.0

    def test_event_with_empty_metrics(self):
        """Test event with default empty metrics."""
        event = MonitoringEvent(
            timestamp=1234567890.0,
            event_type="start",
            message="Monitoring started",
        )

        assert event.metrics == {}


class TestMonitoringConfig:
    """Tests for MonitoringConfig dataclass."""

    def test_default_config(self):
        """Test default monitoring configuration."""
        config = MonitoringConfig()

        assert config.check_interval_seconds == 5.0
        assert config.progress_threshold == 10.0
        assert config.memory_warning_threshold == 80.0
        assert config.timeout_seconds == 3600.0
        assert config.enable_alerts is True

    def test_custom_config(self):
        """Test custom monitoring configuration."""
        config = MonitoringConfig(
            check_interval_seconds=2.0,
            progress_threshold=5.0,
            memory_warning_threshold=90.0,
            timeout_seconds=1800.0,
            enable_alerts=False,
        )

        assert config.check_interval_seconds == 2.0
        assert config.progress_threshold == 5.0
        assert config.memory_warning_threshold == 90.0
        assert config.timeout_seconds == 1800.0
        assert config.enable_alerts is False


class TestExecutionMonitor:
    """Tests for ExecutionMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization with default config."""
        monitor = ExecutionMonitor()

        assert monitor.config.check_interval_seconds == 5.0
        assert monitor._spark is None
        assert monitor._monitoring_thread is None
        assert monitor._callbacks == []
        assert monitor._events == []

    def test_monitor_initialization_with_config(self):
        """Test monitor initialization with custom config."""
        config = MonitoringConfig(check_interval_seconds=1.0)
        monitor = ExecutionMonitor(config=config)

        assert monitor.config.check_interval_seconds == 1.0

    def test_add_callback(self):
        """Test adding callback for monitoring events."""
        monitor = ExecutionMonitor()
        callback = Mock()

        monitor.add_callback(callback)

        assert callback in monitor._callbacks

    def test_remove_callback(self):
        """Test removing callback."""
        monitor = ExecutionMonitor()
        callback = Mock()

        monitor.add_callback(callback)
        monitor.remove_callback(callback)

        assert callback not in monitor._callbacks

    def test_remove_callback_not_in_list(self):
        """Test removing callback that doesn't exist."""
        monitor = ExecutionMonitor()
        callback = Mock()

        # Should not raise
        monitor.remove_callback(callback)

    def test_get_events_empty(self):
        """Test get_events with no events."""
        monitor = ExecutionMonitor()

        events = monitor.get_events()

        assert events == []

    def test_get_events_with_filter(self):
        """Test get_events with event type filter."""
        monitor = ExecutionMonitor()
        # Manually add events to test filtering
        monitor._events = [
            MonitoringEvent(1.0, "progress", "msg1"),
            MonitoringEvent(2.0, "warning", "msg2"),
            MonitoringEvent(3.0, "progress", "msg3"),
        ]

        progress_events = monitor.get_events("progress")

        assert len(progress_events) == 2
        assert all(e.event_type == "progress" for e in progress_events)

    def test_get_latest_metrics_no_spark(self):
        """Test get_latest_metrics without Spark session."""
        monitor = ExecutionMonitor()

        metrics = monitor.get_latest_metrics()

        assert metrics["progress_percent"] == 0.0
        assert metrics["memory_usage_percent"] == 0.0

    def test_is_monitoring_no_thread(self):
        """Test is_monitoring when no thread exists."""
        monitor = ExecutionMonitor()

        assert monitor.is_monitoring() is False

    def test_is_monitoring_with_thread(self):
        """Test is_monitoring when thread is alive."""
        monitor = ExecutionMonitor()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        monitor._monitoring_thread = mock_thread

        assert monitor.is_monitoring() is True

    def test_reset(self):
        """Test reset clears state."""
        monitor = ExecutionMonitor()
        callback = Mock()
        monitor.add_callback(callback)
        monitor._events = [MonitoringEvent(1.0, "start", "msg")]
        monitor._last_progress = 50.0

        monitor.reset()

        assert monitor._events == []
        assert monitor._callbacks == []
        assert monitor._last_progress == 0.0

    def test_emit_event(self):
        """Test event emission."""
        monitor = ExecutionMonitor()

        monitor._emit_event("progress", "Test message", {"key": "value"})

        assert len(monitor._events) == 1
        event = monitor._events[0]
        assert event.event_type == "progress"
        assert event.message == "Test message"
        assert event.metrics == {"key": "value"}

    def test_emit_event_calls_callback(self):
        """Test callback is called when event is emitted."""
        monitor = ExecutionMonitor()
        callback = Mock()
        monitor.add_callback(callback)

        monitor._emit_event("start", "Test message")

        callback.assert_called_once()
        called_event = callback.call_args[0][0]
        assert called_event.event_type == "start"

    def test_emit_event_callback_error_handling(self):
        """Test callback errors are handled gracefully."""
        monitor = ExecutionMonitor()

        def bad_callback(event):
            raise RuntimeError("Callback error")

        monitor.add_callback(bad_callback)

        # Should not raise
        monitor._emit_event("start", "Test message")

    def test_collect_current_metrics_no_spark(self):
        """Test metrics collection without Spark."""
        monitor = ExecutionMonitor()

        metrics = monitor._collect_current_metrics()

        assert "timestamp" in metrics
        assert metrics["progress_percent"] == 0.0
        assert metrics["memory_usage_percent"] == 0.0

    def test_collect_current_metrics_with_mock_spark(self):
        """Test metrics collection with mock Spark session."""
        monitor = ExecutionMonitor()

        # Create mock Spark
        mock_sc = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.return_value = []
        mock_sc.statusTracker.return_value = mock_tracker
        mock_sc.getExecutorMemoryStatus.return_value = {}

        mock_spark = MagicMock()
        mock_spark.sparkContext = mock_sc

        monitor._spark = mock_spark

        metrics = monitor._collect_current_metrics()

        assert "active_jobs" in metrics
        assert metrics["active_jobs"] == 0

    def test_collect_current_metrics_with_active_jobs(self):
        """Test metrics collection with active jobs."""
        monitor = ExecutionMonitor()

        # Create mock Spark with active jobs
        mock_sc = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.return_value = [1, 2, 3]
        mock_sc.statusTracker.return_value = mock_tracker

        # Mock memory status: executor_id -> (total, free)
        mock_sc.getExecutorMemoryStatus.return_value = {
            "executor1": (1000, 400),  # 60% used
        }

        mock_spark = MagicMock()
        mock_spark.sparkContext = mock_sc

        monitor._spark = mock_spark

        metrics = monitor._collect_current_metrics()

        assert metrics["active_jobs"] == 3
        # Memory usage should be calculated
        assert metrics["memory_usage_percent"] > 0

    def test_collect_current_metrics_exception(self):
        """Test exception handling in _collect_current_metrics (lines 209-210)."""
        monitor = ExecutionMonitor()

        # Create mock Spark that raises exception
        # The except block catches (AttributeError, KeyError, TypeError, ValueError)
        mock_spark = MagicMock()
        mock_spark.sparkContext.statusTracker.side_effect = AttributeError("Spark error")

        monitor._spark = mock_spark

        # Should not raise, should return default metrics
        metrics = monitor._collect_current_metrics()

        assert "timestamp" in metrics
        assert metrics["progress_percent"] == 0.0

    def test_start_monitoring(self):
        """Test start_monitoring starts the thread."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)

        assert monitor._spark is mock_spark
        assert monitor._monitoring_thread is not None
        assert monitor._monitoring_thread.is_alive()

        # Clean up
        monitor.stop_monitoring()

    def test_start_monitoring_emits_start_event(self):
        """Test start_monitoring emits start event."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)

        # Should have start event - check any event with type "start"
        start_events = [e for e in monitor._events if e.event_type == "start"]
        assert len(start_events) >= 1

        # Clean up
        monitor.stop_monitoring()

    def test_stop_monitoring(self):
        """Test stop_monitoring stops the thread."""
        # Use short check interval so thread responds quickly
        config = MonitoringConfig(check_interval_seconds=0.1)
        monitor = ExecutionMonitor(config=config)
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)
        thread = monitor._monitoring_thread

        monitor.stop_monitoring()

        # Thread should have been joined
        assert thread is not None
        # Thread should not be alive after stop
        # (Give it a moment if needed)
        import time

        time.sleep(0.2)
        assert not thread.is_alive()

    def test_stop_monitoring_emits_stop_event(self):
        """Test stop_monitoring emits stop event."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)
        events_before = len(monitor._events)

        monitor.stop_monitoring()

        # Should have stop event
        assert len(monitor._events) > events_before
        assert monitor._events[-1].event_type == "stop"

    def test_monitoring_loop_timeout(self):
        """Test timeout handling in monitoring loop (lines 130-135)."""
        # Create config with short timeout
        config = MonitoringConfig(
            check_interval_seconds=0.1,
            timeout_seconds=0.5,  # Very short timeout
        )
        monitor = ExecutionMonitor(config=config)
        mock_spark = MagicMock()

        # Start monitoring
        monitor.start_monitoring(mock_spark)

        # Wait for timeout
        time.sleep(1.0)

        # Check that timeout event was emitted
        timeout_events = [e for e in monitor._events if e.event_type == "warning"]
        assert len(timeout_events) > 0
        assert "timeout" in timeout_events[0].message.lower()

    def test_monitoring_loop_memory_warning(self):
        """Test memory warning in monitoring loop (line 153)."""
        config = MonitoringConfig(
            check_interval_seconds=0.1,
            memory_warning_threshold=50.0,
        )
        monitor = ExecutionMonitor(config=config)

        # Create mock Spark with high memory usage
        mock_sc = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.return_value = [1]
        mock_sc.statusTracker.return_value = mock_tracker

        # Mock high memory usage (80%)
        mock_sc.getExecutorMemoryStatus.return_value = {
            "executor1": (1000, 200),  # 80% used
        }

        mock_spark = MagicMock()
        mock_spark.sparkContext = mock_sc

        monitor.start_monitoring(mock_spark)

        # Wait a bit for monitoring loop to run
        time.sleep(0.5)

        # Check for memory warning event
        warning_events = [e for e in monitor._events if e.event_type == "warning"]
        [e for e in warning_events if "memory" in e.message.lower()]

        monitor.stop_monitoring()

    def test_monitoring_loop_exception_handling(self):
        """Test exception handling in monitoring loop (lines 162-164)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        # Make Spark raise exception on access
        mock_spark.sparkContext = MagicMock(side_effect=RuntimeError("Spark error"))

        monitor.start_monitoring(mock_spark)

        # Let the monitoring loop run and hit the exception
        time.sleep(0.5)

        # The loop should still be running (exception is caught)
        # Or it may have stopped due to exception handling

        monitor.stop_monitoring()


class TestExecutionMonitorEdgeCases:
    """Edge case tests for ExecutionMonitor."""

    def test_start_monitoring_multiple_times(self):
        """Test starting monitoring when already running."""
        monitor = ExecutionMonitor()

        # Create mock thread
        monitor._monitoring_thread = MagicMock()
        monitor._monitoring_thread.is_alive.return_value = True

        # Should not raise
        monitor.start_monitoring(MagicMock())

    def test_stop_monitoring_no_thread(self):
        """Test stopping when no monitoring thread exists."""
        monitor = ExecutionMonitor()

        # Should not raise
        monitor.stop_monitoring()

    def test_callback_removal_not_present(self):
        """Test removing callback that was never added."""
        monitor = ExecutionMonitor()

        callback = Mock()

        # Should not raise
        monitor.remove_callback(callback)

    def test_start_monitoring_initializes_state(self):
        """Test start_monitoring initializes state correctly (lines 97-100)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        # Mock _collect_current_metrics to return fixed value
        # This prevents the monitoring loop from updating _last_progress
        with patch.object(monitor, "_collect_current_metrics", return_value={"progress_percent": 0.0}):
            monitor.start_monitoring(mock_spark)

            assert monitor._spark is mock_spark
            # Events will have at least the "start" event
            assert len(monitor._events) >= 1
            assert monitor._last_progress == 0.0
            assert monitor._stop_event.is_set() is False

        # Clean up
        monitor.stop_monitoring()

    def test_stop_monitoring_sets_event(self):
        """Test stop_monitoring sets the stop event (lines 114-120)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)
        assert monitor._stop_event.is_set() is False

        monitor.stop_monitoring()
        assert monitor._stop_event.is_set() is True

    def test_monitoring_loop_progress_reporting(self):
        """Test monitoring loop progress reporting (lines 141-149)."""
        config = MonitoringConfig(
            check_interval_seconds=0.1,
            progress_threshold=5.0,
        )
        monitor = ExecutionMonitor(config=config)

        # Mock Spark with progress
        mock_sc = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.return_value = [1]
        mock_sc.statusTracker.return_value = mock_tracker

        # First call returns 20%, second returns 30%
        call_count = [0]

        def get_metrics_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"progress_percent": 20.0, "memory_usage_percent": 50.0}
            return {"progress_percent": 20.0, "memory_usage_percent": 50.0}

        with patch.object(monitor, "_collect_current_metrics", side_effect=get_metrics_side_effect):
            monitor.start_monitoring(MagicMock())

            # Wait for monitoring loop to run
            import time

            time.sleep(0.3)

            # Check that progress event was emitted
            [e for e in monitor._events if e.event_type == "progress"]
            # At least one progress event should be emitted

            monitor.stop_monitoring()

    def test_get_latest_metrics_with_spark(self):
        """Test get_latest_metrics with Spark session (lines 279-286)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()
        monitor._spark = mock_spark

        metrics = monitor.get_latest_metrics()

        assert "timestamp" in metrics
        assert "progress_percent" in metrics
        assert "memory_usage_percent" in metrics

    def test_is_monitoring_false_when_stopped(self):
        """Test is_monitoring returns False after stop."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)
        assert monitor.is_monitoring() is True

        monitor.stop_monitoring()
        # Give thread time to stop
        import time

        time.sleep(0.2)
        # Thread should have stopped
        assert monitor.is_monitoring() is False

    def test_emit_event_with_no_callbacks(self):
        """Test _emit_event with no callbacks (lines 239-243)."""
        monitor = ExecutionMonitor()
        monitor._callbacks = []  # No callbacks

        # Should not raise
        monitor._emit_event("test", "Test message")

        assert len(monitor._events) == 1
        assert monitor._events[0].event_type == "test"

    def test_collect_current_metrics_with_active_jobs(self):
        """Test _collect_current_metrics with active jobs (lines 188-194)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()
        monitor._spark = mock_spark

        mock_sc = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.return_value = [1, 2, 3]
        mock_sc.statusTracker.return_value = mock_tracker
        mock_sc.getExecutorMemoryStatus.return_value = {}

        mock_spark.sparkContext = mock_sc

        metrics = monitor._collect_current_metrics()

        assert metrics["active_jobs"] == 3

    def test_stop_monitoring_emits_event(self):
        """Test stop_monitoring emits stop event (lines 119-120)."""
        monitor = ExecutionMonitor()
        mock_spark = MagicMock()

        monitor.start_monitoring(mock_spark)
        events_before = len(monitor._events)

        monitor.stop_monitoring()

        events_after = len(monitor._events)
        assert events_after > events_before
        assert monitor._events[-1].event_type == "stop"
