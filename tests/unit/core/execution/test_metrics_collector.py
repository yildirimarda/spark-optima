# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for MetricsCollector."""

from unittest.mock import MagicMock

from spark_optima.core.bayesian.models import TrialMetrics
from spark_optima.core.execution.metrics_collector import (
    ExecutionMetrics,
    JobMetrics,
    MetricsCollector,
    StageMetrics,
)
from spark_optima.data.profiler import ColumnProfile, DataProfile


class TestMetricsCollector:
    """Test cases for MetricsCollector."""

    def test_initialization(self) -> None:
        """Test collector initialization."""
        collector = MetricsCollector()
        assert collector is not None
        assert collector.spark is None

    def test_initialization_with_spark(self) -> None:
        """Test collector initialization with spark session."""
        mock_spark = MagicMock()
        collector = MetricsCollector(spark=mock_spark)
        assert collector.spark is mock_spark

    def test_start_collection(self) -> None:
        """Test starting collection."""
        collector = MetricsCollector()
        collector.start_collection()
        assert collector._start_time is not None

    def test_stop_collection(self) -> None:
        """Test stopping collection."""
        collector = MetricsCollector()
        collector.start_collection()
        collector.stop_collection()
        assert collector._end_time is not None

    def test_reset(self) -> None:
        """Test resetting collector."""
        collector = MetricsCollector()
        collector.start_collection()
        collector.reset()
        assert collector._start_time is None
        assert collector._end_time is None

    def test_collect_metrics_no_spark(self) -> None:
        """Test collect_metrics when spark is None (lines 221-222)."""
        collector = MetricsCollector(spark=None)

        # Should return empty ExecutionMetrics when no spark
        metrics = collector.collect_metrics()

        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.success is True  # Default value

    def test_collect_metrics_with_spark_mock(self) -> None:
        """Test collect_metrics with mocked spark."""
        mock_spark = MagicMock()
        mock_sc = MagicMock()
        mock_spark.sparkContext = mock_sc
        mock_sc.statusTracker.return_value.getActiveJobsIds.return_value = []
        mock_sc.getExecutorMemoryStatus.return_value = {}

        collector = MetricsCollector(spark=mock_spark)
        collector.start_collection()

        metrics = collector.collect_metrics()

        assert isinstance(metrics, ExecutionMetrics)
        collector.stop_collection()

    def test_collect_metrics_exception_handling(self) -> None:
        """Test collect_metrics exception handling (lines 246-248)."""
        mock_spark = MagicMock()
        # Make spark.sparkContext.statusTracker() raise an exception
        mock_context = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.side_effect = RuntimeError("Spark error")
        mock_context.statusTracker.return_value = mock_tracker
        mock_spark.sparkContext = mock_context

        collector = MetricsCollector(spark=mock_spark)
        collector.start_collection()

        metrics = collector.collect_metrics()

        # Should return ExecutionMetrics with success=False
        assert metrics.success is False
        assert "Spark error" in metrics.error_message

    def test_set_spark_session(self) -> None:
        """Test set_spark_session method."""
        collector = MetricsCollector()
        assert collector.spark is None

        mock_spark = MagicMock()
        collector.set_spark_session(mock_spark)

        assert collector.spark is mock_spark

    def test_get_stage_summary_empty(self) -> None:
        """Test get_stage_summary with no metrics."""
        collector = MetricsCollector()

        summary = collector.get_stage_summary()

        assert summary == {}

    def test_get_stage_summary_with_data(self) -> None:
        """Test get_stage_summary with stage data."""
        from spark_optima.core.execution.metrics_collector import JobMetrics, StageMetrics

        collector = MetricsCollector()

        # Add mock job metrics
        stage1 = StageMetrics(stage_id=1, stage_name="stage1")
        stage2 = StageMetrics(stage_id=2, stage_name="stage2")
        job = JobMetrics(job_id=1, stage_metrics=[stage1, stage2])

        collector._job_metrics = [job]

        summary = collector.get_stage_summary()

        assert len(summary) == 2
        assert 1 in summary
        assert 2 in summary
        assert summary[1]["stage_name"] == "stage1"
        assert summary[2]["stage_name"] == "stage2"

    def test_get_shuffle_summary_empty(self) -> None:
        """Test get_shuffle_summary with no metrics."""
        collector = MetricsCollector()

        summary = collector.get_shuffle_summary()

        assert summary["total_read_gb"] == 0.0
        assert summary["total_write_gb"] == 0.0
        assert summary["total_gb"] == 0.0

    def test_get_shuffle_summary_with_data(self) -> None:
        """Test get_shuffle_summary with shuffle data."""
        from spark_optima.core.execution.metrics_collector import JobMetrics, StageMetrics

        collector = MetricsCollector()

        # Add job with shuffle data
        stage = StageMetrics(
            stage_id=1,
            shuffle_read_bytes=1024**3,  # 1 GB
            shuffle_write_bytes=2 * 1024**3,  # 2 GB
        )
        job = JobMetrics(job_id=1, stage_metrics=[stage])

        collector._job_metrics = [job]

        summary = collector.get_shuffle_summary()

        assert summary["total_read_gb"] == 1.0
        assert summary["total_write_gb"] == 2.0
        assert summary["total_gb"] == 3.0


class TestExecutionMetrics:
    """Test cases for ExecutionMetrics."""

    def test_default_initialization(self) -> None:
        """Test default values."""
        metrics = ExecutionMetrics()
        assert metrics.execution_time_seconds == 0.0
        assert metrics.success is True

    def test_to_trial_metrics(self) -> None:
        """Test conversion to TrialMetrics."""
        metrics = ExecutionMetrics(
            execution_time_seconds=60.0,
            memory_peak_gb=4.0,
            cpu_utilization_percent=75.0,
            shuffle_read_gb=1.0,
            shuffle_write_gb=0.5,
            success=True,
        )
        trial_metrics = metrics.to_trial_metrics()

        assert isinstance(trial_metrics, TrialMetrics)
        assert trial_metrics.execution_time_seconds == 60.0
        assert trial_metrics.memory_peak_gb == 4.0

    def test_to_trial_metrics_with_error(self) -> None:
        """Test conversion to TrialMetrics with error."""
        metrics = ExecutionMetrics(
            execution_time_seconds=30.0,
            success=False,
            error_message="Test error message",
        )
        trial_metrics = metrics.to_trial_metrics()

        assert trial_metrics.success is False
        assert trial_metrics.error_message == "Test error message"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from spark_optima.core.execution.metrics_collector import JobMetrics

        job = JobMetrics(job_id=1)
        metrics = ExecutionMetrics(
            execution_time_seconds=60.0,
            memory_peak_gb=4.0,
            memory_average_gb=2.0,
            cpu_utilization_percent=75.0,
            shuffle_read_gb=1.0,
            shuffle_write_gb=0.5,
            jobs=[job],
            gc_time_seconds=5.0,
            success=True,
        )
        result = metrics.to_dict()

        assert result["execution_time_seconds"] == 60.0
        assert result["memory_peak_gb"] == 4.0
        assert result["memory_average_gb"] == 2.0
        assert result["cpu_utilization_percent"] == 75.0
        assert result["shuffle_read_gb"] == 1.0
        assert result["shuffle_write_gb"] == 0.5
        assert result["num_jobs"] == 1
        assert result["gc_time_seconds"] == 5.0
        assert result["success"] is True
        assert len(result["jobs"]) == 1


class TestJobMetrics:
    """Test cases for JobMetrics."""

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        import time

        start = time.time()
        end = start + 60.0

        job = JobMetrics(
            job_id=1,
            submission_time=start,
            completion_time=end,
        )

        assert job.duration_seconds == 60.0

    def test_duration_zero_when_not_completed(self) -> None:
        """Test duration is 0 when completion time not after submission."""
        job = JobMetrics(
            job_id=1,
            submission_time=100.0,
            completion_time=50.0,  # Before submission
        )

        assert job.duration_seconds == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from spark_optima.core.execution.metrics_collector import StageMetrics

        stage = StageMetrics(stage_id=1, stage_name="stage1")
        job = JobMetrics(
            job_id=1,
            job_name="test_job",
            submission_time=100.0,
            completion_time=160.0,
            stage_metrics=[stage],
            total_tasks=10,
            failed_tasks=2,
        )
        result = job.to_dict()

        assert result["job_id"] == 1
        assert result["job_name"] == "test_job"
        assert result["duration_seconds"] == 60.0
        assert result["total_tasks"] == 10
        assert result["failed_tasks"] == 2
        assert result["num_stages"] == 1
        assert len(result["stages"]) == 1
        assert result["stages"][0]["stage_id"] == 1


class TestStageMetrics:
    """Test cases for StageMetrics."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stage = StageMetrics(
            stage_id=1,
            stage_name="test_stage",
            num_tasks=10,
            executor_run_time=1000,
            executor_cpu_time=800,
            input_bytes=1024,
            output_bytes=2048,
            shuffle_read_bytes=512,
            shuffle_write_bytes=256,
            peak_memory=4096,
        )
        result = stage.to_dict()

        assert result["stage_id"] == 1
        assert result["stage_name"] == "test_stage"
        assert result["num_tasks"] == 10
        assert result["executor_run_time_ms"] == 1000
        assert result["executor_cpu_time_ms"] == 800
        assert result["input_bytes"] == 1024
        assert result["output_bytes"] == 2048
        assert result["shuffle_read_bytes"] == 512
        assert result["shuffle_write_bytes"] == 256
        assert result["peak_memory_bytes"] == 4096


class TestDataProfile:
    """Test cases for DataProfile."""

    def test_get_column_names(self) -> None:
        """Test getting column names."""
        columns = [
            ColumnProfile(name="col1"),
            ColumnProfile(name="col2"),
        ]
        profile = DataProfile(columns=columns)

        names = profile.get_column_names()
        assert names == ["col1", "col2"]

    def test_get_column_profile(self) -> None:
        """Test getting specific column profile."""
        columns = [
            ColumnProfile(name="col1"),
            ColumnProfile(name="col2"),
        ]
        profile = DataProfile(columns=columns)

        col1 = profile.get_column_profile("col1")
        assert col1 is not None
        assert col1.name == "col1"

    def test_get_numeric_columns(self) -> None:
        """Test getting numeric columns."""
        columns = [
            ColumnProfile(name="str_col", data_type="string"),
            ColumnProfile(name="int_col", data_type="int"),
            ColumnProfile(name="double_col", data_type="double"),
        ]
        profile = DataProfile(columns=columns)

        numeric = profile.get_numeric_columns()
        assert len(numeric) == 2
        assert numeric[0].name == "int_col"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        profile = DataProfile(
            path="/test/path",
            format="parquet",
            num_rows=1000,
            num_columns=5,
        )
        result = profile.to_dict()

        assert result["path"] == "/test/path"
        assert result["format"] == "parquet"
        assert result["num_rows"] == 1000


class TestMetricsCollectorNoPySpark:
    """Tests for MetricsCollector when PySpark is not available (lines 25-26)."""

    def test_collect_metrics_pyspark_not_available(self) -> None:
        """Test behavior when PYSPARK_AVAILABLE is False."""
        # This test verifies the code path when PySpark is not available
        # The actual PYSPARK_AVAILABLE variable is set at module level
        # We can test the methods that check this variable

        mock_spark = MagicMock()
        collector = MetricsCollector(spark=mock_spark)

        # Test _collect_job_metrics when PySpark not available
        # We need to mock PYSPARK_AVAILABLE
        import spark_optima.core.execution.metrics_collector as mc_module

        original_value = mc_module.PYSPARK_AVAILABLE

        try:
            # Mock PYSPARK_AVAILABLE to False
            mc_module.PYSPARK_AVAILABLE = False

            # When PYSPARK_AVAILABLE is False, _collect_job_metrics returns empty list
            result = collector._collect_job_metrics()
            assert result == []

            # _collect_memory_metrics should return default values
            memory = collector._collect_memory_metrics()
            assert memory == {"peak_gb": 0.0, "average_gb": 0.0}

            # _collect_shuffle_metrics should return default values
            shuffle = collector._collect_shuffle_metrics()
            assert shuffle == {"read_gb": 0.0, "write_gb": 0.0}

        finally:
            # Restore original value
            mc_module.PYSPARK_AVAILABLE = original_value

    def test_calculate_execution_time_no_start(self) -> None:
        """Test _calculate_execution_time when _start_time is None."""
        collector = MetricsCollector()

        # When _start_time is None, should return 0.0
        duration = collector._calculate_execution_time()
        assert duration == 0.0

    def test_calculate_execution_time_with_start(self) -> None:
        """Test _calculate_execution_time when _start_time is set."""
        import time

        collector = MetricsCollector()
        collector._start_time = time.time() - 10.0  # 10 seconds ago

        duration = collector._calculate_execution_time()
        assert duration >= 10.0

    def test_calculate_execution_time_with_end(self) -> None:
        """Test _calculate_execution_time when _end_time is set."""
        import time

        collector = MetricsCollector()
        collector._start_time = time.time() - 30.0
        collector._end_time = time.time() - 10.0  # 10 seconds ago

        duration = collector._calculate_execution_time()
        assert 19.0 <= duration <= 21.0  # Approximately 20 seconds


class TestMetricsCollectorWithSpark:
    """Tests for MetricsCollector with PySpark available."""

    def test_collect_job_metrics_with_spark(self) -> None:
        """Test _collect_job_metrics when PySpark is available (lines 266-282)."""
        import spark_optima.core.execution.metrics_collector as mc_module

        original_value = mc_module.PYSPARK_AVAILABLE

        try:
            mc_module.PYSPARK_AVAILABLE = True

            mock_spark = MagicMock()
            mock_sc = MagicMock()
            mock_tracker = MagicMock()
            mock_tracker.getActiveJobsIds.return_value = [1, 2]
            mock_sc.statusTracker.return_value = mock_tracker
            mock_spark.sparkContext = mock_sc

            collector = MetricsCollector(spark=mock_spark)
            result = collector._collect_job_metrics()

            # Should return list (may be empty if implementation is simplified)
            assert isinstance(result, list)

        finally:
            mc_module.PYSPARK_AVAILABLE = original_value

    def test_collect_memory_metrics_with_spark(self) -> None:
        """Test _collect_memory_metrics when PySpark is available (lines 297-329)."""
        import spark_optima.core.execution.metrics_collector as mc_module

        original_value = mc_module.PYSPARK_AVAILABLE

        try:
            mc_module.PYSPARK_AVAILABLE = True

            mock_spark = MagicMock()
            mock_sc = MagicMock()
            # Mock memory status: executor_id -> (total, free)
            mock_sc.getExecutorMemoryStatus.return_value = {
                "executor1": (4 * 1024**3, 2 * 1024**3),  # 4GB total, 2GB free
                "executor2": (8 * 1024**3, 6 * 1024**3),  # 8GB total, 6GB free
            }
            mock_spark.sparkContext = mock_sc

            collector = MetricsCollector(spark=mock_spark)
            memory = collector._collect_memory_metrics()

            assert "peak_gb" in memory
            assert "average_gb" in memory
            assert memory["peak_gb"] > 0
            assert memory["average_gb"] > 0

        finally:
            mc_module.PYSPARK_AVAILABLE = original_value

    def test_collect_shuffle_metrics_with_spark(self) -> None:
        """Test _collect_shuffle_metrics when PySpark is available (lines 331-349)."""
        import spark_optima.core.execution.metrics_collector as mc_module

        original_value = mc_module.PYSPARK_AVAILABLE

        try:
            mc_module.PYSPARK_AVAILABLE = True

            mock_spark = MagicMock()
            mock_sc = MagicMock()
            mock_spark.sparkContext = mock_sc

            collector = MetricsCollector(spark=mock_spark)
            shuffle = collector._collect_shuffle_metrics()

            assert "read_gb" in shuffle
            assert "write_gb" in shuffle

        finally:
            mc_module.PYSPARK_AVAILABLE = original_value

    def test_collect_gc_metrics(self) -> None:
        """Test _collect_gc_metrics (lines 351-363)."""
        collector = MetricsCollector()
        gc_time = collector._collect_gc_metrics()

        # Should return 0.0 (placeholder implementation)
        assert gc_time == 0.0

    def test_estimate_cpu_utilization(self) -> None:
        """Test _estimate_cpu_utilization (lines 364-374)."""
        collector = MetricsCollector()
        cpu_util = collector._estimate_cpu_utilization()

        # Should return 0.0 (placeholder implementation)
        assert cpu_util == 0.0

    def test_set_spark_session(self) -> None:
        """Test set_spark_session method (lines 411-418)."""
        collector = MetricsCollector()
        assert collector.spark is None

        mock_spark = MagicMock()
        collector.set_spark_session(mock_spark)

        assert collector.spark is mock_spark

    def test_reset_with_data(self) -> None:
        """Test reset with existing data (lines 421-425)."""
        collector = MetricsCollector()

        # Add some data
        collector._start_time = 100.0
        collector._end_time = 200.0
        stage = StageMetrics(stage_id=1)
        job = JobMetrics(job_id=1, stage_metrics=[stage])
        collector._job_metrics = [job]

        collector.reset()

        assert collector._start_time is None
        assert collector._end_time is None
        assert collector._job_metrics == []

    def test_collect_metrics_with_spark_and_exception(self) -> None:
        """Test collect_metrics when spark raises exception (lines 246-252)."""
        mock_spark = MagicMock()
        # Make spark.sparkContext raise an exception
        mock_context = MagicMock()
        mock_tracker = MagicMock()
        mock_tracker.getActiveJobsIds.side_effect = RuntimeError("Spark error")
        mock_context.statusTracker.return_value = mock_tracker
        mock_spark.sparkContext = mock_context

        collector = MetricsCollector(spark=mock_spark)
        collector.start_collection()

        metrics = collector.collect_metrics()

        # Should return ExecutionMetrics with success=False
        assert metrics.success is False
        assert "Spark error" in metrics.error_message
