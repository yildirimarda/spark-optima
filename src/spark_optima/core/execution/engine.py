# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Execution engine for Spark configuration optimization.

This module provides ExecutionEngine class that orchestrates actual Spark
job execution with configuration application and metrics collection.
Supports both local Docker execution and remote cluster execution
(AWS Glue, Databricks, Azure Synapse).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
    from spark_optima.platforms.models import CostModel, ResourceSpec

from spark_optima.core.bayesian.models import TrialMetrics, TrialResult, TrialStatus
from spark_optima.core.execution.metrics_collector import ExecutionMetrics, MetricsCollector
from spark_optima.core.execution.monitor import ExecutionMonitor, MonitoringConfig
from spark_optima.core.execution.spark_runner import SparkRunner

logger = logging.getLogger(__name__)

# Polling configuration for cluster jobs
_DEFAULT_POLLING_INTERVAL_SECONDS: float = 10.0
_DEFAULT_POLLING_TIMEOUT_SECONDS: float = 3600.0  # 1 hour


class ExecutionEngine:
    """Main execution engine for running Spark jobs.

    This engine manages the complete execution lifecycle:
    - Configuration validation and application
    - Spark session management (local mode)
    - Code execution (local Docker or remote cluster)
    - Metrics collection
    - Resource monitoring

    Supports multiple platforms:
    - local: Docker-based local execution (default)
    - aws_glue: AWS Glue ETL jobs
    - databricks: Databricks clusters (AWS/Azure)
    - azure_synapse: Azure Synapse Spark pools

    Example:
        >>> engine = ExecutionEngine(platform="aws_glue")
        >>> result = engine.execute(
        ...     config={"spark.executor.memory": "4g", ...},
        ...     code_path="./spark_job.py",
        ...     resource_spec=ResourceSpec(cpu_cores=16, memory_gb=64),
        ... )
        >>> print(f"Execution time: {result.metrics.execution_time_seconds:.1f}s")

    """

    def __init__(
        self,
        app_name: str = "SparkOptima",
        enable_monitoring: bool = True,
        use_docker: bool = True,
        platform: str | None = None,
    ) -> None:
        """Initialize the execution engine.

        Args:
            app_name: Application name for Spark sessions.
            enable_monitoring: Whether to enable live monitoring.
            use_docker: If True, use Docker for local execution.
            platform: Platform name ("local", "aws_glue", "databricks", "azure_synapse")
                or None for local Docker execution.

        """
        self.app_name = app_name
        self.enable_monitoring = enable_monitoring
        self._use_docker = use_docker
        self._platform_name = platform or "local"

        # Initialize platform adapter if specified
        self._platform_adapter: Any | None = None
        if self._platform_name != "local":
            self._init_platform_adapter()

        # Initialize local Spark runner (for local/Docker execution)
        self._runner: SparkRunner | None = None
        if self._platform_name == "local":
            try:
                self._runner = SparkRunner(app_name=app_name, use_docker=use_docker)
                self._spark_available = True
            except RuntimeError as e:
                logger.warning(f"Spark not available: {e}")
                self._runner = None
                self._spark_available = False
        else:
            self._spark_available = self._platform_adapter is not None

        self._monitor: ExecutionMonitor | None = None
        if enable_monitoring and self._platform_name == "local":
            self._monitor = ExecutionMonitor(MonitoringConfig(check_interval_seconds=5.0))

    def _init_platform_adapter(self) -> None:
        """Initialize the platform adapter based on platform name."""
        try:
            if self._platform_name == "aws_glue":
                from spark_optima.platforms import AWSGluePlatform

                self._platform_adapter = AWSGluePlatform()
            elif self._platform_name == "databricks":
                from spark_optima.platforms import DatabricksPlatform

                self._platform_adapter = DatabricksPlatform()
            elif self._platform_name == "azure_synapse":
                from spark_optima.platforms import AzureSynapsePlatform

                self._platform_adapter = AzureSynapsePlatform()
            else:
                logger.error(f"Unknown platform: {self._platform_name}")
                self._platform_adapter = None
        except RuntimeError as e:
            logger.warning(f"Platform adapter not available: {e}")
            self._platform_adapter = None

    def execute(
        self,
        config: dict[str, Any],
        code: str | None = None,
        code_path: str | Path | None = None,
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        timeout_seconds: int = 3600,
    ) -> TrialResult:
        """Execute Spark code with given configuration.

        Args:
            config: Spark configuration dictionary.
            code: Python code string (optional).
            code_path: Path to Python file (optional).
            resource_spec: Resource specifications.
            cost_model: Cost model.
            timeout_seconds: Maximum execution time.

        Returns:
            TrialResult with execution metrics.

        Raises:
            RuntimeError: If platform is not available.
            ValueError: If neither code nor code_path is provided.

        """
        start_time = time.time()

        # Route to appropriate execution method
        if self._platform_name != "local":
            return self._execute_cluster(
                config=config,
                code=code,
                code_path=code_path,
                resource_spec=resource_spec,
                cost_model=cost_model,
                timeout_seconds=timeout_seconds,
                start_time=start_time,
            )

        # Local Docker execution
        if not self._spark_available or self._runner is None:
            return self._build_error_result(
                config=config,
                error_message="Spark is not available. Install PySpark to use execution mode.",
                duration=time.time() - start_time,
            )

        # Validate inputs
        if code is None and code_path is None:
            return self._build_error_result(
                config=config,
                error_message="Either code or code_path must be provided",
                duration=time.time() - start_time,
            )

        try:
            # Create Spark session with config
            with self._runner.create_session(config, resource_spec) as spark:
                # Set up monitoring
                if self._monitor and self.enable_monitoring:
                    self._monitor.start_monitoring(spark)
                    metrics_collector = MetricsCollector()
                    metrics_collector.set_spark_session(spark)

                # Execute code
                if code_path:
                    exec_result = self._runner.execute_file(
                        file_path=code_path,
                        config=config,
                        resource_spec=resource_spec,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    exec_result = self._runner.execute_code(
                        code=code,  # type: ignore[arg-type]
                        config=config,
                        resource_spec=resource_spec,
                        timeout_seconds=timeout_seconds,
                    )

                # Stop monitoring
                if self._monitor:
                    self._monitor.stop_monitoring()

                # Collect metrics
                metrics_collector = MetricsCollector()
                metrics_collector.start_collection()
                metrics_collector.set_spark_session(spark)
                metrics_collector.stop_collection()
                execution_metrics = metrics_collector.collect_metrics()

                duration = time.time() - start_time

                return self._build_trial_result(
                    config=config,
                    execution_metrics=execution_metrics,
                    exec_result=exec_result,
                    duration=duration,
                    cost_model=cost_model,
                )

        except (RuntimeError, TimeoutError, AttributeError, ValueError, KeyError, TypeError) as e:
            duration = time.time() - start_time
            logger.error(f"Execution failed: {e}")

            return self._build_error_result(
                config=config,
                error_message=str(e),
                duration=duration,
            )

    def _execute_cluster(
        self,
        config: dict[str, Any],
        code: str | None,
        code_path: str | Path | None,
        resource_spec: ResourceSpec | None,
        cost_model: CostModel | None,
        timeout_seconds: int,
        start_time: float,
    ) -> TrialResult:
        """Execute job on remote cluster platform.

        Args:
            config: Spark configuration.
            code: Python code string.
            code_path: Path to Python file.
            resource_spec: Resource specifications.
            cost_model: Cost model.
            timeout_seconds: Maximum execution time.
            start_time: When execution started.

        Returns:
            TrialResult with execution metrics.

        """
        if self._platform_adapter is None:
            return self._build_error_result(
                config=config,
                error_message=f"Platform {self._platform_name} is not available. Check credentials.",
                duration=time.time() - start_time,
            )

        # Get code path or create temp file
        if code_path is None and code is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                code_path = f.name

        if code_path is None:
            return self._build_error_result(
                config=config,
                error_message="Either code or code_path must be provided",
                duration=time.time() - start_time,
            )

        try:
            # Submit job
            logger.info(f"Submitting job to {self._platform_name}...")
            submit_result = self._platform_adapter.submit_job(
                code_path=str(code_path),
                cluster_config=self._build_cluster_config(config, resource_spec),
                timeout_minutes=timeout_seconds // 60 or 60,
            )

            if not submit_result.get("success", False):
                return self._build_error_result(
                    config=config,
                    error_message=f"Job submission failed: {submit_result.get('error', 'Unknown error')}",
                    duration=time.time() - start_time,
                )

            job_id = submit_result.get("job_id") or submit_result.get("run_id") or submit_result.get("job_run_id")
            if not job_id:
                return self._build_error_result(
                    config=config,
                    error_message="Job submitted but no job ID returned",
                    duration=time.time() - start_time,
                )

            # Poll for job completion
            logger.info(f"Job {job_id} submitted. Polling for completion...")
            result = self._poll_job_completion(job_id, timeout_seconds)

            duration = time.time() - start_time

            return self._build_trial_result(
                config=config,
                execution_metrics=self._extract_metrics_from_result(result),
                exec_result=result,
                duration=duration,
                cost_model=cost_model,
            )

        except (RuntimeError, TimeoutError, ValueError, KeyError, TypeError, ImportError) as e:
            duration = time.time() - start_time
            logger.error(f"Cluster execution failed: {e}")
            return self._build_error_result(
                config=config,
                error_message=str(e),
                duration=duration,
            )

    def _poll_job_completion(
        self,
        job_id: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Poll job status until completion or timeout.

        Args:
            job_id: Job identifier.
            timeout_seconds: Maximum time to wait.

        Returns:
            Dictionary with job results.

        """
        start = time.time()
        interval = _DEFAULT_POLLING_INTERVAL_SECONDS

        while time.time() - start < timeout_seconds:
            try:
                status_result = self._platform_adapter.get_job_status(job_id)  # type: ignore[union-attr]
                status = status_result.get("status", "unknown")

                if status in ("completed", "failed", "timeout", "stopped", "error"):
                    # Get final results
                    results = self._platform_adapter.get_job_results(job_id)  # type: ignore[union-attr]
                    return results

                logger.debug(f"Job {job_id} status: {status}. Waiting...")
                time.sleep(interval)

            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning(f"Error polling job status: {e}")
                time.sleep(interval)

        # Timeout
        return {
            "success": False,
            "error": f"Job {job_id} timed out after {timeout_seconds} seconds",
            "error_type": "TimeoutError",
        }

    def _build_cluster_config(
        self,
        config: dict[str, Any],
        resource_spec: ResourceSpec | None,
    ) -> Any:
        """Build cluster config from Spark config and resource spec.

        Args:
            config: Spark configuration.
            resource_spec: Resource specifications.

        Returns:
            ClusterConfig for the platform.

        """
        from spark_optima.platforms.models import ClusterConfig, WorkerType

        # Try to get worker type from platform adapter
        worker_type = None
        if hasattr(self._platform_adapter, "get_worker_type"):
            # Use medium worker as default
            worker_type = self._platform_adapter.get_worker_type("G.2X") or self._platform_adapter.get_worker_type("m5.xlarge")

        if worker_type is None:
            # Create a default worker type
            worker_type = WorkerType(
                name="default",
                size=None,  # type: ignore[arg-type]
                resources=resource_spec or ResourceSpec(cpu_cores=4, memory_gb=16),
                cost=None,  # type: ignore[arg-type]
                description="Default worker",
            )

        worker_count = 2
        if resource_spec and resource_spec.cpu_cores:
            worker_count = max(2, resource_spec.cpu_cores // 4)

        return ClusterConfig(
            worker_type=worker_type,
            worker_count=worker_count,
            driver_type=None,
            driver_count=1,
            spark_version=config.get("spark.version", "3.5.0"),
            platform_config={},
        )

    def _extract_metrics_from_result(self, result: dict[str, Any]) -> ExecutionMetrics:
        """Extract ExecutionMetrics from platform job result.

        Args:
            result: Job result dictionary.

        Returns:
            ExecutionMetrics instance.

        """
        from spark_optima.core.execution.metrics_collector import ExecutionMetrics

        return ExecutionMetrics(
            execution_time_seconds=result.get("execution_time", 0.0),
            success=result.get("status") == "completed",
            error_message=result.get("error_message", ""),
        )

    def _build_trial_result(
        self,
        config: dict[str, Any],
        execution_metrics: ExecutionMetrics,
        exec_result: dict[str, Any],
        duration: float,
        cost_model: CostModel | None,
    ) -> TrialResult:
        """Build TrialResult from execution data.

        Args:
            config: Applied configuration.
            execution_metrics: Collected execution metrics.
            exec_result: Execution result from runner.
            duration: Total duration.
            cost_model: Cost model.

        Returns:
            TrialResult.

        """
        # Calculate cost if not in metrics
        cost = execution_metrics.to_trial_metrics().cost_estimate_usd
        if cost == 0.0 and cost_model:
            cost = cost_model.calculate(duration / 3600)

        # Build TrialMetrics
        metrics = TrialMetrics(
            execution_time_seconds=execution_metrics.execution_time_seconds,
            memory_peak_gb=execution_metrics.memory_peak_gb,
            cpu_utilization_percent=execution_metrics.cpu_utilization_percent,
            shuffle_read_gb=execution_metrics.shuffle_read_gb,
            shuffle_write_gb=execution_metrics.shuffle_write_gb,
            cost_estimate_usd=cost,
            success=exec_result.get("success", False) and execution_metrics.success,
            error_message=execution_metrics.error_message or exec_result.get("error", ""),
        )

        # Determine status
        if metrics.success:
            status = TrialStatus.COMPLETED
        elif "Timeout" in metrics.error_message:
            status = TrialStatus.FAILED  # Could add TIMEOUT status
        else:
            status = TrialStatus.FAILED

        return TrialResult(
            trial_number=-1,  # Will be set by caller
            configuration=config,
            metrics=metrics,
            status=status,
            duration_seconds=duration,
        )

    def _build_error_result(
        self,
        config: dict[str, Any],
        error_message: str,
        duration: float,
    ) -> TrialResult:
        """Build error TrialResult.

        Args:
            config: Configuration that was being tested.
            error_message: Error description.
            duration: How long execution took before failure.

        Returns:
            TrialResult with FAILED status.

        """
        return TrialResult(
            trial_number=-1,
            configuration=config,
            metrics=TrialMetrics(
                execution_time_seconds=duration,
                success=False,
                error_message=error_message,
            ),
            status=TrialStatus.FAILED,
            duration_seconds=duration,
        )

    def execute_trial(
        self,
        trial_number: int,
        config: dict[str, Any],
        code: str | None = None,
        code_path: str | Path | None = None,
        resource_spec: ResourceSpec | None = None,
        cost_model: CostModel | None = None,
        timeout_seconds: int = 3600,
    ) -> TrialResult:
        """Execute a trial for Bayesian optimization.

        This is a convenience wrapper around execute() that includes
        the trial number.

        Args:
            trial_number: Trial identifier.
            config: Spark configuration.
            code: Python code string.
            code_path: Path to Python file.
            resource_spec: Resource specifications.
            cost_model: Cost model.
            timeout_seconds: Maximum execution time.

        Returns:
            TrialResult with trial_number set.

        """
        result = self.execute(
            config=config,
            code=code,
            code_path=code_path,
            resource_spec=resource_spec,
            cost_model=cost_model,
            timeout_seconds=timeout_seconds,
        )

        # Override trial number
        result.trial_number = trial_number

        return result

    def is_available(self) -> bool:
        """Check if execution engine is available.

        Returns:
            True if execution is available for the current platform.

        """
        if self._platform_name == "local":
            return self._spark_available
        return self._platform_adapter is not None

    def validate_config(
        self,
        config: dict[str, Any],
        spark_version: str | None = None,
    ) -> list[str]:
        """Validate configuration for execution.

        Args:
            config: Configuration to validate.
            spark_version: Target Spark version.

        Returns:
            List of validation errors.

        """
        if self._platform_name == "local":
            if self._runner is None:
                return ["Spark not available"]
            return self._runner.validate_config(config, spark_version)

        # For cluster platforms, do basic validation
        errors = []
        if not config.get("spark.executor.memory"):
            errors.append("Missing spark.executor.memory")
        return errors

    def get_spark_info(self) -> dict[str, Any]:
        """Get information about Spark environment.

        Returns:
            Dictionary with Spark information.

        """
        if self._platform_name == "local":
            if self._runner is None:
                return {"available": False}
            return self._runner.get_spark_info()

        # For cluster platforms
        return {
            "available": self._platform_adapter is not None,
            "platform": self._platform_name,
            "platform_info": (
                self._platform_adapter.get_worker_types() if self._platform_adapter else []
            ),
        }

    def add_monitoring_callback(self, callback: Any) -> None:
        """Add callback for monitoring events.

        Args:
            callback: Function to call on events.

        """
        if self._monitor:
            self._monitor.add_callback(callback)

    def get_monitoring_events(self) -> list[Any]:
        """Get recorded monitoring events.

        Returns:
            List of monitoring events.

        """
        if self._monitor:
            return self._monitor.get_events()
        return []
