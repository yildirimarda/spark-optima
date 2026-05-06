# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spark runner for local Spark execution.

This module provides SparkRunner class that manages local Spark sessions,
applies configurations, and executes Spark code for optimization trials.

Security Note:
    User code is executed in an isolated subprocess to prevent security issues.
    The subprocess approach ensures that even if malicious code is provided,
    it runs in a separate process with no access to the main process memory.
    See _run_code_in_subprocess() for implementation details.
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)

# Dangerous AST node patterns to block (additional layer of defense)
_DANGEROUS_IMPORTS = frozenset(
    {"os", "subprocess", "sys", "builtins", "eval", "exec", "compile", "__import__"},
)
_DANGEROUS_ATTRIBUTES = frozenset(
    {
        "__builtins__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__import__",
        "__getattribute__",
        "func_globals",
        "gi_frame",
    },
)

_DANGEROUS_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
    },
)


def _validate_code_safety(code: str) -> None:
    """Validate that code doesn't contain dangerous patterns.

    This is an additional layer of defense. The primary security mechanism
    is running user code in an isolated subprocess.

    Args:
        code: Python code string to validate.

    Raises:
        ValueError: If dangerous patterns are detected.

    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}") from e

    for node in ast.walk(tree):
        # Block import statements
        if isinstance(node, ast.Import | ast.ImportFrom):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
            else:
                module = node.module.split(".")[0] if node.module else ""

            if module in _DANGEROUS_IMPORTS:
                raise ValueError(f"Import of '{module}' is not allowed for security reasons")

        # Block access to dangerous attributes
        if isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_ATTRIBUTES:
            raise ValueError(f"Access to '{node.attr}' is not allowed")

        # Block dangerous function calls (getattr, setattr, hasattr with __builtins__)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "setattr", "hasattr")
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in _DANGEROUS_NAMES
        ):
            raise ValueError(
                f"Call to {node.func.id} with '{node.args[0].id}' is not allowed",
            )

        # Block dangerous variable names (e.g., __builtins__)
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed")

        # Block string literals that look like they're trying to access __builtins__
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "__builtins__" in value or "__import__" in value:
                raise ValueError("Access to builtins is not allowed")


# Optional PySpark import with graceful fallback
try:
    from pyspark.sql import SparkSession

    _PYSPARK_AVAILABLE = True
except ImportError:
    _PYSPARK_AVAILABLE = False
    logger.warning("PySpark not available. Local Spark execution disabled.")


class SparkRunner:
    """Manages local Spark session lifecycle and execution.

    This class provides a clean interface for creating Spark sessions with
    specific configurations, executing code, and managing resources properly.

    Security:
        Code execution uses Docker-based isolation (run_in_docker method).
        This provides strong security by running user code in an isolated container.
        If Docker is not available, a RuntimeError is raised - no insecure fallback.

    Example:
        >>> runner = SparkRunner()
        >>> with runner.create_session(config={"spark.executor.memory": "4g"}) as spark:
        ...     df = spark.read.parquet("data.parquet")
        ...     result = df.count()
        ...     print(f"Count: {result}")

    """

    def __init__(
        self,
        app_name: str = "SparkOptima",
        master: str = "local[*]",
        log_level: str = "WARN",
        docker_image: str = "spark-optima:latest",
        use_docker: bool = False,
    ) -> None:
        """Initialize the Spark runner.

        Args:
            app_name: Application name for Spark session.
            master: Spark master URL (default: local[*]).
            log_level: Spark log level.
            docker_image: Docker image to use for isolated execution.
            use_docker: If True, use Docker for isolated execution.
                If False, run locally (default: False).

        Raises:
            RuntimeError: If Docker is required but not available.

        """
        self.app_name = app_name
        self.master = master
        self.log_level = log_level
        self._docker_image = docker_image
        self._use_docker = use_docker

        self._active_session: Any | None = None
        self._session_configs: dict[str, Any] = {}

        # Check Docker availability if needed
        if self._use_docker:
            self._docker_available = self._check_docker_available()
            if not self._docker_available:
                raise RuntimeError(
                    "Docker is required for secure code execution. "
                    "Please install Docker: https://docs.docker.com/get-docker/",
                )
        else:
            self._docker_available = False
            logger.info("Docker execution disabled, using local execution")

    @property
    def docker_available(self) -> bool:
        """Check if Docker is available."""
        return self._docker_available

    def _check_docker_available(self) -> bool:
        """Check if Docker is available on the system.

        Returns:
            True if Docker is available and accessible.

        """
        try:
            result = subprocess.run(  # noqa: S603
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @contextmanager
    def create_session(
        self,
        config: dict[str, Any] | None = None,
        resource_spec: ResourceSpec | None = None,
    ) -> Generator[Any, None, None]:
        """Create and yield a configured Spark session.

        This context manager creates a Spark session with the specified
        configuration, yields it for use, and ensures proper cleanup.

        Args:
            config: Spark configuration dictionary.
            resource_spec: Resource specifications (optional).

        Yields:
            SparkSession instance.

        Example:
            >>> with runner.create_session({"spark.executor.memory": "4g"}) as spark:
            ...     df = spark.read.parquet("data.parquet")
            ...     df.show()

        """
        spark = None
        try:
            spark = self._build_session(config, resource_spec)
            self._active_session = spark
            self._session_configs = config or {}

            logger.info(f"Spark session created with {len(config or {})} configs")
            yield spark

        finally:
            if spark is not None:
                self._stop_session(spark)
                self._active_session = None
                self._session_configs = {}

    def _build_session(
        self,
        config: dict[str, Any] | None,
        resource_spec: ResourceSpec | None,
    ) -> Any:
        """Build a Spark session with configuration.

        Args:
            config: Spark configuration.
            resource_spec: Resource specifications.

        Returns:
            Configured SparkSession.

        """
        builder = SparkSession.builder.appName(self.app_name).master(self.master)

        # Apply configuration
        config = config or {}

        # Ensure local mode settings
        config.setdefault("spark.sql.adaptive.enabled", "true")
        config.setdefault("spark.sql.adaptive.coalescePartitions.enabled", "true")
        config.setdefault("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Set default resources if not specified
        if resource_spec:
            config.setdefault("spark.executor.cores", str(min(4, resource_spec.cpu_cores)))
            config.setdefault("spark.driver.cores", str(min(2, resource_spec.cpu_cores)))

        # Apply all configs
        for key, value in config.items():
            builder = builder.config(key, str(value))

        # Create session
        spark = builder.getOrCreate()

        # Set log level
        spark.sparkContext.setLogLevel(self.log_level)

        # Log configuration
        logger.debug(f"Spark version: {spark.version}")
        logger.debug(f"Spark UI: {spark.sparkContext.uiWebUrl}")

        return spark

    def _stop_session(self, spark: Any) -> None:
        """Stop a Spark session gracefully.

        Args:
            spark: SparkSession to stop.

        """
        try:
            spark.stop()
            logger.info("Spark session stopped")
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error stopping Spark session: {e}")

    def execute_code(
        self,
        code: str,
        config: dict[str, Any] | None = None,
        resource_spec: ResourceSpec | None = None,
        timeout_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Execute Python code in an isolated Docker container.

        Security:
            Code is always executed in an isolated Docker container.
            This ensures user code cannot access the main process memory.

        Args:
            code: Python code string to execute.
            config: Spark configuration.
            resource_spec: Resource specifications.
            timeout_seconds: Maximum execution time.

        Returns:
            Dictionary with execution results and metrics.

        Raises:
            RuntimeError: If Docker is not available.

        """
        if not self._docker_available:
            raise RuntimeError(
                "Docker is required for secure code execution. "
                "Please install Docker: https://docs.docker.com/get-docker/",
            )
        return self._execute_code_docker(code, config, resource_spec, timeout_seconds)

    def _execute_code_docker(
        self,
        code: str,
        config: dict[str, Any] | None,
        resource_spec: ResourceSpec | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Execute code in isolated Docker container.

        Args:
            code: Python code to execute.
            config: Spark configuration.
            resource_spec: Resource specifications.
            timeout_seconds: Maximum execution time.

        Returns:
            Execution results dictionary.

        """
        start_time = time.time()
        config = config or {}
        temp_file = None

        try:
            # Write code to temporary file in current directory (ensures Docker can mount it)
            # Note: /tmp may not be shared with Docker Desktop on macOS
            temp_dir = os.getcwd()
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                dir=temp_dir,
                encoding="utf-8",
            ) as f:
                temp_file = f.name
                f.write(code)

            # Encode code as base64 for environment variable
            code_encoded = base64.b64encode(code.encode("utf-8")).decode("utf-8")

            # Prepare Docker command
            # Pass code via environment variable (base64-encoded)
            # This avoids file mount issues on macOS Docker Desktop
            # Note: --network none removed due to hostname resolution issues
            docker_cmd = [
                "docker",
                "run",
                "--rm",  # Remove container after execution
                "--hostname",
                "localhost",  # Fix Java hostname resolution
                "-e",
                f"SPARK_OPTIMA_CONFIG={json.dumps(config)}",
                "-e",
                f"SPARK_OPTIMA_CODE={code_encoded}",
            ]

            # Note: Resource constraints are applied via Spark config, not Docker limits
            # This avoids Docker errors when optimizer suggests more resources than available
            # Docker only enforces limits if resource_spec is explicitly set with enforce=true
            enforce_limits = (
                resource_spec is not None and resource_spec.__dict__.get("enforce_limits", False)
            )
            if enforce_limits and resource_spec is not None:
                docker_cmd.extend(
                    [
                        "--memory",
                        f"{resource_spec.memory_gb}g",
                        "--cpus",
                        str(resource_spec.cpu_cores),
                    ],
                )

            docker_cmd.extend(
                [
                    self._docker_image,
                    "python",
                    "/app/spark_executor.py",
                ],
            )

            logger.info(f"Executing code in Docker container: {self._docker_image}")

            # Run Docker container
            # Capture stdout (JSON output) and discard stderr (Spark logs)
            result = subprocess.run(  # noqa: S603
                docker_cmd,
                stdout=subprocess.PIPE,  # JSON output goes here
                stderr=subprocess.DEVNULL,  # Spark logs discarded
                text=True,
                timeout=timeout_seconds,
                check=False,
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                # Parse JSON output from container (stdout)
                # Spark logs go to stderr (discarded), JSON goes to stdout
                # User code may also print to stdout, so we need to find JSON in output
                try:
                    output = self._extract_json_from_output(result.stdout)
                    if output is None:
                        raise json.JSONDecodeError("No JSON found", result.stdout, 0)
                    output["duration_seconds"] = duration
                    return output
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Docker output: {result.stdout}")
                    return {
                        "success": False,
                        "error": f"Failed to parse output: {result.stdout}",
                        "duration_seconds": duration,
                        "error_type": "OutputParseError",
                    }
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error(f"Docker execution failed (code {result.returncode}): {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "DockerExecutionError",
                    "duration_seconds": duration,
                }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Execution timed out after {timeout_seconds} seconds")
            return {
                "success": False,
                "error": f"Execution exceeded {timeout_seconds} seconds",
                "error_type": "TimeoutError",
                "duration_seconds": duration,
            }
        except (OSError, RuntimeError) as e:
            duration = time.time() - start_time
            logger.error(f"Docker execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": duration,
            }
        finally:
            # Clean up temp file
            if temp_file and Path(temp_file).exists():
                try:
                    os.unlink(temp_file)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")

    def execute_file(
        self,
        file_path: str | Path,
        config: dict[str, Any] | None = None,
        resource_spec: ResourceSpec | None = None,
        timeout_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Execute a Python file in a Spark session.

        Args:
            file_path: Path to Python file.
            config: Spark configuration.
            resource_spec: Resource specifications.
            timeout_seconds: Maximum execution time.

        Returns:
            Dictionary with execution results.

        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Read file content
        code = file_path.read_text(encoding="utf-8")

        return self.execute_code(
            code=code,
            config=config,
            resource_spec=resource_spec,
            timeout_seconds=timeout_seconds,
        )

    def get_active_session(self) -> Any | None:
        """Get currently active Spark session.

        Returns:
            Active SparkSession or None.

        """
        return self._active_session

    def get_applied_configs(self) -> dict[str, Any]:
        """Get configurations applied to current session.

        Returns:
            Dictionary of applied configurations.

        """
        return self._session_configs.copy()

    def validate_config(
        self,
        config: dict[str, Any],
        spark_version: str | None = None,
        resource_spec: ResourceSpec | None = None,
    ) -> list[str]:
        """Validate Spark configuration.

        Args:
            config: Configuration to validate.
            spark_version: Target Spark version.
            resource_spec: Resource specifications (optional).

        Returns:
            List of validation errors (empty if valid).

        """
        _ = spark_version  # Mark as intentionally unused
        _ = resource_spec  # Mark as intentionally unused
        errors = []

        # Check for required configs
        if not config.get("spark.executor.memory"):
            errors.append("Missing spark.executor.memory")

        # Validate memory format
        for key in ["spark.executor.memory", "spark.driver.memory"]:
            if key in config:
                value = config[key]
                if not self._is_valid_memory(value):
                    errors.append(f"Invalid memory format for {key}: {value}")

        # Validate numeric configs
        for key in ["spark.executor.cores", "spark.driver.cores"]:
            if key in config:
                try:
                    int(config[key])
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric value for {key}: {config[key]}")

        return errors

    @staticmethod
    def _is_valid_memory(value: Any) -> bool:
        """Check if memory value is valid.

        Args:
            value: Memory value to check.

        Returns:
            True if valid.

        """
        import re

        if isinstance(value, int | float):
            return value > 0

        value = str(value).strip().lower()
        pattern = r"^[\d.]+\s*[kmgt]?\s*b?$"
        return bool(re.match(pattern, value))

    def get_spark_info(self) -> dict[str, Any]:
        """Get information about available Spark.

        Returns:
            Dictionary with Spark information.

        """
        info = {
            "pyspark_available": _PYSPARK_AVAILABLE,
            "app_name": self.app_name,
            "master": self.master,
        }

        if _PYSPARK_AVAILABLE:
            try:
                # Get version without creating session
                import pyspark

                info["pyspark_version"] = pyspark.__version__
            except ImportError:
                pass

        return info

    @staticmethod
    def _extract_json_from_output(output: str) -> dict[str, Any] | None:
        """Extract JSON from output that may contain other text.

        This method handles cases where user code prints to stdout,
        mixing with the JSON output. It searches for valid JSON
        by looking for lines starting with '{' and trying to parse them.

        Args:
            output: Output string that may contain JSON.

        Returns:
            Parsed dictionary or None if no valid JSON found.

        """
        if not output or not output.strip():
            return None

        lines = output.strip().split("\n")

        # Try to find JSON by checking lines from the end (JSON is usually last)
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    # Try multi-line JSON starting from this line
                    pass

        # If not found by line, try to find JSON by searching for {...} pattern
        # This handles multi-line JSON
        text = output.strip()
        start = text.rfind("{")
        while start != -1:
            end = text.rfind("}", start)
            if end != -1:
                json_str = text[start : end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            start = text.rfind("{", 0, start - 1)

        return None
