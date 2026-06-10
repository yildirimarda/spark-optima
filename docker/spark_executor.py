# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0
"""Spark code executor for Docker-based isolation.

This script runs inside a Docker container, creates a SparkSession
with the given configuration, executes user code, and outputs results as JSON to stderr.
(Logging is disabled to avoid mixing with JSON output.)

Environment Variables:
    SPARK_OPTIMA_CONFIG: JSON string of Spark configuration.
    SPARK_OPTIMA_CODE_PATH: Path to the user code file inside container.
    SPARK_OPTIMA_APP_NAME: Spark application name (default: SparkOptima-Isolated).

Usage:
    docker run --rm \
        -v /path/to/code.py:/code/user_code.py:ro \
        -e SPARK_OPTIMA_CONFIG='{"spark.executor.memory":"4g"}' \
        -e SPARK_OPTIMA_CODE_PATH=/code/user_code.py \
        spark-optima:latest \
        python /app/spark_executor.py

"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

# Disable all logging to avoid mixing with JSON output
logging.disable(logging.CRITICAL + 1)


def main() -> None:
    """Execute Spark code in isolated Docker container."""
    start_time = time.time()

    # Read environment variables
    config_json = os.environ.get("SPARK_OPTIMA_CONFIG", "{}")
    code_path = os.environ.get("SPARK_OPTIMA_CODE_PATH", "")
    code_encoded = os.environ.get("SPARK_OPTIMA_CODE", "")
    app_name = os.environ.get("SPARK_OPTIMA_APP_NAME", "SparkOptima-Isolated")

    # Parse configuration
    try:
        config = json.loads(config_json) if config_json else {}
    except json.JSONDecodeError as e:
        _output_error(f"Invalid config JSON: {e}", start_time)
        sys.exit(1)

    # Read user code (from env var or file path)
    if code_encoded:
        # Decode base64-encoded code from environment variable
        try:
            code = base64.b64decode(code_encoded).decode("utf-8")
        except (base64.binascii.Error, UnicodeDecodeError) as e:
            _output_error(f"Failed to decode code: {e}", start_time)
            sys.exit(1)
    elif code_path:
        # Read code from file (legacy approach)
        code_file = Path(code_path)
        if not code_file.exists():
            _output_error(f"Code file not found: {code_path}", start_time)
            sys.exit(1)
        try:
            code = code_file.read_text(encoding="utf-8")
        except OSError as e:
            _output_error(f"Failed to read code file: {e}", start_time)
            sys.exit(1)
    else:
        _output_error("No code provided (missing SPARK_OPTIMA_CODE or SPARK_OPTIMA_CODE_PATH)", start_time)
        sys.exit(1)

    # Execute the code with Spark
    spark = None
    try:
        from pyspark.sql import SparkSession

        # Build Spark session
        builder = SparkSession.builder.appName(app_name)

        # Set master to local[*] to use all available cores
        # This avoids issues with spark.task.cpus > cores per executor
        builder = builder.master("local[*]")

        # Apply configuration
        config.setdefault("spark.sql.adaptive.enabled", "true")
        config.setdefault("spark.sql.adaptive.coalescePartitions.enabled", "true")
        config.setdefault("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Adjust spark.task.cpus if it's > available cores
        # This avoids "cores per executor >= task cpus" error
        try:
            available_cores = os.cpu_count() or 2  # Fallback to 2 if detection fails
            task_cpus = config.get("spark.task.cpus")
            if task_cpus:
                task_cpus_int = int(task_cpus)
                if task_cpus_int > available_cores:
                    # This is a warning, not error - just adjust
                    config["spark.task.cpus"] = str(available_cores)
        except (ValueError, TypeError):
            pass

        for key, value in config.items():
            # Convert value to string, handling booleans for Spark
            str_value = ("true" if value else "false") if isinstance(value, bool) else str(value)
            builder = builder.config(key, str_value)

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        # Execute user code with limited globals
        exec_globals = {
            "spark": spark,
            "sc": spark.sparkContext,
            "sql": spark.sql,
        }

        exec(code, exec_globals)  # noqa: S102 - isolated in Docker container

        duration = time.time() - start_time

        result = {
            "success": True,
            "duration_seconds": duration,
            "spark_version": spark.version,
            "config_applied": config,
        }

        _output_success(result)

    except Exception as e:
        duration = time.time() - start_time
        # Try to get Java error details if it's a Py4JJavaError
        error_detail = str(e)
        if hasattr(e, "java_exception"):
            try:
                java_exc = e.java_exception
                error_detail = f"{error_detail}\nJava error: {java_exc.getClass().getName()}\n{java_exc.getMessage()}"
                # Try to get stack trace
                stack_trace = java_exc.getStackTrace()
                if stack_trace:
                    error_detail += "\nJava stack trace:"
                    for elem in stack_trace[:10]:  # First 10 elements
                        error_detail += f"\n  at {elem}"
            except Exception:
                pass
        _output_error(error_detail, duration, error_type=type(e).__name__)

    finally:
        if spark is not None:
            with suppress(Exception):
                spark.stop()


def _output_success(result: dict[str, Any]) -> None:
    """Output success result as JSON to stdout."""
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


def _output_error(
    error: str,
    duration: float,
    error_type: str = "ExecutionError",
) -> None:
    """Output error result as JSON to stdout."""
    result = {
        "success": False,
        "error": error,
        "error_type": error_type,
        "duration_seconds": duration,
    }
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    sys.exit(1)


if __name__ == "__main__":
    main()
