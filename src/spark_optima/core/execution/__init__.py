# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Execution engine for Spark configuration optimization.

This module provides capabilities for executing Spark jobs locally
and collecting detailed performance metrics for optimization trials.
"""

from spark_optima.core.execution.engine import ExecutionEngine
from spark_optima.core.execution.event_log import EventLogParser, EventLogSummary, StageSummary
from spark_optima.core.execution.history_server import ApplicationInfo, HistoryServerClient, HistoryServerError
from spark_optima.core.execution.metrics_collector import MetricsCollector
from spark_optima.core.execution.monitor import ExecutionMonitor
from spark_optima.core.execution.spark_runner import SparkRunner

__all__ = [
    "ExecutionEngine",
    "SparkRunner",
    "MetricsCollector",
    "ExecutionMonitor",
    "EventLogParser",
    "EventLogSummary",
    "StageSummary",
    "ApplicationInfo",
    "HistoryServerClient",
    "HistoryServerError",
]
