# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spark History Server REST API client for post-run analysis.

This module provides HistoryServerClient, which queries a Spark History
Server (``/api/v1`` REST endpoints) and condenses a finished application's
metrics into the same EventLogSummary the event-log file parser produces
(see event_log.py), so downstream consumers — including
``EventLogSummary.to_tuning_hints()`` — work unchanged.

Endpoint-to-summary mapping:

- ``/applications/{app_id}``: app name and (latest attempt) duration.
- ``/applications/{app_id}/stages``: per-stage shuffle read/write bytes,
  memory + disk spill, input bytes, task counts, GC and run time.
- ``/applications/{app_id}/stages/{id}/{attempt}/taskSummary``: task
  duration quantiles for per-stage skew and per-task peak execution memory.
- ``/applications/{app_id}/executors``: active executor count.
- ``/applications/{app_id}/environment``: Spark properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any

from spark_optima.core.execution.event_log import BYTES_PER_GB, EventLogSummary, StageSummary

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only when httpx is absent
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

API_SUFFIX = "/api/v1"
COMPLETED_STAGE_STATUS = "COMPLETE"
DRIVER_EXECUTOR_ID = "driver"

# History Server timestamps look like "2026-06-01T10:00:00.000GMT".
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
# Quantiles requested from taskSummary: min, median, max.
_SKEW_QUANTILES = "0.0,0.5,1.0"
_MEDIAN_QUANTILE = 0.5
_MAX_QUANTILE = 1.0


class HistoryServerError(Exception):
    """Raised when the History Server cannot be reached or returns an error."""


@dataclass(frozen=True)
class ApplicationInfo:
    """Light-weight application record from the History Server.

    For multi-attempt applications (e.g. YARN cluster mode retries) the
    fields describe the latest attempt.

    Attributes:
        app_id: Application identifier.
        name: Application name.
        attempt_id: Latest attempt identifier, or None for single-attempt apps.
        attempt_count: Total number of recorded attempts.
        duration_seconds: Wall-clock duration of the latest attempt.
        completed: Whether the latest attempt has finished.
        spark_user: User that submitted the application.

    """

    app_id: str
    name: str = ""
    attempt_id: str | None = None
    attempt_count: int = 0
    duration_seconds: float = 0.0
    completed: bool = False
    spark_user: str = ""


def _as_int(value: Any) -> int:
    """Coerce a JSON value to int, returning 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a History Server timestamp string, returning None on failure."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().removesuffix("GMT").removesuffix("Z")
    try:
        return datetime.strptime(text, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _stage_duration_seconds(stage: dict[str, Any]) -> float:
    """Compute stage wall-clock duration from submission/completion times."""
    submitted = _parse_timestamp(stage.get("submissionTime"))
    completed = _parse_timestamp(stage.get("completionTime"))
    if submitted is None or completed is None or completed <= submitted:
        return 0.0
    return (completed - submitted).total_seconds()


def _attempt_sort_key(attempt: dict[str, Any]) -> tuple[int, int]:
    """Sort key that orders application attempts from oldest to latest."""
    try:
        attempt_number = int(str(attempt.get("attemptId")))
    except ValueError:
        attempt_number = -1
    return (attempt_number, _as_int(attempt.get("startTimeEpoch")))


def _latest_attempt(record: dict[str, Any]) -> dict[str, Any]:
    """Return the latest attempt of an application record (empty if none)."""
    attempts = [attempt for attempt in record.get("attempts") or [] if isinstance(attempt, dict)]
    if not attempts:
        return {}
    return max(attempts, key=_attempt_sort_key)


def _completed_stage_records(payload: Any) -> list[dict[str, Any]]:
    """Filter a /stages payload down to completed stages, latest attempt each.

    Stages are deduplicated by stage id (keeping the highest stage attempt)
    and returned ordered by stage id, mirroring the file parser's output.
    """
    latest: dict[int, dict[str, Any]] = {}
    if not isinstance(payload, list):
        return []
    for record in payload:
        if not isinstance(record, dict) or str(record.get("status", "")).upper() != COMPLETED_STAGE_STATUS:
            continue
        stage_id = _as_int(record.get("stageId"))
        current = latest.get(stage_id)
        if current is None or _as_int(record.get("attemptId")) >= _as_int(current.get("attemptId")):
            latest[stage_id] = record
    return [latest[stage_id] for stage_id in sorted(latest)]


def _active_executor_count(payload: Any) -> int:
    """Count active non-driver executors from an /executors payload."""
    if not isinstance(payload, list):
        return 0
    return sum(
        1
        for entry in payload
        if isinstance(entry, dict) and str(entry.get("id")) != DRIVER_EXECUTOR_ID and entry.get("isActive", True)
    )


def _spark_properties(payload: Any) -> dict[str, str]:
    """Extract Spark properties from an /environment payload."""
    properties: dict[str, str] = {}
    if not isinstance(payload, dict):
        return properties
    for pair in payload.get("sparkProperties") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            properties[str(pair[0])] = str(pair[1])
    return properties


def _skew_from_task_summary(payload: dict[str, Any]) -> float:
    """Compute max/median executor run time ratio from taskSummary quantiles."""
    quantiles = payload.get("quantiles") or []
    run_times = payload.get("executorRunTime") or []
    try:
        median = float(run_times[quantiles.index(_MEDIAN_QUANTILE)])
        maximum = float(run_times[quantiles.index(_MAX_QUANTILE)])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 1.0
    if median <= 0:
        return 1.0
    return maximum / median


def _peak_memory_from_task_summary(payload: dict[str, Any]) -> int:
    """Extract the max per-task peak execution memory (bytes) from taskSummary."""
    peaks = payload.get("peakExecutionMemory")
    if not isinstance(peaks, list):
        return 0
    best = 0
    for value in peaks:
        try:
            best = max(best, int(float(value)))
        except (TypeError, ValueError):
            continue
    return best


class HistoryServerClient:
    """Client for the Spark History Server REST API.

    Fetches application, stage, executor, and environment data over HTTP and
    assembles the same EventLogSummary the event-log file parser produces.

    Example:
        >>> with HistoryServerClient("http://history-server:18080") as client:
        ...     summary = client.fetch_summary("app-20260601100000-0001")
        ...     hints = summary.to_tuning_hints()

    """

    def __init__(self, base_url: str, timeout: float = 10.0, client: httpx.Client | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: History Server base URL, with or without the ``/api/v1``
                suffix and with or without a trailing slash
                (e.g. ``http://host:18080`` or ``http://host:18080/api/v1/``).
            timeout: Request timeout in seconds for the internally created
                HTTP client (ignored when ``client`` is provided).
            client: Optional pre-configured ``httpx.Client`` (e.g. with a mock
                transport in tests). The caller retains ownership; ``close()``
                only closes internally created clients.

        Raises:
            HistoryServerError: If ``base_url`` is empty or httpx is not installed.

        """
        if httpx is None:  # pragma: no cover - exercised only when httpx is absent
            raise HistoryServerError("httpx is required for HistoryServerClient; install it with 'uv add httpx'")
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise HistoryServerError("History Server base URL must not be empty")
        if not normalized.endswith(API_SUFFIX):
            normalized = f"{normalized}{API_SUFFIX}"
        self._base_url = normalized
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            try:
                self._client = httpx.Client(timeout=timeout)
            except (OSError, ValueError) as exc:
                # SSL context setup can fail at construction time (e.g. the CA
                # bundle is unreadable in sandboxed/locked-down environments).
                raise HistoryServerError(f"could not create HTTP client: {exc}") from exc

    @property
    def base_url(self) -> str:
        """Normalized base URL including the ``/api/v1`` suffix."""
        return self._base_url

    def close(self) -> None:
        """Close the underlying HTTP client if it was created internally."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HistoryServerClient:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, closing internally created clients."""
        self.close()

    def list_applications(self, limit: int = 20) -> list[ApplicationInfo]:
        """List applications known to the History Server.

        Args:
            limit: Maximum number of applications to return (passed to the
                server as the ``limit`` query parameter).

        Returns:
            Light-weight application records; for multi-attempt applications
            the fields reflect the latest attempt.

        Raises:
            HistoryServerError: If the server is unreachable or returns an error.

        """
        payload = self._get_json("/applications", params={"limit": limit})
        applications: list[ApplicationInfo] = []
        if not isinstance(payload, list):
            return applications
        for record in payload:
            if not isinstance(record, dict):
                continue
            attempt = _latest_attempt(record)
            attempt_id = attempt.get("attemptId")
            applications.append(
                ApplicationInfo(
                    app_id=str(record.get("id", "")),
                    name=str(record.get("name", "")),
                    attempt_id=str(attempt_id) if attempt_id is not None else None,
                    attempt_count=len(record.get("attempts") or []),
                    duration_seconds=_as_int(attempt.get("duration")) / 1000.0,
                    completed=bool(attempt.get("completed", False)),
                    spark_user=str(attempt.get("sparkUser", "")),
                ),
            )
        return applications

    def fetch_summary(self, app_id: str) -> EventLogSummary:
        """Fetch and assemble an EventLogSummary for an application.

        Produces the same dataclass as ``EventLogParser.parse()`` so that
        ``to_dict()`` and ``to_tuning_hints()`` work unchanged. Multi-attempt
        applications are resolved to their latest attempt.

        Differences from the file parser (REST API limitations):

        - ``skipped_lines`` is always 0 (a file-parsing concept).
        - ``total_tasks``/``failed_tasks`` come from the declared per-stage
          ``numTasks``/``numFailedTasks`` of completed stages rather than
          observed task-end events.
        - ``executor_count_max`` is approximated by the count of active
          non-driver executors from ``/executors``; the REST API exposes no
          concurrent-executor high-water mark.
        - ``peak_execution_memory_gb`` comes from taskSummary quantiles and
          stays at the 0.0 default when taskSummary is unavailable for all
          stages.
        - GC totals use per-stage ``jvmGcTime``/``executorRunTime`` sums
          rather than executor ``totalGCTime``: ``/executors`` omits
          executors that have been removed and includes GC outside task
          execution, while stage sums match the file parser's per-task
          aggregation and pair consistently for ``gc_time_fraction``.

        Args:
            app_id: Application identifier as shown by the History Server.

        Returns:
            EventLogSummary aggregated from the REST API.

        Raises:
            HistoryServerError: If the application does not exist, the server
                is unreachable, or any required endpoint returns an error.

        """
        record = self._get_json(f"/applications/{app_id}", allow_404=True)
        if record is None:
            raise HistoryServerError(f"application not found on History Server ({self._base_url}): {app_id}")
        if not isinstance(record, dict):
            raise HistoryServerError(f"unexpected application payload from History Server for {app_id}")

        attempt = _latest_attempt(record)
        attempt_id = attempt.get("attemptId")
        attempt_segment = str(attempt_id) if attempt_id is not None else None

        stage_records = _completed_stage_records(self._get_json(self._app_path(app_id, attempt_segment, "stages")))

        stages: list[StageSummary] = []
        total_shuffle_read = 0
        total_shuffle_write = 0
        total_spill = 0
        total_input = 0
        total_tasks = 0
        failed_tasks = 0
        total_gc_ms = 0
        total_run_ms = 0
        peak_memory_bytes = 0
        max_skew = 1.0

        for stage in stage_records:
            skew, stage_peak_bytes = self._fetch_stage_task_stats(app_id, attempt_segment, stage)
            max_skew = max(max_skew, skew)
            peak_memory_bytes = max(peak_memory_bytes, stage_peak_bytes)

            shuffle_read = _as_int(stage.get("shuffleReadBytes"))
            shuffle_write = _as_int(stage.get("shuffleWriteBytes"))
            spill = _as_int(stage.get("memoryBytesSpilled")) + _as_int(stage.get("diskBytesSpilled"))
            total_shuffle_read += shuffle_read
            total_shuffle_write += shuffle_write
            total_spill += spill
            total_input += _as_int(stage.get("inputBytes"))
            total_tasks += _as_int(stage.get("numTasks"))
            failed_tasks += _as_int(stage.get("numFailedTasks"))
            total_gc_ms += _as_int(stage.get("jvmGcTime"))
            total_run_ms += _as_int(stage.get("executorRunTime"))

            stages.append(
                StageSummary(
                    stage_id=_as_int(stage.get("stageId")),
                    name=str(stage.get("name", "")),
                    duration_seconds=_stage_duration_seconds(stage),
                    num_tasks=_as_int(stage.get("numTasks")),
                    shuffle_read_gb=shuffle_read / BYTES_PER_GB,
                    shuffle_write_gb=shuffle_write / BYTES_PER_GB,
                    spill_gb=spill / BYTES_PER_GB,
                    skew_ratio=skew,
                ),
            )

        executors_payload = self._get_json(self._app_path(app_id, attempt_segment, "executors"))
        environment_payload = self._get_json(self._app_path(app_id, attempt_segment, "environment"))

        return EventLogSummary(
            app_name=str(record.get("name", "")),
            app_duration_seconds=_as_int(attempt.get("duration")) / 1000.0,
            total_tasks=total_tasks,
            failed_tasks=failed_tasks,
            total_gc_time_seconds=total_gc_ms / 1000.0,
            gc_time_fraction=(total_gc_ms / total_run_ms) if total_run_ms > 0 else 0.0,
            total_shuffle_read_gb=total_shuffle_read / BYTES_PER_GB,
            total_shuffle_write_gb=total_shuffle_write / BYTES_PER_GB,
            total_spill_gb=total_spill / BYTES_PER_GB,
            peak_execution_memory_gb=peak_memory_bytes / BYTES_PER_GB,
            executor_count_max=_active_executor_count(executors_payload),
            input_data_gb=total_input / BYTES_PER_GB,
            max_skew_ratio=max_skew,
            stages=stages,
            spark_conf=_spark_properties(environment_payload),
        )

    def _fetch_stage_task_stats(
        self,
        app_id: str,
        attempt_segment: str | None,
        stage: dict[str, Any],
    ) -> tuple[float, int]:
        """Fetch skew ratio and peak task memory for a stage via taskSummary.

        The taskSummary endpoint returns 404 for some stages (e.g. when task
        data has been evicted); in that case the skew falls back to 1.0 and
        peak memory to 0, and processing continues.

        Returns:
            Tuple of (skew_ratio, peak_execution_memory_bytes).

        """
        stage_id = _as_int(stage.get("stageId"))
        stage_attempt = _as_int(stage.get("attemptId"))
        path = self._app_path(app_id, attempt_segment, f"stages/{stage_id}/{stage_attempt}/taskSummary")
        payload = self._get_json(path, params={"quantiles": _SKEW_QUANTILES}, allow_404=True)
        if not isinstance(payload, dict):
            logger.debug("taskSummary unavailable for stage %s; assuming skew 1.0", stage_id)
            return 1.0, 0
        return _skew_from_task_summary(payload), _peak_memory_from_task_summary(payload)

    def _app_path(self, app_id: str, attempt_segment: str | None, suffix: str) -> str:
        """Build an application-scoped API path, honoring multi-attempt apps."""
        path = f"/applications/{app_id}"
        if attempt_segment:
            path = f"{path}/{attempt_segment}"
        return f"{path}/{suffix}" if suffix else path

    def _get_json(self, path: str, params: dict[str, Any] | None = None, *, allow_404: bool = False) -> Any:
        """Issue a GET request and decode the JSON response.

        Args:
            path: API path relative to the normalized base URL.
            params: Optional query parameters.
            allow_404: When True, a 404 response returns None instead of raising.

        Returns:
            Decoded JSON payload, or None for tolerated 404 responses.

        Raises:
            HistoryServerError: On connection failures, HTTP error statuses,
                or invalid JSON payloads.

        """
        url = f"{self._base_url}{path}"
        try:
            response = self._client.get(url, params=params)
        except (httpx.RequestError, OSError) as exc:
            # OSError covers raw socket-level failures (e.g. sandboxed or
            # firewalled environments) that can escape httpx's own mapping.
            raise HistoryServerError(f"could not reach History Server at {self._base_url}: {exc}") from exc
        if response.status_code == HTTPStatus.NOT_FOUND and allow_404:
            return None
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise HistoryServerError(f"History Server returned HTTP {response.status_code} for {url}")
        try:
            return response.json()
        except ValueError as exc:
            raise HistoryServerError(f"History Server returned invalid JSON for {url}: {exc}") from exc
