# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Spark History Server REST API client."""

from __future__ import annotations

from typing import Any

import pytest

httpx = pytest.importorskip("httpx")

from spark_optima.core.execution.event_log import EventLogSummary  # noqa: E402
from spark_optima.core.execution.history_server import (  # noqa: E402
    HistoryServerClient,
    HistoryServerError,
)

GB = 1024**3


def _attempt(
    attempt_id: str | None = None,
    duration_ms: int = 600_000,
    start_epoch: int = 1_780_000_000_000,
    completed: bool = True,
    user: str = "spark",
) -> dict[str, Any]:
    """Build an application attempt record as returned by the REST API."""
    attempt: dict[str, Any] = {
        "startTime": "2026-06-01T10:00:00.000GMT",
        "endTime": "2026-06-01T10:10:00.000GMT",
        "lastUpdated": "2026-06-01T10:10:05.000GMT",
        "duration": duration_ms,
        "sparkUser": user,
        "completed": completed,
        "appSparkVersion": "3.5.1",
        "startTimeEpoch": start_epoch,
        "endTimeEpoch": start_epoch + duration_ms,
        "lastUpdatedEpoch": start_epoch + duration_ms + 5_000,
    }
    if attempt_id is not None:
        attempt["attemptId"] = attempt_id
    return attempt


def _stage(
    stage_id: int,
    *,
    attempt_id: int = 0,
    status: str = "COMPLETE",
    name: str = "stage",
    num_tasks: int = 0,
    num_failed: int = 0,
    run_time_ms: int = 0,
    gc_time_ms: int = 0,
    shuffle_read: int = 0,
    shuffle_write: int = 0,
    memory_spilled: int = 0,
    disk_spilled: int = 0,
    input_bytes: int = 0,
    submission: str = "2026-06-01T10:00:00.000GMT",
    completion: str = "2026-06-01T10:01:00.000GMT",
) -> dict[str, Any]:
    """Build a StageData record as returned by /stages."""
    return {
        "status": status,
        "stageId": stage_id,
        "attemptId": attempt_id,
        "numTasks": num_tasks,
        "numActiveTasks": 0,
        "numCompleteTasks": num_tasks - num_failed,
        "numFailedTasks": num_failed,
        "numKilledTasks": 0,
        "executorRunTime": run_time_ms,
        "executorCpuTime": run_time_ms * 1_000_000,
        "jvmGcTime": gc_time_ms,
        "submissionTime": submission,
        "firstTaskLaunchedTime": submission,
        "completionTime": completion,
        "inputBytes": input_bytes,
        "inputRecords": 0,
        "outputBytes": 0,
        "outputRecords": 0,
        "shuffleReadBytes": shuffle_read,
        "shuffleReadRecords": 0,
        "shuffleWriteBytes": shuffle_write,
        "shuffleWriteRecords": 0,
        "memoryBytesSpilled": memory_spilled,
        "diskBytesSpilled": disk_spilled,
        "name": name,
        "details": "",
        "schedulingPool": "default",
        "accumulatorUpdates": [],
    }


APP1 = {"id": "app-1", "name": "ETL Pipeline", "attempts": [_attempt()]}

APP1_STAGES = [
    _stage(
        1,
        attempt_id=1,
        name="reduceByKey at etl.py:42",
        num_tasks=100,
        num_failed=2,
        run_time_ms=90_000,
        gc_time_ms=13_000,
        shuffle_read=6 * GB,
        shuffle_write=4 * GB,
        memory_spilled=1 * GB,
        disk_spilled=1 * GB,
        input_bytes=20 * GB,
        submission="2026-06-01T10:00:00.000GMT",
        completion="2026-06-01T10:02:00.000GMT",
    ),
    # Stale earlier attempt of stage 1: must be superseded by attempt 1 above,
    # not summed into the totals (the garbage values would break assertions).
    _stage(1, attempt_id=0, name="reduceByKey at etl.py:42", num_tasks=999, run_time_ms=1, shuffle_read=999 * GB),
    _stage(
        2,
        name="count at etl.py:50",
        num_tasks=50,
        run_time_ms=10_000,
        gc_time_ms=2_000,
        shuffle_read=1 * GB,
        shuffle_write=1 * GB,
        input_bytes=4 * GB,
        submission="2026-06-01T10:02:05.000GMT",
        completion="2026-06-01T10:02:35.000GMT",
    ),
    _stage(3, status="ACTIVE", name="still running", num_tasks=10),
    _stage(4, status="FAILED", name="broken stage", num_tasks=10, num_failed=10),
]

# min / median / max executorRunTime -> skew = 800 / 200 = 4.0.
APP1_STAGE1_TASK_SUMMARY = {
    "quantiles": [0.0, 0.5, 1.0],
    "executorRunTime": [100.0, 200.0, 800.0],
    "executorDeserializeTime": [1.0, 2.0, 3.0],
    "jvmGcTime": [0.0, 10.0, 50.0],
    "peakExecutionMemory": [0.0, float(1 * GB), float(2 * GB)],
    "memoryBytesSpilled": [0.0, 0.0, float(GB)],
}

APP1_EXECUTORS = [
    {"id": "driver", "isActive": True, "totalGCTime": 500, "totalDuration": 0},
    {"id": "1", "isActive": True, "totalGCTime": 9_000, "totalDuration": 60_000},
    {"id": "2", "isActive": True, "totalGCTime": 6_000, "totalDuration": 40_000},
    {"id": "3", "isActive": False, "totalGCTime": 100, "totalDuration": 1_000},
]

APP1_ENVIRONMENT = {
    "runtime": {"javaVersion": "17.0.10", "scalaVersion": "version 2.12.18"},
    "sparkProperties": [
        ["spark.executor.memory", "4g"],
        ["spark.sql.shuffle.partitions", "200"],
    ],
    "hadoopProperties": [["fs.defaultFS", "file:///"]],
    "systemProperties": [],
    "classpathEntries": [],
}

APP_MULTI = {
    "id": "app-multi",
    "name": "Multi Attempt App",
    "attempts": [
        _attempt(attempt_id="1", duration_ms=100_000, start_epoch=1_780_000_000_000, completed=False),
        _attempt(attempt_id="2", duration_ms=300_000, start_epoch=1_780_000_200_000, user="etl"),
    ],
}

MULTI_STAGES = [
    _stage(
        0,
        name="load at multi.py:10",
        num_tasks=10,
        run_time_ms=20_000,
        gc_time_ms=1_000,
        input_bytes=1 * GB,
        submission="2026-06-01T10:00:00.000GMT",
        completion="2026-06-01T10:01:00.000GMT",
    ),
]

MULTI_TASK_SUMMARY = {
    "quantiles": [0.0, 0.5, 1.0],
    "executorRunTime": [100.0, 500.0, 1000.0],
    "peakExecutionMemory": [0.0, 0.0, float(GB // 2)],
}

APP_EMPTY = {"id": "app-empty", "name": "Empty App", "attempts": [_attempt(duration_ms=5_000)]}


def _happy_routes() -> dict[str, Any]:
    """Canned JSON responses keyed by request path for the happy paths."""
    return {
        "/api/v1/applications": [APP1, APP_MULTI],
        "/api/v1/applications/app-1": APP1,
        "/api/v1/applications/app-1/stages": APP1_STAGES,
        "/api/v1/applications/app-1/stages/1/1/taskSummary": APP1_STAGE1_TASK_SUMMARY,
        # Stage 2 taskSummary route intentionally absent -> 404 -> skew fallback.
        "/api/v1/applications/app-1/executors": APP1_EXECUTORS,
        "/api/v1/applications/app-1/environment": APP1_ENVIRONMENT,
        "/api/v1/applications/app-multi": APP_MULTI,
        # Multi-attempt apps are addressed with the attempt id in the path.
        "/api/v1/applications/app-multi/2/stages": MULTI_STAGES,
        "/api/v1/applications/app-multi/2/stages/0/0/taskSummary": MULTI_TASK_SUMMARY,
        "/api/v1/applications/app-multi/2/executors": [
            {"id": "driver", "isActive": True},
            {"id": "1", "isActive": True},
        ],
        "/api/v1/applications/app-multi/2/environment": {"sparkProperties": [["spark.app.name", "multi"]]},
        "/api/v1/applications/app-empty": APP_EMPTY,
        "/api/v1/applications/app-empty/stages": [],
        "/api/v1/applications/app-empty/executors": [],
        "/api/v1/applications/app-empty/environment": {"sparkProperties": []},
    }


def _make_client(
    routes: dict[str, Any],
    requests_log: list[Any] | None = None,
    base_url: str = "http://history:18080",
) -> HistoryServerClient:
    """Build a HistoryServerClient backed by an httpx.MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests_log is not None:
            requests_log.append(request)
        if request.url.path not in routes:
            return httpx.Response(404, json={"message": "no such resource"})
        return httpx.Response(200, json=routes[request.url.path])

    return HistoryServerClient(base_url, client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestFetchSummaryHappyPath:
    """fetch_summary against realistic canned JSON for every endpoint."""

    @pytest.fixture
    def summary(self) -> EventLogSummary:
        """Fetch the summary for the canonical single-attempt app."""
        with _make_client(_happy_routes()) as client:
            return client.fetch_summary("app-1")

    def test_application_fields(self, summary: EventLogSummary) -> None:
        """App name and duration come from the application record."""
        assert summary.app_name == "ETL Pipeline"
        assert summary.app_duration_seconds == pytest.approx(600.0)

    def test_totals(self, summary: EventLogSummary) -> None:
        """Totals aggregate completed stages only, deduplicated by attempt."""
        assert summary.total_tasks == 150
        assert summary.failed_tasks == 2
        assert summary.total_shuffle_read_gb == pytest.approx(7.0)
        assert summary.total_shuffle_write_gb == pytest.approx(5.0)
        assert summary.total_spill_gb == pytest.approx(2.0)
        assert summary.input_data_gb == pytest.approx(24.0)
        assert summary.total_gc_time_seconds == pytest.approx(15.0)
        assert summary.gc_time_fraction == pytest.approx(0.15)
        assert summary.peak_execution_memory_gb == pytest.approx(2.0)
        assert summary.executor_count_max == 2  # active executors, driver excluded
        assert summary.max_skew_ratio == pytest.approx(4.0)
        assert summary.skipped_lines == 0

    def test_stage_summaries(self, summary: EventLogSummary) -> None:
        """Completed stages map onto StageSummary, ordered by stage id."""
        assert [stage.stage_id for stage in summary.stages] == [1, 2]

        stage1, stage2 = summary.stages
        assert stage1.name == "reduceByKey at etl.py:42"
        assert stage1.duration_seconds == pytest.approx(120.0)
        assert stage1.num_tasks == 100
        assert stage1.shuffle_read_gb == pytest.approx(6.0)
        assert stage1.shuffle_write_gb == pytest.approx(4.0)
        assert stage1.spill_gb == pytest.approx(2.0)
        assert stage1.skew_ratio == pytest.approx(4.0)

        assert stage2.duration_seconds == pytest.approx(30.0)
        assert stage2.num_tasks == 50
        assert stage2.shuffle_read_gb == pytest.approx(1.0)
        assert stage2.spill_gb == pytest.approx(0.0)

    def test_spark_conf(self, summary: EventLogSummary) -> None:
        """Spark properties come from /environment sparkProperties pairs."""
        assert summary.spark_conf == {
            "spark.executor.memory": "4g",
            "spark.sql.shuffle.partitions": "200",
        }

    def test_to_dict_round_trip(self, summary: EventLogSummary) -> None:
        """The summary serializes like a file-parsed one."""
        payload = summary.to_dict()
        assert payload["app_name"] == "ETL Pipeline"
        assert payload["stages"][0]["skew_ratio"] == pytest.approx(4.0)


class TestTaskSummaryHandling:
    """Skew computation from the taskSummary endpoint."""

    def test_quantiles_requested(self) -> None:
        """taskSummary is queried with min/median/max quantiles."""
        seen: list[Any] = []
        with _make_client(_happy_routes(), requests_log=seen) as client:
            client.fetch_summary("app-1")
        task_summary_requests = [request for request in seen if request.url.path.endswith("/taskSummary")]
        assert task_summary_requests, "expected taskSummary requests"
        assert all(request.url.params["quantiles"] == "0.0,0.5,1.0" for request in task_summary_requests)

    def test_404_falls_back_to_skew_one(self) -> None:
        """A 404 from taskSummary yields skew 1.0 without aborting the fetch."""
        with _make_client(_happy_routes()) as client:
            summary = client.fetch_summary("app-1")
        stage2 = next(stage for stage in summary.stages if stage.stage_id == 2)
        assert stage2.skew_ratio == pytest.approx(1.0)

    def test_zero_median_guards_division(self) -> None:
        """A zero median run time yields skew 1.0 instead of dividing by zero."""
        routes = _happy_routes()
        routes["/api/v1/applications/app-1/stages/1/1/taskSummary"] = {
            "quantiles": [0.0, 0.5, 1.0],
            "executorRunTime": [0.0, 0.0, 500.0],
        }
        with _make_client(routes) as client:
            summary = client.fetch_summary("app-1")
        stage1 = next(stage for stage in summary.stages if stage.stage_id == 1)
        assert stage1.skew_ratio == pytest.approx(1.0)


class TestTuningHints:
    """The REST-built summary must drive to_tuning_hints() unchanged."""

    def test_hint_keys_match_file_parser_summary(self) -> None:
        """Hint keys equal those of a file-parser EventLogSummary."""
        reference_keys = set(EventLogSummary(input_data_gb=1.0).to_tuning_hints())
        with _make_client(_happy_routes()) as client:
            hints = client.fetch_summary("app-1").to_tuning_hints()
        assert set(hints) == reference_keys

    def test_hint_values(self) -> None:
        """Hints reflect the aggregated REST metrics."""
        with _make_client(_happy_routes()) as client:
            hints = client.fetch_summary("app-1").to_tuning_hints()
        assert hints["skew_factor"] == pytest.approx(4.0)
        assert hints["large_shuffles"] is True  # 12 GB > 10 GB threshold
        assert hints["gc_pressure"] is True  # 0.15 > 0.1 threshold
        assert hints["gc_time_fraction"] == pytest.approx(0.15)
        assert hints["spill_detected"] is True
        assert hints["spill_gb"] == pytest.approx(2.0)
        assert hints["shuffle_total_gb"] == pytest.approx(12.0)
        assert hints["memory_intensive"] is True
        assert hints["data_size_gb"] == pytest.approx(24.0)


class TestMultiAttemptApplications:
    """Multi-attempt apps resolve to the latest attempt."""

    def test_fetch_summary_uses_latest_attempt(self) -> None:
        """Stage/executor/environment paths include the latest attempt id."""
        with _make_client(_happy_routes()) as client:
            summary = client.fetch_summary("app-multi")
        assert summary.app_name == "Multi Attempt App"
        assert summary.app_duration_seconds == pytest.approx(300.0)  # attempt 2, not 1
        assert [stage.stage_id for stage in summary.stages] == [0]
        assert summary.stages[0].skew_ratio == pytest.approx(2.0)
        assert summary.input_data_gb == pytest.approx(1.0)
        assert summary.peak_execution_memory_gb == pytest.approx(0.5)
        assert summary.executor_count_max == 1
        assert summary.spark_conf == {"spark.app.name": "multi"}


class TestListApplications:
    """list_applications light listing."""

    def test_list_applications(self) -> None:
        """Applications are listed with latest-attempt info and limit param."""
        seen: list[Any] = []
        with _make_client(_happy_routes(), requests_log=seen) as client:
            applications = client.list_applications(limit=5)

        assert seen[0].url.params["limit"] == "5"
        assert [app.app_id for app in applications] == ["app-1", "app-multi"]

        single, multi = applications
        assert single.name == "ETL Pipeline"
        assert single.attempt_id is None
        assert single.attempt_count == 1
        assert single.duration_seconds == pytest.approx(600.0)
        assert single.completed is True
        assert single.spark_user == "spark"

        assert multi.attempt_id == "2"  # latest attempt wins
        assert multi.attempt_count == 2
        assert multi.duration_seconds == pytest.approx(300.0)
        assert multi.spark_user == "etl"


class TestEmptyStageList:
    """Applications with no completed stages still produce a valid summary."""

    def test_empty_stage_list(self) -> None:
        """An empty /stages payload yields a zeroed summary with working hints."""
        with _make_client(_happy_routes()) as client:
            summary = client.fetch_summary("app-empty")
        assert summary.app_name == "Empty App"
        assert summary.stages == []
        assert summary.total_tasks == 0
        assert summary.total_shuffle_read_gb == pytest.approx(0.0)
        assert summary.gc_time_fraction == pytest.approx(0.0)
        assert summary.max_skew_ratio == pytest.approx(1.0)
        hints = summary.to_tuning_hints()
        assert "data_size_gb" not in hints  # no input bytes observed
        assert hints["large_shuffles"] is False
        assert hints["memory_intensive"] is False


class TestErrorHandling:
    """Error model: every failure surfaces as HistoryServerError."""

    def test_connection_error(self) -> None:
        """Transport-level failures raise HistoryServerError."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = HistoryServerClient(
            "http://down:18080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(HistoryServerError, match="could not reach History Server"):
            client.list_applications()

    def test_application_not_found(self) -> None:
        """A 404 on the application record raises a clear not-found error."""
        with (
            _make_client(_happy_routes()) as client,
            pytest.raises(HistoryServerError, match="application not found.*missing-app"),
        ):
            client.fetch_summary("missing-app")

    def test_server_error(self) -> None:
        """Non-404 HTTP errors raise HistoryServerError with the status code."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/applications/app-err":
                return httpx.Response(200, json={"id": "app-err", "name": "Err", "attempts": [_attempt()]})
            return httpx.Response(500, json={"message": "boom"})

        client = HistoryServerClient(
            "http://history:18080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(HistoryServerError, match="HTTP 500"):
            client.fetch_summary("app-err")

    def test_invalid_json(self) -> None:
        """A non-JSON 200 response raises HistoryServerError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        client = HistoryServerClient(
            "http://history:18080",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(HistoryServerError, match="invalid JSON"):
            client.list_applications()


class TestBaseUrlNormalization:
    """Base URL handling for all accepted spellings."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://hs:18080",
            "http://hs:18080/",
            "http://hs:18080/api/v1",
            "http://hs:18080/api/v1/",
            "  http://hs:18080  ",
        ],
    )
    def test_normalization_variants(self, base_url: str) -> None:
        """All variants normalize to a single /api/v1 base URL."""
        seen: list[Any] = []
        client = _make_client({"/api/v1/applications": []}, requests_log=seen, base_url=base_url)
        assert client.base_url == "http://hs:18080/api/v1"
        assert client.list_applications() == []
        assert seen[0].url.path == "/api/v1/applications"

    def test_empty_base_url_rejected(self) -> None:
        """An empty base URL raises HistoryServerError."""
        with pytest.raises(HistoryServerError, match="must not be empty"):
            HistoryServerClient("   ")


class TestClientLifecycle:
    """Ownership semantics for injected HTTP clients."""

    def test_close_keeps_injected_client_open(self) -> None:
        """close() leaves caller-provided httpx clients usable."""
        http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])))
        with HistoryServerClient("http://hs:18080", client=http_client) as client:
            assert client.list_applications() == []
        assert not http_client.is_closed
        http_client.close()
