# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the dynamic allocation advisor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from spark_optima.core.execution.dynamic_allocation_advisor import (
    DynamicAllocationAdvisor,
    DynamicAllocationRecommendation,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_event(event_type: str, timestamp: int = 1_000_000, extra: dict | None = None) -> dict:
    event: dict = {"Event": event_type, "Timestamp": timestamp}
    if extra:
        event.update(extra)
    return event


def build_log(tmp_path: Path, events: list[dict]) -> Path:
    log_path = tmp_path / "dyn_alloc_eventlog"
    with open(log_path, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt) + "\n")
    return log_path


class TestDynamicAllocationAdvisor:
    """Tests for DynamicAllocationAdvisor."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DynamicAllocationAdvisor(tmp_path / "missing")

    def test_reconstruct_timeline_from_events(self, tmp_path: Path) -> None:
        events = [
            _make_event("SparkListenerApplicationStart", timestamp=1_000_000, extra={"App Name": "test"}),
            _make_event("SparkListenerExecutorAdded", timestamp=1_001_000, extra={"Executor ID": "1"}),
            _make_event("SparkListenerExecutorAdded", timestamp=1_002_000, extra={"Executor ID": "2"}),
            _make_event("SparkListenerExecutorRemoved", timestamp=1_005_000, extra={"Executor ID": "1"}),
            _make_event("SparkListenerApplicationEnd", timestamp=1_010_000),
        ]
        log_path = build_log(tmp_path, events)
        advisor = DynamicAllocationAdvisor(log_path)
        rec = advisor.analyze()

        assert rec.peak_executors_observed == 2
        assert rec.evidence_summary.startswith("Peak executors observed: 2")

    def test_idle_executor_waste_evidence(self, tmp_path: Path) -> None:
        # Configured max = 10, observed peak = 2 -> waste should be positive
        events = [
            _make_event("SparkListenerApplicationStart", timestamp=1_000_000, extra={"App Name": "waste-test"}),
            {
                "Event": "SparkListenerEnvironmentUpdate",
                "Spark Properties": {
                    "spark.dynamicAllocation.enabled": "true",
                    "spark.dynamicAllocation.maxExecutors": "10",
                },
            },
            _make_event("SparkListenerExecutorAdded", timestamp=1_001_000, extra={"Executor ID": "1"}),
            _make_event("SparkListenerExecutorAdded", timestamp=1_002_000, extra={"Executor ID": "2"}),
            _make_event("SparkListenerApplicationEnd", timestamp=1_060_000),
        ]
        log_path = build_log(tmp_path, events)
        rec = DynamicAllocationAdvisor(log_path).analyze()

        assert rec.configured_max_executors == 10
        assert rec.peak_executors_observed == 2
        assert rec.idle_executor_waste_seconds > 0
        assert "unused executor slots" in rec.evidence_summary

    def test_recommend_shuffle_tracking_when_stage_present(self, tmp_path: Path) -> None:
        events = [
            _make_event("SparkListenerApplicationStart", timestamp=1_000_000, extra={"App Name": "shuffle-test"}),
            {
                "Event": "SparkListenerEnvironmentUpdate",
                "Spark Properties": {
                    "spark.dynamicAllocation.enabled": "true",
                    "spark.dynamicAllocation.shuffleTracking.enabled": "false",
                },
            },
            _make_event(
                "SparkListenerStageCompleted",
                timestamp=1_010_000,
                extra={"Stage Info": {"Stage ID": 0, "Stage Name": "map"}},
            ),
            _make_event("SparkListenerApplicationEnd", timestamp=1_020_000),
        ]
        log_path = build_log(tmp_path, events)
        rec = DynamicAllocationAdvisor(log_path).analyze()

        # Since there's a stage completed event, shuffle potential exists
        assert rec.shuffle_tracking_enabled is True
        assert "shuffle" in rec.evidence_summary.lower() or "Peak executors" in rec.evidence_summary

    def test_current_config_captured(self, tmp_path: Path) -> None:
        events = [
            {
                "Event": "SparkListenerEnvironmentUpdate",
                "Spark Properties": {
                    "spark.dynamicAllocation.minExecutors": "3",
                    "spark.dynamicAllocation.initialExecutors": "1",
                    "spark.shuffle.service.enabled": "true",
                },
            },
            _make_event("SparkListenerApplicationStart", timestamp=1_000_000),
            _make_event("SparkListenerApplicationEnd", timestamp=1_060_000),
        ]
        log_path = build_log(tmp_path, events)
        rec = DynamicAllocationAdvisor(log_path).analyze()

        assert rec.current_config.get("spark.dynamicAllocation.minExecutors") == "3"
        assert rec.current_config.get("spark.dynamicAllocation.initialExecutors") == "1"
        assert rec.current_config.get("spark.shuffle.service.enabled") == "true"

    def test_small_workload_recommendation(self, tmp_path: Path) -> None:
        events = [
            _make_event("SparkListenerApplicationStart", timestamp=1_000_000, extra={"App Name": "small"}),
            _make_event("SparkListenerExecutorAdded", timestamp=1_001_000, extra={"Executor ID": "1"}),
            _make_event("SparkListenerApplicationEnd", timestamp=1_060_000),
        ]
        log_path = build_log(tmp_path, events)
        rec = DynamicAllocationAdvisor(log_path).analyze()

        assert rec.peak_executors_observed == 1
        assert rec.max_executors >= 10  # bounded minimum
        assert rec.min_executors >= 2

    def test_to_dict_roundtrip(self) -> None:
        rec = DynamicAllocationRecommendation(
            min_executors=3,
            max_executors=25,
            initial_executors=3,
            shuffle_tracking_enabled=True,
            shuffle_service_enabled=False,
            idle_executor_waste_seconds=120.5,
            idle_executor_waste_percent=15.2,
            peak_executors_observed=4,
            configured_max_executors=20,
            configured_initial_executors=2,
            configured_min_executors=2,
            evidence_summary="test evidence",
            current_config={"spark.dynamicAllocation.enabled": "true"},
        )
        data = rec.to_dict()
        assert data["min_executors"] == 3
        assert data["max_executors"] == 25
        assert data["idle_executor_waste_seconds"] == 120.5
        assert data["evidence_summary"] == "test evidence"
