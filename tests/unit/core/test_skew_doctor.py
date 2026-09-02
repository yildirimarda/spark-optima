# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the skew doctor (execution + analysis cross-reference)."""

from __future__ import annotations

from spark_optima.analysis.models import (
    AnalysisResult,
    CodeLocation,
    SeverityLevel,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.core.execution.event_log import EventLogSummary, StageSummary
from spark_optima.core.skew_doctor import (
    DEFAULT_SKEW_THRESHOLD,
    SkewDoctor,
    SkewFinding,
    diagnose,
)


class TestSkewDoctorBasics:
    """Basic initialization and threshold behavior."""

    def test_default_threshold_is_1_5(self) -> None:
        doctor = SkewDoctor()
        assert doctor.skew_threshold == DEFAULT_SKEW_THRESHOLD  # 1.5

    def test_custom_threshold(self) -> None:
        doctor = SkewDoctor(skew_threshold=3.0)
        assert doctor.skew_threshold == 3.0

    def test_threshold_clamped_above_1_0(self) -> None:
        doctor = SkewDoctor(skew_threshold=0.5)
        assert doctor.skew_threshold == 1.0


class TestSkewDoctorEmptyInput:
    """No stages or no skew should yield empty findings."""

    def test_no_stages(self) -> None:
        doctor = SkewDoctor()
        findings = doctor.diagnose(EventLogSummary())
        assert findings == []

    def test_all_skew_below_threshold(self) -> None:
        doctor = SkewDoctor(skew_threshold=5.0)
        summary = EventLogSummary(
            stages=[
                StageSummary(
                    stage_id=0,
                    name="balanced stage",
                    skew_ratio=1.2,
                )
            ]
        )
        findings = doctor.diagnose(summary)
        assert findings == []


class TestSkewDoctorSkewDetection:
    """Detecting skewed stages from event summaries."""

    def test_single_skewed_stage(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(
            stages=[
                StageSummary(
                    stage_id=1,
                    name="join at job.py:42",
                    skew_ratio=10.0,
                )
            ]
        )
        findings = doctor.diagnose(summary)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.stage.stage_id == 1
        assert finding.skew_ratio == 10.0
        assert finding.severity == SeverityLevel.CRITICAL  # >= 5.0

    def test_multiple_skewed_stages(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(
            stages=[
                StageSummary(stage_id=0, name="stage 0", skew_ratio=1.1),
                StageSummary(stage_id=1, name="stage 1", skew_ratio=2.5),
                StageSummary(stage_id=2, name="stage 2", skew_ratio=8.0),
            ]
        )
        findings = doctor.diagnose(summary)
        assert len(findings) == 2
        ids = {f.stage.stage_id for f in findings}
        assert ids == {1, 2}

    def test_moderate_skew_is_high_severity(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=0, name="groupBy", skew_ratio=2.0)])
        findings = doctor.diagnose(summary)
        assert findings[0].severity == SeverityLevel.HIGH


class TestSkewDoctorMappingWithAnalysis:
    """Mapping skewed stages back to analysis operations."""

    def test_join_operation_mapped_by_name(self) -> None:
        analysis = AnalysisResult(
            operations=[
                SparkOperation(
                    operation_type=SparkOperationType.JOIN,
                    method_name="join",
                    dataframe_var="df1",
                    arguments=["df2", "'user_id'"],
                    location=CodeLocation(line=10, column=0),
                )
            ]
        )
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(
            stages=[
                StageSummary(
                    stage_id=5,
                    name="join at job.py:10",
                    skew_ratio=4.0,
                )
            ]
        )
        findings = doctor.diagnose(summary, analysis)
        assert len(findings) == 1
        assert findings[0].mapped_operation is not None
        assert findings[0].mapped_operation.method_name == "join"
        assert findings[0].mapped_operation.dataframe_var == "df1"

    def test_groupby_operation_mapped(self) -> None:
        analysis = AnalysisResult(
            operations=[
                SparkOperation(
                    operation_type=SparkOperationType.AGGREGATION,
                    method_name="groupBy",
                    dataframe_var="sales",
                    arguments=["'region'"],
                    location=CodeLocation(line=20, column=4),
                )
            ]
        )
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=3, name="groupBy at job.py:20", skew_ratio=3.5)])
        findings = doctor.diagnose(summary, analysis)
        assert findings[0].mapped_operation is not None
        assert findings[0].mapped_operation.method_name == "groupBy"

    def test_no_analysis_result_maps_none(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=0, name="unknown", skew_ratio=6.0)])
        findings = doctor.diagnose(summary, analysis_result=None)
        assert findings[0].mapped_operation is None
        assert "Salting snippet" in findings[0].recommendation

    def test_fallback_shuffle_hint_mapping(self) -> None:
        # No name match but stage name contains "shuffle".
        analysis = AnalysisResult(
            operations=[
                SparkOperation(
                    operation_type=SparkOperationType.JOIN,
                    method_name="join",
                    dataframe_var="big",
                    arguments=["small", "'id'"],
                    location=CodeLocation(line=5, column=0),
                )
            ]
        )
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=1, name="shuffle sort", skew_ratio=2.0)])
        findings = doctor.diagnose(summary, analysis)
        assert findings[0].mapped_operation is not None


class TestSkewFindingModel:
    """Data model behavior."""

    def test_to_dict_with_operation(self) -> None:
        finding = SkewFinding(
            stage=StageSummary(stage_id=1, name="n", skew_ratio=5.0),
            skew_ratio=5.0,
            severity=SeverityLevel.HIGH,
            mapped_operation=SparkOperation(
                operation_type=SparkOperationType.JOIN,
                method_name="join",
                dataframe_var="d",
                location=CodeLocation(line=1, column=0),
            ),
        )
        d = finding.to_dict()
        assert d["stage_id"] == 1
        assert d["mapped_operation"]["method_name"] == "join"
        assert d["mapped_operation"]["operation_type"] == "JOIN"

    def test_to_dict_without_operation(self) -> None:
        finding = SkewFinding(
            stage=StageSummary(stage_id=2, name="n2", skew_ratio=3.0),
        )
        d = finding.to_dict()
        assert d["mapped_operation"] is None


class TestDiagnoseConvenience:
    """Module-level convenience function."""

    def test_convenience_function_returns_list(self) -> None:
        summary = EventLogSummary(stages=[StageSummary(stage_id=0, name="bad", skew_ratio=7.0)])
        result = diagnose(summary)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SkewFinding)


class TestRecommendationContent:
    """Recommendations contain AQE configs and salting snippets."""

    def test_recommendation_contains_aqe_config(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=1, name="bad", skew_ratio=3.0)])
        findings = doctor.diagnose(summary)
        assert "spark.sql.adaptive.skewJoin.enabled" in findings[0].recommendation
        assert "true" in findings[0].recommendation

    def test_recommendation_contains_salting(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        analysis = AnalysisResult(
            operations=[
                SparkOperation(
                    operation_type=SparkOperationType.JOIN,
                    method_name="join",
                    dataframe_var="df",
                    arguments=["'user_id'"],
                    location=CodeLocation(line=5, column=0),
                )
            ]
        )
        summary = EventLogSummary(stages=[StageSummary(stage_id=1, name="join at job.py:5", skew_ratio=4.0)])
        findings = doctor.diagnose(summary, analysis)
        assert "Salting snippet" in findings[0].recommendation
        assert "user_id" in findings[0].recommendation

    def test_recommendation_type_is_both(self) -> None:
        doctor = SkewDoctor(skew_threshold=1.5)
        summary = EventLogSummary(stages=[StageSummary(stage_id=1, name="bad", skew_ratio=3.0)])
        findings = doctor.diagnose(summary)
        assert findings[0].recommendation_type == "both"
