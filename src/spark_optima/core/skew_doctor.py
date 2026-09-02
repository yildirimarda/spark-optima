# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Skew doctor: cross-reference event-log skew with code analysis.

From event-log task-time distributions this module identifies skewed stages,
maps them back to the responsible join/groupBy via the existing SQL/DataFrame
analyzers, and emits AQE skew-join configurations or salting snippets per
finding — a cross-reference of ``analysis/`` and ``execution/`` that no
mainstream Spark tuning tool provides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spark_optima.analysis.models import (
    AnalysisResult,
    SeverityLevel,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.core.execution.event_log import (
    SKEW_MODERATE_THRESHOLD,
    SKEW_SEVERE_THRESHOLD,
    EventLogSummary,
    StageSummary,
)

logger = logging.getLogger(__name__)

# Default threshold for flagging a stage as skewed.
DEFAULT_SKEW_THRESHOLD = SKEW_MODERATE_THRESHOLD  # 1.5

# AQE skew-join configuration keys emitted for severe findings.
_AQE_SKEW_CONFIG_KEYS = [
    ("spark.sql.adaptive.enabled", "true"),
    ("spark.sql.adaptive.skewJoin.enabled", "true"),
    ("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5.0"),
    ("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB"),
]

# Salting snippet template for Python/PySpark.
_SALTING_SNIPPET_TEMPLATE = """# Manual salting for skewed column '{column}'
from pyspark.sql.functions import rand, lit, explode, array

salt_count = 10

df_salted = df.withColumn(
    "salt",
    (rand() * salt_count).cast("int"),
)
# Duplicate the skewed side or salt the join key accordingly
# Example for a join:
# df_small_salted = small_df.withColumn("salt", explode(array([lit(i) for i in range(salt_count)])))
# df_large_salted = large_df.withColumn("salt", (rand() * salt_count).cast("int"))
# result = df_large_salted.join(df_small_salted, ["{column}", "salt"])
"""


@dataclass
class SkewFinding:
    """A single skew diagnosis linking an event-log stage to code analysis.

    Attributes:
        stage: The skewed stage summary from the event log.
        skew_ratio: Computed max/median task duration ratio.
        severity: Severity derived from the skew ratio.
        mapped_operation: The SparkOperation (join/groupBy) mapped from
            ``analysis/`` when available; ``None`` when no code analysis was
            provided or no matching operation could be found.
        recommendation_type: ``"aqe_config"`` or ``"salting_snippet"`` (or
            ``"both"`` when both are emitted).
        recommendation: Human-readable recommendation string containing either
            AQE config keys, a salting snippet, or both.
        explanation: Short explanation of why the stage is skewed and how the
            recommendation addresses it.

    """

    stage: StageSummary
    skew_ratio: float = 1.0
    severity: SeverityLevel = SeverityLevel.LOW
    mapped_operation: SparkOperation | None = None
    recommendation_type: str = "both"
    recommendation: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage.stage_id,
            "stage_name": self.stage.name,
            "skew_ratio": self.skew_ratio,
            "severity": self.severity.value,
            "mapped_operation": {
                "method_name": self.mapped_operation.method_name,
                "operation_type": self.mapped_operation.operation_type.name,
                "dataframe_var": self.mapped_operation.dataframe_var,
                "location_line": self.mapped_operation.location.line if self.mapped_operation.location else None,
            }
            if self.mapped_operation
            else None,
            "recommendation_type": self.recommendation_type,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
        }


class SkewDoctor:
    """Diagnose skewed Spark stages by cross-referencing execution and analysis.

    Example:
        >>> doctor = SkewDoctor()
        >>> findings = doctor.diagnose(event_summary, analysis_result)
        >>> for f in findings:
        ...     print(f"Stage {f.stage.stage_id}: skew={f.skew_ratio:.1f} -> {f.recommendation_type}")

    """

    def __init__(self, skew_threshold: float = DEFAULT_SKEW_THRESHOLD) -> None:
        self.skew_threshold = max(skew_threshold, 1.0)

    def diagnose(
        self,
        event_summary: EventLogSummary,
        analysis_result: AnalysisResult | None = None,
    ) -> list[SkewFinding]:
        """Identify skewed stages and map them back to code operations.

        Args:
            event_summary: Parsed event log summary (from ``EventLogParser`` or
                ``HistoryServerClient``).
            analysis_result: Optional result from ``SmellDetector`` / parser so
                that join/groupBy operations can be mapped to the skewed stage.

        Returns:
            List of ``SkewFinding`` objects, one per skewed stage that exceeds
            the configured threshold.

        """
        findings: list[SkewFinding] = []

        if not event_summary.stages:
            logger.debug("No stages in event summary; nothing to diagnose.")
            return findings

        for stage in event_summary.stages:
            if stage.skew_ratio < self.skew_threshold:
                continue

            severity = SeverityLevel.CRITICAL if stage.skew_ratio >= SKEW_SEVERE_THRESHOLD else SeverityLevel.HIGH

            mapped_op = self._map_operation(stage, analysis_result)
            rec_type, recommendation = self._build_recommendation(stage, mapped_op, severity)
            explanation = self._build_explanation(stage, mapped_op, severity)

            findings.append(
                SkewFinding(
                    stage=stage,
                    skew_ratio=stage.skew_ratio,
                    severity=severity,
                    mapped_operation=mapped_op,
                    recommendation_type=rec_type,
                    recommendation=recommendation,
                    explanation=explanation,
                )
            )

            logger.info(
                "Skew finding: stage %d (%s) skew=%.2f severity=%s mapped_op=%s",
                stage.stage_id,
                stage.name,
                stage.skew_ratio,
                severity.value,
                mapped_op.method_name if mapped_op else "none",
            )

        return findings

    def _map_operation(
        self,
        stage: StageSummary,
        analysis_result: AnalysisResult | None,
    ) -> SparkOperation | None:
        """Map a skewed stage back to the responsible join/groupBy operation.

        The mapping strategy, in order of preference:

        1. If ``analysis_result`` is available, inspect ``operations`` for
           ``JOIN`` or ``AGGREGATION`` types whose ``method_name`` matches the
           stage name (e.g. ``groupBy``, ``join``).
        2. Fall back to selecting the first ``JOIN`` or ``AGGREGATION``
           operation from the analysis when the stage name suggests a shuffle
           (contains ``join``, ``group`` or ``groupBy`` in lower case).
        3. Return ``None`` when no analysis is provided or no operation matches.

        Args:
            stage: The skewed stage being diagnosed.
            analysis_result: Optional analysis result.

        Returns:
            The matched ``SparkOperation`` or ``None``.

        """
        if analysis_result is None:
            return None

        operations = analysis_result.operations
        name_lower = (stage.name or "").lower()

        # Try explicit name-based matching first.
        for op in operations:
            op_name = (op.method_name or "").lower()
            # Stage names often contain the method name (e.g. "join at ...").
            if op_name in name_lower and op.operation_type in (
                SparkOperationType.JOIN,
                SparkOperationType.AGGREGATION,
            ):
                return op
            # If stage mentions aggregation/shuffle patterns but we have
            # an operation of the matching type, prefer it.
            if op_name == "join" and "join" in name_lower and op.operation_type == SparkOperationType.JOIN:
                return op
            if (
                op_name in ("groupby", "groupbykey")
                and ("group" in name_lower or "groupby" in name_lower)
                and op.operation_type == SparkOperationType.AGGREGATION
            ):
                return op

        # Fallback: pick the first shuffle-sensitive operation (JOIN or AGGREGATION)
        # when the stage name hints at a shuffle operation.
        shuffle_hints = ("join", "group", "shuffle", "sort", "repartition", "distinct")
        is_shuffle_hint = any(hint in name_lower for hint in shuffle_hints)
        if is_shuffle_hint:
            for op in operations:
                if op.operation_type in (
                    SparkOperationType.JOIN,
                    SparkOperationType.AGGREGATION,
                ):
                    return op

        return None

    def _build_recommendation(
        self,
        stage: StageSummary,
        mapped_op: SparkOperation | None,
        severity: SeverityLevel,
    ) -> tuple[str, str]:
        """Generate the AQE config / salting snippet recommendation.

        Args:
            stage: The skewed stage.
            mapped_op: The mapped operation (may be ``None``).
            severity: Severity level derived from the skew ratio.

        Returns:
            Tuple of (recommendation_type, recommendation_text).

        """
        lines: list[str] = []
        op_ref = f" ({mapped_op.method_name} on '{mapped_op.dataframe_var}')" if mapped_op else ""

        # AQE configuration is always recommended for severe/moderate skew.
        lines.append(
            f"# AQE skew-join configuration for stage {stage.stage_id}{op_ref}\n"
            f"spark.conf.set('spark.sql.adaptive.enabled', 'true')\n"
            f"spark.conf.set('spark.sql.adaptive.skewJoin.enabled', 'true')\n"
            f"spark.conf.set('spark.sql.adaptive.skewJoin.skewedPartitionFactor', '5.0')\n"
            f"spark.conf.set('spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes', '256MB')"
        )

        # Salting snippet is emitted when we have a mapped operation with a
        # concrete column reference or when severity is at least HIGH.
        if mapped_op is not None:
            column = self._extract_skew_column(mapped_op)
            if column:
                snippet = _SALTING_SNIPPET_TEMPLATE.format(column=column)
                lines.append(f"\n# Salting snippet for column '{column}'{op_ref}\n{snippet}")
            else:
                # Even without a concrete column reference, provide a generic
                # salting snippet so the user has actionable code.
                snippet = _SALTING_SNIPPET_TEMPLATE.format(column="skewed_key")
                lines.append(f"\n# Salting snippet (adapt column name){op_ref}\n{snippet}")
        else:
            snippet = _SALTING_SNIPPET_TEMPLATE.format(column="skewed_key")
            lines.append(f"\n# Salting snippet (adapt column name){op_ref}\n{snippet}")

        recommendation_text = "\n".join(lines)
        recommendation_type = "both"
        return recommendation_type, recommendation_text

    def _extract_skew_column(self, op: SparkOperation) -> str | None:
        """Extract a likely skew column name from the operation arguments.

        Heuristic: look for quoted string arguments that contain common
        high-cardinality / skew-prone column names (``user_id``,
        ``customer_id``, etc.) or any column argument on a ``groupBy`` /
        ``join`` operation.

        Args:
            op: The mapped Spark operation.

        Returns:
            The inferred column name, or ``None`` when no clear column can be
            extracted.

        """
        if not op.arguments:
            return None

        # Common skew-prone column indicators.
        skew_indicators = [
            "user_id",
            "customer_id",
            "user",
            "customer",
            "account",
            "order_id",
        ]

        # For join operations, arguments are often ``[other_df, key]`` or
        # ``[other_df, ["col1", "col2"]]``.
        args_str = " ".join(str(arg) for arg in op.arguments).lower()
        for indicator in skew_indicators:
            if indicator in args_str:
                # Try to recover a quoted column name matching the indicator.
                for arg in op.arguments:
                    arg_str = str(arg)
                    if indicator.replace("_", "") in arg_str.lower():
                        # Strip quotes for a clean column name.
                        clean = arg_str.strip('"').strip("'")
                        if clean and not clean.startswith("["):
                            return clean
                        # For list-style keys like ["user_id"] try to extract
                        # the first quoted token.
                        if clean.startswith("[") and clean.endswith("]"):
                            inner = clean[1:-1]
                            tokens = [t.strip().strip("\"'") for t in inner.split(",")]
                            if tokens:
                                return tokens[0]
                return indicator.replace("_", "_")  # best-effort fallback

        # If no indicator matched, return the first quoted argument that
        # looks like a column name.
        for arg in op.arguments:
            arg_str = str(arg).strip()
            if arg_str.startswith(("'", '"')) and arg_str.endswith(("'", '"')):
                return arg_str[1:-1]
            # Handle list-form keys like ["user_id", ...].
            if arg_str.startswith("[") and arg_str.endswith("]"):
                inner = arg_str[1:-1]
                tokens = [t.strip().strip("\"'") for t in inner.split(",")]
                for token in tokens:
                    if token and not token.startswith("["):
                        return token

        return None

    def _build_explanation(
        self,
        stage: StageSummary,
        mapped_op: SparkOperation | None,
        severity: SeverityLevel,
    ) -> str:
        op_ref = (
            f" mapped to operation '{mapped_op.method_name}' ({mapped_op.operation_type.name})"
            if mapped_op
            else " (no matching operation found in analysis)"
        )
        return (
            f"Stage '{stage.name}' (id={stage.stage_id}) shows a task-time skew ratio of "
            f"{stage.skew_ratio:.1f}{op_ref}. "
            f"This indicates that some keys have significantly more records than others, "
            f"causing some partitions to run much longer than the median. "
            f"Enable AQE skew-join handling and consider manual salting for the skewed column."
        )


def diagnose(
    event_summary: EventLogSummary,
    analysis_result: AnalysisResult | None = None,
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
) -> list[SkewFinding]:
    """Convenience function for diagnosing skew from event and analysis data.

    Args:
        event_summary: Parsed event log summary.
        analysis_result: Optional code analysis result.
        skew_threshold: Minimum skew ratio to trigger a finding.

    Returns:
        List of ``SkewFinding`` objects.

    """
    doctor = SkewDoctor(skew_threshold=skew_threshold)
    return doctor.diagnose(event_summary, analysis_result)
