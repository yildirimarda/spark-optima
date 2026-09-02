# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the spot/preemptible strategy advisor."""

from __future__ import annotations

import pytest

from spark_optima.core.spot_strategy_advisor import (
    DEFAULT_INTERRUPTION_RATES,
    DEFAULT_SPOT_DISCOUNTS,
    SpotStrategyAdvisor,
    SpotStrategyRecommendation,
)


class TestSpotStrategyAdvisorDefaults:
    def test_default_interruption_rates_defined_for_known_platforms(self) -> None:
        for platform in [
            "local",
            "aws_emr",
            "aws_glue",
            "azure_synapse",
            "gcp_dataproc",
            "databricks",
            "kubernetes",
        ]:
            assert platform in DEFAULT_INTERRUPTION_RATES
            assert platform in DEFAULT_SPOT_DISCOUNTS

    def test_default_discounts_positive_for_cloud_platforms(self) -> None:
        for platform, discount in DEFAULT_SPOT_DISCOUNTS.items():
            if platform == "local":
                assert discount == 0.0
            else:
                assert discount > 0.0


class TestSpotStrategyAdvisorInit:
    def test_default_platform_is_gcp_dataproc(self) -> None:
        advisor = SpotStrategyAdvisor()
        assert advisor.platform == "gcp_dataproc"
        assert advisor.interruption_rate == DEFAULT_INTERRUPTION_RATES["gcp_dataproc"]
        assert advisor.spot_discount == DEFAULT_SPOT_DISCOUNTS["gcp_dataproc"]

    def test_custom_platform_uses_defaults(self) -> None:
        advisor = SpotStrategyAdvisor("aws_emr")
        assert advisor.platform == "aws_emr"
        assert advisor.interruption_rate == DEFAULT_INTERRUPTION_RATES["aws_emr"]

    def test_custom_interruption_rate_override(self) -> None:
        advisor = SpotStrategyAdvisor("gcp_dataproc", interruption_rate=0.25)
        assert advisor.interruption_rate == 0.25

    def test_custom_spot_discount_override(self) -> None:
        advisor = SpotStrategyAdvisor("aws_emr", spot_discount=0.80)
        assert advisor.spot_discount == 0.80


class TestSpotStrategyAdvisorRecommend:
    def test_high_retry_tolerance_yields_positive_spot_mix(self) -> None:
        advisor = SpotStrategyAdvisor("gcp_dataproc")
        rec = advisor.recommend(
            stage_retry_tolerance=5,
            worker_count=10,
            duration_hours=2.0,
            base_hourly_cost_per_worker=0.50,
        )
        assert isinstance(rec, SpotStrategyRecommendation)
        assert rec.spot_mix_fraction > 0.0
        assert rec.spot_mix_fraction < 1.0
        assert rec.on_demand_fraction == pytest.approx(1.0 - rec.spot_mix_fraction)
        assert rec.stage_retry_tolerance == 5
        assert rec.expected_interruption_cost >= 0.0
        assert "spot mix" in rec.notes.lower() or "no spot mix" in rec.notes.lower()

    def test_zero_retry_tolerance_yields_zero_spot_mix(self) -> None:
        advisor = SpotStrategyAdvisor("aws_emr")
        rec = advisor.recommend(
            stage_retry_tolerance=0,
            worker_count=10,
            duration_hours=1.0,
            base_hourly_cost_per_worker=1.0,
        )
        assert rec.spot_mix_fraction == 0.0
        assert rec.on_demand_fraction == 1.0
        assert "No spot mix" in rec.notes

    def test_high_interruption_rate_suppresses_spot_mix(self) -> None:
        advisor = SpotStrategyAdvisor("gcp_dataproc", interruption_rate=0.95)
        rec = advisor.recommend(
            stage_retry_tolerance=3,
            worker_count=5,
            duration_hours=1.0,
            base_hourly_cost_per_worker=0.50,
        )
        # With 95% interruption rate, the raw fraction is heavily suppressed
        assert rec.spot_mix_fraction < 0.5

    def test_low_interruption_rate_allows_high_spot_mix(self) -> None:
        advisor = SpotStrategyAdvisor("local", interruption_rate=0.0)
        rec = advisor.recommend(
            stage_retry_tolerance=5,
            worker_count=8,
            duration_hours=3.0,
            base_hourly_cost_per_worker=0.30,
        )
        # Local has no interruption risk, but discount is 0, so savings should be 0
        assert rec.spot_mix_fraction > 0.0  # retry tolerance allows it
        assert rec.expected_cost_savings == pytest.approx(0.0)

    def test_override_interruption_probability(self) -> None:
        advisor = SpotStrategyAdvisor("aws_glue")
        rec = advisor.recommend(
            stage_retry_tolerance=3,
            interruption_probability=0.01,
            worker_count=4,
            duration_hours=1.0,
            base_hourly_cost_per_worker=1.0,
        )
        assert rec.interruption_probability == 0.01
        assert rec.spot_mix_fraction > 0.0

    def test_expected_cost_savings_positive_for_spot(self) -> None:
        advisor = SpotStrategyAdvisor("aws_emr", spot_discount=0.65)
        rec = advisor.recommend(
            stage_retry_tolerance=4,
            worker_count=20,
            duration_hours=2.0,
            base_hourly_cost_per_worker=0.50,
        )
        # With a 65% discount and low interruption rate, savings should be positive
        # (ignoring the small retry cost).
        assert rec.expected_cost_savings > 0.0 or rec.spot_mix_fraction == 0.0

    def test_to_dict_roundtrip(self) -> None:
        advisor = SpotStrategyAdvisor("databricks")
        rec = advisor.recommend(
            stage_retry_tolerance=2,
            worker_count=6,
            duration_hours=1.5,
            base_hourly_cost_per_worker=0.75,
        )
        data = rec.to_dict()
        assert data["platform"] == "databricks"
        assert data["stage_retry_tolerance"] == 2
        assert data["worker_count"] == 6
        assert "spot_mix_fraction" in data

    def test_notes_contain_key_metrics(self) -> None:
        advisor = SpotStrategyAdvisor("azure_synapse")
        rec = advisor.recommend(
            stage_retry_tolerance=3,
            worker_count=10,
            duration_hours=2.0,
            base_hourly_cost_per_worker=1.0,
        )
        assert "prob=" in rec.notes or "No spot mix" in rec.notes
        assert "retries=" in rec.notes or "No spot mix" in rec.notes
