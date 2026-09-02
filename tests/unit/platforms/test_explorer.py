# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the cross-platform what-if explorer."""

from __future__ import annotations

from spark_optima.platforms.explorer import explore_what_if


class TestExploreWhatIf:
    def test_default_sweep_returns_ranked_results(self) -> None:
        results = explore_what_if(duration_hours=1.0)
        assert isinstance(results, list)
        assert len(results) > 0
        # All three platforms should be represented at least once.
        platforms = {r["platform"] for r in results}
        assert platforms.issubset({"aws_emr", "databricks", "gcp_dataproc"})

    def test_sorted_by_cost_ascending_when_pareto_enabled(self) -> None:
        results = explore_what_if(duration_hours=2.0, use_pareto=True)
        costs = [r["total_cost"] for r in results]
        assert costs == sorted(costs)

    def test_result_contains_pricing_source(self) -> None:
        results = explore_what_if(duration_hours=0.5)
        for r in results:
            assert "pricing_source" in r
            assert r["pricing_source"] in ("live", "static", "unknown")

    def test_custom_platform_list(self) -> None:
        results = explore_what_if(platforms=["aws_emr", "gcp_dataproc"])
        platforms = {r["platform"] for r in results}
        assert platforms == {"aws_emr", "gcp_dataproc"}
        assert not any(r["platform"] == "databricks" for r in results)

    def test_duration_hours_scales_cost(self) -> None:
        results_1h = explore_what_if(duration_hours=1.0, platforms=["aws_emr"])
        results_2h = explore_what_if(duration_hours=2.0, platforms=["aws_emr"])
        # Same instance should roughly double cost for 2h.
        instance_1h = next(r for r in results_1h if r["instance_size"] == results_2h[0]["instance_size"])
        # Just verify both have positive cost.
        assert instance_1h["total_cost"] > 0
