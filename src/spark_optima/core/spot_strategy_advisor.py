# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Spot/preemptible strategy advisor: recommend spot-vs-on-demand mix.

This module provides a ``SpotStrategyAdvisor`` that recommends what fraction
of executors should run as spot/preemptible instances per platform.  Expected
interruption cost is modeled from the workload's stage retry tolerance
(i.e., how many retries the workload can absorb before a preemption becomes
costlier than the spot discount it provides).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Approximate spot/preemptible interruption rates per platform (fraction of
# preemptible executors expected to be interrupted per hour).
DEFAULT_INTERRUPTION_RATES: dict[str, float] = {
    "local": 0.0,
    "aws_emr": 0.06,
    "aws_glue": 0.03,
    "azure_synapse": 0.07,
    "gcp_dataproc": 0.14,
    "databricks": 0.10,
    "kubernetes": 0.11,
}

# Approximate spot discounts on the compute portion per platform.
DEFAULT_SPOT_DISCOUNTS: dict[str, float] = {
    "local": 0.0,
    "aws_emr": 0.65,
    "aws_glue": 0.45,
    "azure_synapse": 0.35,
    "gcp_dataproc": 0.65,
    "databricks": 0.40,
    "kubernetes": 0.55,
}


@dataclass
class SpotStrategyRecommendation:
    """Recommendation produced by the spot/preemptible strategy advisor."""

    platform: str = ""
    spot_mix_fraction: float = 0.0
    on_demand_fraction: float = 1.0
    stage_retry_tolerance: int = 0
    interruption_probability: float = 0.0
    expected_interruption_cost: float = 0.0
    expected_cost_savings: float = 0.0
    expected_total_cost: float = 0.0
    notes: str = ""
    worker_count: int = 0
    duration_hours: float = 0.0
    base_hourly_cost_per_worker: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "spot_mix_fraction": self.spot_mix_fraction,
            "on_demand_fraction": self.on_demand_fraction,
            "stage_retry_tolerance": self.stage_retry_tolerance,
            "interruption_probability": self.interruption_probability,
            "expected_interruption_cost": self.expected_interruption_cost,
            "expected_cost_savings": self.expected_cost_savings,
            "expected_total_cost": self.expected_total_cost,
            "notes": self.notes,
            "worker_count": self.worker_count,
            "duration_hours": self.duration_hours,
            "base_hourly_cost_per_worker": self.base_hourly_cost_per_worker,
        }


class SpotStrategyAdvisor:
    """Recommend spot-vs-on-demand executor mix per platform.

    The recommendation balances the cost savings from spot/preemptible
    instances against the expected interruption cost.  Interruption cost is
    modeled as:

    .. math::

        E[cost] = retry_tolerance x p(interrupt) x cost_of_retry

    where ``cost_of_retry`` is approximated as half the base hourly cost of
    a worker times the job duration (representing lost progress and
    re-execution overhead).

    Example:
        >>> advisor = SpotStrategyAdvisor("gcp_dataproc")
        >>> rec = advisor.recommend(
        ...     stage_retry_tolerance=3,
        ...     worker_count=10,
        ...     duration_hours=2.0,
        ...     base_hourly_cost_per_worker=0.50,
        ... )
        >>> print(f"Spot mix: {rec.spot_mix_fraction:.0%}")

    """

    def __init__(
        self,
        platform: str = "gcp_dataproc",
        interruption_rate: float | None = None,
        spot_discount: float | None = None,
    ) -> None:
        """Initialize the advisor for a specific platform.

        Args:
            platform: Platform identifier (e.g. ``gcp_dataproc``, ``aws_emr``).
            interruption_rate: Optional override for the platform's default
                interruption probability.
            spot_discount: Optional override for the platform's default spot
                discount.

        """
        self.platform = platform.lower()
        self.interruption_rate = (
            interruption_rate if interruption_rate is not None else DEFAULT_INTERRUPTION_RATES.get(self.platform, 0.10)
        )
        self.spot_discount = (
            spot_discount if spot_discount is not None else DEFAULT_SPOT_DISCOUNTS.get(self.platform, 0.0)
        )

    def recommend(
        self,
        stage_retry_tolerance: int = 2,
        worker_count: int = 10,
        duration_hours: float = 1.0,
        base_hourly_cost_per_worker: float = 1.0,
        interruption_probability: float | None = None,
    ) -> SpotStrategyRecommendation:
        """Recommend a spot-vs-on-demand mix.

        Args:
            stage_retry_tolerance: Maximum number of stage retries the workload
                can tolerate before interruption cost exceeds savings.
            worker_count: Total number of worker executors.
            duration_hours: Expected job duration in hours.
            base_hourly_cost_per_worker: On-demand hourly cost for one worker.
            interruption_probability: Optional override for interruption rate.

        Returns:
            A ``SpotStrategyRecommendation`` with the recommended mix and
            expected cost modeling.

        """
        prob = interruption_probability if interruption_probability is not None else self.interruption_rate

        # Higher retry tolerance allows a larger spot fraction.
        # The raw fraction is capped at 90% and scaled by (1 - interruption_prob)
        # so very high interruption rates suppress the recommendation.
        raw_fraction = min(0.9, max(0.0, stage_retry_tolerance / 5.0))
        spot_mix = max(0.0, min(0.9, raw_fraction * (1.0 - prob)))
        on_demand_mix = 1.0 - spot_mix

        # Base cost (all on-demand)
        base_cost = worker_count * base_hourly_cost_per_worker * duration_hours

        # Spot cost (spot workers get the discount; master nodes always stay
        # on-demand, but for simplicity this model applies to the worker pool).
        spot_hourly = base_hourly_cost_per_worker * (1.0 - self.spot_discount)
        spot_cost = (spot_mix * worker_count * spot_hourly * duration_hours) + (
            on_demand_mix * worker_count * base_hourly_cost_per_worker * duration_hours
        )

        # Expected interruption cost
        # Approximate retry cost as 50% of base hourly cost * duration,
        # scaled by retry tolerance and interruption probability.
        retry_cost_factor = 0.5 * base_hourly_cost_per_worker * duration_hours
        expected_retry_cost = stage_retry_tolerance * prob * retry_cost_factor * worker_count * spot_mix
        expected_total_cost = spot_cost + expected_retry_cost
        cost_savings = base_cost - expected_total_cost

        notes_parts: list[str] = []
        if spot_mix == 0.0:
            notes_parts.append("No spot mix recommended (low retry tolerance or high interruption rate).")
        else:
            notes_parts.append(
                f"{spot_mix:.0%} spot mix recommended for retry_tolerance={stage_retry_tolerance}, "
                f"interruption_prob={prob:.2f}."
            )
        notes_parts.append(
            f"Expected interruption cost: {expected_retry_cost:.2f} (prob={prob:.2f}, retries={stage_retry_tolerance})."
        )

        return SpotStrategyRecommendation(
            platform=self.platform,
            spot_mix_fraction=spot_mix,
            on_demand_fraction=on_demand_mix,
            stage_retry_tolerance=stage_retry_tolerance,
            interruption_probability=prob,
            expected_interruption_cost=expected_retry_cost,
            expected_cost_savings=cost_savings,
            expected_total_cost=expected_total_cost,
            notes="; ".join(notes_parts),
            worker_count=worker_count,
            duration_hours=duration_hours,
            base_hourly_cost_per_worker=base_hourly_cost_per_worker,
        )
