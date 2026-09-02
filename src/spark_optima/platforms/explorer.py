# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Cross-platform what-if explorer.

Given a tuned workload, sweep instance families/sizes across EMR,
Databricks and Dataproc using the live-pricing module and report
"same SLA, cheapest platform" as a ranked table.  This extends the
existing Pareto frontier across platforms by treating cost (for a
fixed duration / SLA) as the primary comparison metric.
"""

from __future__ import annotations

import logging
from typing import Any

from spark_optima.platforms import get_platform
from spark_optima.platforms.live_pricing import get_live_hourly_rate, is_live_pricing_enabled
from spark_optima.platforms.models import ResourceSpec

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["aws_emr", "databricks", "gcp_dataproc"]


def explore_what_if(
    workload_resources: ResourceSpec | None = None,
    duration_hours: float = 1.0,
    platforms: list[str] | None = None,
    region: str | None = None,
    spark_version: str = "3.5.0",
    use_pareto: bool = True,
) -> list[dict[str, Any]]:
    """Sweep instance families/sizes across platforms and rank by cost.

    Args:
        workload_resources: Target total resources (defaults to a small
            representative spec if None).
        duration_hours: Expected job duration for cost estimation.
        platforms: List of platform identifiers to sweep.  Defaults to
            ``["aws_emr", "databricks", "gcp_dataproc"]``.
        region: Optional cloud region override.
        spark_version: Spark version for cluster recommendations.
        use_pareto: When True, the returned list is sorted by ascending
            cost and tagged with ``pricing_source`` so it acts as a
            cross-platform Pareto extension (same SLA, cheapest platform).

    Returns:
        List of result dictionaries, one per platform/instance
        combination, sorted by total cost ascending.

    """
    if workload_resources is None:
        workload_resources = ResourceSpec(cpu_cores=16, memory_gb=64)

    if platforms is None:
        platforms = SUPPORTED_PLATFORMS.copy()

    results: list[dict[str, Any]] = []

    for platform_name in platforms:
        try:
            # Region defaults are handled by each platform constructor,
            # but we allow an explicit override for live pricing.
            if platform_name == "aws_emr":
                platform = get_platform(platform_name, region=region or "us-east-1")
            elif platform_name == "gcp_dataproc":
                platform = get_platform(platform_name, region=region or "us-central1")
            elif platform_name == "databricks":
                # Databricks uses compound region keys; keep default.
                platform = get_platform(platform_name)
            else:
                platform = get_platform(platform_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unsupported platform '%s': %s", platform_name, exc)
            continue

        worker_types = platform.get_worker_types()
        if not worker_types:
            continue

        for worker in worker_types:
            try:
                config = platform.recommend_config(
                    resources=workload_resources,
                    spark_version=spark_version,
                )
                # Force the worker type to the current sweep instance so
                # we evaluate every family/size independently.
                from dataclasses import replace

                config = replace(config, worker_type=worker)
                # Ensure driver is at least the same family for consistency.
                driver = platform.get_worker_type(worker.name) or worker
                config = replace(config, driver_type=driver)

                cost = platform.estimate_cost(config, duration_hours=duration_hours)
                total_cost = cost.get("total_cost", 0.0) if isinstance(cost, dict) else float(cost)

                # Try to resolve live pricing label for this instance.
                pricing_source = cost.get("pricing_source", "unknown") if isinstance(cost, dict) else "unknown"

                # If live pricing is enabled and the cost came from live rates,
                # surface the instance-level rate for transparency.
                instance_rate: float | None = None
                if is_live_pricing_enabled():
                    try:
                        if platform_name == "aws_emr":
                            rate = get_live_hourly_rate(
                                platform_name,
                                region=(region or getattr(platform, "region", "us-east-1")),
                                instance_type=worker.name,
                            )
                        elif platform_name == "gcp_dataproc":
                            rate = get_live_hourly_rate(
                                platform_name,
                                region=(region or getattr(platform, "region", "us-central1")),
                                instance_type=worker.name,
                                vcpus=worker.resources.cpu_cores,
                                memory_gb=worker.resources.memory_gb,
                            )
                        else:
                            rate = None
                        instance_rate = rate
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Live rate lookup failed for %s/%s: %s", platform_name, worker.name, exc)
                        instance_rate = None

                results.append({
                    "platform": platform_name,
                    "platform_display": getattr(platform, "display_name", platform_name),
                    "instance_family": worker.name.split(".")[0] if "." in worker.name else worker.name.split("-")[0] if "-" in worker.name else "other",
                    "instance_size": worker.name,
                    "instance_type": worker.name,
                    "resources": worker.resources.to_dict(),
                    "worker_type": worker.to_dict(),
                    "cluster_config": config.to_dict() if hasattr(config, "to_dict") else config,
                    "duration_hours": duration_hours,
                    "total_cost": total_cost,
                    "pricing_source": pricing_source,
                    "instance_rate_live": instance_rate,
                    "region": region or (getattr(platform, "region", None)),
                })
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping %s/%s due to error: %s", platform_name, worker.name, exc)
                continue

    if use_pareto:
        results.sort(key=lambda r: r["total_cost"])
    else:
        results.sort(key=lambda r: (r["platform"], r["total_cost"]))

    return results
