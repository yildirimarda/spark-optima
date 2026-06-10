# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Static regional pricing multipliers for cloud platform adapters.

This module provides curated, static price multipliers per cloud region,
relative to each platform's baseline region (multiplier 1.0). The values
are curated approximations of typical regional price differences — they
are NOT live quotes and may drift from actual cloud provider pricing.
Live pricing API integration is deliberately deferred (see PLAN.md backlog).

Platform keys and their baseline regions:

| Platform key    | Baseline region | Region key format            |
|-----------------|-----------------|------------------------------|
| ``aws_glue``    | us-east-1       | AWS region (``us-east-1``)   |
| ``aws_emr``     | us-east-1       | AWS region (``us-east-1``)   |
| ``azure_synapse`` | eastus        | Azure region (``eastus``)    |
| ``databricks``  | aws:us-east-1   | ``<cloud>:<region>`` (e.g. ``aws:us-east-1``, ``azure:eastus``) |

Databricks runs on multiple clouds, so its table is keyed by a compound
``<cloud_provider>:<region>`` identifier reusing the AWS and Azure tables.

Example:
    >>> from spark_optima.platforms.pricing import get_region_multiplier
    >>> get_region_multiplier("aws_glue", "eu-west-1")
    1.1
    >>> get_region_multiplier("databricks", "azure:westeurope")
    1.1

"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Curated AWS region multipliers relative to us-east-1 (baseline = 1.0).
# Approximations of typical on-demand price differences; not live quotes.
_AWS_REGION_MULTIPLIERS: dict[str, float] = {
    "us-east-1": 1.0,  # N. Virginia (baseline)
    "us-east-2": 1.0,  # Ohio
    "us-west-1": 1.1,  # N. California
    "us-west-2": 1.0,  # Oregon
    "ca-central-1": 1.05,  # Canada Central
    "eu-west-1": 1.1,  # Ireland
    "eu-west-2": 1.15,  # London
    "eu-central-1": 1.15,  # Frankfurt
    "ap-south-1": 1.05,  # Mumbai
    "ap-southeast-1": 1.15,  # Singapore
    "ap-southeast-2": 1.2,  # Sydney
    "ap-northeast-1": 1.2,  # Tokyo
    "sa-east-1": 1.4,  # Sao Paulo
}

# Curated Azure region multipliers relative to eastus (baseline = 1.0).
# Approximations of typical pay-as-you-go price differences; not live quotes.
_AZURE_REGION_MULTIPLIERS: dict[str, float] = {
    "eastus": 1.0,  # East US (baseline)
    "eastus2": 1.0,  # East US 2
    "centralus": 1.05,  # Central US
    "westus": 1.05,  # West US
    "westus2": 1.0,  # West US 2
    "northeurope": 1.1,  # Ireland
    "westeurope": 1.1,  # Netherlands
    "uksouth": 1.1,  # London
    "southeastasia": 1.15,  # Singapore
    "australiaeast": 1.2,  # New South Wales
    "japaneast": 1.2,  # Tokyo
    "brazilsouth": 1.4,  # Sao Paulo
}

# Curated GCP region multipliers relative to us-central1 (baseline = 1.0).
# Approximations of typical Compute Engine price differences; not live quotes.
_GCP_REGION_MULTIPLIERS: dict[str, float] = {
    "us-central1": 1.0,  # Iowa (baseline)
    "us-east1": 1.0,  # South Carolina
    "us-east4": 1.05,  # N. Virginia
    "us-west1": 1.0,  # Oregon
    "us-west2": 1.1,  # Los Angeles
    "europe-west1": 1.1,  # Belgium
    "europe-west2": 1.15,  # London
    "europe-west3": 1.15,  # Frankfurt
    "asia-south1": 1.05,  # Mumbai
    "asia-southeast1": 1.15,  # Singapore
    "asia-northeast1": 1.2,  # Tokyo
    "southamerica-east1": 1.4,  # Sao Paulo
}

# Region price multipliers per platform, relative to each platform's
# baseline region (multiplier 1.0). Databricks is keyed by the compound
# "<cloud_provider>:<region>" identifier because the adapter models both
# the cloud (aws/azure) and the region.
REGION_MULTIPLIERS: dict[str, dict[str, float]] = {
    "aws_glue": dict(_AWS_REGION_MULTIPLIERS),
    "aws_emr": dict(_AWS_REGION_MULTIPLIERS),
    "azure_synapse": dict(_AZURE_REGION_MULTIPLIERS),
    "gcp_dataproc": dict(_GCP_REGION_MULTIPLIERS),
    "databricks": {
        **{f"aws:{region}": multiplier for region, multiplier in _AWS_REGION_MULTIPLIERS.items()},
        **{f"azure:{region}": multiplier for region, multiplier in _AZURE_REGION_MULTIPLIERS.items()},
    },
}


def get_region_multiplier(platform: str, region: str | None) -> float:
    """Get the price multiplier for a platform region.

    The lookup is case-insensitive. Unknown platforms or regions fall back
    to the baseline multiplier (1.0) with a logged warning so cost
    estimation never fails due to an unrecognized region.

    Args:
        platform: Platform identifier (e.g., "aws_glue", "databricks").
            Databricks regions use the compound "<cloud>:<region>" form
            (e.g., "aws:us-east-1", "azure:westeurope").
        region: Region identifier, or None.

    Returns:
        Price multiplier relative to the platform's baseline region
        (1.0 for the baseline, unknown regions, or None).

    """
    table = REGION_MULTIPLIERS.get(platform.lower())
    if table is None:
        logger.warning(
            "No regional pricing table for platform '%s'; using baseline multiplier 1.0",
            platform,
        )
        return 1.0

    if region is None:
        logger.warning(
            "No region specified for platform '%s'; using baseline multiplier 1.0",
            platform,
        )
        return 1.0

    multiplier = table.get(region.lower())
    if multiplier is None:
        logger.warning(
            "Unknown region '%s' for platform '%s'; using baseline multiplier 1.0",
            region,
            platform,
        )
        return 1.0

    return multiplier


def get_supported_regions(platform: str) -> list[str]:
    """List the regions with curated pricing multipliers for a platform.

    Args:
        platform: Platform identifier (e.g., "aws_glue", "databricks").

    Returns:
        Sorted list of supported region identifiers (empty for unknown
        platforms).

    """
    table = REGION_MULTIPLIERS.get(platform.lower())
    if table is None:
        return []
    return sorted(table)
