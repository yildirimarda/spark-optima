# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the static regional pricing multipliers.

This module contains tests for the curated region multiplier tables and
the lookup helpers used by the cloud platform adapters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spark_optima.platforms.pricing import (
    REGION_MULTIPLIERS,
    get_region_multiplier,
    get_supported_regions,
)

if TYPE_CHECKING:
    import pytest

PRICING_LOGGER = "spark_optima.platforms.pricing"


class TestRegionMultiplierTables:
    """Test cases for the REGION_MULTIPLIERS tables."""

    def test_all_platforms_have_tables(self) -> None:
        """Test that all four cloud platforms have multiplier tables."""
        for platform in ("aws_glue", "aws_emr", "databricks", "azure_synapse"):
            assert platform in REGION_MULTIPLIERS
            assert len(REGION_MULTIPLIERS[platform]) >= 8

    def test_baseline_regions_are_one(self) -> None:
        """Test that each platform's baseline region has multiplier 1.0."""
        assert REGION_MULTIPLIERS["aws_glue"]["us-east-1"] == 1.0
        assert REGION_MULTIPLIERS["aws_emr"]["us-east-1"] == 1.0
        assert REGION_MULTIPLIERS["azure_synapse"]["eastus"] == 1.0
        assert REGION_MULTIPLIERS["databricks"]["aws:us-east-1"] == 1.0
        assert REGION_MULTIPLIERS["databricks"]["azure:eastus"] == 1.0

    def test_multipliers_are_positive(self) -> None:
        """Test that all multipliers are positive floats."""
        for platform, table in REGION_MULTIPLIERS.items():
            for region, multiplier in table.items():
                assert multiplier > 0, f"{platform}/{region} has non-positive multiplier"

    def test_databricks_keys_are_cloud_prefixed(self) -> None:
        """Test that databricks regions use the compound <cloud>:<region> form."""
        for region in REGION_MULTIPLIERS["databricks"]:
            assert region.startswith(("aws:", "azure:"))

    def test_databricks_reuses_aws_and_azure_tables(self) -> None:
        """Test that databricks multipliers match the AWS/Azure tables."""
        for region, multiplier in REGION_MULTIPLIERS["aws_glue"].items():
            assert REGION_MULTIPLIERS["databricks"][f"aws:{region}"] == multiplier
        for region, multiplier in REGION_MULTIPLIERS["azure_synapse"].items():
            assert REGION_MULTIPLIERS["databricks"][f"azure:{region}"] == multiplier


class TestGetRegionMultiplier:
    """Test cases for get_region_multiplier."""

    def test_known_region(self) -> None:
        """Test lookup of known non-baseline regions."""
        assert get_region_multiplier("aws_glue", "eu-west-1") == 1.1
        assert get_region_multiplier("aws_emr", "sa-east-1") == 1.4
        assert get_region_multiplier("azure_synapse", "westeurope") == 1.1
        assert get_region_multiplier("databricks", "aws:ap-northeast-1") == 1.2
        assert get_region_multiplier("databricks", "azure:japaneast") == 1.2

    def test_baseline_region(self) -> None:
        """Test that baseline regions return 1.0."""
        assert get_region_multiplier("aws_glue", "us-east-1") == 1.0
        assert get_region_multiplier("aws_emr", "us-east-1") == 1.0
        assert get_region_multiplier("azure_synapse", "eastus") == 1.0
        assert get_region_multiplier("databricks", "aws:us-east-1") == 1.0

    def test_case_insensitive_region(self) -> None:
        """Test that region lookup is case-insensitive."""
        assert get_region_multiplier("aws_glue", "EU-WEST-1") == 1.1
        assert get_region_multiplier("azure_synapse", "WestEurope") == 1.1
        assert get_region_multiplier("databricks", "AWS:EU-CENTRAL-1") == 1.15

    def test_case_insensitive_platform(self) -> None:
        """Test that platform lookup is case-insensitive."""
        assert get_region_multiplier("AWS_GLUE", "eu-west-1") == 1.1

    def test_unknown_region_returns_one_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that an unknown region falls back to 1.0 with a warning."""
        with caplog.at_level(logging.WARNING, logger=PRICING_LOGGER):
            multiplier = get_region_multiplier("aws_glue", "mars-north-1")

        assert multiplier == 1.0
        assert any("mars-north-1" in record.message for record in caplog.records)

    def test_none_region_returns_one_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that a None region falls back to 1.0 with a warning."""
        with caplog.at_level(logging.WARNING, logger=PRICING_LOGGER):
            multiplier = get_region_multiplier("azure_synapse", None)

        assert multiplier == 1.0
        assert any("No region specified" in record.message for record in caplog.records)

    def test_unknown_platform_returns_one_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that an unknown platform falls back to 1.0 with a warning."""
        with caplog.at_level(logging.WARNING, logger=PRICING_LOGGER):
            multiplier = get_region_multiplier("unknown_platform", "us-east-1")

        assert multiplier == 1.0
        assert any("unknown_platform" in record.message for record in caplog.records)

    def test_known_region_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that known lookups do not log warnings."""
        with caplog.at_level(logging.WARNING, logger=PRICING_LOGGER):
            get_region_multiplier("aws_glue", "us-east-1")
            get_region_multiplier("databricks", "azure:eastus")

        assert len(caplog.records) == 0


class TestGetSupportedRegions:
    """Test cases for get_supported_regions."""

    def test_supported_regions_aws(self) -> None:
        """Test supported regions listing for AWS platforms."""
        regions = get_supported_regions("aws_glue")

        assert isinstance(regions, list)
        assert "us-east-1" in regions
        assert "eu-west-1" in regions
        assert regions == sorted(regions)
        assert get_supported_regions("aws_emr") == regions

    def test_supported_regions_azure_synapse(self) -> None:
        """Test supported regions listing for Azure Synapse."""
        regions = get_supported_regions("azure_synapse")

        assert "eastus" in regions
        assert "brazilsouth" in regions

    def test_supported_regions_databricks(self) -> None:
        """Test supported regions listing for Databricks (compound keys)."""
        regions = get_supported_regions("databricks")

        assert "aws:us-east-1" in regions
        assert "azure:eastus" in regions

    def test_supported_regions_case_insensitive(self) -> None:
        """Test that platform lookup is case-insensitive."""
        assert get_supported_regions("Azure_Synapse") == get_supported_regions("azure_synapse")

    def test_supported_regions_unknown_platform(self) -> None:
        """Test that an unknown platform returns an empty list."""
        assert get_supported_regions("unknown_platform") == []
