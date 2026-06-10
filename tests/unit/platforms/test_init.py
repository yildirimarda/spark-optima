# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for platforms __init__.py to achieve 100% coverage.

This module tests the registry functions: get_platform, list_platforms,
and get_platform_info.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from spark_optima.platforms import (
    AWSEMRPlatform,
    AWSGluePlatform,
    AzureSynapsePlatform,
    DatabricksPlatform,
    LocalPlatform,
    get_platform,
    get_platform_info,
    list_platforms,
)


class TestGetPlatform:
    """Test cases for get_platform function."""

    def test_get_platform_local(self) -> None:
        """Test getting local platform."""
        platform = get_platform("local")
        assert isinstance(platform, LocalPlatform)
        assert platform.name == "local"

    def test_get_platform_databricks(self) -> None:
        """Test getting databricks platform."""
        platform = get_platform("databricks")
        assert isinstance(platform, DatabricksPlatform)

    def test_get_platform_aws_glue(self) -> None:
        """Test getting aws_glue platform."""
        platform = get_platform("aws_glue")
        assert isinstance(platform, AWSGluePlatform)

    def test_get_platform_aws_emr(self) -> None:
        """Test getting aws_emr platform."""
        platform = get_platform("aws_emr")
        assert isinstance(platform, AWSEMRPlatform)

    def test_get_platform_azure_synapse(self) -> None:
        """Test getting azure_synapse platform."""
        platform = get_platform("azure_synapse")
        assert isinstance(platform, AzureSynapsePlatform)

    def test_get_platform_case_insensitive(self) -> None:
        """Test that platform name is case insensitive."""
        platform = get_platform("LOCAL")
        assert isinstance(platform, LocalPlatform)

    def test_get_platform_with_kwargs(self) -> None:
        """Test get_platform with additional kwargs."""
        platform = get_platform("aws_glue", region="us-west-2")
        assert isinstance(platform, AWSGluePlatform)

    def test_get_platform_invalid_name(self) -> None:
        """Test get_platform raises ValueError for invalid platform (lines 69-70)."""
        with pytest.raises(ValueError, match="Unknown platform: 'invalid_platform'"):
            get_platform("invalid_platform")

    def test_get_platform_invalid_name_includes_valid(self) -> None:
        """Test error message includes valid platform names (lines 69-70)."""
        with pytest.raises(ValueError, match="local") as exc_info:
            get_platform("nonexistent")
        error_message = str(exc_info.value)
        assert "aws_glue" in error_message
        assert "databricks" in error_message
        assert "azure_synapse" in error_message


class TestListPlatforms:
    """Test cases for list_platforms function (line 84)."""

    def test_list_platforms_returns_list(self) -> None:
        """Test that list_platforms returns a list."""
        platforms = list_platforms()
        assert isinstance(platforms, list)

    def test_list_platforms_contains_expected(self) -> None:
        """Test that list_platforms contains expected platforms."""
        platforms = list_platforms()
        assert "local" in platforms
        assert "aws_glue" in platforms
        assert "aws_emr" in platforms
        assert "databricks" in platforms
        assert "azure_synapse" in platforms

    def test_list_platforms_no_unexpected(self) -> None:
        """Test that list_platforms doesn't contain unexpected platforms."""
        platforms = list_platforms()
        # Should only contain the 5 known platforms
        assert len(platforms) == 5

    def test_list_platforms_returns_copy(self) -> None:
        """Test that list_platforms returns a copy (not the original)."""
        platforms = list_platforms()
        # Modifying the returned list shouldn't affect future calls
        platforms.append("new_platform")
        platforms2 = list_platforms()
        assert "new_platform" not in platforms2


class TestGetPlatformInfo:
    """Test cases for get_platform_info function (lines 94-110)."""

    def test_get_platform_info_returns_dict(self) -> None:
        """Test that get_platform_info returns a dictionary."""
        info = get_platform_info()
        assert isinstance(info, dict)

    def test_get_platform_info_contains_all_platforms(self) -> None:
        """Test that info contains all platforms."""
        info = get_platform_info()
        assert "local" in info
        assert "aws_glue" in info
        assert "aws_emr" in info
        assert "databricks" in info
        assert "azure_synapse" in info

    def test_get_platform_info_structure(self) -> None:
        """Test that each platform info has correct structure."""
        info = get_platform_info()
        for platform_name, platform_info in info.items():
            assert "name" in platform_info
            assert "display_name" in platform_info
            assert "description" in platform_info
            assert platform_info["name"] == platform_name

    def test_get_platform_info_values(self) -> None:
        """Test that platform info has valid values."""
        info = get_platform_info()
        for _platform_name, platform_info in info.items():
            assert isinstance(platform_info["name"], str)
            assert isinstance(platform_info["display_name"], str)
            # Description can be empty string
            assert isinstance(platform_info["description"], str)

    @patch("spark_optima.platforms.PLATFORM_REGISTRY")
    def test_get_platform_info_handles_exceptions(self, mock_registry: MagicMock) -> None:
        """Test that get_platform_info handles exceptions gracefully (lines 104-109)."""
        # Create a mock that raises RuntimeError (which is caught by the except clause)
        mock_platform_class = MagicMock(side_effect=RuntimeError("Test error"))
        mock_platform_class.__name__ = "MockPlatform"
        mock_registry.items.return_value = [("test_platform", mock_platform_class)]

        info = get_platform_info()

        assert "test_platform" in info
        assert info["test_platform"]["name"] == "test_platform"
        assert info["test_platform"]["display_name"] == "MockPlatform"
        assert info["test_platform"]["description"] == ""

    def test_get_platform_info_local_platform(self) -> None:
        """Test get_platform_info for local platform specifically."""
        info = get_platform_info()
        local_info = info["local"]
        assert local_info["name"] == "local"
        assert isinstance(local_info["display_name"], str)
        assert len(local_info["display_name"]) > 0

    def test_get_platform_info_aws_glue_platform(self) -> None:
        """Test get_platform_info for AWS Glue platform specifically."""
        info = get_platform_info()
        glue_info = info["aws_glue"]
        assert glue_info["name"] == "aws_glue"
        assert isinstance(glue_info["display_name"], str)

    def test_get_platform_info_databricks_platform(self) -> None:
        """Test get_platform_info for Databricks platform specifically."""
        info = get_platform_info()
        databricks_info = info["databricks"]
        assert databricks_info["name"] == "databricks"
        assert isinstance(databricks_info["display_name"], str)

    def test_get_platform_info_aws_emr_platform(self) -> None:
        """Test get_platform_info for AWS EMR platform specifically."""
        info = get_platform_info()
        emr_info = info["aws_emr"]
        assert emr_info["name"] == "aws_emr"
        assert isinstance(emr_info["display_name"], str)

    def test_get_platform_info_azure_synapse_platform(self) -> None:
        """Test get_platform_info for Azure Synapse platform specifically."""
        info = get_platform_info()
        synapse_info = info["azure_synapse"]
        assert synapse_info["name"] == "azure_synapse"
        assert isinstance(synapse_info["display_name"], str)
