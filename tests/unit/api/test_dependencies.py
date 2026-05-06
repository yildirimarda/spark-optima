# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for API dependencies."""

from __future__ import annotations

from spark_optima.api.dependencies import (
    APIMetadata,
    OptimizationService,
    get_all_platforms_metadata,
    get_optimization_service,
    get_optimizer,
    get_platform_metadata,
)


class TestOptimizationService:
    """Tests for OptimizationService."""

    def test_initialization(self) -> None:
        """Test service initialization."""
        service = OptimizationService()
        assert service.config_database is not None
        assert service._optimizer_cache == {}

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        service = OptimizationService()
        service._optimizer_cache["test"] = object()  # Add dummy entry
        service.clear_cache()
        assert service._optimizer_cache == {}

    def test_get_available_spark_versions(self) -> None:
        """Test getting available Spark versions."""
        service = OptimizationService()
        versions = service.get_available_spark_versions()
        assert isinstance(versions, list)
        assert len(versions) > 0

    def test_validate_spark_version_valid(self) -> None:
        """Test validating a valid Spark version."""
        service = OptimizationService()
        # Get a valid version from available list
        versions = service.get_available_spark_versions()
        if versions:
            assert service.validate_spark_version(versions[0]) is True

    def test_validate_spark_version_invalid(self) -> None:
        """Test validating an invalid Spark version."""
        service = OptimizationService()
        assert service.validate_spark_version("99.99.99") is False


class TestGetOptimizationService:
    """Tests for get_optimization_service function."""

    def test_singleton_pattern(self) -> None:
        """Test that get_optimization_service returns singleton."""
        service1 = get_optimization_service()
        service2 = get_optimization_service()
        assert service1 is service2


class TestPlatformMetadata:
    """Tests for platform metadata functions."""

    def test_get_platform_metadata_local(self) -> None:
        """Test getting local platform metadata."""
        metadata = get_platform_metadata("local")
        assert metadata is not None
        assert "display_name" in metadata
        assert "supported_spark_versions" in metadata

    def test_get_platform_metadata_aws_glue(self) -> None:
        """Test getting AWS Glue platform metadata."""
        metadata = get_platform_metadata("aws_glue")
        assert metadata is not None
        assert "display_name" in metadata

    def test_get_platform_metadata_databricks(self) -> None:
        """Test getting Databricks platform metadata."""
        metadata = get_platform_metadata("databricks")
        assert metadata is not None
        assert "display_name" in metadata

    def test_get_platform_metadata_azure_synapse(self) -> None:
        """Test getting Azure Synapse platform metadata."""
        metadata = get_platform_metadata("azure_synapse")
        assert metadata is not None
        assert "display_name" in metadata

    def test_get_platform_metadata_invalid(self) -> None:
        """Test getting invalid platform metadata."""
        metadata = get_platform_metadata("invalid_platform")
        assert metadata is None

    def test_get_all_platforms_metadata(self) -> None:
        """Test getting all platforms metadata."""
        all_metadata = get_all_platforms_metadata()
        assert isinstance(all_metadata, dict)
        assert "local" in all_metadata
        assert "aws_glue" in all_metadata
        assert "databricks" in all_metadata
        assert "azure_synapse" in all_metadata


class TestAPIMetadata:
    """Tests for APIMetadata class."""

    def test_api_metadata_attributes(self) -> None:
        """Test API metadata has required attributes."""
        assert APIMetadata.TITLE == "Spark Optima API"
        assert APIMetadata.VERSION == "0.1.0"
        assert isinstance(APIMetadata.DESCRIPTION, str)
        assert isinstance(APIMetadata.CONTACT, dict)
        assert isinstance(APIMetadata.LICENSE_INFO, dict)


class TestGetOptimizer:
    """Tests for get_optimizer dependency function."""

    def test_get_optimizer_creates_instance(self) -> None:
        """Test that get_optimizer creates an optimizer instance."""
        optimizer = get_optimizer(
            platform="local",
            spark_version="3.5.0",
            optimization_mode="simulation",
        )
        assert optimizer is not None
        assert optimizer.platform == "local"
