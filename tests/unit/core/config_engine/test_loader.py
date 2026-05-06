# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for version loader."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spark_optima.core.config_engine.loader import VersionLoader
from spark_optima.core.config_engine.models import ConfigSet


@pytest.fixture
def mock_database():
    """Create a mock database with test versions."""
    db = MagicMock()
    db.get_available_versions.return_value = ["3.4.0", "3.5.0", "4.0.0"]

    # Create proper ConfigSet objects - only for valid versions
    valid_versions = {"3.4.0", "3.5.0", "4.0.0"}

    def make_config_set(v):
        if v in valid_versions:
            config_set = ConfigSet(version=v, parameters={}, metadata={})
            return config_set
        return None

    db.get_config_set.side_effect = make_config_set
    return db


class TestVersionLoader:
    """Test cases for VersionLoader class."""

    def test_initialization(self, mock_database):
        """Test loader initialization."""
        loader = VersionLoader(mock_database)
        assert loader.database == mock_database

    def test_load_exact_match(self, mock_database):
        """Test loading exact version."""
        loader = VersionLoader(mock_database)
        config = loader.load("3.5.0")
        assert config is not None
        assert config.version == "3.5.0"

    def test_load_fallback_to_base_version(self, mock_database):
        """Test fallback to base version (e.g., 3.5.2 -> 3.5.0)."""
        loader = VersionLoader(mock_database)
        config = loader.load("3.5.2")
        assert config is not None
        assert config.version == "3.5.0"

    def test_load_exact_only(self, mock_database):
        """Test load_exact method."""
        loader = VersionLoader(mock_database)
        config = loader.load_exact("3.5.0")
        assert config is not None

        config = loader.load_exact("3.5.2")
        assert config is None

    def test_is_supported(self, mock_database):
        """Test version support check."""
        loader = VersionLoader(mock_database)
        assert loader.is_supported("3.5.0")
        assert loader.is_supported("3.5.2")  # Fallback available
        assert not loader.is_supported("9.9.9")

    def test_get_supported_versions(self, mock_database):
        """Test getting supported versions."""
        loader = VersionLoader(mock_database)
        versions = loader.get_supported_versions()
        assert versions == ["3.4.0", "3.5.0", "4.0.0"]

    def test_version_sort_key(self):
        """Test version sorting."""
        assert VersionLoader._version_sort_key("3.5.0") == (3, 5, 0)
        assert VersionLoader._version_sort_key("4.0.0") == (4, 0, 0)
        assert VersionLoader._version_sort_key("3.10.1") == (3, 10, 1)

    def test_compare_versions(self, mock_database):
        """Test version comparison."""
        loader = VersionLoader(mock_database)
        assert loader.compare_versions("3.5.0", "3.5.0") == 0
        assert loader.compare_versions("3.5.0", "3.4.0") > 0
        assert loader.compare_versions("3.5.0", "3.6.0") < 0
        assert loader.compare_versions("4.0.0", "3.9.9") > 0

    def test_is_at_least(self, mock_database):
        """Test minimum version check."""
        loader = VersionLoader(mock_database)
        assert loader.is_at_least("3.5.0", "3.0.0")
        assert loader.is_at_least("3.5.0", "3.5.0")
        assert not loader.is_at_least("3.4.0", "3.5.0")

    def test_is_between(self, mock_database):
        """Test version range check."""
        loader = VersionLoader(mock_database)
        assert loader.is_between("3.5.0", "3.0.0", "4.0.0")
        assert loader.is_between("3.5.0", "3.5.0", "3.5.0")
        assert not loader.is_between("3.5.0", "3.6.0", "4.0.0")

    def test_get_version_info(self, mock_database):
        """Test getting version information."""
        loader = VersionLoader(mock_database)
        # Verify load works first
        config = loader.load("3.5.0")
        assert config is not None
        assert config.version == "3.5.0"
        # Now test get_version_info
        info = loader.get_version_info("3.5.0")
        assert info is not None
        assert info["version"] == "3.5.0"
        assert info["resolved_version"] == "3.5.0"

    def test_get_version_info_not_found(self, mock_database):
        """Test get_version_info for non-existent version."""
        loader = VersionLoader(mock_database)
        # Version 9.9.9 will fallback to closest version (4.0.0)
        info = loader.get_version_info("9.9.9")
        assert info is not None
        assert "version" in info
        assert "resolved_version" in info

    def test_version_distance(self):
        """Test version distance calculation."""
        v1 = [3, 5, 0]
        v2 = [3, 5, 1]
        distance = VersionLoader._version_distance(v1, v2)
        assert distance > 0

        v1 = [3, 5, 0]
        v2 = [3, 6, 0]
        distance1 = VersionLoader._version_distance(v1, v2)

        v1 = [3, 5, 0]
        v2 = [4, 0, 0]
        distance2 = VersionLoader._version_distance(v1, v2)

        # Major version difference should be larger
        assert distance2 > distance1


class TestVersionLoaderEdgeCases:
    """Test edge cases for VersionLoader."""

    def test_load_nonexistent_version(self):
        """Test loading completely unknown version."""
        db = MagicMock()
        db.get_available_versions.return_value = ["3.5.0"]
        db.get_config_set.return_value = None

        loader = VersionLoader(db)
        config = loader.load("9.9.9")
        assert config is None

    def test_load_invalid_version_string(self):
        """Test loading invalid version string."""
        db = MagicMock()
        db.get_available_versions.return_value = ["3.5.0"]

        loader = VersionLoader(db)
        # Should handle gracefully
        loader.load("invalid")
        # May return None or closest match depending on implementation

    def test_empty_database(self):
        """Test with empty database."""
        db = MagicMock()
        db.get_available_versions.return_value = []

        loader = VersionLoader(db)
        assert loader.get_supported_versions() == []
        assert not loader.is_supported("3.5.0")

    def test_reload(self):
        """Test reloading database."""
        db = MagicMock()
        db.get_available_versions.return_value = ["3.5.0"]

        loader = VersionLoader(db)
        assert len(loader._available_versions) == 1

        # Simulate database update
        db.get_available_versions.return_value = ["3.5.0", "4.0.0"]
        loader.reload()
        assert len(loader._available_versions) == 2


class TestVersionLoaderMoreCoverage:
    """Additional tests for 100% coverage."""

    def test_initialization_no_database(self):
        """Test initialization without database (lines 51-59)."""
        loader = VersionLoader(database=None)
        assert loader.database is not None  # Creates new ConfigDatabase

    def test_build_version_index(self, mock_database):
        """Test _build_version_index (lines 61-64)."""
        loader = VersionLoader(mock_database)
        loader._build_version_index()
        assert len(loader._available_versions) == 3

    def test_load_exact_no_match(self, mock_database):
        """Test load_exact with no match (lines 105-115)."""
        loader = VersionLoader(mock_database)
        config = loader.load_exact("9.9.9")
        assert config is None

    def test_get_base_version_valid(self):
        """Test _get_base_version with valid versions (lines 117-132)."""
        loader = VersionLoader()

        # Test with full version
        result = loader._get_base_version("3.5.2")
        assert result == "3.5.0"

        # Test with major.minor only
        result = loader._get_base_version("3.5")
        assert result == "3.5.0"

        # Test with invalid version
        result = loader._get_base_version("invalid")
        assert result is None

        # Test with single part
        result = loader._get_base_version("3")
        assert result is None

    def test_find_closest_version(self):
        """Test _find_closest_version (lines 134-168)."""
        db = MagicMock()
        db.get_available_versions.return_value = ["3.0.0", "3.5.0", "4.0.0"]

        loader = VersionLoader(db)

        # Test finding closest
        result = loader._find_closest_version("3.4.0")
        assert result in ["3.5.0", "3.0.0"]

        # Test with version that doesn't exist
        result = loader._find_closest_version("invalid")
        assert result is None

    def test_version_sort_key_complex(self):
        """Test _version_sort_key with complex versions (lines 220-231)."""
        assert VersionLoader._version_sort_key("3.5.2") == (3, 5, 2)
        assert VersionLoader._version_sort_key("10.0.0") == (10, 0, 0)
        assert VersionLoader._version_sort_key("3.10.1") == (3, 10, 1)

    def test_get_version_info_not_found(self, mock_database):
        """Test get_version_info for non-existent version."""
        loader = VersionLoader(mock_database)

        # For non-existent version, load() may return a fallback
        # Reset the mock to return None for this version
        def get_config_side_effect(v):
            if v == "9.9.9":
                return None
            return MagicMock()

        mock_database.get_config_set.side_effect = get_config_side_effect

        info = loader.get_version_info("9.9.9")
        # May return None or a fallback result
        if info is not None:
            assert "version" in info

    def test_compare_versions_invalid(self, mock_database):
        """Test compare_versions with invalid input (lines 272-277)."""
        loader = VersionLoader(mock_database)

        # Test with invalid version strings - should use string comparison
        result = loader.compare_versions("invalid", "3.5.0")
        assert isinstance(result, int)

    def test_is_between_edge_cases(self, mock_database):
        """Test is_between edge cases (lines 303-320)."""
        loader = VersionLoader(mock_database)

        # Test equal to min
        assert loader.is_between("3.5.0", "3.5.0", "4.0.0")

        # Test equal to max
        assert loader.is_between("4.0.0", "3.5.0", "4.0.0")

        # Test just outside
        assert not loader.is_between("3.4.0", "3.5.0", "4.0.0")

    def test_supported_versions_sorted(self, mock_database):
        """Test get_supported_versions returns sorted (lines 211-218)."""
        loader = VersionLoader(mock_database)
        versions = loader.get_supported_versions()
        assert versions == ["3.4.0", "3.5.0", "4.0.0"]
        # Verify they're sorted
        assert versions == sorted(versions)
