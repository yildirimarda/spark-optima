# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Version loader for Spark configuration.

This module provides the VersionLoader class for managing and resolving
Spark version configurations, including version matching and fallback logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from spark_optima.core.config_engine.database import ConfigDatabase

if TYPE_CHECKING:
    from spark_optima.core.config_engine.models import ConfigSet

logger = logging.getLogger(__name__)


class VersionLoader:
    """Loader for resolving Spark versions to configuration sets.

    This class handles version matching, fallback logic, and version-specific
    configuration loading. It supports exact version matching as well as
    compatibility matching for minor version differences.

    Attributes:
        database: ConfigDatabase instance for accessing configurations.
        _version_map: Mapping of major.minor versions to specific config files.

    Example:
        >>> loader = VersionLoader()
        >>> config_set = loader.load("3.5.2")  # Falls back to 3.5.0 config
        >>> config_set = loader.load_exact("3.5.0")

    """

    # Version to config file mapping
    SUPPORTED_VERSIONS = {
        "3.0": "3.0.0",
        "3.1": "3.1.0",
        "3.2": "3.2.0",
        "3.3": "3.3.0",
        "3.4": "3.4.0",
        "3.5": "3.5.0",
        "4.0": "4.0.0",
        "4.1": "4.1.0",
    }

    def __init__(self, database: ConfigDatabase | None = None) -> None:
        """Initialize the version loader.

        Args:
            database: ConfigDatabase instance. If None, creates a new one.

        """
        self.database = database or ConfigDatabase()
        self._build_version_index()

    def _build_version_index(self) -> None:
        """Build index of available versions from the database."""
        self._available_versions = set(self.database.get_available_versions())
        logger.debug(f"Available versions: {self._available_versions}")

    def load(self, version: str) -> ConfigSet | None:
        """Load configuration for a version with fallback logic.

        This method attempts to find the best matching configuration for
        the given version. It tries exact match first, then falls back to
        the closest major.minor version.

        Args:
            version: Spark version string (e.g., "3.5.2", "3.5", "4.0.0").

        Returns:
            ConfigSet for the version, or None if not found.

        Example:
            >>> loader.load("3.5.2")  # Returns 3.5.0 config
            >>> loader.load("3.5")    # Returns 3.5.0 config
            >>> loader.load("4.0.0")  # Returns 4.0.0 config

        """
        # Try exact match first
        if version in self._available_versions:
            logger.debug(f"Exact version match: {version}")
            return self.database.get_config_set(version)

        # Try to match major.minor version
        base_version = self._get_base_version(version)
        if base_version and base_version in self._available_versions:
            logger.debug(f"Fallback to base version: {version} -> {base_version}")
            return self.database.get_config_set(base_version)

        # Try to find closest version
        closest = self._find_closest_version(version)
        if closest:
            logger.debug(f"Fallback to closest version: {version} -> {closest}")
            return self.database.get_config_set(closest)

        logger.warning(f"No configuration found for version: {version}")
        return None

    def load_exact(self, version: str) -> ConfigSet | None:
        """Load configuration for exact version only.

        Args:
            version: Exact Spark version string.

        Returns:
            ConfigSet if exact match found, None otherwise.

        """
        return self.database.get_config_set(version)

    def _get_base_version(self, version: str) -> str | None:
        """Get the base version (major.minor.0) for a version string.

        Args:
            version: Version string like "3.5.2" or "3.5".

        Returns:
            Base version string like "3.5.0", or None if invalid.

        """
        parts = version.split(".")
        if len(parts) < 2:
            return None

        major_minor = f"{parts[0]}.{parts[1]}"
        return self.SUPPORTED_VERSIONS.get(major_minor)

    def _find_closest_version(self, version: str) -> str | None:
        """Find the closest available version.

        Args:
            version: Target version string.

        Returns:
            Closest available version string, or None.

        """
        if not self._available_versions:
            return None

        try:
            target_parts = [int(x) for x in version.split(".")]
        except ValueError:
            return None

        best_match = None
        best_distance = float("inf")

        for avail in self._available_versions:
            try:
                avail_parts = [int(x) for x in avail.split(".")]
            except ValueError:
                continue

            # Calculate version distance
            distance = self._version_distance(target_parts, avail_parts)

            if distance < best_distance:
                best_distance = distance
                best_match = avail

        return best_match

    @staticmethod
    def _version_distance(v1: list[int], v2: list[int]) -> int:
        """Calculate distance between two version tuples.

        Args:
            v1: First version parts.
            v2: Second version parts.

        Returns:
            Distance metric (lower is closer).

        """
        # Pad shorter version
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0] * (max_len - len(v1))
        v2 = v2 + [0] * (max_len - len(v2))

        # Weight major version differences more heavily
        distance = 0
        for i, (p1, p2) in enumerate(zip(v1, v2, strict=False)):
            weight = 1000 ** (max_len - i - 1)
            distance += abs(p1 - p2) * weight

        return distance

    def is_supported(self, version: str) -> bool:
        """Check if a version is supported (has configuration).

        Args:
            version: Spark version string.

        Returns:
            True if configuration exists for the version.

        """
        if version in self._available_versions:
            return True

        base = self._get_base_version(version)
        return base in self._available_versions if base else False

    def get_supported_versions(self) -> list[str]:
        """Get list of all supported versions.

        Returns:
            Sorted list of supported version strings.

        """
        return sorted(self._available_versions, key=self._version_sort_key)

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, ...]:
        """Create sort key for version strings.

        Args:
            version: Version string like "3.5.0".

        Returns:
            Tuple of integers for proper version sorting.

        """
        return tuple(int(x) for x in version.split("."))

    def get_version_info(self, version: str) -> dict[str, Any] | None:
        """Get information about a specific version.

        Args:
            version: Spark version string.

        Returns:
            Dictionary with version information, or None.

        """
        config_set = self.load(version)
        if config_set is None:
            return None

        return {
            "version": version,
            "resolved_version": config_set.version,
            "parameter_count": len(config_set),
            "categories": sorted(
                {param.category.value for param in config_set.parameters.values()},
            ),
        }

    def compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings.

        Args:
            version1: First version string.
            version2: Second version string.

        Returns:
            Negative if v1 < v2, 0 if equal, positive if v1 > v2.

        """
        try:
            parts1 = [int(x) for x in version1.split(".")]
            parts2 = [int(x) for x in version2.split(".")]
        except ValueError:
            # Fall back to string comparison for non-standard versions
            return (version1 > version2) - (version1 < version2)

        # Pad to same length
        max_len = max(len(parts1), len(parts2))
        parts1 = parts1 + [0] * (max_len - len(parts1))
        parts2 = parts2 + [0] * (max_len - len(parts2))

        for p1, p2 in zip(parts1, parts2, strict=False):
            if p1 != p2:
                return p1 - p2

        return 0

    def is_at_least(self, version: str, minimum: str) -> bool:
        """Check if version is at least the minimum.

        Args:
            version: Version to check.
            minimum: Minimum required version.

        Returns:
            True if version >= minimum.

        """
        return self.compare_versions(version, minimum) >= 0

    def is_between(self, version: str, min_version: str, max_version: str) -> bool:
        """Check if version is within a range.

        Args:
            version: Version to check.
            min_version: Minimum version (inclusive).
            max_version: Maximum version (inclusive).

        Returns:
            True if min_version <= version <= max_version.

        """
        return self.compare_versions(version, min_version) >= 0 and self.compare_versions(version, max_version) <= 0

    def get_parameters_introduced_in(self, since_version: str) -> dict[str, list[str]]:
        """Get parameters introduced since a specific version.

        Args:
            since_version: Version to check from.

        Returns:
            Dictionary mapping versions to lists of parameter names.

        """
        result: dict[str, list[str]] = {}

        for avail_version in self._available_versions:
            if self.compare_versions(avail_version, since_version) >= 0:
                config_set = self.database.get_config_set(avail_version)
                if config_set:
                    new_params = [
                        name for name, param in config_set.parameters.items() if param.since_version == avail_version
                    ]
                    if new_params:
                        result[avail_version] = new_params

        return result

    def reload(self) -> None:
        """Reload the database and version index."""
        self.database.reload()
        self._build_version_index()
