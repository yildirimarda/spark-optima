# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Shared unit parsing helpers for Spark configuration values.

These module-level helpers convert unit-suffixed strings to canonical base
units (bytes for size values, seconds for durations). They are used both by
the load-time constraint canonicalization in
:mod:`spark_optima.core.config_engine.models` and by the candidate-value
parsing in :mod:`spark_optima.core.config_engine.validator`, so the database
bounds and validated values are guaranteed to share one parser.
"""

from __future__ import annotations

import re
from typing import Any

# Byte size multipliers
BYTE_UNITS: dict[str, int] = {
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
}

# Duration multipliers (in seconds)
DURATION_UNITS: dict[str, float] = {
    "ms": 0.001,
    "s": 1,
    "sec": 1,
    "m": 60,
    "min": 60,
    "h": 3600,
    "hr": 3600,
    "d": 86400,
    "day": 86400,
}

_BYTES_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([a-z]*)$")
_DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([a-z]+)$")


def parse_bytes(value: str | int) -> int:
    """Parse a byte string to an integer number of bytes.

    Args:
        value: Byte string like "4g", "512m", or an integer (already bytes).

    Returns:
        Number of bytes. Bare numbers (no unit suffix) are returned as-is.

    Raises:
        ValueError: If the string cannot be parsed.

    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip().lower()

        # Handle -1 for unlimited
        if value == "-1" or value == "infinity":
            return -1

        # Extract number and unit
        match = _BYTES_PATTERN.match(value)
        if not match:
            raise ValueError(f"Invalid byte string: {value}")

        num_str = match.group(1)
        unit = match.group(2)

        # Validate that we actually have a number
        try:
            num = float(num_str)
        except ValueError as e:
            raise ValueError(f"Invalid byte string: {value}") from e

        # Empty unit is valid (plain number)
        if unit == "":
            return int(num)

        multiplier = BYTE_UNITS.get(unit)
        if multiplier is None:
            raise ValueError(f"Invalid byte unit: {unit}")
        return int(num * multiplier)

    raise ValueError(f"Cannot parse bytes from: {value}")


def parse_duration(value: str | int) -> int:
    """Parse a duration string to seconds.

    Args:
        value: Duration string like "5m", "1h", or integer seconds.

    Returns:
        Duration in seconds. Bare numbers (no unit suffix) mean seconds.

    Raises:
        ValueError: If the string cannot be parsed.

    """
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip().lower()

        # Handle "infinity"
        if value == "infinity":
            return -1

        # Check for special keywords
        if value == "daily":
            return 86400

        # Extract number and unit
        match = _DURATION_PATTERN.match(value)
        if match:
            num = float(match.group(1))
            unit = match.group(2)
            multiplier = DURATION_UNITS.get(unit, 1)
            return int(num * multiplier)

        # Try parsing as just a number (seconds)
        try:
            return int(value)
        except ValueError:
            pass

    raise ValueError(f"Cannot parse duration from: {value}")


def has_byte_unit_suffix(value: Any) -> bool:
    """Check whether a value carries an explicit byte unit suffix.

    Only strings like "512m" or "4gb" — a number followed by a known byte
    unit — qualify. Bare numbers (int or suffixless strings like "4096"),
    "-1"/"infinity" sentinels, and unparsable values do not.

    Args:
        value: Candidate value to inspect.

    Returns:
        True if the value is a string with an explicit byte unit suffix.

    """
    if not isinstance(value, str):
        return False

    match = _BYTES_PATTERN.match(value.strip().lower())
    if not match:
        return False

    unit = match.group(2)
    return unit != "" and unit in BYTE_UNITS
