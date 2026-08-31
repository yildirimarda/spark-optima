# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for PySpark dependency range in pyproject.toml."""

from __future__ import annotations

from pathlib import Path


def test_pyspark_range_includes_4_2() -> None:
    """The pyproject.toml must include PySpark 4.2 in the supported range."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()
    assert "pyspark>=4.1.1,<4.3.0" in content
