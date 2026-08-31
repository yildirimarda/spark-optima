# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Test that the Spark 4.2 CI matrix proposal exists and is well-formed."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_spark_42_matrix_proposal_exists() -> None:
    proposal_path = Path(__file__).parent.parent.parent / "ci-proposals" / "spark-42-matrix.yml"
    assert proposal_path.exists(), f"Proposal file missing: {proposal_path}"


def test_spark_42_matrix_proposal_is_valid_yaml() -> None:
    proposal_path = Path(__file__).parent.parent.parent / "ci-proposals" / "spark-42-matrix.yml"
    content = proposal_path.read_text()
    data = yaml.safe_load(content)
    assert "jobs" in data
    assert "spark-42" in data["jobs"]


def test_spark_42_matrix_proposal_has_pyspark_42_version() -> None:
    proposal_path = Path(__file__).parent.parent.parent / "ci-proposals" / "spark-42-matrix.yml"
    content = proposal_path.read_text()
    assert "4.2.0" in content
    assert "pyspark-version" in content
