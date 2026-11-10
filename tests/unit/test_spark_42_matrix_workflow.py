# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Test that the Spark 4.2 CI matrix workflow exists and is well-formed."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_spark_42_matrix_workflow_exists() -> None:
    workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "spark-42-matrix.yml"
    assert workflow_path.exists(), f"Workflow file missing: {workflow_path}"


def test_spark_42_matrix_workflow_is_valid_yaml() -> None:
    workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "spark-42-matrix.yml"
    content = workflow_path.read_text()
    data = yaml.safe_load(content)
    assert "jobs" in data
    assert "spark-42" in data["jobs"]


def test_spark_42_matrix_workflow_has_pyspark_42_version() -> None:
    workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "spark-42-matrix.yml"
    content = workflow_path.read_text()
    assert "4.2.0" in content
    assert "pyspark-version" in content


def test_spark_42_matrix_workflow_installs_into_the_project_venv() -> None:
    """The pinned PySpark must land in the uv venv and survive later `uv run`s."""
    workflow_path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "spark-42-matrix.yml"
    content = workflow_path.read_text()
    assert "uv pip install" in content
    assert "uv run pip install" not in content
    assert "uv run pytest" not in content, "a plain `uv run` re-syncs to uv.lock"
    assert "uv run --no-sync pytest" in content
