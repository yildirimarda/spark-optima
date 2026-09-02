# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the `spark-optima check` CLI guardrail command."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import pytest
from typer.testing import CliRunner

from spark_optima.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCheckCommand:
    def test_check_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["check", "--help"])
        assert result.exit_code == 0
        assert "guardrail" in result.output.lower()

    def test_check_missing_repo(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["check", "--repo", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_check_baseline_not_found(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["check", "--repo", str(tmp_path)])
        assert result.exit_code == 2
        assert "baseline file not found" in result.output

    def test_check_create_baseline(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(app, ["check", "--repo", str(tmp_path), "--create-baseline"])
        assert result.exit_code == 0
        assert "Baseline created" in result.output
        baseline_path = tmp_path / ".spark-optima" / "check-baseline.json"
        assert baseline_path.exists()
        data = json.loads(baseline_path.read_text())
        assert "smells" in data
        assert "config_issues" in data

    def test_check_with_baseline_pass(self, runner: CliRunner, tmp_path: Path) -> None:
        # Create a baseline with no smells
        baseline_dir = tmp_path / ".spark-optima"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / "check-baseline.json"
        baseline_path.write_text(json.dumps({"smells": {}, "config_issues": []}))

        result = runner.invoke(app, ["check", "--repo", str(tmp_path), "--baseline", str(baseline_path)])
        # Should exit 0 when no new smells/issues
        assert result.exit_code == 0
        assert "Passed" in result.output or "No new smells" in result.output

    def test_check_json_output(self, runner: CliRunner, tmp_path: Path) -> None:
        baseline_path = tmp_path / ".spark-optima" / "check-baseline.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({"smells": {}, "config_issues": []}))

        result = runner.invoke(
            app, ["check", "--repo", str(tmp_path), "--baseline", str(baseline_path), "--output", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload.get("failed") is False
