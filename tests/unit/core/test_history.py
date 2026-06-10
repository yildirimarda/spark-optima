# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the optimization history store.

This module contains tests for the SQLite-backed OptimizationHistory class,
covering persistence, querying, filtering, and database path resolution.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

from spark_optima.core.history import HISTORY_DB_ENV_VAR, HistoryEntry, OptimizationHistory

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample optimization result dictionary."""
    return {
        "configuration": {
            "spark.executor.memory": "4g",
            "spark.executor.cores": 4,
            "spark.sql.adaptive.enabled": "true",
        },
        "estimated_time_minutes": 12.5,
        "confidence_score": 0.85,
        "code_suggestions": [],
        "platform_specific": {"platform": "local", "spark_version": "3.5.0"},
        "metadata": {"platform": "local", "spark_version": "3.5.0"},
    }


@pytest.fixture
def history_store(tmp_path: Path) -> Iterator[OptimizationHistory]:
    """Create a history store backed by a temporary database."""
    store = OptimizationHistory(db_path=tmp_path / "history.db")
    yield store
    store.close()


class TestOptimizationHistoryPaths:
    """Test cases for database path resolution."""

    def test_explicit_path_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that nested parent directories are created for the DB file."""
        db_path = tmp_path / "deeply" / "nested" / "history.db"

        with OptimizationHistory(db_path=db_path) as store:
            assert store.db_path == db_path

        assert db_path.exists()

    def test_env_var_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that SPARK_OPTIMA_HISTORY_DB overrides the default path."""
        env_db = tmp_path / "env" / "custom.db"
        monkeypatch.setenv(HISTORY_DB_ENV_VAR, str(env_db))

        with OptimizationHistory() as store:
            assert store.db_path == env_db

        assert env_db.exists()

    def test_constructor_arg_beats_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that an explicit constructor path wins over the env var."""
        monkeypatch.setenv(HISTORY_DB_ENV_VAR, str(tmp_path / "env.db"))
        explicit_db = tmp_path / "explicit.db"

        with OptimizationHistory(db_path=explicit_db) as store:
            assert store.db_path == explicit_db

        assert explicit_db.exists()
        assert not (tmp_path / "env.db").exists()

    def test_default_path_uses_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the default path resolves to ~/.spark_optima/history.db."""
        monkeypatch.delenv(HISTORY_DB_ENV_VAR, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        with OptimizationHistory() as store:
            assert store.db_path == tmp_path / ".spark_optima" / "history.db"

        assert (tmp_path / ".spark_optima" / "history.db").exists()


class TestOptimizationHistorySave:
    """Test cases for saving entries."""

    def test_save_returns_incrementing_ids(
        self, history_store: OptimizationHistory, sample_result: dict[str, Any]
    ) -> None:
        """Test that save returns sequential entry ids."""
        first_id = history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")
        second_id = history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        assert first_id == 1
        assert second_id == 2

    def test_save_persists_all_fields(self, history_store: OptimizationHistory, sample_result: dict[str, Any]) -> None:
        """Test that all fields round-trip through the database."""
        entry_id = history_store.save(
            sample_result,
            platform="databricks",
            spark_version="3.5.0",
            mode="execution",
            code_path="/tmp/job.py",
        )

        entry = history_store.get(entry_id)
        assert entry is not None
        assert isinstance(entry, HistoryEntry)
        assert entry.entry_id == entry_id
        assert entry.platform == "databricks"
        assert entry.spark_version == "3.5.0"
        assert entry.mode == "execution"
        assert entry.code_path == "/tmp/job.py"
        assert entry.estimated_time_minutes == pytest.approx(12.5)
        assert entry.confidence_score == pytest.approx(0.85)
        assert entry.configuration == sample_result["configuration"]
        assert entry.result == sample_result
        # created_at must be a valid ISO-8601 timestamp
        assert datetime.fromisoformat(entry.created_at) is not None

    def test_save_without_code_path(self, history_store: OptimizationHistory, sample_result: dict[str, Any]) -> None:
        """Test that code_path is optional and stored as None."""
        entry_id = history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        entry = history_store.get(entry_id)
        assert entry is not None
        assert entry.code_path is None

    def test_save_defaults_missing_metrics_to_zero(self, history_store: OptimizationHistory) -> None:
        """Test that missing metrics default to 0.0."""
        entry_id = history_store.save(
            {"configuration": {"spark.executor.memory": "2g"}},
            platform="local",
            spark_version="3.5.0",
            mode="simulation",
        )

        entry = history_store.get(entry_id)
        assert entry is not None
        assert entry.estimated_time_minutes == 0.0
        assert entry.confidence_score == 0.0

    def test_save_rejects_non_dict(self, history_store: OptimizationHistory) -> None:
        """Test that save raises TypeError for non-dict input."""
        with pytest.raises(TypeError, match="result_dict must be a dict"):
            history_store.save(
                "not a dict",  # type: ignore[arg-type]
                platform="local",
                spark_version="3.5.0",
                mode="simulation",
            )


class TestOptimizationHistoryQuery:
    """Test cases for listing and fetching entries."""

    def test_get_missing_returns_none(self, history_store: OptimizationHistory) -> None:
        """Test that get returns None for unknown ids."""
        assert history_store.get(999) is None

    def test_list_entries_empty(self, history_store: OptimizationHistory) -> None:
        """Test that listing an empty store returns an empty list."""
        assert history_store.list_entries() == []

    def test_list_entries_newest_first(self, history_store: OptimizationHistory, sample_result: dict[str, Any]) -> None:
        """Test that entries are returned newest first."""
        ids = [
            history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")
            for _ in range(3)
        ]

        entries = history_store.list_entries()
        assert [e.entry_id for e in entries] == sorted(ids, reverse=True)

    def test_list_entries_limit(self, history_store: OptimizationHistory, sample_result: dict[str, Any]) -> None:
        """Test that the limit parameter caps the number of entries."""
        for _ in range(5):
            history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        entries = history_store.list_entries(limit=2)
        assert len(entries) == 2

    def test_list_entries_platform_filter(
        self, history_store: OptimizationHistory, sample_result: dict[str, Any]
    ) -> None:
        """Test filtering entries by platform."""
        history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")
        history_store.save(sample_result, platform="databricks", spark_version="3.5.0", mode="simulation")

        local_entries = history_store.list_entries(platform="local")
        assert len(local_entries) == 1
        assert local_entries[0].platform == "local"

        missing_entries = history_store.list_entries(platform="aws_glue")
        assert missing_entries == []

    def test_list_entries_invalid_limit(self, history_store: OptimizationHistory) -> None:
        """Test that a limit below 1 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            history_store.list_entries(limit=0)


class TestOptimizationHistoryClear:
    """Test cases for clearing the store."""

    def test_clear_returns_deleted_count(
        self, history_store: OptimizationHistory, sample_result: dict[str, Any]
    ) -> None:
        """Test that clear deletes all entries and reports the count."""
        for _ in range(3):
            history_store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        assert history_store.clear() == 3
        assert history_store.list_entries() == []

    def test_clear_empty_store(self, history_store: OptimizationHistory) -> None:
        """Test that clearing an empty store returns zero."""
        assert history_store.clear() == 0


class TestOptimizationHistoryLifecycle:
    """Test cases for connection lifecycle management."""

    def test_context_manager_closes_connection(self, tmp_path: Path, sample_result: dict[str, Any]) -> None:
        """Test that the context manager closes the connection on exit."""
        with OptimizationHistory(db_path=tmp_path / "history.db") as store:
            store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        with pytest.raises(sqlite3.ProgrammingError):
            store.list_entries()

    def test_data_persists_across_instances(self, tmp_path: Path, sample_result: dict[str, Any]) -> None:
        """Test that saved entries survive reopening the database."""
        db_path = tmp_path / "history.db"
        with OptimizationHistory(db_path=db_path) as store:
            entry_id = store.save(sample_result, platform="local", spark_version="3.5.0", mode="simulation")

        with OptimizationHistory(db_path=db_path) as reopened:
            entry = reopened.get(entry_id)
            assert entry is not None
            assert entry.configuration == sample_result["configuration"]
