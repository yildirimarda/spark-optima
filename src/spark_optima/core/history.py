# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Optimization history persistence for Spark Optima.

This module provides the OptimizationHistory class, a lightweight SQLite-backed
store for past optimization results. It powers the ``spark-optima history`` CLI
command and the auto-save behaviour of ``spark-optima optimize``.

The database location is resolved in the following order:

1. Explicit ``db_path`` constructor argument.
2. The ``SPARK_OPTIMA_HISTORY_DB`` environment variable.
3. The default ``~/.spark_optima/history.db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

HISTORY_DB_ENV_VAR = "SPARK_OPTIMA_HISTORY_DB"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS optimization_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    platform TEXT NOT NULL,
    spark_version TEXT NOT NULL,
    mode TEXT NOT NULL,
    estimated_time_minutes REAL NOT NULL DEFAULT 0.0,
    confidence_score REAL NOT NULL DEFAULT 0.0,
    code_path TEXT,
    config_json TEXT NOT NULL,
    result_json TEXT NOT NULL
)
"""


def _default_db_path() -> Path:
    """Resolve the default history database path.

    Returns:
        Path from the ``SPARK_OPTIMA_HISTORY_DB`` environment variable if set,
        otherwise ``~/.spark_optima/history.db``.

    """
    env_path = os.environ.get(HISTORY_DB_ENV_VAR)
    if env_path:
        return Path(env_path)
    return Path.home() / ".spark_optima" / "history.db"


@dataclass
class HistoryEntry:
    """A single persisted optimization run.

    Attributes:
        entry_id: Auto-incremented database identifier.
        created_at: ISO-8601 timestamp of when the entry was saved (UTC).
        platform: Target platform of the optimization (e.g. "local").
        spark_version: Spark version the configuration was optimized for.
        mode: Optimization mode ("simulation" or "execution").
        estimated_time_minutes: Predicted execution time in minutes.
        confidence_score: Confidence level of the optimization (0.0 to 1.0).
        code_path: Path of the analyzed Spark code file, if any.
        configuration: Optimized Spark configuration key-value pairs.
        result: Full optimization result dictionary as saved.

    """

    entry_id: int
    created_at: str
    platform: str
    spark_version: str
    mode: str
    estimated_time_minutes: float
    confidence_score: float
    code_path: str | None
    configuration: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> HistoryEntry:
        """Build a HistoryEntry from a database row.

        Args:
            row: SQLite row with all ``optimization_history`` columns.

        Returns:
            Populated HistoryEntry instance.

        """
        return cls(
            entry_id=int(row["id"]),
            created_at=str(row["created_at"]),
            platform=str(row["platform"]),
            spark_version=str(row["spark_version"]),
            mode=str(row["mode"]),
            estimated_time_minutes=float(row["estimated_time_minutes"]),
            confidence_score=float(row["confidence_score"]),
            code_path=row["code_path"],
            configuration=json.loads(row["config_json"]),
            result=json.loads(row["result_json"]),
        )


class OptimizationHistory:
    """SQLite-backed store for optimization results.

    This class persists optimization results so past runs can be listed,
    inspected, and compared. It uses only the Python standard library and
    parameterized SQL queries.

    Example:
        >>> with OptimizationHistory() as history:
        ...     entry_id = history.save(
        ...         result.to_dict(),
        ...         platform="local",
        ...         spark_version="3.5.0",
        ...         mode="simulation",
        ...         code_path="job.py",
        ...     )
        ...     entries = history.list_entries(limit=10)

    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the history store and ensure the schema exists.

        Args:
            db_path: Optional database file path. Falls back to the
                ``SPARK_OPTIMA_HISTORY_DB`` environment variable, then to
                ``~/.spark_optima/history.db``. Parent directories are
                created as needed.

        """
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)

    def __enter__(self) -> OptimizationHistory:
        """Enter the context manager.

        Returns:
            This OptimizationHistory instance.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the database connection."""
        self.close()

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def save(
        self,
        result_dict: dict[str, Any],
        *,
        platform: str,
        spark_version: str,
        mode: str,
        code_path: str | None = None,
    ) -> int:
        """Persist an optimization result.

        Args:
            result_dict: Full result dictionary (e.g. ``OptimizationResult.to_dict()``).
            platform: Target platform name.
            spark_version: Spark version the run was optimized for.
            mode: Optimization mode ("simulation" or "execution").
            code_path: Path of the analyzed code file, if any.

        Returns:
            The auto-assigned history entry id.

        Raises:
            TypeError: If ``result_dict`` is not a dictionary.

        """
        if not isinstance(result_dict, dict):
            raise TypeError(f"result_dict must be a dict, got {type(result_dict).__name__}")

        configuration = result_dict.get("configuration") or {}
        estimated_time = float(result_dict.get("estimated_time_minutes") or 0.0)
        confidence = float(result_dict.get("confidence_score") or 0.0)
        created_at = datetime.now(timezone.utc).isoformat()

        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO optimization_history "
                "(created_at, platform, spark_version, mode, estimated_time_minutes, "
                "confidence_score, code_path, config_json, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at,
                    str(platform),
                    str(spark_version),
                    str(mode),
                    estimated_time,
                    confidence,
                    str(code_path) if code_path is not None else None,
                    json.dumps(configuration),
                    json.dumps(result_dict),
                ),
            )
        return int(cursor.lastrowid or 0)

    def list_entries(self, platform: str | None = None, limit: int = 20) -> list[HistoryEntry]:
        """List stored entries, newest first.

        Args:
            platform: Optional platform name to filter by.
            limit: Maximum number of entries to return (must be >= 1).

        Returns:
            List of HistoryEntry objects ordered by id descending.

        Raises:
            ValueError: If ``limit`` is less than 1.

        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        if platform:
            cursor = self._conn.execute(
                "SELECT * FROM optimization_history WHERE platform = ? ORDER BY id DESC LIMIT ?",
                (platform, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM optimization_history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [HistoryEntry.from_row(row) for row in cursor.fetchall()]

    def get(self, entry_id: int) -> HistoryEntry | None:
        """Fetch a single entry by id.

        Args:
            entry_id: History entry identifier.

        Returns:
            HistoryEntry if found, None otherwise.

        """
        cursor = self._conn.execute(
            "SELECT * FROM optimization_history WHERE id = ?",
            (entry_id,),
        )
        row = cursor.fetchone()
        return HistoryEntry.from_row(row) if row is not None else None

    def clear(self) -> int:
        """Delete all stored entries.

        Returns:
            Number of deleted entries.

        """
        with self._conn:
            cursor = self._conn.execute("SELECT COUNT(*) FROM optimization_history")
            count = int(cursor.fetchone()[0])
            self._conn.execute("DELETE FROM optimization_history")
        return count
