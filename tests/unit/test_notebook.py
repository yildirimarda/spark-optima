# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the notebook cell magic (%%spark_optima)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNotebookImports:
    """Test module import and flags."""

    def test_import_notebook_module(self) -> None:
        from spark_optima.notebook import (
            HAS_IPYTHON,
            load_ipython_extension,
            spark_optima,
        )

        assert callable(spark_optima)
        assert callable(load_ipython_extension)
        # HAS_IPYTHON depends on environment; just verify it is a bool
        assert isinstance(HAS_IPYTHON, bool)


class TestProfileExtraction:
    """Test session profiling helpers."""

    def test_build_profile_with_none(self) -> None:
        from spark_optima.notebook import _build_profile

        profile = _build_profile(None)
        assert profile["session_available"] is False
        assert profile.get("conf_keys") == []

    @patch("spark_optima.notebook.PYSPARK_AVAILABLE", True)
    def test_extract_session_config_empty_spark(self) -> None:
        from spark_optima.notebook import _extract_session_config

        mock_spark = MagicMock()
        mock_spark.sparkContext.getConf.side_effect = Exception("no conf")
        conf = _extract_session_config(mock_spark)
        assert conf == {}

    @patch("spark_optima.notebook.PYSPARK_AVAILABLE", True)
    def test_build_profile_with_mock_spark(self) -> None:
        from spark_optima.notebook import _build_profile

        mock_spark = MagicMock()
        mock_spark.sparkContext.appName = "MockApp"
        mock_spark.sparkContext.master = "local[*]"
        mock_spark.version = "4.0.0"
        mock_conf = MagicMock()
        mock_conf.getAll.return_value = [("spark.app.name", "MockApp")]
        mock_conf.get.side_effect = lambda k, default="": {
            ("spark.app.name", "MockApp"): "MockApp",
        }.get(str(k), default)
        mock_spark.sparkContext.getConf.return_value = mock_conf

        profile = _build_profile(mock_spark)
        assert profile["session_available"] is True
        assert profile.get("app_name") == "MockApp"
        assert profile.get("master") == "local[*]"
        assert profile.get("version_hint") == "4.0.0"
        assert profile.get("platform_hint") == "local"
        assert profile.get("conf_count", 0) >= 1


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_recommendations_from_empty_profile(self) -> None:
        from spark_optima.notebook import _generate_recommendations

        recommendations = _generate_recommendations({})
        assert isinstance(recommendations, list)
        # At minimum we expect the memory, adaptive, dynamic allocation,
        # serialization, and shuffle compression recommendations.
        assert len(recommendations) >= 4

    def test_recommendations_with_cell_code(self) -> None:
        from spark_optima.notebook import _generate_recommendations

        recommendations = _generate_recommendations(
            {"conf": {"spark.sql.adaptive.enabled": "true"}},
            cell_code="df = spark.range(10)\n",
        )
        # Should contain some recommendations; cell analysis may fail gracefully
        assert isinstance(recommendations, list)


class TestCellMagicOutput:
    """Test the cell magic prints recommendations without crashing."""

    @patch("spark_optima.notebook.PYSPARK_AVAILABLE", False)
    def test_spark_optima_without_pyspark(self, capsys) -> None:
        from spark_optima.notebook import spark_optima

        result = spark_optima()
        assert result is None
        captured = capsys.readouterr()
        assert "PySpark is not installed" in captured.out

    @patch("spark_optima.notebook.PYSPARK_AVAILABLE", True)
    def test_spark_optima_no_active_session(self, capsys) -> None:
        from spark_optima.notebook import spark_optima

        # Mock out the session lookup so nothing is found
        with patch("spark_optima.notebook._get_active_spark_session", return_value=None):
            result = spark_optima()
        assert result is None
        captured = capsys.readouterr()
        # Should indicate no active session
        assert "No active SparkSession found" in captured.out or "Spark Optima" in captured.out

    @patch("spark_optima.notebook.PYSPARK_AVAILABLE", True)
    def test_spark_optima_with_mock_session(self, capsys) -> None:
        from spark_optima.notebook import spark_optima

        mock_spark = MagicMock()
        mock_spark.sparkContext.appName = "TestMagic"
        mock_spark.sparkContext.master = "local[*]"
        mock_spark.version = "3.5.0"
        mock_conf = MagicMock()
        conf_items = [
            ("spark.executor.memory", "4g"),
            ("spark.sql.adaptive.enabled", "true"),
        ]
        mock_conf.getAll.return_value = conf_items
        mock_conf.get.side_effect = lambda k, default="": dict(conf_items).get(str(k), default)
        mock_spark.sparkContext.getConf.return_value = mock_conf

        with patch("spark_optima.notebook._get_active_spark_session", return_value=mock_spark):
            result = spark_optima()
        assert result is None
        captured = capsys.readouterr()
        assert "Spark Optima" in captured.out
        assert "TestMagic" in captured.out


class TestLoadExtension:
    """Test the extension loader functions."""

    def test_load_ipython_extension_raises_without_ipython(self) -> None:
        from spark_optima.notebook import HAS_IPYTHON, load_ipython_extension

        if HAS_IPYTHON:
            pytest.skip("IPython is available; cannot test import error path")
        mock_ipython = MagicMock()
        with pytest.raises(ImportError, match="IPython is required"):
            load_ipython_extension(mock_ipython)

    def test_unload_ipython_extension_no_op(self) -> None:
        from spark_optima.notebook import unload_ipython_extension

        mock_ipython = MagicMock()
        # Should not raise
        unload_ipython_extension(mock_ipython)
