# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Notebook integration — %%spark_optima cell magic.

Usage in a Jupyter / Databricks notebook::

    %load_ext spark_optima.notebook
    %%spark_optima
    # Optional: put Spark code here to analyze
    df = spark.read.parquet("data/")

Or import directly::

    from spark_optima.notebook import load_ipython_extension
    load_ipython_extension(get_ipython())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Optional PySpark import (same pattern used across the package)
try:
    from pyspark.sql import SparkSession

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

# Optional IPython import
try:
    from IPython.core.magic import register_cell_magic
    from IPython.display import HTML, display
    from IPython.terminal.interactiveshell import InteractiveShell

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    register_cell_magic = None  # type: ignore[misc, assignment]
    display = None  # type: ignore[misc, assignment]
    InteractiveShell = None  # type: ignore[misc, assignment]


def _get_active_spark_session(user_ns: dict[str, Any] | None = None) -> Any | None:
    """Find the active SparkSession from user namespace or global session."""
    if not PYSPARK_AVAILABLE:
        return None

    # Try user namespace first
    if user_ns is not None:
        for key in ("spark", "session", "sc"):
            if key in user_ns:
                val = user_ns[key]
                # A SparkSession has .sparkContext; an RDD-based spark has .spark
                if hasattr(val, "sparkContext"):
                    return val
        # If user has a variable that is the SparkContext directly
        if "sc" in user_ns:
            sc = user_ns["sc"]
            if hasattr(sc, "getConf"):
                # Reconstruct session from context if needed
                try:
                    return SparkSession.builder.getOrCreate()
                except Exception:
                    pass

    # Fall back to Spark's own active session tracking
    try:
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _extract_session_config(spark: Any) -> dict[str, Any]:
    """Extract current Spark configuration from a live session."""
    config: dict[str, Any] = {}
    try:
        conf = spark.sparkContext.getConf()
        config = {str(k): str(conf.get(k, "")) for k in conf.getAll()}
    except Exception as exc:
        logger.debug("Failed to read Spark conf: %s", exc)
    return config


def _build_profile(spark: Any) -> dict[str, Any]:
    """Build a profile dictionary from a live SparkSession."""
    profile: dict[str, Any] = {
        "session_available": spark is not None,
        "conf_keys": [],
        "platform_hint": "local",
        "version_hint": "3.5.0",
    }
    if spark is None:
        return profile

    try:
        profile["app_name"] = spark.sparkContext.appName
    except Exception:
        profile["app_name"] = "unknown"

    try:
        profile["master"] = spark.sparkContext.master
        master_str = str(profile.get("master", ""))
        if "local" in master_str:
            profile["platform_hint"] = "local"
        elif master_str.startswith("yarn") or master_str.startswith("k8s://"):
            profile["platform_hint"] = "kubernetes"
    except Exception:
        profile["master"] = "unknown"

    try:
        profile["version_hint"] = spark.version
    except Exception:
        profile["version_hint"] = "3.5.0"

    conf = _extract_session_config(spark)
    profile["conf"] = conf
    profile["conf_keys"] = list(conf.keys())
    profile["conf_count"] = len(conf)
    return profile


def _generate_recommendations(
    profile: dict[str, Any],
    cell_code: str = "",
    platform: str = "local",
) -> list[str]:
    """Generate inline recommendations from session profile and optional cell code."""
    recommendations: list[str] = []
    conf = profile.get("conf", {})

    # Memory recommendations
    memory_params = {
        "spark.executor.memory",
        "spark.driver.memory",
    }
    missing_memory = [p for p in memory_params if p not in conf]
    if missing_memory:
        recommendations.append(f"Set memory parameters explicitly: {', '.join(missing_memory)}")

    # Adaptive Query Execution
    aqe_key = "spark.sql.adaptive.enabled"
    aqe_value = conf.get(aqe_key, "false").lower()
    if aqe_value not in ("true", "1", "yes", "on"):
        recommendations.append("Enable Adaptive Query Execution: spark.conf.set('spark.sql.adaptive.enabled', 'true')")

    # Dynamic allocation
    da_key = "spark.dynamicAllocation.enabled"
    da_value = conf.get(da_key, "false").lower()
    if da_value in ("true", "1", "yes", "on"):
        max_exec_key = "spark.dynamicAllocation.maxExecutors"
        if max_exec_key not in conf:
            recommendations.append("Dynamic allocation enabled — consider setting spark.dynamicAllocation.maxExecutors")
    else:
        recommendations.append(
            "Consider enabling dynamic allocation for variable workloads: "
            "spark.conf.set('spark.dynamicAllocation.enabled', 'true')"
        )

    # Serialization
    serializer_key = "spark.serializer"
    serializer_value = conf.get(serializer_key, "")
    if "kryo" not in serializer_value.lower():
        recommendations.append(
            "Consider Kryo serialization for faster shuffle/serialization: "
            "spark.conf.set('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')"
        )

    # Shuffle compression
    shuffle_compress_key = "spark.shuffle.compress"
    shuffle_compress_value = conf.get(shuffle_compress_key, "true").lower()
    if shuffle_compress_value not in ("true", "1", "yes", "on"):
        recommendations.append("Enable shuffle compression: spark.conf.set('spark.shuffle.compress', 'true')")

    # Code analysis (optional) when cell content is non-empty
    if cell_code.strip():
        try:
            from spark_optima.analysis.recommender import RecommendationEngine

            engine = RecommendationEngine()
            analysis = engine.analyze_source(cell_code, language="auto")
            for rec in analysis.recommendations:
                severity = getattr(rec.smell, "severity", None)
                severity_str = severity.value if severity else "medium"
                recommendations.append(f"[{severity_str.upper()}] {rec.suggestion}")
        except Exception as exc:
            logger.debug("Cell code analysis skipped: %s", exc)
            recommendations.append(f"[INFO] Cell code present; analysis skipped ({exc})")

    # Heuristic recommendations based on profile
    try:
        from spark_optima.core.optimizer import Optimizer
        from spark_optima.platforms.models import ResourceSpec

        version_hint = profile.get("version_hint", "3.5.0")
        optimizer = Optimizer(
            platform=platform,
            spark_version=version_hint,
            optimization_mode="simulation",
        )
        # Use rough resource estimate from session if available
        resources = ResourceSpec(cpu_cores=4, memory_gb=16)
        heuristic_config = optimizer.heuristic_engine.evaluate(
            resources=resources,
            platform=platform,
        )
        # Pick a short subset of high-value recommendations to display inline
        top_params = sorted(
            heuristic_config.items(),
            key=lambda item: (
                0 if item[0] in ("spark.executor.memory", "spark.executor.cores", "spark.sql.adaptive.enabled") else 1,
            ),
        )
        recommendations.append(
            f"Heuristic baseline (platform={platform}, version={version_hint}): {dict(top_params[:6])}"
        )
    except Exception as exc:
        logger.debug("Heuristic baseline unavailable: %s", exc)
        recommendations.append(f"Heuristic baseline unavailable: {exc}")

    return recommendations


def spark_optima(line: str = "", cell: str = "") -> Any:  # noqa: D401
    """%%spark_optima cell magic.

    Profiles the active SparkSession and prints inline configuration
    recommendations. Non-empty cell contents are analyzed for code smells.

    Args:
        line: Cell magic arguments (unused, reserved).
        cell: Cell body (optional Spark code to analyze).

    Returns:
        None — output is printed inline.
    """
    if not PYSPARK_AVAILABLE:
        print("[spark_optima] PySpark is not installed; cannot profile SparkSession.")
        return None

    user_ns: dict[str, Any] | None = None
    spark_session: Any = None
    if HAS_IPYTHON:
        try:
            # Try to get the current interactive shell for user namespace
            ipython_shell = __import__("IPython").get_ipython()  # type: ignore[attr-defined]
            if ipython_shell is not None:
                user_ns = getattr(ipython_shell, "user_ns", None)
        except Exception:
            pass

    spark_session = _get_active_spark_session(user_ns)
    profile = _build_profile(spark_session)

    # Determine platform for recommendations
    master_hint = str(profile.get("master", ""))
    if "local" in master_hint:
        platform_hint = "local"
    elif master_hint.startswith("yarn"):
        platform_hint = "aws_emr"
    elif master_hint.startswith("k8s://") or master_hint.startswith("spark://"):
        platform_hint = "kubernetes"
    else:
        platform_hint = profile.get("platform_hint", "local")

    recommendations = _generate_recommendations(profile, cell, platform=platform_hint)

    # Inline output (works in both standard notebooks and Databricks)
    header = "=" * 60
    print(header)
    print("  Spark Optima — Notebook Profile & Recommendations")
    print(header)
    print(f"App name : {profile.get('app_name', 'unknown')}")
    print(f"Master   : {profile.get('master', 'unknown')}")
    print(f"Version  : {profile.get('version_hint', 'unknown')}")
    print(f"Conf keys: {profile.get('conf_count', 0)}")
    # Show a few representative config values
    conf = profile.get("conf", {})
    for key in sorted(conf):
        if any(sub in key for sub in ("memory", "executor", "driver", "adaptive", "serializer", "shuffle")):
            print(f"  {key} = {conf[key]}")
    print("-" * 60)
    print("Recommendations:")
    for idx, rec in enumerate(recommendations, start=1):
        # Wrap long lines for readability
        lines = rec.split("; ")
        for line in lines:
            # Keep print concise — wrap at ~100 chars
            wrapped = line
            if len(wrapped) > 100:
                wrapped = wrapped[:97] + "..."
            print(f"  {idx}. {wrapped}")
    print(header)

    # If IPython display is available, emit a structured HTML card
    if HAS_IPYTHON and display is not None:
        try:
            html_parts = [
                "<div style='font-family:sans-serif; padding:12px; border:1px solid #ddd; border-radius:8px;'>",
                "<h3 style='margin-top:0;'>Spark Optima</h3>",
                f"<p><strong>App:</strong> {profile.get('app_name', 'unknown')}</p>",
                f"<p><strong>Platform hint:</strong> {platform_hint}</p>",
                f"<p><strong>Config keys:</strong> {profile.get('conf_count', 0)}</p>",
                "<hr/>",
                "<ul>",
            ]
            for rec in recommendations:
                safe_rec = rec.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_parts.append(f"<li>{safe_rec}</li>")
            html_parts.append("</ul></div>")
            display(HTML("".join(html_parts)))
        except Exception:
            # Graceful degradation: HTML display is optional
            pass

    return None


def load_ipython_extension(ipython: Any) -> None:  # noqa: D401
    """Load the %%spark_optima magic into an IPython session.

    Args:
        ipython: The active ``InteractiveShell`` instance.
    """
    if not HAS_IPYTHON:
        raise ImportError(
            "IPython is required to load the spark_optima notebook extension. Install it with: pip install ipython"
        )
    ipython.register_magic_function(spark_optima, "cell", "spark_optima")


def unload_ipython_extension(ipython: Any) -> None:  # noqa: D401
    """Unload the cell magic (no-op for this extension)."""
    pass


# Auto-register when imported inside an active IPython environment
if HAS_IPYTHON:
    try:
        ipython_obj = __import__("IPython").get_ipython()  # type: ignore[attr-defined]
        if ipython_obj is not None and not getattr(ipython_obj, "_spark_optima_registered", False):
            load_ipython_extension(ipython_obj)
            ipython_obj._spark_optima_registered = True  # type: ignore[attr-defined]
    except Exception:
        pass
