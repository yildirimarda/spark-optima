# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Reusable core module for validate and import operations.

This module extracts the validation and import logic previously embedded
in `cli/main.py` so it can be reused by both the CLI and the REST API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from spark_optima.core.config_engine.loader import VersionLoader
from spark_optima.core.config_engine.validator import ConfigValidator
from spark_optima.core.optimizer import Optimizer

# Matches "key value", "key=value", and "key = value" properties lines
_PROPERTIES_LINE_RE = re.compile(r"^(\S+?)(?:\s*=\s*|\s+)(.+)$")


def parse_config_file(config_path: Path | str) -> dict[str, Any]:
    """Parse a Spark configuration file in properties or JSON format.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary of parameter names to raw values.

    Raises:
        ValueError: If the file cannot be read or parsed.
    """
    config_path_obj = Path(config_path)
    try:
        content = config_path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Error reading config file: {exc}") from exc

    if config_path_obj.suffix.lower() == ".json" or content.lstrip().startswith("{"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Error parsing JSON config: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{config_path_obj} must contain a JSON object")
        return {str(key): value for key, value in data.items()}

    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROPERTIES_LINE_RE.match(line)
        if match is None:
            raise ValueError(f"cannot parse line {line_number} of {config_path_obj}: {raw_line!r}")
        config[match.group(1)] = match.group(2).strip()
    return config


def validation_issue(severity: str, param: str, message: str, check: str) -> dict[str, str]:
    """Build a single validation issue record."""
    return {"severity": severity, "param": param, "message": message, "check": check}


def config_bool(value: Any) -> bool | None:
    """Interpret a raw config value as a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def config_int(value: Any) -> int | None:
    """Interpret a raw config value as an integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def memory_to_gb(value: Any, validator: ConfigValidator) -> float | None:
    """Convert a Spark byte-size value to gigabytes."""
    from spark_optima.core.config_engine.models import ParameterType

    if value is None:
        return None
    try:
        bytes_value = validator.normalize_value(str(value).strip(), ParameterType.BYTES)
    except (ValueError, TypeError):
        return None
    if not isinstance(bytes_value, int) or bytes_value <= 0:
        return None
    return bytes_value / float(1024**3)


def coerce_typed_value(
    value: Any,
    param_type: Any,
    validator: ConfigValidator,
) -> tuple[Any, str | None]:
    """Coerce a raw config value into the parameter's type."""
    from spark_optima.core.config_engine.models import ParameterType

    if not isinstance(value, str):
        return value, None

    text = value.strip()
    if param_type == ParameterType.BOOLEAN:
        boolean = config_bool(text)
        if boolean is None:
            return value, f"expected a boolean (true/false), got '{value}'"
        return boolean, None
    if param_type == ParameterType.INTEGER:
        try:
            return int(text), None
        except ValueError:
            return value, f"expected an integer, got '{value}'"
    if param_type == ParameterType.FLOAT:
        try:
            return float(text), None
        except ValueError:
            return value, f"expected a number, got '{value}'"
    if param_type == ParameterType.BYTES and not validator.is_valid_bytes(text):
        return value, f"expected a byte size like '4g' or '512m', got '{value}'"
    if param_type == ParameterType.DURATION and not validator.is_valid_duration(text):
        return value, f"expected a duration like '60s' or '5m', got '{value}'"
    return text, None


def collect_db_issues(
    config: dict[str, Any],
    config_set: Any,
    validator: ConfigValidator,
) -> list[dict[str, str]]:
    """Check a config against the Spark parameter database."""
    issues: list[dict[str, str]] = []
    for param_name in sorted(config):
        raw_value = config[param_name]
        db_param = config_set.parameters.get(param_name)
        if db_param is None:
            issues.append(
                validation_issue(
                    "warning",
                    param_name,
                    f"unknown parameter (not in the Spark {config_set.version} parameter database)",
                    "unknown_parameter",
                ),
            )
            continue

        if db_param.deprecated_in and db_param.is_deprecated_in(config_set.version):
            message = f"deprecated since Spark {db_param.deprecated_in}"
            if db_param.alternatives:
                message += f"; use {', '.join(db_param.alternatives)} instead"
            issues.append(validation_issue("warning", param_name, message, "deprecated_parameter"))

        coerced, type_error = coerce_typed_value(raw_value, db_param.param_type, validator)
        if type_error is not None:
            issues.append(validation_issue("error", param_name, type_error, "invalid_value"))
            continue

        if not validator.validate(db_param, coerced):
            for error in validator.get_errors():
                issues.append(validation_issue("error", param_name, str(error), "invalid_value"))
    return issues


def collect_platform_issues(
    config: dict[str, Any],
    platform_name: str,
    validator: ConfigValidator,
) -> list[dict[str, str]]:
    """Check executor memory/cores against a platform's resource constraints."""
    from spark_optima.platforms import get_platform

    try:
        platform_obj = get_platform(platform_name)
    except (ValueError, RuntimeError, ImportError) as exc:
        raise ValueError(str(exc)) from exc

    constraints = platform_obj.constraints
    issues: list[dict[str, str]] = []

    executor_memory_gb = memory_to_gb(config.get("spark.executor.memory"), validator)
    if executor_memory_gb is not None:
        if executor_memory_gb > constraints.max_memory_gb:
            issues.append(
                validation_issue(
                    "error",
                    "spark.executor.memory",
                    f"{executor_memory_gb:.1f} GB exceeds the {platform_obj.name} maximum of "
                    f"{constraints.max_memory_gb:.0f} GB per worker",
                    "platform_constraint",
                ),
            )
        elif executor_memory_gb < constraints.min_memory_gb:
            issues.append(
                validation_issue(
                    "error",
                    "spark.executor.memory",
                    f"{executor_memory_gb:.1f} GB is below the {platform_obj.name} minimum of "
                    f"{constraints.min_memory_gb:.0f} GB per worker",
                    "platform_constraint",
                ),
            )

    executor_cores = config_int(config.get("spark.executor.cores"))
    if executor_cores is not None:
        if executor_cores > constraints.max_cores:
            issues.append(
                validation_issue(
                    "error",
                    "spark.executor.cores",
                    f"{executor_cores} cores exceeds the {platform_obj.name} maximum of "
                    f"{constraints.max_cores} cores per worker",
                    "platform_constraint",
                ),
            )
        elif executor_cores < constraints.min_cores:
            issues.append(
                validation_issue(
                    "error",
                    "spark.executor.cores",
                    f"{executor_cores} cores is below the {platform_obj.name} minimum of "
                    f"{constraints.min_cores} cores per worker",
                    "platform_constraint",
                ),
            )

    return issues


def collect_anti_pattern_issues(
    config: dict[str, Any],
    resolved_version: str,
    loader: VersionLoader,
    validator: ConfigValidator,
) -> list[dict[str, str]]:
    """Check a config against a curated list of Spark anti-patterns."""
    issues: list[dict[str, str]] = []

    # Driver memory larger than executor memory
    driver_gb = memory_to_gb(config.get("spark.driver.memory"), validator)
    executor_gb = memory_to_gb(config.get("spark.executor.memory"), validator)
    if driver_gb is not None and executor_gb is not None and driver_gb > executor_gb:
        issues.append(
            validation_issue(
                "warning",
                "spark.driver.memory",
                f"driver memory ({driver_gb:.1f} GB) is larger than executor memory ({executor_gb:.1f} GB); "
                "the driver rarely needs more memory than the executors",
                "driver_memory_exceeds_executor",
            ),
        )

    # Dynamic allocation misconfigurations
    if config_bool(config.get("spark.dynamicAllocation.enabled")):
        min_executors = config_int(config.get("spark.dynamicAllocation.minExecutors"))
        max_executors = config_int(config.get("spark.dynamicAllocation.maxExecutors"))
        if min_executors is not None and max_executors is not None:
            if max_executors < min_executors:
                issues.append(
                    validation_issue(
                        "error",
                        "spark.dynamicAllocation.maxExecutors",
                        f"maxExecutors ({max_executors}) is less than minExecutors ({min_executors})",
                        "dynamic_allocation_bounds",
                    ),
                )
            elif max_executors == min_executors:
                issues.append(
                    validation_issue(
                        "warning",
                        "spark.dynamicAllocation.maxExecutors",
                        f"maxExecutors equals minExecutors ({max_executors}); dynamic allocation cannot scale",
                        "dynamic_allocation_bounds",
                    ),
                )

        shuffle_service = config_bool(config.get("spark.shuffle.service.enabled"))
        shuffle_tracking = config_bool(config.get("spark.dynamicAllocation.shuffleTracking.enabled"))
        if not shuffle_service and not shuffle_tracking:
            issues.append(
                validation_issue(
                    "warning",
                    "spark.dynamicAllocation.enabled",
                    "dynamic allocation needs spark.shuffle.service.enabled=true or "
                    "spark.dynamicAllocation.shuffleTracking.enabled=true to release executors safely",
                    "dynamic_allocation_shuffle",
                ),
            )

    # Java serializer combined with Kryo-dependent settings
    serializer = str(config.get("spark.serializer", "") or "").strip()
    if serializer.endswith("JavaSerializer"):
        kryo_settings = sorted(name for name in config if name.startswith("spark.kryo"))
        if kryo_settings:
            issues.append(
                validation_issue(
                    "warning",
                    "spark.serializer",
                    f"Java serializer is configured but Kryo settings are present "
                    f"({', '.join(kryo_settings)}); they will have no effect",
                    "serializer_mismatch",
                ),
            )

    # AQE disabled on Spark >= 3.2
    aqe_enabled = config_bool(config.get("spark.sql.adaptive.enabled"))
    if aqe_enabled is False and loader.is_at_least(resolved_version, "3.2.0"):
        issues.append(
            validation_issue(
                "warning",
                "spark.sql.adaptive.enabled",
                f"Adaptive Query Execution is disabled; on Spark {resolved_version} (>= 3.2) "
                "AQE is mature and usually improves performance",
                "aqe_disabled",
            ),
        )

    return issues


def diff_configs(
    current: dict[str, Any],
    recommended: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Compute the difference between a current and a recommended config."""
    shared = set(current) & set(recommended)
    changed = sorted(key for key in shared if str(current[key]).strip() != str(recommended[key]).strip())
    only_in_current = sorted(set(current) - set(recommended))
    only_in_recommended = sorted(set(recommended) - set(current))
    return changed, only_in_current, only_in_recommended


def validate_config(
    config_path: Path | str,
    platform: str = "",
    spark_version: str = "3.5.0",
) -> dict[str, Any]:
    """Run the full validation pipeline and return a structured result."""
    config_path_obj = Path(config_path)
    if not config_path_obj.is_file():
        raise ValueError(f"config file not found: {config_path_obj}")

    config = parse_config_file(config_path_obj)
    loader = VersionLoader()
    config_set = loader.load(spark_version)
    if config_set is None:
        raise ValueError(f"no parameter database available for Spark {spark_version}")

    validator = ConfigValidator()
    issues = collect_db_issues(config, config_set, validator)
    if platform:
        issues.extend(collect_platform_issues(config, platform, validator))
    issues.extend(collect_anti_pattern_issues(config, config_set.version, loader, validator))

    severity_rank = {"error": 0, "warning": 1}
    issues.sort(key=lambda issue: (severity_rank.get(issue["severity"], 2), issue["param"]))
    error_issues = [issue for issue in issues if issue["severity"] == "error"]
    warning_issues = [issue for issue in issues if issue["severity"] == "warning"]

    return {
        "config_file": str(config_path_obj),
        "spark_version": config_set.version,
        "platform": platform or None,
        "parameter_count": len(config),
        "issues": issues,
        "error_count": len(error_issues),
        "warning_count": len(warning_issues),
        "valid": not error_issues,
    }


def import_config(
    config_path: Path | str,
    code_path: Path | str,
    platform: str = "local",
    spark_version: str = "3.5.0",
    data_size_gb: float = 0.0,
    bayesian_trials: int = 50,
) -> dict[str, Any]:
    """Import an existing Spark config, run optimization, and return the diff."""
    config_path_obj = Path(config_path)
    if not config_path_obj.is_file():
        raise ValueError(f"config file not found: {config_path_obj}")

    current = parse_config_file(config_path_obj)

    optimizer = Optimizer(
        platform=platform,
        spark_version=spark_version,
        optimization_mode="simulation",
    )

    data_profile = {"size_gb": data_size_gb} if data_size_gb else None

    result = optimizer.optimize(
        code_path=Path(code_path),
        data_profile=data_profile,
        use_bayesian=True,
        bayesian_trials=bayesian_trials,
    )

    recommended: dict[str, Any] = result.configuration or {}
    changed, only_in_current, only_in_recommended = diff_configs(current, recommended)

    return {
        "current": current,
        "recommended": recommended,
        "diff": {
            "changed": {key: {"current": current[key], "recommended": recommended[key]} for key in changed},
            "only_in_current": {key: current[key] for key in only_in_current},
            "only_in_recommended": {key: recommended[key] for key in only_in_recommended},
        },
        "estimated_time_minutes": result.estimated_time_minutes,
    }
