# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Output formatters for Spark Optima CLI.

This module provides various output formatters for exporting
optimization results in different formats suitable for different
platforms and use cases.
"""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING, Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from pathlib import Path

    from spark_optima.core.result import OptimizationResult

console = Console()


class ConfigExporter:
    """Configuration exporter for various formats and platforms.

    This class handles exporting optimization results to different
    formats suitable for Local, AWS Glue, Databricks, and Azure Synapse.

    Attributes:
        result: The optimization result to export.
        platform: Target platform for the export.
    """

    def __init__(self, result: OptimizationResult, platform: str) -> None:
        """Initialize the config exporter.

        Args:
            result: Optimization result to export.
            platform: Target platform identifier.
        """
        self.result = result
        self.platform = platform

    def export_json(self, indent: int = 2) -> str:
        """Export configuration as JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON formatted string.
        """
        return json.dumps(self.result.to_dict(), indent=indent)

    def export_yaml(self) -> str:
        """Export configuration as YAML.

        Returns:
            YAML formatted string.
        """
        return yaml.dump(self.result.to_dict())

    def export_spark_submit(self) -> str:
        """Export as spark-submit command arguments.

        Returns:
            String with spark-submit compatible arguments.
        """
        lines = ["# Spark Submit Configuration", ""]

        # Build spark-submit command
        cmd_parts = ["spark-submit"]

        for key, value in self.result.configuration.items():
            cmd_parts.append(f'--conf "{key}={value}"')

        cmd_parts.append("your_script.py")

        lines.append(" \\\n    ".join(cmd_parts))
        lines.append("")

        # Also export as environment variables
        lines.append("# Or as environment variables:")
        for key, value in self.result.configuration.items():
            env_key = key.replace(".", "_").upper()
            lines.append(f'export {env_key}="{value}"')

        return "\n".join(lines)

    def export_databricks_json(self) -> str:
        """Export as Databricks cluster configuration JSON.

        Returns:
            Databricks-compatible JSON configuration.
        """
        config = {
            "cluster_name": "spark-optima-optimized",
            "spark_version": self.result.platform_specific.get(
                "spark_version",
                "3.5.0-x-scala2.12",
            ),
            "node_type_id": self.result.platform_specific.get("cluster_config", {}).get(
                "node_type_id",
                "Standard_DS3_v2",
            ),
            "spark_conf": self.result.configuration,
            "autotermination_minutes": 30,
        }

        # Add autoscaling if dynamic allocation is enabled
        if self.result.configuration.get("spark.dynamicAllocation.enabled") == "true":
            config["autoscale"] = {
                "min_workers": int(
                    self.result.configuration.get("spark.dynamicAllocation.minExecutors", "1"),
                ),
                "max_workers": int(
                    self.result.configuration.get("spark.dynamicAllocation.maxExecutors", "8"),
                ),
            }

        return json.dumps(config, indent=2)

    def export_databricks_cli(self) -> str:
        """Export as Databricks CLI commands.

        Returns:
            Databricks CLI commands for cluster creation.
        """
        config = self.export_databricks_json()

        lines = [
            "# Databricks CLI Cluster Creation Commands",
            "",
            "# First, save this configuration to a file:",
            "# cat > cluster_config.json << 'EOF'",
            config,
            "# EOF",
            "",
            "# Then create the cluster using databricks CLI:",
            "databricks clusters create --json-file cluster_config.json",
            "",
            "# Or create directly:",
            "databricks clusters create --json '",
            config.replace("'", "'\"'\"'"),
            "'",
        ]

        return "\n".join(lines)

    def export_aws_glue(self) -> str:
        """Export as AWS Glue job configuration.

        Returns:
            AWS Glue job configuration in JSON format.
        """
        # Convert Spark config to Glue DefaultArguments format
        default_args = {}
        for key, value in self.result.configuration.items():
            default_args[f"--{key}"] = value

        config = {
            "Name": "spark-optima-job",
            "Description": "Spark job optimized by Spark Optima",
            "Role": "YOUR_GLUE_SERVICE_ROLE",
            "ExecutionProperty": {"MaxConcurrentRuns": 1},
            "Command": {
                "Name": "glueetl",
                "ScriptLocation": "s3://your-bucket/script.py",
                "PythonVersion": "3.9",
            },
            "DefaultArguments": {
                **default_args,
                "--enable-metrics": "true",
                "--enable-continuous-cloudwatch-log": "true",
            },
            "MaxRetries": 0,
            "Timeout": 2880,
            "GlueVersion": self.result.platform_specific.get("glue_version", "4.0"),
            "NumberOfWorkers": int(self.result.configuration.get("spark.executor.instances", "2")),
            "WorkerType": "G.1X",
        }

        return json.dumps(config, indent=2)

    def export_aws_cli(self) -> str:
        """Export as AWS CLI command for Glue job creation.

        Returns:
            AWS CLI command for creating a Glue job.
        """
        config = self.export_aws_glue()

        lines = [
            "# AWS Glue Job Creation Command",
            "",
            "# Using AWS CLI:",
            "aws glue create-job \\",
            f"    --cli-input-json '{config}'",
            "",
            "# Or save to file and use:",
            "# aws glue create-job --cli-input-json file://glue_job.json",
        ]

        return "\n".join(lines)

    def export_azure_synapse(self) -> str:
        """Export as Azure Synapse Spark pool configuration.

        Returns:
            Azure Synapse configuration.
        """
        # Build spark.conf settings
        spark_conf_lines = []
        for key, value in self.result.configuration.items():
            spark_conf_lines.append(f'spark.conf.set("{key}", "{value}")')

        config = {
            "name": "spark-optima-pool",
            "sparkVersion": self.result.platform_specific.get("spark_pool_version", "3.5"),
            "nodeSize": "Small",
            "nodeSizeFamily": "MemoryOptimized",
            "autoScale": {
                "enabled": self.result.configuration.get("spark.dynamicAllocation.enabled", "false") == "true",
                "minNodeCount": int(
                    self.result.configuration.get("spark.dynamicAllocation.minExecutors", "3"),
                ),
                "maxNodeCount": int(
                    self.result.configuration.get("spark.dynamicAllocation.maxExecutors", "10"),
                ),
            },
            "sparkConfigProperties": {
                "configurationType": "Customized",
                "properties": self.result.configuration,
            },
        }

        lines = [
            "# Azure Synapse Spark Pool Configuration",
            "",
            "# Spark Configuration (Python):",
            *spark_conf_lines,
            "",
            "# Pool Configuration (JSON):",
            json.dumps(config, indent=2),
        ]

        return "\n".join(lines)

    def export_environment_variables(self) -> str:
        """Export as shell environment variables.

        Returns:
            Shell export statements for configuration.
        """
        lines = ["# Spark Configuration Environment Variables", ""]

        for key, value in self.result.configuration.items():
            # Convert spark.executor.memory to SPARK_EXECUTOR_MEMORY
            env_key = key.replace(".", "_").upper()
            lines.append(f'export {env_key}="{value}"')

        lines.append("")
        lines.append("# To apply these settings:")
        lines.append("# source spark_config.env")

        return "\n".join(lines)

    def _sorted_config_items(self) -> list[tuple[str, str]]:
        """Return configuration items sorted by key with stringified values.

        Returns:
            List of (key, value) tuples sorted by key, values cast to str.
        """
        return sorted((key, str(value)) for key, value in self.result.configuration.items())

    def export_airflow_dag(self) -> str:
        """Export as a ready-to-edit Airflow DAG Python snippet.

        The generated operator is selected based on the exporter's platform:

        - ``databricks``: DatabricksSubmitRunOperator with the optimized config
          in ``new_cluster.spark_conf``.
        - ``aws_glue``: GlueJobOperator with the optimized config passed as
          ``--conf`` style DefaultArguments.
        - anything else (local/generic): SparkSubmitOperator with ``conf={...}``.

        Returns:
            Python source for an Airflow DAG with TODO placeholders to edit.
        """
        if self.platform == "databricks":
            operator_import = (
                "from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator"
            )
        elif self.platform == "aws_glue":
            operator_import = "from airflow.providers.amazon.aws.operators.glue import GlueJobOperator"
        else:
            operator_import = "from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator"

        lines = [
            "# Airflow DAG generated by Spark Optima.",
            "#",
            "# Placeholders to edit before deploying:",
            '#   - dag_id: unique DAG identifier (default "spark_optima_job")',
            "#   - schedule: cron expression or preset (None = manual trigger only)",
            "#   - application/script path: location of your Spark application",
            "#   - connection ids and cloud settings (marked with TODO)",
            "",
            "from __future__ import annotations",
            "",
            "import pendulum",
            "",
            "from airflow import DAG",
            operator_import,
            "",
            "# Optimized Spark configuration produced by Spark Optima.",
            "SPARK_CONF = {",
        ]
        for key, value in self._sorted_config_items():
            lines.append(f'    "{key}": "{value}",')
        lines.append("}")
        lines.append("")

        if self.platform == "aws_glue":
            # Glue receives Spark settings through the special --conf default argument.
            conf_chain = " --conf ".join(f"{key}={value}" for key, value in self._sorted_config_items())
            lines.extend(
                [
                    "# Glue passes Spark settings through the special --conf default argument.",
                    "DEFAULT_ARGUMENTS = {",
                    f'    "--conf": "{conf_chain}",',
                    '    "--enable-metrics": "true",',
                    "}",
                    "",
                ],
            )

        lines.extend(
            [
                "with DAG(",
                '    dag_id="spark_optima_job",  # TODO: set a unique dag_id',
                '    schedule=None,  # TODO: set a schedule, e.g. "@daily"',
                '    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),',
                "    catchup=False,",
                '    tags=["spark-optima"],',
                ") as dag:",
            ],
        )

        if self.platform == "databricks":
            spark_version = self.result.platform_specific.get("spark_version", "3.5.0-x-scala2.12")
            node_type_id = self.result.platform_specific.get("cluster_config", {}).get(
                "node_type_id",
                "Standard_DS3_v2",
            )
            lines.extend(
                [
                    "    spark_task = DatabricksSubmitRunOperator(",
                    '        task_id="run_spark_job",',
                    '        databricks_conn_id="databricks_default",  # TODO: set your Databricks connection',
                    "        new_cluster={",
                    f'            "spark_version": "{spark_version}",',
                    f'            "node_type_id": "{node_type_id}",  # TODO: adjust node type',
                    '            "num_workers": 2,  # TODO: adjust worker count',
                    '            "spark_conf": SPARK_CONF,',
                    "        },",
                    "        spark_python_task={",
                    '            "python_file": "dbfs:/path/to/your_script.py",  # TODO: set application path',
                    "        },",
                    "    )",
                ],
            )
        elif self.platform == "aws_glue":
            glue_version = self.result.platform_specific.get("glue_version", "4.0")
            lines.extend(
                [
                    "    spark_task = GlueJobOperator(",
                    '        task_id="run_spark_job",',
                    '        job_name="spark-optima-job",  # TODO: set your Glue job name',
                    '        script_location="s3://your-bucket/script.py",  # TODO: set application path',
                    '        iam_role_name="YOUR_GLUE_SERVICE_ROLE",  # TODO: set Glue service role',
                    "        create_job_kwargs={",
                    f'            "GlueVersion": "{glue_version}",',
                    '            "DefaultArguments": DEFAULT_ARGUMENTS,',
                    "        },",
                    "    )",
                ],
            )
        else:
            lines.extend(
                [
                    "    spark_task = SparkSubmitOperator(",
                    '        task_id="run_spark_job",',
                    '        application="/path/to/your_script.py",  # TODO: set application path',
                    '        conn_id="spark_default",  # TODO: set your Spark connection',
                    "        conf=SPARK_CONF,",
                    "    )",
                ],
            )

        return "\n".join(lines)

    def export_kubernetes_configmap(self) -> str:
        """Export as a Kubernetes ConfigMap YAML manifest.

        The ``data`` section carries the optimized configuration rendered as
        ``spark-defaults.conf`` content (one ``key value`` pair per line),
        suitable for mounting into spark-on-k8s driver/executor pods.

        Returns:
            YAML manifest for a ConfigMap named ``spark-optima-config``.
        """
        lines = [
            "# Kubernetes ConfigMap generated by Spark Optima.",
            "# Mount the spark-defaults.conf key into spark-on-k8s pods (e.g. at",
            "# /opt/spark/conf) or reference it from your SparkApplication spec.",
            "apiVersion: v1",
            "kind: ConfigMap",
            "metadata:",
            "  name: spark-optima-config",
            "data:",
            "  spark-defaults.conf: |",
        ]
        for key, value in self._sorted_config_items():
            lines.append(f"    {key} {value}")

        return "\n".join(lines)

    def export_aws_emr(self) -> str:
        """Export as AWS EMR cluster configurations JSON.

        Produces a JSON list with a single ``spark-defaults`` classification
        entry whose ``Properties`` carry the optimized configuration (all
        values as strings). Usage::

            aws emr create-cluster --configurations file://emr_config.json ...

        Returns:
            JSON string compatible with ``aws emr create-cluster --configurations``.
        """
        configurations = [
            {
                "Classification": "spark-defaults",
                "Properties": dict(self._sorted_config_items()),
            },
        ]
        return json.dumps(configurations, indent=2, sort_keys=True)

    def _pareto_points(self) -> list[dict[str, Any]]:
        """Return the Pareto frontier points stored in the result metadata.

        Returns:
            List of frontier point dictionaries, each with ``trial_number``,
            ``objective_values``, and ``configuration`` keys.

        Raises:
            ValueError: If the result has no Pareto frontier (single-objective run).
        """
        frontier = (self.result.metadata or {}).get("pareto_frontier")
        if not frontier:
            raise ValueError(
                "Result has no Pareto frontier. Rerun optimization with multiple "
                "--objective flags (e.g. --objective minimize_time --objective minimize_cost).",
            )
        return list(frontier)

    def _pareto_objective_names(self, points: list[dict[str, Any]]) -> list[str]:
        """Return objective names for the Pareto frontier in deterministic order.

        Prefers the objective order recorded in result metadata (the order the
        user requested); falls back to the sorted union of objective names
        found across all frontier points.

        Args:
            points: Pareto frontier point dictionaries.

        Returns:
            Ordered list of objective names.
        """
        names = (self.result.metadata or {}).get("objectives")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        return sorted({key for point in points for key in (point.get("objective_values") or {})})

    def export_pareto_json(self, indent: int = 2) -> str:
        """Export the Pareto frontier of a multi-objective run as JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON object with ``objectives``, ``n_points``, and ``points`` keys.

        Raises:
            ValueError: If the result has no Pareto frontier.
        """
        points = self._pareto_points()
        payload = {
            "objectives": self._pareto_objective_names(points),
            "n_points": len(points),
            "points": points,
        }
        return json.dumps(payload, indent=indent)

    def export_pareto_csv(self) -> str:
        """Export the Pareto frontier of a multi-objective run as CSV.

        Each frontier point becomes one row. Column order is deterministic:
        ``trial`` first, then one column per objective (in the order recorded
        in the result metadata), then the sorted union of configuration
        parameter names across all points. Missing values are left empty.
        Rows are sorted by trial number.

        Returns:
            CSV document as a string.

        Raises:
            ValueError: If the result has no Pareto frontier.
        """
        points = self._pareto_points()
        objective_names = self._pareto_objective_names(points)
        param_names = sorted({key for point in points for key in (point.get("configuration") or {})})

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["trial", *objective_names, *param_names])
        for point in sorted(points, key=lambda p: p.get("trial_number", -1)):
            objective_values = point.get("objective_values") or {}
            configuration = point.get("configuration") or {}
            row: list[Any] = [point.get("trial_number", "")]
            row.extend(objective_values.get(name, "") for name in objective_names)
            row.extend(configuration.get(name, "") for name in param_names)
            writer.writerow(row)
        return buffer.getvalue()

    def export_properties_file(self) -> str:
        """Export as Spark properties file.

        Returns:
            spark-defaults.conf compatible format.
        """
        lines = ["# Spark Configuration - Generated by Spark Optima", ""]

        for key, value in self.result.configuration.items():
            lines.append(f"{key} {value}")

        lines.append("")
        lines.append("# Save this to $SPARK_HOME/conf/spark-defaults.conf")

        return "\n".join(lines)

    def save_to_file(self, filepath: Path, format_type: str = "json") -> None:
        """Save configuration to a file.

        Args:
            filepath: Path to save the file.
            format_type: Export format type.

        Raises:
            ValueError: If format_type is not supported.
        """
        formatters = {
            "json": self.export_json,
            "yaml": self.export_yaml,
            "spark-submit": self.export_spark_submit,
            "databricks-json": self.export_databricks_json,
            "databricks-cli": self.export_databricks_cli,
            "aws-glue": self.export_aws_glue,
            "aws-cli": self.export_aws_cli,
            "azure-synapse": self.export_azure_synapse,
            "env": self.export_environment_variables,
            "properties": self.export_properties_file,
            "airflow": self.export_airflow_dag,
            "kubernetes": self.export_kubernetes_configmap,
            "emr": self.export_aws_emr,
            "pareto-json": self.export_pareto_json,
            "pareto-csv": self.export_pareto_csv,
        }

        if format_type not in formatters:
            raise ValueError(
                f"Unsupported format: {format_type}. Supported formats: {list(formatters.keys())}",
            )

        content = formatters[format_type]()  # type: ignore[operator]
        filepath.write_text(content)
        console.print(f"[green]✓[/green] Configuration saved to [cyan]{filepath}[/cyan]")


def display_results_table(result: OptimizationResult) -> None:
    """Display optimization results in a formatted table.

    Args:
        result: Optimization result to display.
    """
    # Main results panel
    console.print(
        Panel.fit(
            "[bold green]✅ Optimization Complete![/bold green]\n"
            f"[dim]Estimated time: {result.estimated_time_minutes:.1f} minutes | "
            f"Confidence: {result.confidence_score:.1%}[/dim]",
            border_style="green",
        ),
    )
    console.print()

    # Key metrics table
    metrics = Table(title="Optimization Results", show_header=False)
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Value", style="green")

    metrics.add_row("Configuration Keys", str(len(result.configuration)))
    metrics.add_row("Estimated Time", f"{result.estimated_time_minutes:.1f} min")
    metrics.add_row("Confidence Score", f"{result.confidence_score:.1%}")
    metrics.add_row("Code Suggestions", str(len(result.code_suggestions)))

    console.print(metrics)
    console.print()

    # Configuration preview
    if result.configuration:
        console.print("[bold]Key Configuration Settings:[/bold]")
        config_table = Table(show_header=False)
        config_table.add_column("Key", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="yellow")

        # Show important configs first
        priority_keys = [
            "spark.executor.memory",
            "spark.executor.cores",
            "spark.driver.memory",
            "spark.sql.shuffle.partitions",
            "spark.default.parallelism",
            "spark.sql.adaptive.enabled",
            "spark.dynamicAllocation.enabled",
        ]

        shown = 0
        for key in priority_keys:
            if key in result.configuration:
                config_table.add_row(key, str(result.configuration[key]))
                shown += 1

        # Fill with other configs
        for key, value in result.configuration.items():
            if key not in priority_keys and shown < 15:
                config_table.add_row(key, str(value))
                shown += 1

        if len(result.configuration) > 15:
            config_table.add_row("...", f"(+{len(result.configuration) - 15} more)")

        console.print(config_table)
        console.print()

    # Code suggestions
    if result.code_suggestions:
        console.print("[bold]Code Improvement Suggestions:[/bold]")

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_suggestions = sorted(
            result.code_suggestions,
            key=lambda x: severity_order.get(x.severity, 4),
        )

        for i, suggestion in enumerate(sorted_suggestions[:5], 1):
            severity_color = {
                "critical": "red",
                "high": "orange3",
                "medium": "yellow",
                "low": "blue",
            }.get(suggestion.severity, "white")

            console.print(
                f"\n  [{i}] Line {suggestion.line_number}: "
                f"[{severity_color}]{suggestion.severity.upper()}[/{severity_color}]",
            )
            console.print(f"      [dim]{suggestion.description}[/dim]")
            console.print(f"      [green]→ {suggestion.suggestion}[/green]")

        if len(result.code_suggestions) > 5:
            console.print(f"\n  ... and {len(result.code_suggestions) - 5} more suggestions")

        console.print()

    # Export hints
    console.print("[dim]Export options:[/dim]")
    console.print("  --output json    # Full JSON export")
    console.print("  --output yaml    # YAML format")
    console.print("  Use 'export' command for platform-specific formats")


def print_platform_export_help(platform: str) -> None:
    """Print help for platform-specific exports.

    Args:
        platform: Target platform.
    """
    help_texts = {
        "local": """
Local Mode Export Options:
  --format spark-submit    # spark-submit command arguments
  --format env             # Environment variables
  --format properties      # spark-defaults.conf format
  --format airflow         # Airflow DAG (SparkSubmitOperator)
  --format kubernetes      # Kubernetes ConfigMap (spark-on-k8s)
  --format emr             # AWS EMR --configurations JSON
        """,
        "databricks": """
Databricks Export Options:
  --format databricks-json  # Cluster configuration JSON
  --format databricks-cli   # Databricks CLI commands
  --format airflow          # Airflow DAG (DatabricksSubmitRunOperator)
  --format json             # Generic JSON export
        """,
        "aws_glue": """
AWS Glue Export Options:
  --format aws-glue    # Glue job configuration JSON
  --format aws-cli     # AWS CLI command
  --format airflow     # Airflow DAG (GlueJobOperator)
  --format emr         # AWS EMR --configurations JSON
  --format json        # Generic JSON export
        """,
        "azure_synapse": """
Azure Synapse Export Options:
  --format azure-synapse   # Synapse Spark pool config
  --format json            # Generic JSON export
        """,
    }

    text = help_texts.get(platform, "Use --format json or --format yaml")
    console.print(Panel(text.strip(), title="Export Formats", border_style="blue"))
