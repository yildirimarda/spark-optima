# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Output formatters for Spark Optima CLI.

This module provides various output formatters for exporting
optimization results in different formats suitable for different
platforms and use cases.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
                "enabled": self.result.configuration.get("spark.dynamicAllocation.enabled", "false")
                == "true",
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
        }

        if format_type not in formatters:
            raise ValueError(
                f"Unsupported format: {format_type}. "
                f"Supported formats: {list(formatters.keys())}",
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
        """,
        "databricks": """
Databricks Export Options:
  --format databricks-json  # Cluster configuration JSON
  --format databricks-cli   # Databricks CLI commands
  --format json             # Generic JSON export
        """,
        "aws_glue": """
AWS Glue Export Options:
  --format aws-glue    # Glue job configuration JSON
  --format aws-cli     # AWS CLI command
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
