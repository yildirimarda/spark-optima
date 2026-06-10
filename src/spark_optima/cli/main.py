# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Main CLI entry point for Spark Optima.

This module provides the command-line interface for Spark Optima,
allowing users to run configuration optimization from the terminal.
"""

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spark_optima import __version__
from spark_optima.core.optimizer import Optimizer

logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="spark-optima",
    help="Intelligent Apache Spark configuration optimization tool",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Spark Optima[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Spark Optima - Intelligent Spark Configuration Optimization."""


@app.command()
def optimize(
    code_file: str = "",
    code_path: str = typer.Option(
        "",
        "--code-path",
        "-c",
        help="Path to Spark application code file (alternative to positional arg)",
    ),
    platform: str = typer.Option(
        "local",
        "--platform",
        "-p",
        help="Target platform (local, databricks, aws_glue, aws_emr, azure_synapse)",
    ),
    spark_version: str = typer.Option(
        "3.5.0",
        "--spark-version",
        "-s",
        help="Spark version to optimize for",
    ),
    data_size_gb: float = typer.Option(
        0.0,
        "--data-size",
        "-d",
        help="Data size in GB",
    ),
    data_format: str = typer.Option(
        "parquet",
        "--data-format",
        "-f",
        help="Data format (parquet, delta, json, csv, orc)",
    ),
    max_memory_gb: float = typer.Option(
        0.0,
        "--max-memory",
        "-m",
        help="Maximum memory constraint in GB",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json, yaml)",
    ),
    mode: str = typer.Option(
        "simulation",
        "--mode",
        help="Optimization mode (simulation or execution)",
    ),
    bayesian_trials: int = typer.Option(
        50,
        "--bayesian-trials",
        help="Number of Bayesian optimization trials",
    ),
) -> None:
    """Optimize Spark configuration for the given code.

    This command analyzes the provided Spark code and data characteristics
    to find the optimal configuration using hybrid optimization.

    Example:
        $ spark-optima optimize -c ./my_job.py -p databricks -d 100

    """
    # Convert empty strings to None for compatibility
    if not code_file:
        code_file = None  # type: ignore[assignment]
    if not code_path:
        code_path = None  # type: ignore[assignment]
    if not data_size_gb:
        data_size_gb = None  # type: ignore[assignment]
    if not max_memory_gb:
        max_memory_gb = None  # type: ignore[assignment]

    # Use positional arg if provided, otherwise use option
    actual_code_path = code_file if code_file is not None else code_path

    # Check code_path is provided
    if actual_code_path is None:
        console.print("[bold red]Error:[/bold red] code file path is required")
        raise typer.Exit(1)

    # Convert to Path
    code_path_obj = Path(actual_code_path)

    # Display header
    console.print(
        Panel.fit(
            "[bold blue]🔥 Spark Optima[/bold blue]\n[dim]Intelligent Spark Configuration Optimization[/dim]",
            border_style="blue",
        ),
    )

    # Show input summary
    input_table = Table(title="Input Parameters", show_header=False)
    input_table.add_column("Parameter", style="cyan")
    input_table.add_column("Value", style="green")
    input_table.add_row("Code Path", str(code_path_obj))
    input_table.add_row("Platform", platform)
    input_table.add_row("Spark Version", spark_version)
    input_table.add_row("Data Size", f"{data_size_gb} GB" if data_size_gb else "Auto-detect")
    input_table.add_row("Data Format", data_format)
    input_table.add_row("Max Memory", f"{max_memory_gb} GB" if max_memory_gb else "Unlimited")
    input_table.add_row("Mode", mode)
    console.print(input_table)
    console.print()

    # Initialize optimizer
    with console.status("[bold green]Initializing optimizer..."):
        try:
            optimizer = Optimizer(
                platform=platform,
                spark_version=spark_version,
                optimization_mode=mode,
            )
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1) from e

    # Prepare data profile
    data_profile: dict[str, Any] = {"format": data_format}
    if data_size_gb:
        data_profile["size_gb"] = data_size_gb

    # Prepare resource constraints
    resource_constraints = {}
    if max_memory_gb:
        resource_constraints["max_memory_gb"] = max_memory_gb

    # Run optimization
    with console.status("[bold green]Running optimization..."):
        try:
            result = optimizer.optimize(
                code_path=code_path_obj,
                data_profile=data_profile if data_profile else None,
                resource_constraints=resource_constraints if resource_constraints else None,
                use_bayesian=True,
                bayesian_trials=bayesian_trials,
            )
        except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
            console.print(f"[bold red]Optimization failed:[/bold red] {e}")
            raise typer.Exit(1) from e

    # Display results using formatters
    from spark_optima.cli.formatters import (
        ConfigExporter,
        display_results_table,
    )

    if output_format == "json":
        exporter = ConfigExporter(result, platform)
        typer.echo(exporter.export_json())
    elif output_format == "yaml":
        exporter = ConfigExporter(result, platform)
        typer.echo(exporter.export_yaml())
    else:  # table
        display_results_table(result)

    # Auto-save to optimization history (best effort — never break the optimize flow)
    try:
        from spark_optima.core.history import OptimizationHistory

        with OptimizationHistory() as history_store:
            entry_id = history_store.save(
                result.to_dict(),
                platform=platform,
                spark_version=spark_version,
                mode=mode,
                code_path=str(code_path_obj),
            )
        console.print(
            f"\n[dim]Saved to history (id={entry_id}). View with: spark-optima history --show {entry_id}[/dim]",
        )
    except Exception as e:  # history persistence must never break optimization
        logger.warning("Failed to save optimization result to history: %s", e)

    # Save hint
    console.print("\n[dim]Tip: Save results with --output json > result.json[/dim]")
    console.print(
        "[dim]      Then export with: spark-optima export -r result.json -f <format>[/dim]",
    )


@app.command()
def analyze(
    code_file: str = "",
    code_path: str = typer.Option(
        "",
        "--code-path",
        "-c",
        help="Path to Spark application code file (alternative to positional arg)",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """Analyze Spark code for optimization opportunities.

    This command analyzes the provided Spark code and identifies
    potential issues and optimization opportunities.

    Example:
        $ spark-optima analyze -c ./my_job.py

    """
    # Convert empty strings to None for compatibility
    if not code_file:
        code_file = None  # type: ignore[assignment]
    if not code_path:
        code_path = None  # type: ignore[assignment]

    # Use positional arg if provided, otherwise use option
    actual_code_path = code_file if code_file is not None else code_path

    # Check code_path is provided
    if actual_code_path is None:
        console.print("[bold red]Error:[/bold red] code file path is required")
        raise typer.Exit(1)

    # Convert to Path
    code_path_obj = Path(actual_code_path)

    # Display header
    console.print(
        Panel.fit(
            "[bold blue]🔍 Spark Code Analysis[/bold blue]\n[dim]Identify optimization opportunities[/dim]",
            border_style="blue",
        ),
    )

    # Run analysis
    from spark_optima.analysis.recommender import analyze_code

    with console.status("[bold green]Analyzing code..."):
        try:
            source_code = code_path_obj.read_text(encoding="utf-8")
            result = analyze_code(source_code)
        except (OSError, RuntimeError, ValueError, AttributeError, TypeError) as e:
            console.print(f"[bold red]Analysis failed:[/bold red] {e}")
            raise typer.Exit(1) from e

    # Display results
    if output_format == "json":
        import json

        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        # Table format
        table = Table(title="Code Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Operations Count", str(len(result.operations)))
        table.add_row("Code Smells", str(len(result.smells)))
        table.add_row("Recommendations", str(len(result.recommendations)))
        console.print(table)

        if result.smells:
            smell_table = Table(title="Code Smells")
            smell_table.add_column("Line", style="cyan")
            smell_table.add_column("Type", style="yellow")
            smell_table.add_column("Severity", style="red")
            for smell in result.smells:
                line_number = smell.location.line if smell.location is not None else "Unknown"
                smell_table.add_row(str(line_number), smell.smell_type, str(smell.severity))
            console.print(smell_table)


# Create platforms subgroup
platforms_app = typer.Typer(help="Platform management commands")
app.add_typer(platforms_app, name="platforms")


@platforms_app.callback()
def platforms_callback() -> None:
    """Platform management commands."""


@platforms_app.command("list")
def platforms_list() -> None:
    """List supported platforms.

    Example:
        $ spark-optima platforms list
    """
    from spark_optima.platforms import list_platforms

    # Display header
    console.print(
        Panel.fit(
            "[bold blue]🖥️ Supported Platforms[/bold blue]\n[dim]Available execution platforms[/dim]",
            border_style="blue",
        ),
    )

    platforms_list_data = list_platforms()

    table = Table(title="Supported Platforms")
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Description", style="dim")

    for platform_name in platforms_list_data:
        from spark_optima.platforms import get_platform

        try:
            platform = get_platform(platform_name)
            table.add_row(platform.name, platform.display_name, platform.description or "")
        except (RuntimeError, ValueError, AttributeError, TypeError):
            table.add_row(platform_name, platform_name, "")

    console.print(table)


@app.command()
def platforms(
    _list_all: bool = typer.Option(
        True,
        "--list",
        "-l",
        help="List all supported platforms",
    ),
) -> None:
    """List supported platforms and their capabilities (legacy command).

    Example:
        $ spark-optima platforms

    """
    platforms_list()


@app.command()
def wizard() -> None:
    """Launch interactive configuration wizard.

    Guides you through the optimization process step by step,
    collecting all necessary information interactively.

    Example:
        $ spark-optima wizard

    """
    from spark_optima.cli.wizard import run_wizard
    from spark_optima.platforms.models import ResourceSpec

    # Run the wizard
    config = run_wizard()

    if not config:
        return

    # Extract configuration
    code_path_obj = Path(config["code_path"])
    platform = config["platform"]
    spark_version = config["spark_version"]
    output_format = config.get("output_format", "table")

    # Create resource spec
    resources = ResourceSpec(
        cpu_cores=config["resources"]["cpu_cores"],
        memory_gb=config["resources"]["memory_gb"],
    )

    # Prepare data profile
    data_profile = config.get("data_profile", {})

    # Prepare constraints
    constraints = config.get("constraints", {})

    # Display start message
    console.print("\n[bold green]Starting optimization...[/bold green]\n")

    # Initialize optimizer
    with console.status("[bold green]Initializing optimizer..."):
        try:
            optimizer = Optimizer(
                platform=platform,
                spark_version=spark_version,
                optimization_mode="simulation",
            )
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1) from e

    # Run optimization
    with console.status("[bold green]Running optimization..."):
        try:
            result = optimizer.optimize(
                code_path=code_path_obj,
                resources=resources,
                data_profile=data_profile if data_profile else None,
                resource_constraints=constraints if constraints else None,
                use_bayesian=config.get("use_bayesian", True),
                bayesian_trials=config.get("bayesian_trials", 50),
            )
        except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
            console.print(f"[bold red]Optimization failed:[/bold red] {e}")
            raise typer.Exit(1) from e

    # Display results using formatters
    from spark_optima.cli.formatters import (
        ConfigExporter,
        display_results_table,
    )

    if output_format in ("json", "yaml"):
        exporter = ConfigExporter(result, platform)
        if output_format == "json":
            typer.echo(exporter.export_json())
        else:
            typer.echo(exporter.export_yaml())
    else:
        display_results_table(result)


@app.command()
def export(
    result_file: str = typer.Option(
        "",
        "--result-file",
        "-r",
        help="Path to optimization result JSON file",
        exists=True,
        readable=True,
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help=(
            "Export format (json, yaml, databricks-json, aws-glue, "
            "azure-synapse, env, properties, airflow, kubernetes, emr)"
        ),
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    ),
) -> None:
    """Export optimization result to various formats.

    Converts a previously saved optimization result to platform-specific
    configuration formats for easy deployment.

    Example:
        $ spark-optima export -r result.json -f databricks-json -o cluster.json
        $ spark-optima export -r result.json -f aws-glue
        $ spark-optima export -r result.json -f env -o spark_config.env

    """
    import json

    from spark_optima.cli.formatters import (
        ConfigExporter,
        print_platform_export_help,
    )
    from spark_optima.core.result import OptimizationResult

    # Convert empty strings to None for compatibility
    if not result_file:
        result_file = None  # type: ignore[assignment]
    if not output:
        output = None  # type: ignore[assignment]

    # Check result_file is provided
    if result_file is None:
        console.print("[bold red]Error:[/bold red] --result-file is required")
        raise typer.Exit(1)

    # Convert to Path objects
    result_file_obj = Path(result_file)
    output_obj = Path(output) if output else None

    console.print(
        Panel.fit(
            "[bold blue]📤 Export Configuration[/bold blue]",
            border_style="blue",
        ),
    )

    # Load the result
    try:
        with open(result_file_obj) as f:
            result_data = json.load(f)
        result = OptimizationResult(**result_data)
        platform = result.platform_specific.get("platform", "local")
    except (
        OSError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        AttributeError,
    ) as e:
        console.print(f"[bold red]Error loading result file:[/bold red] {e}")
        raise typer.Exit(1) from e

    # Show help if requested format is 'help'
    if format == "help":
        print_platform_export_help(platform)
        return

    # Create exporter
    exporter = ConfigExporter(result, platform)

    # Export
    try:
        if output_obj:
            exporter.save_to_file(output_obj, format)
        else:
            # Print to stdout
            formatters = {
                "json": exporter.export_json,
                "yaml": exporter.export_yaml,
                "spark-submit": exporter.export_spark_submit,
                "databricks-json": exporter.export_databricks_json,
                "databricks-cli": exporter.export_databricks_cli,
                "aws-glue": exporter.export_aws_glue,
                "aws-cli": exporter.export_aws_cli,
                "azure-synapse": exporter.export_azure_synapse,
                "env": exporter.export_environment_variables,
                "properties": exporter.export_properties_file,
                "airflow": exporter.export_airflow_dag,
                "kubernetes": exporter.export_kubernetes_configmap,
                "emr": exporter.export_aws_emr,
            }

            if format not in formatters:
                console.print(f"[bold red]Unknown format:[/bold red] {format}")
                console.print(f"Supported formats: {', '.join(formatters.keys())}")
                raise typer.Exit(1)

            typer.echo(formatters[format]())  # type: ignore[operator]

    except ValueError as e:
        console.print(f"[bold red]Export error:[/bold red] {e}")
        raise typer.Exit(1) from e


def _load_result_dict(result_path: Path) -> dict[str, Any]:
    """Load an optimization result JSON file into a dictionary.

    Args:
        result_path: Path to a result JSON file produced by the optimize/export commands.

    Returns:
        Parsed result dictionary.

    Raises:
        typer.Exit: If the file cannot be read or does not contain a JSON object.

    """
    try:
        with open(result_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error loading result file:[/bold red] {e}")
        raise typer.Exit(1) from e

    if not isinstance(data, dict):
        console.print(f"[bold red]Error:[/bold red] {result_path} does not contain a JSON object")
        raise typer.Exit(1)
    return data


def _extract_metrics(result_data: dict[str, Any]) -> dict[str, float]:
    """Extract comparable numeric metrics from a result dictionary.

    Args:
        result_data: Parsed optimization result dictionary.

    Returns:
        Mapping of metric name to numeric value (time, confidence, cost if present).

    """
    metrics: dict[str, float] = {}
    for key in ("estimated_time_minutes", "confidence_score"):
        value = result_data.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    # Cost is optional and may live at several locations depending on the platform
    cost_candidates = (
        result_data.get("estimated_cost"),
        (result_data.get("metadata") or {}).get("estimated_cost"),
        (result_data.get("platform_specific") or {}).get("estimated_cost"),
    )
    for candidate in cost_candidates:
        if isinstance(candidate, (int, float)):
            metrics["estimated_cost"] = float(candidate)
            break
    return metrics


def _display_history_entry(entry: Any) -> None:
    """Display the full details of a single history entry.

    Args:
        entry: HistoryEntry instance to display.

    """
    console.print(
        Panel.fit(
            f"[bold blue]History Entry #{entry.entry_id}[/bold blue]\n[dim]{entry.created_at}[/dim]",
            border_style="blue",
        ),
    )

    summary_table = Table(title="Run Summary", show_header=False)
    summary_table.add_column("Field", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Platform", entry.platform)
    summary_table.add_row("Spark Version", entry.spark_version)
    summary_table.add_row("Mode", entry.mode)
    summary_table.add_row("Estimated Time", f"{entry.estimated_time_minutes:.1f} min")
    summary_table.add_row("Confidence", f"{entry.confidence_score:.1%}")
    summary_table.add_row("Code Path", entry.code_path or "-")
    console.print(summary_table)

    config_table = Table(title="Optimized Configuration")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")
    for param, value in sorted(entry.configuration.items()):
        config_table.add_row(param, str(value))
    console.print(config_table)


@app.command()
def history(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of entries to list",
    ),
    platform: str = typer.Option(
        "",
        "--platform",
        "-p",
        help="Filter entries by platform (local, databricks, aws_glue, azure_synapse)",
    ),
    show: int = typer.Option(
        0,
        "--show",
        help="Show full details of the entry with this ID",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete all history entries",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt for --clear",
    ),
) -> None:
    """Browse past optimization runs stored in the local history database.

    Optimization results are saved automatically by the optimize command to
    ~/.spark_optima/history.db (override with SPARK_OPTIMA_HISTORY_DB).

    Example:
        $ spark-optima history --limit 10
        $ spark-optima history --show 3
        $ spark-optima history --clear --yes

    """
    from spark_optima.core.history import OptimizationHistory

    with OptimizationHistory() as history_store:
        if clear:
            if not yes and not typer.confirm("Delete ALL optimization history entries?"):
                console.print("[yellow]Aborted.[/yellow]")
                return
            deleted = history_store.clear()
            console.print(f"[green]✓[/green] Deleted {deleted} history entries")
            return

        if show:
            entry = history_store.get(show)
            if entry is None:
                console.print(f"[bold red]Error:[/bold red] no history entry with id {show}")
                raise typer.Exit(1)
            _display_history_entry(entry)
            return

        try:
            entries = history_store.list_entries(platform=platform or None, limit=limit)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1) from e

    if not entries:
        console.print("[yellow]No optimization history found.[/yellow]")
        return

    table = Table(title="Optimization History")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Created (UTC)", style="dim")
    table.add_column("Platform", style="green")
    table.add_column("Spark", style="green")
    table.add_column("Mode", style="dim")
    table.add_column("Est. Time", justify="right")
    table.add_column("Confidence", justify="right")
    for entry in entries:
        table.add_row(
            str(entry.entry_id),
            entry.created_at[:19].replace("T", " "),
            entry.platform,
            entry.spark_version,
            entry.mode,
            f"{entry.estimated_time_minutes:.1f} min",
            f"{entry.confidence_score:.1%}",
        )
    console.print(table)


@app.command()
def compare(
    result_a: str = typer.Option(
        ...,
        "--result-a",
        "-a",
        help="Path to the first optimization result JSON file",
    ),
    result_b: str = typer.Option(
        ...,
        "--result-b",
        "-b",
        help="Path to the second optimization result JSON file",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """Compare two optimization result files.

    Shows configuration parameters that differ between the two results,
    parameters present in only one of them, and metric deltas.

    Example:
        $ spark-optima compare -a result1.json -b result2.json
        $ spark-optima compare -a result1.json -b result2.json --output json

    """
    data_a = _load_result_dict(Path(result_a))
    data_b = _load_result_dict(Path(result_b))

    config_a: dict[str, Any] = data_a.get("configuration") or {}
    config_b: dict[str, Any] = data_b.get("configuration") or {}

    shared_keys = set(config_a) & set(config_b)
    changed = {key: (config_a[key], config_b[key]) for key in shared_keys if config_a[key] != config_b[key]}
    only_in_a = {key: config_a[key] for key in set(config_a) - set(config_b)}
    only_in_b = {key: config_b[key] for key in set(config_b) - set(config_a)}

    metrics_a = _extract_metrics(data_a)
    metrics_b = _extract_metrics(data_b)
    metric_deltas = {
        name: {"a": metrics_a[name], "b": metrics_b[name], "delta": metrics_b[name] - metrics_a[name]}
        for name in metrics_a
        if name in metrics_b
    }

    if output_format == "json":
        diff = {
            "changed": {key: {"a": a_value, "b": b_value} for key, (a_value, b_value) in sorted(changed.items())},
            "only_in_a": {key: only_in_a[key] for key in sorted(only_in_a)},
            "only_in_b": {key: only_in_b[key] for key in sorted(only_in_b)},
            "metrics": metric_deltas,
        }
        console.print_json(data=diff)
        return

    console.print(
        Panel.fit(
            f"[bold blue]Configuration Comparison[/bold blue]\n[dim]A: {result_a}\nB: {result_b}[/dim]",
            border_style="blue",
        ),
    )

    if not changed and not only_in_a and not only_in_b:
        console.print("[green]Configurations are identical.[/green]")
    else:
        config_table = Table(title="Configuration Differences")
        config_table.add_column("Parameter", style="cyan", no_wrap=True)
        config_table.add_column("Value A", style="green")
        config_table.add_column("Value B", style="yellow")
        for key in sorted(changed):
            a_value, b_value = changed[key]
            config_table.add_row(key, str(a_value), str(b_value))
        for key in sorted(only_in_a):
            config_table.add_row(key, str(only_in_a[key]), "[dim]-[/dim]")
        for key in sorted(only_in_b):
            config_table.add_row(key, "[dim]-[/dim]", str(only_in_b[key]))
        console.print(config_table)

    if metric_deltas:
        metrics_table = Table(title="Metric Deltas")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("A", justify="right")
        metrics_table.add_column("B", justify="right")
        metrics_table.add_column("Delta (B - A)", justify="right")
        for name, delta in metric_deltas.items():
            metrics_table.add_row(
                name,
                f"{delta['a']:.2f}",
                f"{delta['b']:.2f}",
                f"{delta['delta']:+.2f}",
            )
        console.print(metrics_table)


@app.command()
def explain(
    result_file: str = typer.Option(
        ...,
        "--result-file",
        "-r",
        help="Path to optimization result JSON file",
    ),
) -> None:
    """Explain the rationale behind each parameter in an optimization result.

    Rationale is sourced from the heuristic rule registry, falling back to the
    Spark parameter database description, and finally to a Bayesian tuning note.

    Example:
        $ spark-optima explain -r result.json

    """
    from spark_optima.core.config_engine.loader import VersionLoader
    from spark_optima.core.heuristics.rules import RuleRegistry

    result_data = _load_result_dict(Path(result_file))
    configuration: dict[str, Any] = result_data.get("configuration") or {}
    if not configuration:
        console.print("[yellow]Result file contains no configuration to explain.[/yellow]")
        return

    metadata = result_data.get("metadata") or {}
    platform_specific = result_data.get("platform_specific") or {}
    spark_version = str(
        metadata.get("spark_version") or platform_specific.get("spark_version") or "3.5.0",
    )

    registry = RuleRegistry()
    try:
        config_set = VersionLoader().load(spark_version)
    except (OSError, ValueError, KeyError):
        config_set = None

    console.print(
        Panel.fit(
            "[bold blue]Configuration Explanation[/bold blue]\n"
            f"[dim]Spark {spark_version} — {len(configuration)} parameters[/dim]",
            border_style="blue",
        ),
    )

    table = Table(title="Parameter Rationale")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Rationale")
    for param, value in sorted(configuration.items()):
        rule = registry.get_rule(param)
        db_param = config_set.parameters.get(param) if config_set else None
        if rule is not None and rule.description:
            source = "heuristic"
            rationale = rule.description
        elif db_param is not None and db_param.description:
            source = "database"
            rationale = db_param.description
        else:
            source = "bayesian"
            rationale = "Tuned by Bayesian optimization"
        table.add_row(param, str(value), source, rationale)
    console.print(table)


if __name__ == "__main__":
    app()
