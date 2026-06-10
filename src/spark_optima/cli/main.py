# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Main CLI entry point for Spark Optima.

This module provides the command-line interface for Spark Optima,
allowing users to run configuration optimization from the terminal.
"""

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spark_optima import __version__
from spark_optima.core.optimizer import Optimizer

if TYPE_CHECKING:
    from spark_optima.core.config_engine.loader import VersionLoader
    from spark_optima.core.config_engine.models import ConfigSet, ParameterType
    from spark_optima.core.config_engine.validator import ConfigValidator
    from spark_optima.core.execution.event_log import EventLogSummary

logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="spark-optima",
    help="Intelligent Apache Spark configuration optimization tool",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()

# Module-level option singleton (mutable list default would trip ruff B008 inline)
_OBJECTIVE_OPTION = typer.Option(
    [],
    "--objective",
    help=(
        "Optimization objective; repeat the flag for multi-objective runs, e.g. "
        "--objective minimize_time --objective minimize_cost (default: minimize_time)"
    ),
)


def _silence_optuna_logs() -> None:
    """Keep machine-readable stdout free of Optuna per-trial log lines."""
    import logging as _logging

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:  # pragma: no cover - optuna is a hard dependency
        pass
    _logging.getLogger("optuna").setLevel(_logging.WARNING)


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
        help="Target platform (local, databricks, aws_glue, aws_emr, azure_synapse, gcp_dataproc, kubernetes)",
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
    objective: list[str] = _OBJECTIVE_OPTION,
    event_log: str = typer.Option(
        "",
        "--event-log",
        "-e",
        help="Path to a Spark event log from a past run to enrich optimization inputs",
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

    # Validate objectives (deduplicated, order-preserving); empty means default
    from spark_optima.core.bayesian.objectives import ObjectiveFunctionFactory

    objectives: list[str] | None = list(dict.fromkeys(objective)) or None
    valid_objectives = ObjectiveFunctionFactory.get_available_objectives()
    invalid_objectives = [name for name in (objectives or []) if name not in valid_objectives]
    if invalid_objectives:
        console.print(
            f"[bold red]Error:[/bold red] unknown objective(s): {', '.join(invalid_objectives)}",
        )
        console.print(f"Valid objectives: {', '.join(valid_objectives)}")
        raise typer.Exit(1)

    # Parse the event log early so inferred values feed the input summary
    event_summary = None
    data_size_inferred = False
    if event_log:
        from spark_optima.core.execution.event_log import EventLogParser

        try:
            event_summary = EventLogParser(event_log).parse()
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] event log not found: {event_log}")
            raise typer.Exit(1) from None
        except (OSError, ValueError) as e:
            console.print(f"[bold red]Error parsing event log:[/bold red] {e}")
            raise typer.Exit(1) from e

        if data_size_gb is None and event_summary.input_data_gb > 0:
            data_size_gb = round(event_summary.input_data_gb, 2)
            data_size_inferred = True

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
    input_table.add_row("Objectives", ", ".join(objectives) if objectives else "minimize_time (default)")
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
    resource_constraints: dict[str, Any] = {}
    if max_memory_gb:
        resource_constraints["max_memory_gb"] = max_memory_gb

    # Enrich inputs with tuning hints derived from the event log
    if event_summary is not None:
        hints = event_summary.to_tuning_hints()
        hint_size = hints.pop("data_size_gb", None)
        if "size_gb" not in data_profile and hint_size:
            data_profile["size_gb"] = hint_size
        for key, value in hints.items():
            resource_constraints.setdefault(key, value)

        notes = []
        if data_size_inferred:
            notes.append(f"data size {data_size_gb} GB")
        notes.append(f"skew factor {hints['skew_factor']:.1f}")
        if hints["gc_pressure"]:
            notes.append(f"high GC pressure ({hints['gc_time_fraction']:.0%} of executor time)")
        if hints["spill_detected"]:
            notes.append(f"spill detected ({hints['spill_gb']:.2f} GB)")
        if hints["large_shuffles"]:
            notes.append(f"large shuffles ({hints['shuffle_total_gb']:.1f} GB)")
        console.print(f"[dim]Inferred from event log: {', '.join(notes)}[/dim]\n")

    # Run optimization
    if output_format in ("json", "yaml"):
        _silence_optuna_logs()
    with console.status("[bold green]Running optimization..."):
        try:
            result = optimizer.optimize(
                code_path=code_path_obj,
                data_profile=data_profile if data_profile else None,
                resource_constraints=resource_constraints if resource_constraints else None,
                use_bayesian=True,
                bayesian_trials=bayesian_trials,
                objectives=objectives,
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
    if "pareto_frontier" in result.metadata:
        console.print(
            "[dim]      Multi-objective run: inspect trade-offs with: spark-optima pareto -r result.json[/dim]",
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


def _build_tuning_advice(summary: "EventLogSummary", hints: dict[str, Any]) -> list[tuple[str, str]]:
    """Translate event log findings into plain-language tuning advice.

    Args:
        summary: Parsed event log summary.
        hints: Tuning hints produced by EventLogSummary.to_tuning_hints().

    Returns:
        List of (finding, advice) pairs; empty when no bottleneck was found.

    """
    from spark_optima.core.execution.event_log import SKEW_MODERATE_THRESHOLD, SKEW_SEVERE_THRESHOLD

    advice: list[tuple[str, str]] = []
    if hints["gc_pressure"]:
        advice.append(
            (
                f"High GC time ({hints['gc_time_fraction']:.0%} of executor run time)",
                "Increase executor memory or switch to G1GC (-XX:+UseG1GC)",
            ),
        )
    skew = hints["skew_factor"]
    if skew > SKEW_SEVERE_THRESHOLD:
        advice.append(
            (
                f"Severe task skew (max/median duration ratio {skew:.1f})",
                "Enable AQE skew join handling (spark.sql.adaptive.skewJoin.enabled=true)",
            ),
        )
    elif skew > SKEW_MODERATE_THRESHOLD:
        advice.append(
            (
                f"Moderate task skew (max/median duration ratio {skew:.1f})",
                "Review partitioning keys; AQE skew join handling can split hot partitions",
            ),
        )
    if hints["spill_detected"]:
        advice.append(
            (
                f"{hints['spill_gb']:.2f} GB spilled to memory/disk",
                "Increase executor memory or raise spark.sql.shuffle.partitions to shrink per-task data",
            ),
        )
    if hints["large_shuffles"]:
        advice.append(
            (
                f"Large shuffle volume ({hints['shuffle_total_gb']:.1f} GB)",
                "Increase shuffle partitions and enable AQE partition coalescing",
            ),
        )
    if summary.failed_tasks > 0:
        advice.append(
            (
                f"{summary.failed_tasks} failed task(s)",
                "Check executor logs; repeated task failures often indicate memory pressure",
            ),
        )
    return advice


@app.command("analyze-log")
def analyze_log(
    log_path: str = typer.Option(
        "",
        "--log-path",
        "-l",
        help="Path to a Spark event log file (plain JSON lines or .gz)",
    ),
    history_server: str = typer.Option(
        "",
        "--history-server",
        help="Spark History Server base URL (e.g. http://localhost:18080)",
    ),
    app_id: str = typer.Option(
        "",
        "--app-id",
        help="Application id on the History Server (omit to list recent applications)",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
    top_stages: int = typer.Option(
        10,
        "--top-stages",
        "-n",
        help="Number of slowest stages to display in table mode",
    ),
) -> None:
    """Analyze a Spark run from an event log file or a History Server.

    Parses the event log (JSON lines, optionally gzip-compressed) or fetches
    the same metrics from a running Spark History Server, and reports
    application, stage, GC, shuffle, spill, and skew metrics, together with
    plain-language tuning advice.

    Example:
        $ spark-optima analyze-log -l ./application_1234_eventlog
        $ spark-optima analyze-log -l ./eventlog.gz --output json
        $ spark-optima analyze-log --history-server http://localhost:18080
        $ spark-optima analyze-log --history-server http://localhost:18080 --app-id app-1234

    """
    from spark_optima.core.execution.event_log import EventLogParser

    if history_server:
        from spark_optima.core.execution.history_server import HistoryServerClient, HistoryServerError

        try:
            with HistoryServerClient(history_server) as client:
                if not app_id:
                    apps = client.list_applications()
                    if output_format == "json":
                        typer.echo(
                            json.dumps(
                                [
                                    {
                                        "app_id": a.app_id,
                                        "name": a.name,
                                        "duration_seconds": a.duration_seconds,
                                        "completed": a.completed,
                                        "spark_user": a.spark_user,
                                    }
                                    for a in apps
                                ],
                                indent=2,
                            ),
                        )
                        return
                    apps_table = Table(title="History Server Applications")
                    apps_table.add_column("App ID", style="cyan")
                    apps_table.add_column("Name", style="green")
                    apps_table.add_column("Duration", style="yellow")
                    apps_table.add_column("Completed", style="magenta")
                    for a in apps:
                        apps_table.add_row(
                            a.app_id,
                            a.name,
                            f"{a.duration_seconds:.0f}s",
                            "yes" if a.completed else "no",
                        )
                    console.print(apps_table)
                    console.print(
                        "[dim]Analyze one with: spark-optima analyze-log --history-server ... --app-id <id>[/dim]",
                    )
                    return
                summary = client.fetch_summary(app_id)
        except HistoryServerError as e:
            console.print(f"[bold red]History Server error:[/bold red] {e}")
            raise typer.Exit(1) from e
    elif log_path:
        log_path_obj = Path(log_path)
        if not log_path_obj.is_file():
            console.print(f"[bold red]Error:[/bold red] event log not found: {log_path_obj}")
            raise typer.Exit(1)

        try:
            summary = EventLogParser(log_path_obj).parse()
        except (OSError, ValueError) as e:
            console.print(f"[bold red]Error parsing event log:[/bold red] {e}")
            raise typer.Exit(1) from e
    else:
        console.print("[bold red]Error:[/bold red] provide --log-path or --history-server")
        raise typer.Exit(1)

    hints = summary.to_tuning_hints()

    if output_format == "json":
        payload = summary.to_dict()
        payload["tuning_hints"] = hints
        typer.echo(json.dumps(payload, indent=2))
        return

    # Table mode
    console.print(
        Panel.fit(
            f"[bold blue]📊 Spark Event Log Analysis[/bold blue]\n"
            f"[dim]{summary.app_name or app_id or Path(log_path).name}[/dim]",
            border_style="blue",
        ),
    )

    summary_table = Table(title="Run Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Application", summary.app_name or "-")
    summary_table.add_row("Duration", f"{summary.app_duration_seconds:.1f} s")
    summary_table.add_row("Total Tasks", str(summary.total_tasks))
    summary_table.add_row("Failed Tasks", str(summary.failed_tasks))
    summary_table.add_row("Max Executors", str(summary.executor_count_max))
    summary_table.add_row("Input Data", f"{summary.input_data_gb:.2f} GB")
    summary_table.add_row("Shuffle Read", f"{summary.total_shuffle_read_gb:.2f} GB")
    summary_table.add_row("Shuffle Write", f"{summary.total_shuffle_write_gb:.2f} GB")
    summary_table.add_row("Spill (memory+disk)", f"{summary.total_spill_gb:.2f} GB")
    summary_table.add_row("Peak Execution Memory", f"{summary.peak_execution_memory_gb:.2f} GB")
    summary_table.add_row("GC Time", f"{summary.total_gc_time_seconds:.1f} s")
    summary_table.add_row("GC Fraction", f"{summary.gc_time_fraction:.1%}")
    summary_table.add_row("Max Skew Ratio", f"{summary.max_skew_ratio:.1f}")
    summary_table.add_row("Spark Conf Entries", str(len(summary.spark_conf)))
    if summary.skipped_lines:
        summary_table.add_row("Skipped Lines", str(summary.skipped_lines))
    console.print(summary_table)

    if summary.stages:
        slowest = sorted(summary.stages, key=lambda stage: stage.duration_seconds, reverse=True)[:top_stages]
        stage_table = Table(title=f"Top {len(slowest)} Stages by Duration")
        stage_table.add_column("ID", style="cyan", justify="right")
        stage_table.add_column("Name", style="dim", max_width=40)
        stage_table.add_column("Duration (s)", justify="right")
        stage_table.add_column("Tasks", justify="right")
        stage_table.add_column("Shuffle R (GB)", justify="right")
        stage_table.add_column("Shuffle W (GB)", justify="right")
        stage_table.add_column("Spill (GB)", justify="right")
        stage_table.add_column("Skew", justify="right")
        for stage in slowest:
            stage_table.add_row(
                str(stage.stage_id),
                stage.name or "-",
                f"{stage.duration_seconds:.1f}",
                str(stage.num_tasks),
                f"{stage.shuffle_read_gb:.2f}",
                f"{stage.shuffle_write_gb:.2f}",
                f"{stage.spill_gb:.2f}",
                f"{stage.skew_ratio:.1f}",
            )
        console.print(stage_table)

    advice = _build_tuning_advice(summary, hints)
    if advice:
        hints_table = Table(title="Tuning Hints")
        hints_table.add_column("Finding", style="yellow")
        hints_table.add_column("Advice", style="green")
        for finding, recommendation in advice:
            hints_table.add_row(finding, recommendation)
        console.print(hints_table)
    else:
        console.print("[green]No obvious bottlenecks detected in this run.[/green]")

    console.print(
        "\n[dim]Tip: feed this run into optimization with: "
        f"spark-optima optimize -c <code.py> --event-log {log_path_obj}[/dim]",
    )


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
                objectives=config.get("objectives"),
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
            "azure-synapse, env, properties, airflow, kubernetes, emr, "
            "pareto-json, pareto-csv)"
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
                "pareto-json": exporter.export_pareto_json,
                "pareto-csv": exporter.export_pareto_csv,
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


def _pareto_objective_names(metadata: dict[str, Any], frontier: list[dict[str, Any]]) -> list[str]:
    """Return Pareto objective names in deterministic order.

    Prefers the objective order recorded in result metadata, falling back to
    the sorted union of objective names across frontier points.

    Args:
        metadata: Result metadata dictionary.
        frontier: Pareto frontier point dictionaries.

    Returns:
        Ordered list of objective names.

    """
    names = metadata.get("objectives")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return sorted({key for point in frontier for key in (point.get("objective_values") or {})})


def _objective_direction(name: str) -> str:
    """Return the optimization direction ("minimize" or "maximize") for an objective.

    Args:
        name: Objective name (e.g. "minimize_time").

    Returns:
        Direction string; unknown objectives default to "minimize".

    """
    from spark_optima.core.bayesian.objectives import ObjectiveFunctionFactory

    try:
        return ObjectiveFunctionFactory.create(name).direction
    except (ValueError, TypeError):
        return "minimize"


def _pareto_value(point: dict[str, Any], name: str) -> float | None:
    """Extract a numeric objective value from a Pareto point.

    Args:
        point: Pareto frontier point dictionary.
        name: Objective name.

    Returns:
        Float value or None when missing or non-numeric.

    """
    value = (point.get("objective_values") or {}).get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_metric(value: Any) -> str:
    """Format an objective value for display.

    Args:
        value: Objective value (usually numeric).

    Returns:
        Compact string representation.

    """
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return "-" if value is None else str(value)


def _select_pareto_config_columns(frontier: list[dict[str, Any]], max_columns: int = 4) -> list[str]:
    """Pick a few key configuration parameters to show as table columns.

    Args:
        frontier: Pareto frontier point dictionaries.
        max_columns: Maximum number of configuration columns.

    Returns:
        List of parameter names (priority parameters first, then alphabetical).

    """
    available = {key for point in frontier for key in (point.get("configuration") or {})}
    priority = [
        "spark.executor.memory",
        "spark.executor.cores",
        "spark.executor.instances",
        "spark.sql.shuffle.partitions",
        "spark.driver.memory",
        "spark.default.parallelism",
    ]
    columns = [key for key in priority if key in available]
    columns.extend(key for key in sorted(available) if key not in columns)
    return columns[:max_columns]


def _build_pareto_tradeoff_lines(
    frontier: list[dict[str, Any]],
    objective_names: list[str],
) -> list[str]:
    """Build trade-off summary lines (one per objective) for the Pareto frontier.

    For each objective the best point is identified (respecting its
    minimize/maximize direction) and its values on the other objectives are
    shown relative to their own best values.

    Args:
        frontier: Pareto frontier point dictionaries.
        objective_names: Ordered objective names.

    Returns:
        Human-readable summary lines; empty when no numeric values exist.

    """
    best_points: dict[str, dict[str, Any]] = {}
    for name in objective_names:
        candidates = [point for point in frontier if _pareto_value(point, name) is not None]
        if not candidates:
            continue
        direction = _objective_direction(name)
        chooser = max if direction == "maximize" else min
        best_points[name] = chooser(candidates, key=lambda point: _pareto_value(point, name) or 0.0)

    lines: list[str] = []
    for name in objective_names:
        best = best_points.get(name)
        if best is None:
            continue
        parts = [f"Best {name}: trial {best.get('trial_number')} ({_format_metric(_pareto_value(best, name))})"]
        deltas = []
        for other in objective_names:
            if other == name or other not in best_points:
                continue
            own_value = _pareto_value(best, other)
            best_value = _pareto_value(best_points[other], other)
            if own_value is None or best_value is None:
                continue
            delta_text = f"{other} {_format_metric(own_value)}"
            if best_value != 0:
                delta_pct = (own_value - best_value) / abs(best_value) * 100
                delta_text += f" ({delta_pct:+.1f}% vs best)"
            deltas.append(delta_text)
        if deltas:
            parts.append(", ".join(deltas))
        lines.append(" — ".join(parts))
    return lines


@app.command()
def pareto(
    result_file: str = typer.Option(
        ...,
        "--result-file",
        "-r",
        help="Path to optimization result JSON file",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """Display the Pareto frontier of a multi-objective optimization result.

    Shows the non-dominated trade-off points found by a multi-objective run
    (e.g. minimize_time vs minimize_cost) with their objective values and key
    configuration parameters, plus a trade-off summary.

    Example:
        $ spark-optima pareto -r result.json
        $ spark-optima pareto -r result.json --output json

    """
    result_data = _load_result_dict(Path(result_file))
    metadata: dict[str, Any] = result_data.get("metadata") or {}
    frontier = metadata.get("pareto_frontier")

    if not frontier:
        console.print("[bold red]Error:[/bold red] this result has no Pareto frontier.")
        console.print(
            "[dim]Rerun optimization with multiple objectives, e.g.:[/dim]\n"
            "[dim]  spark-optima optimize -c job.py "
            "--objective minimize_time --objective minimize_cost[/dim]",
        )
        raise typer.Exit(1)

    objective_names = _pareto_objective_names(metadata, frontier)

    if output_format == "json":
        payload = {
            "objectives": objective_names,
            "n_points": len(frontier),
            "points": frontier,
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(
        Panel.fit(
            "[bold blue]Pareto Frontier[/bold blue]\n"
            f"[dim]{len(frontier)} non-dominated point(s) — objectives: {', '.join(objective_names)}[/dim]",
            border_style="blue",
        ),
    )

    config_columns = _select_pareto_config_columns(frontier)
    table = Table(title="Pareto Frontier Points")
    table.add_column("Trial", style="cyan", justify="right")
    for name in objective_names:
        table.add_column(name, style="green", justify="right")
    for param in config_columns:
        table.add_column(param, style="yellow")

    for point in sorted(frontier, key=lambda p: p.get("trial_number", -1)):
        configuration = point.get("configuration") or {}
        row = [str(point.get("trial_number", "-"))]
        row.extend(_format_metric(_pareto_value(point, name)) for name in objective_names)
        row.extend(str(configuration[param]) if param in configuration else "-" for param in config_columns)
        table.add_row(*row)
    console.print(table)

    tradeoff_lines = _build_pareto_tradeoff_lines(frontier, objective_names)
    if tradeoff_lines:
        console.print("\n[bold]Trade-off summary:[/bold]")
        for line in tradeoff_lines:
            console.print(f"  {line}")

    console.print(
        f"\n[dim]Tip: export the frontier with: spark-optima export -r {result_file} -f pareto-csv[/dim]",
    )


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


# =============================================================================
# v1.4 — validate / import / templates (Workstream S)
# =============================================================================

# Matches "key value", "key=value", and "key = value" properties lines
_PROPERTIES_LINE_RE = re.compile(r"^(\S+?)(?:\s*=\s*|\s+)(.+)$")


def _parse_config_file(config_path: Path) -> dict[str, Any]:
    """Parse a Spark configuration file in properties or JSON format.

    Supports the spark-defaults.conf properties format ("key value" or
    "key=value" lines with # comments) and JSON dictionaries. The format is
    detected from the file extension and content.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary of parameter names to raw values.

    Raises:
        typer.Exit: If the file cannot be read or parsed.

    """
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as e:
        console.print(f"[bold red]Error reading config file:[/bold red] {e}")
        raise typer.Exit(1) from e

    if config_path.suffix.lower() == ".json" or content.lstrip().startswith("{"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error parsing JSON config:[/bold red] {e}")
            raise typer.Exit(1) from e
        if not isinstance(data, dict):
            console.print(f"[bold red]Error:[/bold red] {config_path} must contain a JSON object")
            raise typer.Exit(1)
        return {str(key): value for key, value in data.items()}

    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROPERTIES_LINE_RE.match(line)
        if match is None:
            console.print(
                f"[bold red]Error:[/bold red] cannot parse line {line_number} of {config_path}: {raw_line!r}",
            )
            raise typer.Exit(1)
        config[match.group(1)] = match.group(2).strip()
    return config


def _validation_issue(severity: str, param: str, message: str, check: str) -> dict[str, str]:
    """Build a single validation issue record.

    Args:
        severity: Issue severity ("error" or "warning").
        param: Parameter name the issue refers to.
        message: Human-readable issue description.
        check: Machine-readable name of the check that produced the issue.

    Returns:
        Issue dictionary.

    """
    return {"severity": severity, "param": param, "message": message, "check": check}


def _config_bool(value: Any) -> bool | None:
    """Interpret a raw config value as a boolean.

    Args:
        value: Raw value (bool or "true"/"false" string).

    Returns:
        Boolean value, or None when the value is not a recognizable boolean.

    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _config_int(value: Any) -> int | None:
    """Interpret a raw config value as an integer.

    Args:
        value: Raw value (int or numeric string).

    Returns:
        Integer value, or None when the value is not a valid integer.

    """
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


def _memory_to_gb(value: Any, validator: "ConfigValidator") -> float | None:
    """Convert a Spark byte-size value (e.g. "4g") to gigabytes.

    Args:
        value: Raw memory value from a config file.
        validator: Validator used to parse byte strings.

    Returns:
        Size in GB, or None when the value cannot be parsed.

    """
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


def _coerce_typed_value(
    value: Any,
    param_type: "ParameterType",
    validator: "ConfigValidator",
) -> tuple[Any, str | None]:
    """Coerce a raw config value (usually a string) into the parameter's type.

    Args:
        value: Raw value from the parsed config file.
        param_type: Expected parameter type from the database.
        validator: Validator used for byte/duration format checks.

    Returns:
        Tuple of (coerced value, error message or None when the value is valid).

    """
    from spark_optima.core.config_engine.models import ParameterType

    if not isinstance(value, str):
        # JSON configs may carry natively typed values; the validator checks them
        return value, None

    text = value.strip()
    if param_type == ParameterType.BOOLEAN:
        boolean = _config_bool(text)
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


def _collect_db_issues(
    config: dict[str, Any],
    config_set: "ConfigSet",
    validator: "ConfigValidator",
) -> list[dict[str, str]]:
    """Check a config against the Spark parameter database.

    Flags unknown parameters, parameters deprecated in the target version,
    and values that fail type/format or constraint validation.

    Args:
        config: Parsed user configuration.
        config_set: Parameter database for the resolved Spark version.
        validator: Shared validator instance.

    Returns:
        List of validation issues.

    """
    from spark_optima.core.config_engine.models import ParameterType

    issues: list[dict[str, str]] = []
    for param_name in sorted(config):
        raw_value = config[param_name]
        db_param = config_set.parameters.get(param_name)
        if db_param is None:
            issues.append(
                _validation_issue(
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
            issues.append(_validation_issue("warning", param_name, message, "deprecated_parameter"))

        coerced, type_error = _coerce_typed_value(raw_value, db_param.param_type, validator)
        if type_error is not None:
            issues.append(_validation_issue("error", param_name, type_error, "invalid_value"))
            continue

        if db_param.param_type in (ParameterType.BYTES, ParameterType.DURATION):
            # The database expresses byte/duration min/max constraints in mixed
            # units, so numeric range checks would produce false positives.
            # Format validity was checked during coercion; also apply the
            # database regex pattern when one exists.
            pattern = db_param.constraints.pattern
            if pattern and isinstance(coerced, str) and not re.match(pattern, coerced):
                issues.append(
                    _validation_issue(
                        "error",
                        param_name,
                        f"value '{coerced}' does not match pattern '{pattern}'",
                        "invalid_value",
                    ),
                )
            continue

        if not validator.validate(db_param, coerced):
            for error in validator.get_errors():
                issues.append(_validation_issue("error", param_name, str(error), "invalid_value"))
    return issues


def _collect_platform_issues(
    config: dict[str, Any],
    platform_name: str,
    validator: "ConfigValidator",
) -> list[dict[str, str]]:
    """Check executor memory/cores against a platform's resource constraints.

    Args:
        config: Parsed user configuration.
        platform_name: Platform identifier from PLATFORM_REGISTRY.
        validator: Shared validator instance (for memory parsing).

    Returns:
        List of validation issues.

    Raises:
        typer.Exit: If the platform name is unknown.

    """
    from spark_optima.platforms import get_platform

    try:
        platform_obj = get_platform(platform_name)
    except (ValueError, RuntimeError, ImportError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e

    constraints = platform_obj.constraints
    issues: list[dict[str, str]] = []

    executor_memory_gb = _memory_to_gb(config.get("spark.executor.memory"), validator)
    if executor_memory_gb is not None:
        if executor_memory_gb > constraints.max_memory_gb:
            issues.append(
                _validation_issue(
                    "error",
                    "spark.executor.memory",
                    f"{executor_memory_gb:.1f} GB exceeds the {platform_obj.name} maximum of "
                    f"{constraints.max_memory_gb:.0f} GB per worker",
                    "platform_constraint",
                ),
            )
        elif executor_memory_gb < constraints.min_memory_gb:
            issues.append(
                _validation_issue(
                    "error",
                    "spark.executor.memory",
                    f"{executor_memory_gb:.1f} GB is below the {platform_obj.name} minimum of "
                    f"{constraints.min_memory_gb:.0f} GB per worker",
                    "platform_constraint",
                ),
            )

    executor_cores = _config_int(config.get("spark.executor.cores"))
    if executor_cores is not None:
        if executor_cores > constraints.max_cores:
            issues.append(
                _validation_issue(
                    "error",
                    "spark.executor.cores",
                    f"{executor_cores} cores exceeds the {platform_obj.name} maximum of "
                    f"{constraints.max_cores} cores per worker",
                    "platform_constraint",
                ),
            )
        elif executor_cores < constraints.min_cores:
            issues.append(
                _validation_issue(
                    "error",
                    "spark.executor.cores",
                    f"{executor_cores} cores is below the {platform_obj.name} minimum of "
                    f"{constraints.min_cores} cores per worker",
                    "platform_constraint",
                ),
            )

    return issues


def _collect_anti_pattern_issues(
    config: dict[str, Any],
    resolved_version: str,
    loader: "VersionLoader",
    validator: "ConfigValidator",
) -> list[dict[str, str]]:
    """Check a config against a curated list of Spark anti-patterns.

    Args:
        config: Parsed user configuration.
        resolved_version: Spark version resolved from the parameter database.
        loader: Version loader used for version comparisons.
        validator: Shared validator instance (for memory parsing).

    Returns:
        List of validation issues.

    """
    issues: list[dict[str, str]] = []

    # Driver memory larger than executor memory
    driver_gb = _memory_to_gb(config.get("spark.driver.memory"), validator)
    executor_gb = _memory_to_gb(config.get("spark.executor.memory"), validator)
    if driver_gb is not None and executor_gb is not None and driver_gb > executor_gb:
        issues.append(
            _validation_issue(
                "warning",
                "spark.driver.memory",
                f"driver memory ({driver_gb:.1f} GB) is larger than executor memory ({executor_gb:.1f} GB); "
                "the driver rarely needs more memory than the executors",
                "driver_memory_exceeds_executor",
            ),
        )

    # Dynamic allocation misconfigurations
    if _config_bool(config.get("spark.dynamicAllocation.enabled")):
        min_executors = _config_int(config.get("spark.dynamicAllocation.minExecutors"))
        max_executors = _config_int(config.get("spark.dynamicAllocation.maxExecutors"))
        if min_executors is not None and max_executors is not None:
            if max_executors < min_executors:
                issues.append(
                    _validation_issue(
                        "error",
                        "spark.dynamicAllocation.maxExecutors",
                        f"maxExecutors ({max_executors}) is less than minExecutors ({min_executors})",
                        "dynamic_allocation_bounds",
                    ),
                )
            elif max_executors == min_executors:
                issues.append(
                    _validation_issue(
                        "warning",
                        "spark.dynamicAllocation.maxExecutors",
                        f"maxExecutors equals minExecutors ({max_executors}); dynamic allocation cannot scale",
                        "dynamic_allocation_bounds",
                    ),
                )

        shuffle_service = _config_bool(config.get("spark.shuffle.service.enabled"))
        shuffle_tracking = _config_bool(config.get("spark.dynamicAllocation.shuffleTracking.enabled"))
        if not shuffle_service and not shuffle_tracking:
            issues.append(
                _validation_issue(
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
                _validation_issue(
                    "warning",
                    "spark.serializer",
                    f"Java serializer is configured but Kryo settings are present "
                    f"({', '.join(kryo_settings)}); they will have no effect",
                    "serializer_mismatch",
                ),
            )

    # AQE disabled on Spark >= 3.2
    aqe_enabled = _config_bool(config.get("spark.sql.adaptive.enabled"))
    if aqe_enabled is False and loader.is_at_least(resolved_version, "3.2.0"):
        issues.append(
            _validation_issue(
                "warning",
                "spark.sql.adaptive.enabled",
                f"Adaptive Query Execution is disabled; on Spark {resolved_version} (>= 3.2) "
                "AQE is mature and usually improves performance",
                "aqe_disabled",
            ),
        )

    return issues


@app.command()
def validate(
    config_file: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a Spark configuration file (spark-defaults.conf properties or JSON dict)",
    ),
    platform: str = typer.Option(
        "",
        "--platform",
        "-p",
        help=(
            "Validate against a platform's resource constraints "
            "(local, databricks, aws_glue, aws_emr, azure_synapse, gcp_dataproc, kubernetes)"
        ),
    ),
    spark_version: str = typer.Option(
        "3.5.0",
        "--spark-version",
        "-s",
        help="Spark version whose parameter database is used",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """Validate a Spark configuration file against the parameter database.

    Checks for unknown parameters, deprecated parameters, invalid values,
    platform constraint violations (with --platform), and a curated list of
    configuration anti-patterns. Exits with code 1 if any error is found;
    warnings alone exit 0.

    Example:
        $ spark-optima validate -c spark-defaults.conf
        $ spark-optima validate -c config.json -p aws_glue -s 3.5.0 --output json

    """
    from spark_optima.core.config_engine.loader import VersionLoader
    from spark_optima.core.config_engine.validator import ConfigValidator

    config_path = Path(config_file)
    if not config_path.is_file():
        console.print(f"[bold red]Error:[/bold red] config file not found: {config_path}")
        raise typer.Exit(1)

    config = _parse_config_file(config_path)

    loader = VersionLoader()
    config_set = loader.load(spark_version)
    if config_set is None:
        console.print(f"[bold red]Error:[/bold red] no parameter database available for Spark {spark_version}")
        raise typer.Exit(1)

    validator = ConfigValidator()
    issues = _collect_db_issues(config, config_set, validator)
    if platform:
        issues.extend(_collect_platform_issues(config, platform, validator))
    issues.extend(_collect_anti_pattern_issues(config, config_set.version, loader, validator))

    severity_rank = {"error": 0, "warning": 1}
    issues.sort(key=lambda issue: (severity_rank.get(issue["severity"], 2), issue["param"]))
    error_issues = [issue for issue in issues if issue["severity"] == "error"]
    warning_issues = [issue for issue in issues if issue["severity"] == "warning"]

    if output_format == "json":
        payload = {
            "config_file": str(config_path),
            "spark_version": config_set.version,
            "platform": platform or None,
            "parameter_count": len(config),
            "issues": issues,
            "error_count": len(error_issues),
            "warning_count": len(warning_issues),
            "valid": not error_issues,
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        platform_note = f" — platform {platform}" if platform else ""
        console.print(
            Panel.fit(
                "[bold blue]Configuration Validation[/bold blue]\n"
                f"[dim]{config_path} — Spark {config_set.version}{platform_note}[/dim]",
                border_style="blue",
            ),
        )
        if not issues:
            console.print(f"[green]No issues found in {len(config)} parameter(s).[/green]")
        else:
            table = Table(title="Validation Issues")
            table.add_column("Severity", style="bold", no_wrap=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Message")
            for issue in issues:
                style = "red" if issue["severity"] == "error" else "yellow"
                table.add_row(f"[{style}]{issue['severity']}[/{style}]", issue["param"], issue["message"])
            console.print(table)
            console.print(
                f"\n{len(config)} parameter(s) checked: "
                f"[red]{len(error_issues)} error(s)[/red], [yellow]{len(warning_issues)} warning(s)[/yellow]",
            )

    if error_issues:
        raise typer.Exit(1)


def _diff_configs(
    current: dict[str, Any],
    recommended: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Compute the difference between a current and a recommended config.

    Values are compared as trimmed strings so "200" and 200 are equal.

    Args:
        current: User's existing configuration.
        recommended: Optimizer-recommended configuration.

    Returns:
        Tuple of sorted key lists: (changed, only_in_current, only_in_recommended).

    """
    shared = set(current) & set(recommended)
    changed = sorted(key for key in shared if str(current[key]).strip() != str(recommended[key]).strip())
    only_in_current = sorted(set(current) - set(recommended))
    only_in_recommended = sorted(set(recommended) - set(current))
    return changed, only_in_current, only_in_recommended


@app.command("import")
def import_config(
    config_file: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to your existing Spark configuration (spark-defaults.conf properties or JSON dict)",
    ),
    code_path: str = typer.Option(
        ...,
        "--code",
        help="Path to the Spark application code file",
    ),
    platform: str = typer.Option(
        "local",
        "--platform",
        "-p",
        help="Target platform (local, databricks, aws_glue, aws_emr, azure_synapse, gcp_dataproc, kubernetes)",
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
    bayesian_trials: int = typer.Option(
        50,
        "--bayesian-trials",
        help="Number of Bayesian optimization trials",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """Import an existing Spark config and diff it against an optimized one.

    Parses your current configuration, runs a normal optimization for the
    given code, and shows what the optimizer would change: parameters with
    different values, parameters only in your config, and parameters only in
    the recommendation.

    Example:
        $ spark-optima import -c spark-defaults.conf --code ./my_job.py -p databricks -d 100
        $ spark-optima import -c config.json --code ./my_job.py --output json

    """
    config_path_obj = Path(config_file)
    if not config_path_obj.is_file():
        console.print(f"[bold red]Error:[/bold red] config file not found: {config_path_obj}")
        raise typer.Exit(1)

    current = _parse_config_file(config_path_obj)

    if output_format != "json":
        console.print(
            Panel.fit(
                "[bold blue]Import & Diff Configuration[/bold blue]\n"
                f"[dim]{config_path_obj} ({len(current)} parameters) vs optimized recommendation[/dim]",
                border_style="blue",
            ),
        )

    try:
        optimizer = Optimizer(
            platform=platform,
            spark_version=spark_version,
            optimization_mode="simulation",
        )
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e

    data_profile = {"size_gb": data_size_gb} if data_size_gb else None

    if output_format == "json":
        _silence_optuna_logs()

    try:
        result = optimizer.optimize(
            code_path=Path(code_path),
            data_profile=data_profile,
            use_bayesian=True,
            bayesian_trials=bayesian_trials,
        )
    except (RuntimeError, ValueError, KeyError, AttributeError, TypeError) as e:
        console.print(f"[bold red]Optimization failed:[/bold red] {e}")
        raise typer.Exit(1) from e

    recommended: dict[str, Any] = result.configuration or {}
    changed, only_in_current, only_in_recommended = _diff_configs(current, recommended)

    if output_format == "json":
        payload = {
            "current": current,
            "recommended": recommended,
            "diff": {
                "changed": {key: {"current": current[key], "recommended": recommended[key]} for key in changed},
                "only_in_current": {key: current[key] for key in only_in_current},
                "only_in_recommended": {key: recommended[key] for key in only_in_recommended},
            },
            "estimated_time_minutes": result.estimated_time_minutes,
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    if not changed and not only_in_current and not only_in_recommended:
        console.print("[green]Your configuration already matches the recommendation.[/green]")
    else:
        diff_table = Table(title="Current vs Recommended Configuration")
        diff_table.add_column("Parameter", style="cyan", no_wrap=True)
        diff_table.add_column("Current", style="yellow")
        diff_table.add_column("Recommended", style="green")
        for key in changed:
            diff_table.add_row(key, str(current[key]), str(recommended[key]))
        if only_in_current:
            diff_table.add_section()
            diff_table.add_row("[bold dim]Only in current[/bold dim]", "", "")
            for key in only_in_current:
                diff_table.add_row(key, str(current[key]), "[dim]-[/dim]")
        if only_in_recommended:
            diff_table.add_section()
            diff_table.add_row("[bold dim]Only in recommended[/bold dim]", "", "")
            for key in only_in_recommended:
                diff_table.add_row(key, "[dim]-[/dim]", str(recommended[key]))
        console.print(diff_table)

        unchanged_count = len(set(current) & set(recommended)) - len(changed)
        if unchanged_count:
            console.print(f"[dim]{unchanged_count} parameter(s) already match and are not shown.[/dim]")

    console.print(
        f"\nEstimated time with recommended configuration: [green]{result.estimated_time_minutes:.1f} min[/green]",
    )
    console.print("[dim]Tip: validate your current config with: spark-optima validate -c <file>[/dim]")


@app.command()
def templates(
    show: str = typer.Option(
        "",
        "--show",
        help="Show full details of one template (e.g. etl-batch)",
    ),
    output_format: str = typer.Option(
        "table",
        "--output",
        "-o",
        help="Output format (table, json)",
    ),
) -> None:
    """List curated workload templates or inspect one in detail.

    Templates are curated baseline Spark configurations for common workload
    archetypes (batch ETL, streaming, ML training, interactive analytics).
    Use them as a starting point and layer your own settings on top.

    Example:
        $ spark-optima templates
        $ spark-optima templates --show etl-batch
        $ spark-optima templates --show streaming --output json

    """
    from spark_optima.core.templates import TemplateRegistry

    registry = TemplateRegistry()

    if show:
        try:
            template = registry.get_template(show)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1) from e

        if output_format == "json":
            typer.echo(json.dumps(template.to_dict(), indent=2))
            return

        console.print(
            Panel.fit(
                f"[bold blue]{template.display_name}[/bold blue]\n[dim]{template.description}[/dim]",
                border_style="blue",
            ),
        )

        if template.workload_traits:
            console.print("[bold]Workload traits:[/bold]")
            for trait in template.workload_traits:
                console.print(f"  - {trait}")
            console.print()

        config_table = Table(title="Curated Configuration")
        config_table.add_column("Parameter", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="green")
        config_table.add_column("Why", style="dim")
        for param_name, parameter in template.config.items():
            config_table.add_row(param_name, str(parameter.value), parameter.comment)
        console.print(config_table)

        if template.recommended_for:
            console.print("\n[bold green]Recommended for:[/bold green]")
            for item in template.recommended_for:
                console.print(f"  - {item}")
        if template.not_recommended_for:
            console.print("\n[bold yellow]Not recommended for:[/bold yellow]")
            for item in template.not_recommended_for:
                console.print(f"  - {item}")
        return

    all_templates = registry.list_templates()

    if output_format == "json":
        typer.echo(json.dumps([template.to_dict() for template in all_templates], indent=2))
        return

    console.print(
        Panel.fit(
            "[bold blue]Workload Templates[/bold blue]\n[dim]Curated baseline Spark configurations[/dim]",
            border_style="blue",
        ),
    )

    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="green")
    table.add_column("Parameters", justify="right")
    table.add_column("Description", style="dim")
    for template in all_templates:
        table.add_row(template.name, template.display_name, str(len(template.config)), template.description)
    console.print(table)

    console.print("\n[dim]Tip: inspect one with: spark-optima templates --show <name>[/dim]")


if __name__ == "__main__":
    app()
