# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Interactive configuration wizard for Spark Optima CLI.

This module provides an interactive step-by-step wizard for configuring
Spark optimization jobs through the command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from spark_optima.api.dependencies import get_all_platforms_metadata, get_optimization_service
from spark_optima.core.bayesian.objectives import ObjectiveFunctionFactory
from spark_optima.core.execution.event_log import EventLogParser

console = Console()


class ConfigurationWizard:
    """Interactive configuration wizard.

    This class guides users through a step-by-step process to configure
    their Spark optimization job, collecting all necessary information
    through interactive prompts.

    Attributes:
        config: Dictionary storing the collected configuration.
        current_step: Current step number in the wizard.
        total_steps: Total number of steps in the wizard.
    """

    TOTAL_STEPS = 7

    DEFAULT_OBJECTIVE = "minimize_time"

    OBJECTIVE_DESCRIPTIONS: dict[str, str] = {
        "minimize_time": "Fastest job completion",
        "minimize_cost": "Lowest estimated run cost",
        "maximize_success": "Highest reliability under the time budget",
        "minimize_memory": "Smallest memory footprint",
    }

    # Curated subset of export formats shown in the wizard (full set lives in
    # the `spark-optima export` command dispatch).
    EXPORT_FORMATS: tuple[tuple[str, str], ...] = (
        ("spark-submit", "Ready-to-run spark-submit command"),
        ("properties", "spark-defaults.conf properties file"),
        ("databricks-json", "Databricks cluster spec (REST API / Jobs)"),
        ("aws-glue", "AWS Glue job parameters"),
        ("emr", "AWS EMR cluster configurations JSON"),
        ("kubernetes", "Kubernetes ConfigMap manifest"),
        ("airflow", "Airflow DAG snippet"),
        ("pareto-json", "Pareto frontier of a multi-objective run"),
    )

    ADDITIONAL_EXPORT_FORMATS: tuple[str, ...] = (
        "json",
        "yaml",
        "env",
        "databricks-cli",
        "aws-cli",
        "azure-synapse",
        "pareto-csv",
    )

    def __init__(self) -> None:
        """Initialize the configuration wizard."""
        self.config: dict[str, Any] = {}
        self.current_step = 0

    def _print_header(self) -> None:
        """Print the wizard header."""
        console.print(
            Panel.fit(
                "[bold blue]🧙 Spark Optima Configuration Wizard[/bold blue]\n"
                "[dim]Follow the steps to configure your Spark optimization[/dim]",
                border_style="blue",
            ),
        )
        console.print()

    def _print_progress(self) -> None:
        """Print the current progress indicator."""
        self.current_step += 1
        progress = f"[cyan]Step {self.current_step}/{self.TOTAL_STEPS}[/cyan]"
        console.print(f"\n{progress}", style="bold")
        console.print("─" * 50, style="dim")

    def _step_platform_selection(self) -> None:
        """Step 1: Platform selection.

        Prompts the user to select a target platform for the Spark job.
        """
        self._print_progress()
        console.print("[bold]Select your target platform:[/bold]\n")

        platforms_metadata = get_all_platforms_metadata()

        # Display platform options
        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="green")

        platform_options: dict[str, str] = {}
        for i, (key, meta) in enumerate(platforms_metadata.items(), 1):
            platform_options[str(i)] = key
            platform_options[key] = key
            table.add_row(f"  [{i}]", f"{meta['display_name']} - {meta['description']}")

        console.print(table)
        console.print()

        # Get user selection
        prompt_text = f"Enter choice ({', '.join(platform_options.keys())})"

        while True:
            choice = Prompt.ask(prompt_text, console=console)
            if choice in platform_options:
                self.config["platform"] = platform_options[choice]
                break
            console.print("[red]Invalid choice. Please try again.[/red]")

        console.print(
            f"[green]✓[/green] Selected platform: "
            f"[cyan]{platforms_metadata[self.config['platform']]['display_name']}[/cyan]",
        )

    def _step_spark_version(self) -> None:
        """Step 2: Spark version selection.

        Prompts the user to select a Spark version.
        """
        self._print_progress()
        console.print("[bold]Select Spark version:[/bold]\n")

        service = get_optimization_service()
        available_versions = service.get_available_spark_versions()

        # Show recent versions
        recent_versions = available_versions[-5:] if len(available_versions) > 5 else available_versions

        console.print("[dim]Available versions:[/dim]")
        for i, version in enumerate(recent_versions, 1):
            marker = " (recommended)" if version == "3.5.0" else ""
            console.print(f"  [{i}] {version}{marker}")

        console.print()

        # Get user input
        version = Prompt.ask(
            "Enter Spark version",
            default="3.5.0",
            console=console,
        )

        # Validate version
        if not service.validate_spark_version(version):
            console.print(
                f"[yellow]Warning: Version {version} may not be fully supported. "
                f"Using best-match configuration.[/yellow]",
            )

        self.config["spark_version"] = version
        console.print(f"[green]✓[/green] Spark version: [cyan]{version}[/cyan]")

    def _step_resource_constraints(self) -> None:
        """Step 3: Resource constraints.

        Prompts the user to enter resource constraints.
        """
        self._print_progress()
        console.print("[bold]Configure resource constraints:[/bold]\n")

        # CPU cores
        cpu_cores = IntPrompt.ask(
            "Available CPU cores",
            default=4,
            console=console,
        )
        self.config["resources"] = {"cpu_cores": cpu_cores}

        # Memory
        memory_gb = FloatPrompt.ask(
            "Available memory (GB)",
            default=16.0,
            console=console,
        )
        self.config["resources"]["memory_gb"] = memory_gb

        # Optional constraints
        console.print("\n[dim]Optional constraints (press Enter to skip):[/dim]")

        max_memory = FloatPrompt.ask(
            "Maximum memory constraint (GB)",
            default=None,
            console=console,
        )
        if max_memory:
            self.config["constraints"] = self.config.get("constraints", {})
            self.config["constraints"]["max_memory_gb"] = max_memory

        max_cost = FloatPrompt.ask(
            "Maximum cost per hour (USD)",
            default=None,
            console=console,
        )
        if max_cost:
            self.config["constraints"] = self.config.get("constraints", {})
            self.config["constraints"]["max_cost_per_hour"] = max_cost

        console.print("[green]✓[/green] Resources configured")

    def _prompt_event_log(self, max_attempts: int = 3) -> float | None:
        """Optionally enrich the data profile from a Spark event log.

        Asks whether an event log from a previous run is available and, if so,
        parses it to infer the input data size and derive tuning hints. Hints
        are merged into ``config["constraints"]`` (without overriding values
        the user already entered) so they reach the optimizer as resource
        constraints, mirroring the `optimize --event-log` behavior.

        Args:
            max_attempts: Maximum number of attempts to get a valid file path.

        Returns:
            Inferred input data size in GB, or None when unavailable.
        """
        has_log = Confirm.ask(
            "Do you have a Spark event log from a previous run?",
            default=False,
            console=console,
        )
        if not has_log:
            return None

        summary = None
        for _ in range(max_attempts):
            log_path = Prompt.ask(
                "Path to event log file (.gz accepted, Enter to skip)",
                default="",
                console=console,
            )
            if not log_path:
                return None

            path = Path(log_path)
            if not path.is_file():
                console.print(f"[red]File not found: {log_path}[/red]")
                continue

            try:
                summary = EventLogParser(path).parse()
            except (OSError, ValueError) as e:
                console.print(f"[yellow]Could not parse event log ({e}). Continuing without it.[/yellow]")
                return None

            self.config["event_log"] = str(path.absolute())
            break

        if summary is None:
            console.print("[yellow]Max attempts reached. Continuing without event log.[/yellow]")
            return None

        hints = summary.to_tuning_hints()
        raw_size = hints.pop("data_size_gb", None)
        inferred_size: float | None = round(float(raw_size), 2) if raw_size else None

        # Merge hints into constraints without overriding user-entered values
        constraints = self.config.setdefault("constraints", {})
        for key, value in hints.items():
            constraints.setdefault(key, value)

        notes = []
        if inferred_size:
            notes.append(f"data size {inferred_size} GB")
        notes.append(f"skew factor {hints['skew_factor']:.1f}")
        if hints["gc_pressure"]:
            notes.append(f"high GC pressure ({hints['gc_time_fraction']:.0%} of executor time)")
        if hints["spill_detected"]:
            notes.append(f"spill detected ({hints['spill_gb']:.2f} GB)")
        if hints["large_shuffles"]:
            notes.append(f"large shuffles ({hints['shuffle_total_gb']:.1f} GB)")

        self.config["event_log_inference"] = ", ".join(notes)
        console.print(f"[green]✓[/green] Event log parsed. [dim]Inferred: {self.config['event_log_inference']}[/dim]")

        return inferred_size

    def _step_data_profile(self) -> None:
        """Step 4: Data profile.

        Prompts the user to enter data characteristics, optionally pre-filled
        from a Spark event log of a previous run.
        """
        self._print_progress()
        console.print("[bold]Configure data characteristics:[/bold]\n")

        # Optional event-log enrichment (pre-fills the data size below)
        inferred_size = self._prompt_event_log()

        # Data size (defaults to the event-log inference when available)
        size_gb = FloatPrompt.ask(
            "Data size (GB)",
            default=inferred_size if inferred_size else 10.0,
            console=console,
        )
        self.config["data_profile"] = {"size_gb": size_gb}

        # Data format
        console.print("\n[dim]Data format options:[/dim]")
        formats = ["parquet", "delta", "json", "csv", "orc", "avro"]
        for i, fmt in enumerate(formats, 1):
            console.print(f"  [{i}] {fmt}")

        format_choice = Prompt.ask(
            "Select data format",
            default="parquet",
            console=console,
        )

        # Handle numeric or text input
        if format_choice.isdigit() and 1 <= int(format_choice) <= len(formats):
            self.config["data_profile"]["format"] = formats[int(format_choice) - 1]
        else:
            self.config["data_profile"]["format"] = format_choice

        # Compression
        compression = Prompt.ask(
            "Compression codec (optional)",
            default="",
            console=console,
        )
        if compression:
            self.config["data_profile"]["compression"] = compression

        console.print("[green]✓[/green] Data profile configured")

    def _step_spark_code(self, max_attempts: int = 3) -> None:
        """Step 5: Spark code input.

        Prompts the user for the Spark code file path.

        Args:
            max_attempts: Maximum number of attempts to get a valid file path.
        """
        self._print_progress()
        console.print("[bold]Provide Spark code:[/bold]\n")

        # File path
        for _ in range(max_attempts):
            code_path = Prompt.ask(
                "Path to Spark code file (.py)",
                console=console,
            )

            path = Path(code_path)
            if path.exists() and path.suffix == ".py":
                self.config["code_path"] = str(path.absolute())
                console.print(
                    f"[green]✓[/green] Code file: [cyan]{self.config['code_path']}[/cyan]",
                )
                return
            elif not path.exists():
                console.print(f"[red]File not found: {code_path}[/red]")
            else:
                console.print("[red]File must be a Python file (.py)[/red]")

        console.print("[yellow]Max attempts reached. Skipping code file.[/yellow]")

    def _step_objectives(self) -> None:
        """Step 6: Optimization objectives.

        Prompts the user to select one or more optimization objectives.
        Selecting more than one enables multi-objective (Pareto) search.
        """
        self._print_progress()
        console.print("[bold]Select optimization objectives:[/bold]\n")

        available = ObjectiveFunctionFactory.get_available_objectives()

        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="green")
        for i, name in enumerate(available, 1):
            suffix = " (default)" if name == self.DEFAULT_OBJECTIVE else ""
            table.add_row(f"  [{i}]", f"{name}{suffix} - {self.OBJECTIVE_DESCRIPTIONS.get(name, '')}")
        console.print(table)
        console.print(
            "\n[dim]Selecting more than one objective enables multi-objective (Pareto) search: "
            "you get a frontier of trade-off configurations instead of a single best one.[/dim]\n",
        )

        while True:
            raw = Prompt.ask(
                "Objectives (comma-separated numbers or names)",
                default="1",
                console=console,
            )

            selected: list[str] = []
            invalid: list[str] = []
            for token in (part.strip() for part in raw.split(",")):
                if not token:
                    continue
                if token.isdigit() and 1 <= int(token) <= len(available):
                    name = available[int(token) - 1]
                elif token in available:
                    name = token
                else:
                    invalid.append(token)
                    continue
                if name not in selected:
                    selected.append(name)

            if invalid:
                console.print(
                    f"[red]Unknown objective(s): {', '.join(invalid)}. Valid options: {', '.join(available)}[/red]",
                )
                continue

            if not selected:
                selected = [self.DEFAULT_OBJECTIVE]
            break

        self.config["objectives"] = selected
        console.print(f"[green]✓[/green] Objectives: [cyan]{', '.join(selected)}[/cyan]")

    def _print_export_formats(self) -> None:
        """Show the curated set of export formats available after a run."""
        console.print(
            "\n[dim]After the run, export the result with: spark-optima export -r result.json -f <format>[/dim]",
        )
        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="dim")
        for name, description in self.EXPORT_FORMATS:
            table.add_row(f"  {name}", description)
        console.print(table)
        console.print(f"[dim]Also available: {', '.join(self.ADDITIONAL_EXPORT_FORMATS)}[/dim]\n")

    def _step_optimization_settings(self) -> None:
        """Step 7: Optimization and output settings.

        Prompts the user for advanced optimization settings and shows the
        export formats available once the run completes.
        """
        self._print_progress()
        console.print("[bold]Optimization settings:[/bold]\n")

        # Optimization mode
        use_bayesian = Confirm.ask(
            "Use Bayesian optimization? (recommended)",
            default=True,
            console=console,
        )
        self.config["use_bayesian"] = use_bayesian

        if use_bayesian:
            multi_objective = len(self.config.get("objectives", [])) > 1
            console.print(
                "[dim]Guidance: ~20 trials for a quick look, 50 for a balanced search, "
                "100+ for thorough or multi-objective (Pareto) searches.[/dim]",
            )
            trials = IntPrompt.ask(
                "Number of optimization trials",
                default=100 if multi_objective else 50,
                console=console,
            )
            self.config["bayesian_trials"] = trials

        # Output format
        console.print("\n[dim]Output format options: table, json, yaml[/dim]")
        output_format = Prompt.ask(
            "Output format",
            default="table",
            console=console,
        )
        self.config["output_format"] = output_format

        # Show what the result can be exported to afterwards
        self._print_export_formats()

        console.print("[green]✓[/green] Optimization settings configured")

    def _print_summary(self) -> None:
        """Print configuration summary."""
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold green]✅ Configuration Complete![/bold green]",
                border_style="green",
            ),
        )

        summary = Table(title="Configuration Summary", show_header=False)
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Platform", self.config.get("platform", "N/A"))
        summary.add_row("Spark Version", self.config.get("spark_version", "N/A"))
        summary.add_row(
            "Resources",
            f"{self.config.get('resources', {}).get('cpu_cores', 'N/A')} cores, "
            f"{self.config.get('resources', {}).get('memory_gb', 'N/A')} GB",
        )
        summary.add_row(
            "Data",
            f"{self.config.get('data_profile', {}).get('size_gb', 'N/A')} GB "
            f"({self.config.get('data_profile', {}).get('format', 'N/A')})",
        )
        summary.add_row("Code Path", self.config.get("code_path", "N/A"))

        objectives = self.config.get("objectives", [self.DEFAULT_OBJECTIVE])
        summary.add_row("Objectives", ", ".join(objectives))

        if self.config.get("event_log"):
            summary.add_row("Event Log", self.config["event_log"])
        if self.config.get("event_log_inference"):
            summary.add_row("Inferred From Log", self.config["event_log_inference"])

        use_bayesian = self.config.get("use_bayesian", True)
        bayesian_display = str(use_bayesian)
        if use_bayesian and self.config.get("bayesian_trials"):
            bayesian_display = f"True ({self.config['bayesian_trials']} trials)"
        summary.add_row("Bayesian Opt.", bayesian_display)
        summary.add_row("Output Format", self.config.get("output_format", "table"))

        console.print(summary)

        # The wizard handoff in the CLI currently runs the default single
        # objective; surface how to apply a custom selection explicitly.
        if objectives != [self.DEFAULT_OBJECTIVE]:
            flags = " ".join(f"--objective {name}" for name in objectives)
            console.print(
                "[dim]Note: if your selected objectives are not applied by this run, "
                f"use: spark-optima optimize -c <code> {flags}[/dim]",
            )

        console.print()

    def run(self) -> dict[str, Any]:
        """Run the interactive configuration wizard.

        Guides the user through all configuration steps and returns
        the collected configuration.

        Returns:
            Dictionary with all collected configuration values.
        """
        self._print_header()

        # Run each step
        self._step_platform_selection()
        self._step_spark_version()
        self._step_resource_constraints()
        self._step_data_profile()
        self._step_spark_code()
        self._step_objectives()
        self._step_optimization_settings()

        # Show summary
        self._print_summary()

        # Confirm
        confirmed = Confirm.ask(
            "Proceed with optimization?",
            default=True,
            console=console,
        )

        if not confirmed:
            console.print("[yellow]Optimization cancelled.[/yellow]")
            return {}

        return self.config


def run_wizard() -> dict[str, Any]:
    """Convenience function to run the configuration wizard.

    Returns:
        Collected configuration or empty dict if cancelled.
    """
    wizard = ConfigurationWizard()
    return wizard.run()
