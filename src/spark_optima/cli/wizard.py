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

    TOTAL_STEPS = 6

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
        list(platform_options.keys())
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

    def _step_data_profile(self) -> None:
        """Step 4: Data profile.

        Prompts the user to enter data characteristics.
        """
        self._print_progress()
        console.print("[bold]Configure data characteristics:[/bold]\n")

        # Data size
        size_gb = FloatPrompt.ask(
            "Data size (GB)",
            default=10.0,
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

    def _step_optimization_settings(self) -> None:
        """Step 6: Optimization settings (bonus step).

        Prompts the user for advanced optimization settings.
        """
        console.print("\n[bold]Optimization settings:[/bold]\n")

        # Optimization mode
        use_bayesian = Confirm.ask(
            "Use Bayesian optimization? (recommended)",
            default=True,
            console=console,
        )
        self.config["use_bayesian"] = use_bayesian

        if use_bayesian:
            trials = IntPrompt.ask(
                "Number of optimization trials",
                default=50,
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
        summary.add_row("Bayesian Opt.", str(self.config.get("use_bayesian", True)))
        summary.add_row("Output Format", self.config.get("output_format", "table"))

        console.print(summary)
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
