# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Workload templates for Spark Optima.

This module provides curated baseline Spark configurations for common
workload archetypes (batch ETL, streaming, ML training, interactive
analytics). Templates are stored as YAML files in ``data/templates`` at the
repository root and are loaded through the TemplateRegistry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TemplateParameter:
    """A single curated Spark parameter inside a workload template.

    Attributes:
        value: Curated parameter value.
        comment: Short rationale explaining why the value was chosen.

    """

    value: Any
    comment: str = ""


@dataclass
class WorkloadTemplate:
    """Curated baseline Spark configuration for a workload archetype.

    Attributes:
        name: Template identifier (e.g. "etl-batch").
        display_name: Human-readable template name.
        description: What the template is for.
        workload_traits: Characteristics of the targeted workload.
        config: Curated Spark parameters with per-parameter rationale.
        recommended_for: Scenarios where the template is a good fit.
        not_recommended_for: Scenarios where the template should be avoided.

    """

    name: str
    display_name: str
    description: str
    workload_traits: list[str] = field(default_factory=list)
    config: dict[str, TemplateParameter] = field(default_factory=dict)
    recommended_for: list[str] = field(default_factory=list)
    not_recommended_for: list[str] = field(default_factory=list)

    def config_values(self) -> dict[str, Any]:
        """Return the template configuration as plain parameter/value pairs.

        Returns:
            Dictionary mapping parameter names to their curated values.

        """
        return {name: parameter.value for name, parameter in self.config.items()}

    def apply_to(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge a user configuration on top of the template baseline.

        The template acts as the base layer; any parameter present in the
        user configuration wins over the template value.

        Args:
            config: User-provided Spark configuration.

        Returns:
            Merged configuration dictionary.

        """
        merged = self.config_values()
        merged.update(config)
        return merged

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary representation.

        Returns:
            Dictionary with all template fields.

        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "workload_traits": list(self.workload_traits),
            "config": {
                name: {"value": parameter.value, "comment": parameter.comment}
                for name, parameter in self.config.items()
            },
            "recommended_for": list(self.recommended_for),
            "not_recommended_for": list(self.not_recommended_for),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadTemplate:
        """Create a template from a parsed YAML dictionary.

        Args:
            data: Parsed template data.

        Returns:
            WorkloadTemplate instance.

        Raises:
            ValueError: If a required field is missing or empty.

        """
        for required in ("name", "display_name", "description"):
            if not data.get(required):
                raise ValueError(f"Template is missing required field: {required}")

        config: dict[str, TemplateParameter] = {}
        for param_name, param_data in (data.get("config") or {}).items():
            if isinstance(param_data, dict):
                config[str(param_name)] = TemplateParameter(
                    value=param_data.get("value"),
                    comment=str(param_data.get("comment", "")),
                )
            else:
                # Allow the shorthand "param: value" form without a comment
                config[str(param_name)] = TemplateParameter(value=param_data)

        return cls(
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            description=str(data["description"]).strip(),
            workload_traits=[str(trait) for trait in data.get("workload_traits") or []],
            config=config,
            recommended_for=[str(item) for item in data.get("recommended_for") or []],
            not_recommended_for=[str(item) for item in data.get("not_recommended_for") or []],
        )


class TemplateRegistry:
    """Registry of curated workload templates loaded from YAML files.

    Templates live in ``data/templates`` at the repository root, resolved the
    same way the Spark parameter database resolves ``data/configs``.

    Attributes:
        template_dir: Directory containing template YAML files.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [template.name for template in registry.list_templates()]
        >>> template = registry.get_template("etl-batch")
        >>> merged = template.apply_to({"spark.executor.memory": "8g"})

    """

    def __init__(self, template_dir: str | Path | None = None) -> None:
        """Initialize the registry and load all templates.

        Args:
            template_dir: Directory containing template YAML files.
                Defaults to data/templates relative to the package root.

        """
        if template_dir is None:
            # Default to the package data/templates directory (same path
            # resolution as ConfigDatabase in core/config_engine/database.py)
            package_root = Path(__file__).parent.parent.parent.parent
            template_dir = package_root / "data" / "templates"

        self.template_dir = Path(template_dir)
        self._templates: dict[str, WorkloadTemplate] = {}
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        """Load all template files from the template directory."""
        if not self.template_dir.exists():
            logger.warning(f"Template directory not found: {self.template_dir}")
            return

        for template_file in sorted(self.template_dir.glob("*.yaml")):
            try:
                self._load_template_file(template_file)
            except (OSError, yaml.YAMLError, ValueError, KeyError, AttributeError) as e:
                logger.error(f"Failed to load template file {template_file}: {e}")

    def _load_template_file(self, filepath: Path) -> None:
        """Load a single template YAML file.

        Args:
            filepath: Path to the template YAML file.

        Raises:
            ValueError: If the file does not contain a valid template mapping.

        """
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Template file {filepath} must contain a YAML mapping")

        template = WorkloadTemplate.from_dict(data)
        self._templates[template.name] = template
        logger.debug(f"Loaded template '{template.name}' with {len(template.config)} parameters")

    def list_templates(self) -> list[WorkloadTemplate]:
        """List all loaded templates.

        Returns:
            Templates sorted by name.

        """
        return [self._templates[name] for name in sorted(self._templates)]

    def get_template(self, name: str) -> WorkloadTemplate:
        """Get a template by name.

        Args:
            name: Template identifier (e.g. "etl-batch").

        Returns:
            WorkloadTemplate instance.

        Raises:
            ValueError: If no template with that name exists.

        """
        template = self._templates.get(name)
        if template is None:
            available = ", ".join(sorted(self._templates)) or "none"
            raise ValueError(f"Unknown template: '{name}'. Available templates: {available}")
        return template

    def __contains__(self, name: str) -> bool:
        """Check if a template name is registered."""
        return name in self._templates

    def __len__(self) -> int:
        """Return the number of loaded templates."""
        return len(self._templates)
