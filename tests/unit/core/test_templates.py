# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Unit tests for the workload template registry (Workstream S)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spark_optima.core.templates import TemplateParameter, TemplateRegistry, WorkloadTemplate

if TYPE_CHECKING:
    from pathlib import Path

BUNDLED_TEMPLATE_NAMES = {"etl-batch", "streaming", "ml-training", "interactive"}


@pytest.fixture
def registry() -> TemplateRegistry:
    """Create a registry loading the bundled templates."""
    return TemplateRegistry()


class TestWorkloadTemplateModel:
    """Test cases for the WorkloadTemplate dataclass."""

    @staticmethod
    def _sample_template() -> WorkloadTemplate:
        """Build a small template for merge tests."""
        return WorkloadTemplate(
            name="sample",
            display_name="Sample",
            description="A sample template",
            workload_traits=["trait"],
            config={
                "spark.sql.shuffle.partitions": TemplateParameter(value="200", comment="baseline"),
                "spark.serializer": TemplateParameter(
                    value="org.apache.spark.serializer.KryoSerializer",
                    comment="faster",
                ),
            },
            recommended_for=["tests"],
            not_recommended_for=["production"],
        )

    def test_config_values_returns_plain_dict(self) -> None:
        """config_values flattens TemplateParameter objects to raw values."""
        template = self._sample_template()

        values = template.config_values()

        assert values == {
            "spark.sql.shuffle.partitions": "200",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        }

    def test_apply_to_user_config_wins(self) -> None:
        """apply_to merges with the template as base and user values winning."""
        template = self._sample_template()

        merged = template.apply_to(
            {
                "spark.sql.shuffle.partitions": "999",
                "spark.executor.memory": "8g",
            },
        )

        # User override wins over the template value
        assert merged["spark.sql.shuffle.partitions"] == "999"
        # Template-only parameters are kept
        assert merged["spark.serializer"] == "org.apache.spark.serializer.KryoSerializer"
        # User-only parameters are added
        assert merged["spark.executor.memory"] == "8g"

    def test_apply_to_does_not_mutate_inputs(self) -> None:
        """apply_to leaves both the template and the user config untouched."""
        template = self._sample_template()
        user_config = {"spark.sql.shuffle.partitions": "999"}

        template.apply_to(user_config)

        assert template.config["spark.sql.shuffle.partitions"].value == "200"
        assert user_config == {"spark.sql.shuffle.partitions": "999"}

    def test_apply_to_empty_user_config(self) -> None:
        """apply_to with an empty user config returns the template baseline."""
        template = self._sample_template()

        assert template.apply_to({}) == template.config_values()

    def test_to_dict_shape(self) -> None:
        """to_dict produces a JSON-serializable representation."""
        template = self._sample_template()

        data = template.to_dict()

        assert data["name"] == "sample"
        assert data["display_name"] == "Sample"
        assert data["config"]["spark.serializer"]["comment"] == "faster"
        assert data["recommended_for"] == ["tests"]
        assert data["not_recommended_for"] == ["production"]

    def test_from_dict_requires_core_fields(self) -> None:
        """from_dict rejects templates missing required fields."""
        with pytest.raises(ValueError, match="display_name"):
            WorkloadTemplate.from_dict({"name": "x", "description": "y"})

    def test_from_dict_supports_shorthand_config(self) -> None:
        """from_dict accepts plain values without a comment wrapper."""
        template = WorkloadTemplate.from_dict(
            {
                "name": "short",
                "display_name": "Short",
                "description": "shorthand config",
                "config": {"spark.executor.cores": "4"},
            },
        )

        assert template.config["spark.executor.cores"].value == "4"
        assert template.config["spark.executor.cores"].comment == ""


class TestTemplateRegistryBundled:
    """Test cases for the bundled data/templates YAML files."""

    def test_all_four_templates_load(self, registry: TemplateRegistry) -> None:
        """All four curated templates load from data/templates."""
        names = {template.name for template in registry.list_templates()}

        assert names == BUNDLED_TEMPLATE_NAMES
        assert len(registry) == 4

    def test_templates_have_complete_schema(self, registry: TemplateRegistry) -> None:
        """Every bundled template carries the full schema with rationale."""
        for template in registry.list_templates():
            assert template.name
            assert template.display_name
            assert template.description
            assert template.workload_traits, f"{template.name} has no workload traits"
            assert template.config, f"{template.name} has no config"
            assert template.recommended_for, f"{template.name} has no recommended_for"
            assert template.not_recommended_for, f"{template.name} has no not_recommended_for"
            for param_name, parameter in template.config.items():
                assert param_name.startswith("spark."), f"{template.name}: {param_name}"
                assert parameter.value is not None, f"{template.name}: {param_name} has no value"
                assert parameter.comment, f"{template.name}: {param_name} has no comment"

    def test_streaming_template_disables_dynamic_allocation(self, registry: TemplateRegistry) -> None:
        """Streaming keeps a fixed executor footprint with backpressure."""
        values = registry.get_template("streaming").config_values()

        assert values["spark.dynamicAllocation.enabled"] == "false"
        assert values["spark.streaming.backpressure.enabled"] == "true"

    def test_etl_batch_template_enables_aqe_and_elasticity(self, registry: TemplateRegistry) -> None:
        """Batch ETL enables AQE and dynamic allocation with shuffle tracking."""
        values = registry.get_template("etl-batch").config_values()

        assert values["spark.sql.adaptive.enabled"] == "true"
        assert values["spark.dynamicAllocation.enabled"] == "true"
        assert values["spark.dynamicAllocation.shuffleTracking.enabled"] == "true"

    def test_ml_training_template_is_memory_oriented(self, registry: TemplateRegistry) -> None:
        """ML training raises memory fraction, uses Kryo and off-heap memory."""
        values = registry.get_template("ml-training").config_values()

        assert values["spark.memory.fraction"] == "0.8"
        assert values["spark.memory.offHeap.enabled"] == "true"
        assert values["spark.serializer"].endswith("KryoSerializer")

    def test_interactive_template_is_broadcast_friendly(self, registry: TemplateRegistry) -> None:
        """Interactive raises the broadcast threshold and lowers parallelism."""
        values = registry.get_template("interactive").config_values()

        assert values["spark.sql.autoBroadcastJoinThreshold"] == "100m"
        assert values["spark.sql.shuffle.partitions"] == "64"

    def test_get_template_unknown_name(self, registry: TemplateRegistry) -> None:
        """Unknown template names raise a ValueError listing the options."""
        with pytest.raises(ValueError, match="etl-batch"):
            registry.get_template("does-not-exist")

    def test_contains_operator(self, registry: TemplateRegistry) -> None:
        """The registry supports membership checks by name."""
        assert "etl-batch" in registry
        assert "does-not-exist" not in registry


class TestTemplateRegistryCustomDir:
    """Test cases for registry behavior with custom directories."""

    def test_custom_directory_is_used(self, tmp_path: Path) -> None:
        """The registry loads templates from a custom directory."""
        (tmp_path / "custom.yaml").write_text(
            "name: custom\n"
            "display_name: Custom\n"
            "description: custom template\n"
            "config:\n"
            "  spark.executor.memory:\n"
            "    value: '4g'\n"
            "    comment: test value\n",
        )

        registry = TemplateRegistry(template_dir=tmp_path)

        assert len(registry) == 1
        assert registry.get_template("custom").config_values() == {"spark.executor.memory": "4g"}

    def test_missing_directory_yields_empty_registry(self, tmp_path: Path) -> None:
        """A missing template directory produces an empty registry."""
        registry = TemplateRegistry(template_dir=tmp_path / "nope")

        assert len(registry) == 0
        assert registry.list_templates() == []

    def test_invalid_yaml_file_is_skipped(self, tmp_path: Path) -> None:
        """Invalid template files are skipped without breaking the registry."""
        (tmp_path / "broken.yaml").write_text("- just\n- a\n- list\n")
        (tmp_path / "good.yaml").write_text(
            "name: good\ndisplay_name: Good\ndescription: ok\n",
        )

        registry = TemplateRegistry(template_dir=tmp_path)

        assert len(registry) == 1
        assert "good" in registry
