# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the heuristic rules."""

from __future__ import annotations

from spark_optima.core.config_engine.models import ParameterCategory
from spark_optima.core.heuristics.rules import HeuristicRuleDef, RuleRegistry


class TestHeuristicRuleDef:
    """Test cases for HeuristicRuleDef."""

    def test_can_apply_with_dependencies(self):
        """Test can_apply with dependencies."""
        rule = HeuristicRuleDef(
            param_name="spark.executor.memory",
            category=ParameterCategory.MEMORY,
            formula="total_memory / num_executors",
            depends_on=["total_memory", "num_executors"],
        )

        assert rule.can_apply({"total_memory", "num_executors"}, "local", "3.5.0")
        assert not rule.can_apply({"total_memory"}, "local", "3.5.0")

    def test_can_apply_platform_filter(self):
        """Test can_apply with platform filtering."""
        rule = HeuristicRuleDef(
            param_name="spark.dynamicAllocation.enabled",
            category=ParameterCategory.DYNAMIC_ALLOCATION,
            applies_to=["aws_glue", "databricks"],
        )

        assert rule.can_apply(set(), "aws_glue", "3.5.0")
        assert rule.can_apply(set(), "databricks", "3.5.0")
        assert not rule.can_apply(set(), "local", "3.5.0")

    def test_can_apply_version_filter(self):
        """Test can_apply with version filtering."""
        rule = HeuristicRuleDef(
            param_name="spark.sql.adaptive.enabled",
            category=ParameterCategory.SQL,
            min_version="3.0.0",
        )

        assert rule.can_apply(set(), "local", "3.0.0")
        assert rule.can_apply(set(), "local", "3.5.0")
        assert rule.can_apply(set(), "local", "4.0.0")
        assert not rule.can_apply(set(), "local", "2.4.0")

    def test_version_in_range_with_max(self):
        """Test version range with max version."""
        rule = HeuristicRuleDef(
            param_name="test.param",
            category=ParameterCategory.RUNTIME,
            min_version="3.0.0",
            max_version="3.5.0",
        )

        assert rule._version_in_range("3.0.0")
        assert rule._version_in_range("3.3.0")
        assert rule._version_in_range("3.5.0")
        assert not rule._version_in_range("2.4.0")
        assert not rule._version_in_range("4.0.0")


class TestRuleRegistry:
    """Test cases for RuleRegistry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = RuleRegistry()

    def test_registry_has_rules(self):
        """Test that registry has rules."""
        rules = self.registry.get_all_rules()
        assert len(rules) > 0

    def test_get_rules_by_category(self):
        """Test getting rules by category."""
        memory_rules = self.registry.get_rules_by_category(ParameterCategory.MEMORY)
        assert len(memory_rules) > 0
        assert all(r.category == ParameterCategory.MEMORY for r in memory_rules)

    def test_get_rules_by_priority(self):
        """Test getting rules by priority."""
        high_rules = self.registry.get_rules_by_priority("high")
        assert len(high_rules) > 0
        assert all(r.priority == "high" for r in high_rules)

    def test_get_rules_for_platform(self):
        """Test getting rules for platform."""
        local_rules = self.registry.get_rules_for_platform("local")
        assert len(local_rules) > 0
        assert all("local" in r.applies_to for r in local_rules)

    def test_get_rule_by_name(self):
        """Test getting rule by parameter name."""
        rule = self.registry.get_rule("spark.executor.memory")
        assert rule is not None
        assert rule.param_name == "spark.executor.memory"
        assert rule.category == ParameterCategory.MEMORY

    def test_get_rule_not_found(self):
        """Test getting non-existent rule."""
        rule = self.registry.get_rule("nonexistent.parameter")
        assert rule is None

    def test_add_custom_rule(self):
        """Test adding custom rule."""
        custom_rule = HeuristicRuleDef(
            param_name="custom.param",
            category=ParameterCategory.RUNTIME,
            formula="100",
            base_value=100,
        )

        initial_count = len(self.registry.get_all_rules())
        self.registry.add_rule(custom_rule)
        final_count = len(self.registry.get_all_rules())

        assert final_count == initial_count + 1
        assert self.registry.get_rule("custom.param") == custom_rule

    def test_get_applicable_rules(self):
        """Test getting applicable rules."""
        variables = {"total_memory_gb", "num_executors"}
        rules = self.registry.get_applicable_rules(
            variables=variables,
            platform="local",
            version="3.5.0",
        )

        assert len(rules) > 0
        # All returned rules should have their dependencies satisfied
        for rule in rules:
            assert all(dep in variables for dep in rule.depends_on)

    def test_get_applicable_rules_with_category(self):
        """Test getting applicable rules with category filter."""
        variables = {"total_memory_gb", "num_executors"}
        rules = self.registry.get_applicable_rules(
            variables=variables,
            platform="local",
            version="3.5.0",
            category=ParameterCategory.MEMORY,
        )

        assert len(rules) > 0
        assert all(r.category == ParameterCategory.MEMORY for r in rules)
