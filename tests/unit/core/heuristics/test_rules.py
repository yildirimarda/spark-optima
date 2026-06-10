# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the heuristic rules."""

from __future__ import annotations

from spark_optima.core.config_engine.models import ParameterCategory
from spark_optima.core.heuristics.evaluator import FormulaEvaluator
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


class TestSpeculationRules:
    """Test cases for speculative execution rules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = RuleRegistry()
        self.evaluator = FormulaEvaluator()

    def test_speculation_rule_present(self):
        """Test spark.speculation rule exists with skew/long-running condition."""
        rule = self.registry.get_rule("spark.speculation")

        assert rule is not None
        assert rule.category == ParameterCategory.SCHEDULER
        assert rule.priority == "medium"
        assert rule.conditions == {"large_shuffles": True}
        assert self.evaluator.evaluate(rule.formula) is True

    def test_speculation_quantile_rule(self):
        """Test spark.speculation.quantile rule evaluates to 0.75."""
        rule = self.registry.get_rule("spark.speculation.quantile")

        assert rule is not None
        assert rule.priority == "medium"
        assert rule.conditions == {"large_shuffles": True}
        assert self.evaluator.evaluate(rule.formula) == 0.75

    def test_speculation_multiplier_rule(self):
        """Test spark.speculation.multiplier rule evaluates to 1.5."""
        rule = self.registry.get_rule("spark.speculation.multiplier")

        assert rule is not None
        assert rule.priority == "medium"
        assert rule.conditions == {"large_shuffles": True}
        assert self.evaluator.evaluate(rule.formula) == 1.5

    def test_speculation_rules_apply_on_all_platforms(self):
        """Test speculation rules apply to all default platforms."""
        rule = self.registry.get_rule("spark.speculation")

        assert rule is not None
        for platform in ("local", "databricks", "aws_glue", "azure_synapse"):
            assert rule.can_apply(set(), platform, "3.5.0")


class TestDataAwareDynamicAllocationRules:
    """Test cases for data-aware dynamic allocation bounds."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = RuleRegistry()
        self.evaluator = FormulaEvaluator()

    def _data_aware_rule(self) -> HeuristicRuleDef:
        """Return the data-size scaled maxExecutors rule."""
        rules = [
            r
            for r in self.registry.get_all_rules()
            if r.param_name == "spark.dynamicAllocation.maxExecutors" and "data_size_gb" in r.depends_on
        ]
        assert len(rules) == 1
        return rules[0]

    def test_data_aware_max_executors_present(self):
        """Test the data-aware maxExecutors rule exists alongside the capacity rule."""
        all_max = [r for r in self.registry.get_all_rules() if r.param_name == "spark.dynamicAllocation.maxExecutors"]
        assert len(all_max) == 2

        rule = self._data_aware_rule()
        assert rule.priority == "high"
        assert rule.conditions == {"data_size_gb": ">0"}
        assert rule.applies_to == ["databricks", "aws_glue", "aws_emr", "azure_synapse", "gcp_dataproc", "kubernetes"]

    def test_data_aware_max_executors_formula_scales_with_data(self):
        """Test formula scales with data size: ceil(data_size_gb / 4)."""
        rule = self._data_aware_rule()

        assert self.evaluator.evaluate(rule.formula, {"data_size_gb": 100}) == 25
        assert self.evaluator.evaluate(rule.formula, {"data_size_gb": 101}) == 26

    def test_data_aware_max_executors_formula_bounds(self):
        """Test formula is bounded to the 10-128 range."""
        rule = self._data_aware_rule()

        # Lower bound: tiny data still keeps 10 executors available
        assert self.evaluator.evaluate(rule.formula, {"data_size_gb": 1}) == 10
        # Upper bound: huge data is capped at 128 executors
        assert self.evaluator.evaluate(rule.formula, {"data_size_gb": 10000}) == 128

    def test_data_aware_max_executors_platform_filter(self):
        """Test the rule only applies on cloud platforms with data size known."""
        rule = self._data_aware_rule()

        assert rule.can_apply({"data_size_gb"}, "databricks", "3.5.0")
        assert not rule.can_apply({"data_size_gb"}, "local", "3.5.0")
        assert not rule.can_apply(set(), "databricks", "3.5.0")


class TestAqeFineTuningRules:
    """Test cases for AQE fine-tuning rules."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = RuleRegistry()
        self.evaluator = FormulaEvaluator()

    def test_advisory_partition_size_rule_present(self):
        """Test advisoryPartitionSizeInBytes rule exists with sane metadata."""
        rule = self.registry.get_rule("spark.sql.adaptive.advisoryPartitionSizeInBytes")

        assert rule is not None
        assert rule.category == ParameterCategory.SQL
        assert rule.min_version == "3.0.0"
        assert rule.base_value == "64m"
        assert "data_size_gb" in rule.depends_on

    def test_advisory_partition_size_formula_data_aware(self):
        """Test advisory partition size stays in the 64-128MB byte range."""
        rule = self.registry.get_rule("spark.sql.adaptive.advisoryPartitionSizeInBytes")
        assert rule is not None

        # Large data: clamped to 128MB upper bound
        value = self.evaluator.evaluate(
            rule.formula,
            {"data_size_gb": 100, "spark_sql_shuffle_partitions": 400},
        )
        assert value == 128 * 1024 * 1024

        # Small data: clamped to 64MB lower bound
        value = self.evaluator.evaluate(
            rule.formula,
            {"data_size_gb": 1, "spark_sql_shuffle_partitions": 200},
        )
        assert value == 64 * 1024 * 1024

        # Mid-range data lands between the bounds
        value = self.evaluator.evaluate(
            rule.formula,
            {"data_size_gb": 20, "spark_sql_shuffle_partitions": 200},
        )
        assert 64 * 1024 * 1024 <= value <= 128 * 1024 * 1024

    def test_skewed_partition_factor_rules(self):
        """Test default and skew-conditioned skewedPartitionFactor rules."""
        rules = [
            r
            for r in self.registry.get_all_rules()
            if r.param_name == "spark.sql.adaptive.skewJoin.skewedPartitionFactor"
        ]
        assert len(rules) == 2

        default_rules = [r for r in rules if not r.conditions]
        skewed_rules = [r for r in rules if r.conditions]
        assert len(default_rules) == 1
        assert len(skewed_rules) == 1

        # Default stays at 5x median
        assert self.evaluator.evaluate(default_rules[0].formula) == 5.0

        # Skewed data lowers the threshold to 3x median
        skewed = skewed_rules[0]
        assert skewed.conditions == {"skew_factor": ">1.5"}
        assert skewed.priority == "high"
        assert self.evaluator.evaluate(skewed.formula) == 3.0

    def test_skewed_partition_threshold_rule(self):
        """Test skewedPartitionThresholdInBytes is set to 256m."""
        rule = self.registry.get_rule(
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
        )

        assert rule is not None
        assert rule.base_value == "256m"
        assert rule.min_version == "3.0.0"
