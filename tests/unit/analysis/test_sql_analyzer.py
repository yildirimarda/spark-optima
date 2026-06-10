# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the sqlglot-based SQL analyzer."""

from __future__ import annotations

import pytest

from spark_optima.analysis.models import SeverityLevel
from spark_optima.analysis.sql_analyzer import SQLAnalyzer, SQLFinding


@pytest.fixture
def analyzer() -> SQLAnalyzer:
    """Provide a fresh SQLAnalyzer instance."""
    return SQLAnalyzer()


def finding_types(findings: list[SQLFinding]) -> list[str]:
    """Extract the finding_type of each finding."""
    return [f.finding_type for f in findings]


def by_type(findings: list[SQLFinding], finding_type: str) -> list[SQLFinding]:
    """Filter findings by type."""
    return [f for f in findings if f.finding_type == finding_type]


class TestSelectStar:
    """Tests for sql_select_star detection."""

    def test_top_level_select_star_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that a top-level SELECT * is flagged."""
        findings = by_type(analyzer.analyze("SELECT * FROM events"), "sql_select_star")
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.LOW
        assert "SELECT *" in findings[0].fragment

    def test_qualified_star_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that a table-qualified star (t.*) is flagged."""
        findings = by_type(analyzer.analyze("SELECT e.* FROM events e"), "sql_select_star")
        assert len(findings) == 1

    def test_subquery_select_star_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that SELECT * inside a subquery is flagged."""
        sql = "SELECT id FROM (SELECT * FROM raw_events) sub"
        findings = by_type(analyzer.analyze(sql), "sql_select_star")
        assert len(findings) == 1

    def test_count_star_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that COUNT(*) inside an aggregate is not flagged."""
        findings = analyzer.analyze("SELECT COUNT(*) FROM events")
        assert by_type(findings, "sql_select_star") == []

    def test_explicit_columns_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an explicit column list is not flagged."""
        findings = analyzer.analyze("SELECT id, name FROM events")
        assert by_type(findings, "sql_select_star") == []

    def test_star_and_count_star_mixed(self, analyzer: SQLAnalyzer) -> None:
        """Test that a projection star is flagged once even next to COUNT(*)."""
        sql = "SELECT *, COUNT(*) OVER () AS total FROM events"
        findings = by_type(analyzer.analyze(sql), "sql_select_star")
        assert len(findings) == 1

    def test_one_finding_per_select(self, analyzer: SQLAnalyzer) -> None:
        """Test that multiple stars in one SELECT produce a single finding."""
        sql = "SELECT a.*, b.* FROM a JOIN b ON a.id = b.id"
        findings = by_type(analyzer.analyze(sql), "sql_select_star")
        assert len(findings) == 1

    def test_case_insensitive(self, analyzer: SQLAnalyzer) -> None:
        """Test that lowercase select * is flagged."""
        findings = by_type(analyzer.analyze("select * from events"), "sql_select_star")
        assert len(findings) == 1


class TestCartesianJoin:
    """Tests for sql_cartesian_join detection."""

    def test_explicit_cross_join_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an explicit CROSS JOIN is flagged HIGH."""
        findings = by_type(analyzer.analyze("SELECT a.id FROM a CROSS JOIN b"), "sql_cartesian_join")
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.HIGH

    def test_explicit_cross_join_with_where_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an explicit CROSS JOIN is flagged even with a WHERE clause."""
        sql = "SELECT a.id FROM a CROSS JOIN b WHERE a.x > 1"
        findings = by_type(analyzer.analyze(sql), "sql_cartesian_join")
        assert len(findings) == 1

    def test_comma_join_without_where_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that comma-separated FROM tables without WHERE are flagged."""
        findings = by_type(analyzer.analyze("SELECT a.id FROM a, b"), "sql_cartesian_join")
        assert len(findings) == 1
        assert "implicit" in findings[0].description.lower()

    def test_comma_join_with_where_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that a comma join constrained by a WHERE clause is not flagged."""
        sql = "SELECT a.id FROM a, b WHERE a.id = b.id"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_cartesian_join") == []

    def test_comma_join_with_unrelated_where_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test the conservative rule: any WHERE at all suppresses comma-join flags."""
        sql = "SELECT a.id FROM a, b WHERE a.x > 1"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_cartesian_join") == []

    def test_equi_join_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an explicit JOIN with an ON condition is not flagged."""
        sql = "SELECT u.id FROM users u JOIN orders o ON u.id = o.user_id"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_cartesian_join") == []

    def test_cross_in_string_literal_not_explicit(self, analyzer: SQLAnalyzer) -> None:
        """Test that 'cross join' inside a string literal does not mark explicitness."""
        sql = "SELECT a.id FROM a, b WHERE a.name = 'cross join'"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_cartesian_join") == []

    def test_lowercase_cross_join_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that lowercase cross join is flagged."""
        findings = by_type(analyzer.analyze("select a.id from a cross join b"), "sql_cartesian_join")
        assert len(findings) == 1


class TestOrderByWithoutLimit:
    """Tests for sql_orderby_without_limit detection."""

    def test_orderby_without_limit_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an outermost ORDER BY without LIMIT is flagged."""
        findings = by_type(analyzer.analyze("SELECT id FROM t ORDER BY id"), "sql_orderby_without_limit")
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.MEDIUM

    def test_orderby_with_limit_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that ORDER BY with LIMIT is not flagged."""
        findings = analyzer.analyze("SELECT id FROM t ORDER BY id LIMIT 10")
        assert by_type(findings, "sql_orderby_without_limit") == []

    def test_no_orderby_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that a query without ORDER BY is not flagged."""
        findings = analyzer.analyze("SELECT id FROM t")
        assert by_type(findings, "sql_orderby_without_limit") == []

    def test_subquery_orderby_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that ORDER BY in a subquery (not outermost) is not flagged."""
        sql = "SELECT id FROM (SELECT id FROM t ORDER BY id) sub"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_orderby_without_limit") == []

    def test_union_with_outer_orderby_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that ORDER BY attached to a UNION ALL without LIMIT is flagged."""
        sql = "SELECT id FROM a UNION ALL SELECT id FROM b ORDER BY id"
        findings = by_type(analyzer.analyze(sql), "sql_orderby_without_limit")
        assert len(findings) == 1

    def test_cte_outer_orderby_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that the outer ORDER BY of a CTE query is flagged."""
        sql = "WITH x AS (SELECT id FROM t) SELECT id FROM x ORDER BY id"
        findings = by_type(analyzer.analyze(sql), "sql_orderby_without_limit")
        assert len(findings) == 1


class TestUnionInsteadOfUnionAll:
    """Tests for sql_union_instead_of_union_all detection."""

    def test_union_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that a plain UNION is flagged LOW with advisory wording."""
        findings = by_type(
            analyzer.analyze("SELECT id FROM a UNION SELECT id FROM b"), "sql_union_instead_of_union_all"
        )
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.LOW
        assert "UNION ALL" in findings[0].suggestion

    def test_union_all_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that UNION ALL is not flagged."""
        findings = analyzer.analyze("SELECT id FROM a UNION ALL SELECT id FROM b")
        assert by_type(findings, "sql_union_instead_of_union_all") == []

    def test_chained_unions_each_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that each distinct UNION in a chain is flagged."""
        sql = "SELECT id FROM a UNION SELECT id FROM b UNION SELECT id FROM c"
        findings = by_type(analyzer.analyze(sql), "sql_union_instead_of_union_all")
        assert len(findings) == 2

    def test_intersect_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that INTERSECT is not reported as a UNION finding."""
        findings = analyzer.analyze("SELECT id FROM a INTERSECT SELECT id FROM b")
        assert by_type(findings, "sql_union_instead_of_union_all") == []


class TestLeadingWildcardLike:
    """Tests for sql_leading_wildcard_like detection."""

    def test_leading_percent_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that LIKE '%abc' is flagged."""
        findings = by_type(analyzer.analyze("SELECT id FROM t WHERE name LIKE '%abc'"), "sql_leading_wildcard_like")
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.MEDIUM

    def test_leading_underscore_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that LIKE '_abc' is flagged."""
        findings = by_type(analyzer.analyze("SELECT id FROM t WHERE name LIKE '_abc'"), "sql_leading_wildcard_like")
        assert len(findings) == 1

    def test_trailing_wildcard_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that LIKE 'abc%' (literal prefix) is not flagged."""
        findings = analyzer.analyze("SELECT id FROM t WHERE name LIKE 'abc%'")
        assert by_type(findings, "sql_leading_wildcard_like") == []

    def test_inner_wildcard_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that LIKE 'a%c' (literal prefix) is not flagged."""
        findings = analyzer.analyze("SELECT id FROM t WHERE name LIKE 'a%c'")
        assert by_type(findings, "sql_leading_wildcard_like") == []

    def test_ilike_leading_wildcard_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that ILIKE '%abc' is flagged."""
        findings = by_type(analyzer.analyze("SELECT id FROM t WHERE name ILIKE '%abc'"), "sql_leading_wildcard_like")
        assert len(findings) == 1

    def test_not_like_leading_wildcard_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that NOT LIKE '%abc' is also flagged (same scan cost)."""
        findings = by_type(analyzer.analyze("SELECT id FROM t WHERE name NOT LIKE '%abc'"), "sql_leading_wildcard_like")
        assert len(findings) == 1

    def test_non_literal_pattern_skipped(self, analyzer: SQLAnalyzer) -> None:
        """Test that a non-literal LIKE pattern (column reference) is skipped."""
        findings = analyzer.analyze("SELECT id FROM t WHERE name LIKE pattern_col")
        assert by_type(findings, "sql_leading_wildcard_like") == []


class TestInSubquery:
    """Tests for sql_in_subquery detection."""

    def test_in_subquery_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that IN (SELECT ...) is flagged LOW with advisory wording."""
        sql = "SELECT id FROM orders WHERE user_id IN (SELECT id FROM vip_users)"
        findings = by_type(analyzer.analyze(sql), "sql_in_subquery")
        assert len(findings) == 1
        assert findings[0].severity == SeverityLevel.LOW
        assert "EXISTS" in findings[0].suggestion

    def test_not_in_subquery_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that NOT IN (SELECT ...) is flagged."""
        sql = "SELECT id FROM orders WHERE user_id NOT IN (SELECT id FROM banned)"
        findings = by_type(analyzer.analyze(sql), "sql_in_subquery")
        assert len(findings) == 1

    def test_in_value_list_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that IN (1, 2, 3) with a literal list is not flagged."""
        findings = analyzer.analyze("SELECT id FROM orders WHERE status IN (1, 2, 3)")
        assert by_type(findings, "sql_in_subquery") == []

    def test_exists_not_flagged(self, analyzer: SQLAnalyzer) -> None:
        """Test that an EXISTS predicate produces no IN-subquery finding."""
        sql = "SELECT id FROM orders o WHERE EXISTS (SELECT 1 FROM vip_users v WHERE v.id = o.user_id)"
        findings = analyzer.analyze(sql)
        assert by_type(findings, "sql_in_subquery") == []


class TestRobustness:
    """Tests for multi-statement input, parse failures, and edge cases."""

    def test_multiple_statements(self, analyzer: SQLAnalyzer) -> None:
        """Test that findings are collected across semicolon-separated statements."""
        sql = "SELECT * FROM a; SELECT id FROM b UNION SELECT id FROM c"
        findings = analyzer.analyze(sql)
        assert finding_types(findings).count("sql_select_star") == 1
        assert finding_types(findings).count("sql_union_instead_of_union_all") == 1
        assert analyzer.parse_failed is False

    def test_unparseable_sql_no_crash(self, analyzer: SQLAnalyzer) -> None:
        """Test that unparseable SQL returns no findings and sets parse_failed."""
        findings = analyzer.analyze("this is not sql at all ???")
        assert findings == []
        assert analyzer.parse_failed is True

    def test_parse_failed_resets_between_calls(self, analyzer: SQLAnalyzer) -> None:
        """Test that parse_failed resets on the next successful analyze call."""
        analyzer.analyze("definitely ::: not sql")
        assert analyzer.parse_failed is True
        analyzer.analyze("SELECT id FROM t")
        assert analyzer.parse_failed is False

    def test_empty_string(self, analyzer: SQLAnalyzer) -> None:
        """Test that an empty string yields no findings and no parse failure."""
        assert analyzer.analyze("") == []
        assert analyzer.parse_failed is False

    def test_whitespace_only_string(self, analyzer: SQLAnalyzer) -> None:
        """Test that a whitespace-only string yields no findings."""
        assert analyzer.analyze("   \n\t  ") == []
        assert analyzer.parse_failed is False

    def test_clean_query_no_findings(self, analyzer: SQLAnalyzer) -> None:
        """Test that a well-formed query produces no findings at all."""
        sql = (
            "SELECT u.id, u.name FROM users u "
            "JOIN orders o ON u.id = o.user_id WHERE o.amount > 100 "
            "ORDER BY o.amount DESC LIMIT 50"
        )
        assert analyzer.analyze(sql) == []

    def test_multiple_findings_in_one_query(self, analyzer: SQLAnalyzer) -> None:
        """Test that one query can produce several distinct findings."""
        sql = "SELECT * FROM a CROSS JOIN b WHERE a.name LIKE '%x' ORDER BY a.id"
        types = finding_types(analyzer.analyze(sql))
        assert "sql_select_star" in types
        assert "sql_cartesian_join" in types
        assert "sql_leading_wildcard_like" in types
        assert "sql_orderby_without_limit" in types

    def test_fragment_truncation(self, analyzer: SQLAnalyzer) -> None:
        """Test that very long fragments are truncated with an ellipsis."""
        cols = ", ".join(f"col_{i}" for i in range(60))
        sql = f"SELECT * FROM (SELECT {cols} FROM wide_table) t"
        findings = by_type(analyzer.analyze(sql), "sql_select_star")
        assert len(findings) == 1
        assert len(findings[0].fragment) <= 160
        assert findings[0].fragment.endswith("...")

    def test_finding_fields_populated(self, analyzer: SQLAnalyzer) -> None:
        """Test that every finding carries non-empty fields."""
        sql = "SELECT * FROM a CROSS JOIN b"
        for finding in analyzer.analyze(sql):
            assert finding.finding_type
            assert isinstance(finding.severity, SeverityLevel)
            assert finding.description
            assert finding.suggestion
            assert finding.fragment
