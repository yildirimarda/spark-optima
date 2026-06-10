# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""sqlglot-based analysis of Spark SQL statements.

This module provides the SQLAnalyzer class that parses SQL strings (as passed
to ``spark.sql()``) with sqlglot's Spark dialect and walks the resulting AST
to detect performance anti-patterns such as ``SELECT *``, cartesian joins,
unbounded global sorts, ``UNION`` deduplication, leading-wildcard ``LIKE``
patterns, and ``IN (subquery)`` predicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError
from sqlglot.tokens import TokenType

from spark_optima.analysis.models import SeverityLevel

logger = logging.getLogger(__name__)

# Maximum length of a SQL fragment embedded in a finding before truncation
_MAX_FRAGMENT_LENGTH = 160


@dataclass
class SQLFinding:
    """A single anti-pattern found in a SQL statement.

    Attributes:
        finding_type: Machine-readable finding identifier (e.g. 'sql_select_star').
        severity: Severity level of the finding.
        description: Human-readable description of the issue.
        suggestion: Actionable advice on how to address the issue.
        fragment: The offending SQL snippet (rendered via sqlglot's .sql()).

    """

    finding_type: str
    severity: SeverityLevel
    description: str
    suggestion: str
    fragment: str


class SQLAnalyzer:
    """Analyzer for Spark SQL statements using sqlglot.

    Parses SQL with the Spark dialect and walks the expression tree to find
    performance anti-patterns. Parsing failures never propagate to callers:
    unparseable input yields no findings and sets :attr:`parse_failed`.

    Attributes:
        parse_failed: True if the most recent :meth:`analyze` call could not
            parse its input, False otherwise.

    Example:
        >>> analyzer = SQLAnalyzer()
        >>> findings = analyzer.analyze("SELECT * FROM events")
        >>> findings[0].finding_type
        'sql_select_star'

    """

    DIALECT = "spark"

    def __init__(self) -> None:
        """Initialize the SQL analyzer."""
        self.parse_failed: bool = False

    def analyze(self, sql: str) -> list[SQLFinding]:
        """Analyze a SQL string for anti-patterns.

        Handles multiple semicolon-separated statements. Unparseable SQL is
        skipped gracefully: an empty list is returned and ``parse_failed`` is
        set to True instead of raising.

        Args:
            sql: SQL text in Spark dialect (one or more statements).

        Returns:
            List of SQLFinding objects, empty when nothing is detected or the
            input cannot be parsed.

        """
        self.parse_failed = False
        if not sql or not sql.strip():
            return []

        try:
            statements = sqlglot.parse(sql, read=self.DIALECT)
        except ParseError as e:
            logger.debug(f"Skipping unparseable SQL: {e}")
            self.parse_failed = True
            return []

        # sqlglot canonicalizes comma-separated FROM tables into CROSS joins,
        # so the original token stream is the only reliable way to tell an
        # explicit CROSS JOIN apart from an old-style implicit comma join.
        has_explicit_cross = self._has_explicit_cross_join(sql)

        findings: list[SQLFinding] = []
        for statement in statements:
            if statement is None:
                continue
            findings.extend(self._check_select_star(statement))
            findings.extend(self._check_cartesian_join(statement, has_explicit_cross))
            findings.extend(self._check_orderby_without_limit(statement))
            findings.extend(self._check_union_distinct(statement))
            findings.extend(self._check_leading_wildcard_like(statement))
            findings.extend(self._check_in_subquery(statement))
        return findings

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_select_star(self, statement: exp.Expr) -> list[SQLFinding]:
        """Flag SELECT * projections at any query level.

        Star expressions inside function calls (e.g. ``COUNT(*)``) are not
        projections and are deliberately not flagged.

        Args:
            statement: Parsed SQL statement.

        Returns:
            List of findings, at most one per SELECT.

        """
        findings: list[SQLFinding] = []
        for select in statement.find_all(exp.Select):
            if any(self._is_star_projection(projection) for projection in select.expressions):
                findings.append(
                    SQLFinding(
                        finding_type="sql_select_star",
                        severity=SeverityLevel.LOW,
                        description=(
                            "SELECT * reads every column and disables column pruning. "
                            "List only the columns you need so Spark can skip the rest."
                        ),
                        suggestion=(
                            "Replace * with an explicit column list. For columnar formats "
                            "like Parquet/ORC this lets Spark skip entire column chunks on disk."
                        ),
                        fragment=self._fragment(select),
                    )
                )
        return findings

    def _check_cartesian_join(self, statement: exp.Expr, has_explicit_cross: bool) -> list[SQLFinding]:
        """Flag cartesian products from CROSS JOIN or unconstrained comma joins.

        Explicit ``CROSS JOIN`` is always flagged. Comma-separated FROM tables
        (which sqlglot canonicalizes to CROSS joins) are flagged conservatively:
        only when the enclosing SELECT has no WHERE clause at all, since a WHERE
        usually carries the join condition in old-style implicit joins.

        Args:
            statement: Parsed SQL statement.
            has_explicit_cross: Whether the original SQL contains a CROSS keyword.

        Returns:
            List of findings, one per cartesian join.

        """
        findings: list[SQLFinding] = []
        for select in statement.find_all(exp.Select):
            has_where = select.args.get("where") is not None
            for join in select.args.get("joins") or []:
                if (join.args.get("kind") or "").upper() != "CROSS":
                    continue
                if join.args.get("on") is not None or join.args.get("using") is not None:
                    continue
                if has_where and not has_explicit_cross:
                    # Conservative: a comma join constrained by a WHERE clause is
                    # most likely an old-style equi-join, not a cartesian product.
                    continue
                if has_explicit_cross:
                    description = (
                        "CROSS JOIN produces a cartesian product: the result size is "
                        "the product of both input row counts."
                    )
                else:
                    description = (
                        "Comma-separated tables in FROM without a WHERE clause form an "
                        "implicit cartesian product: the result size is the product of "
                        "both input row counts."
                    )
                findings.append(
                    SQLFinding(
                        finding_type="sql_cartesian_join",
                        severity=SeverityLevel.HIGH,
                        description=description,
                        suggestion=(
                            "Use an explicit JOIN with an ON condition. If the cross join "
                            "is intentional, keep one side tiny and consider a broadcast hint."
                        ),
                        fragment=self._fragment(join),
                    )
                )
        return findings

    def _check_orderby_without_limit(self, statement: exp.Expr) -> list[SQLFinding]:
        """Flag an outermost query that has ORDER BY but no LIMIT.

        Only the outermost query of each statement is checked: an ORDER BY in
        a subquery has different semantics and is often combined with LIMIT
        in the outer query.

        Args:
            statement: Parsed SQL statement.

        Returns:
            List with at most one finding.

        """
        query: exp.Expr = statement
        if isinstance(query, exp.Subquery):
            query = query.unnest()
        if not isinstance(query, exp.Select | exp.SetOperation):
            return []
        if query.args.get("order") is None or query.args.get("limit") is not None:
            return []
        return [
            SQLFinding(
                finding_type="sql_orderby_without_limit",
                severity=SeverityLevel.MEDIUM,
                description=(
                    "ORDER BY without LIMIT performs a full global sort: the entire "
                    "dataset is range-partitioned and shuffled just to be ordered."
                ),
                suggestion=(
                    "Add LIMIT N for top-N queries (Spark rewrites ORDER BY + LIMIT into a "
                    "cheap TakeOrdered), or drop the ORDER BY if downstream consumers do "
                    "not rely on row order."
                ),
                fragment=self._fragment(query),
            )
        ]

    def _check_union_distinct(self, statement: exp.Expr) -> list[SQLFinding]:
        """Flag UNION (distinct) which deduplicates via an extra shuffle.

        Args:
            statement: Parsed SQL statement.

        Returns:
            List of findings, one per distinct UNION.

        """
        findings: list[SQLFinding] = []
        for union in statement.find_all(exp.Union):
            if not union.args.get("distinct"):
                continue
            findings.append(
                SQLFinding(
                    finding_type="sql_union_instead_of_union_all",
                    severity=SeverityLevel.LOW,
                    description=(
                        "UNION deduplicates the combined result, which requires a full "
                        "shuffle and aggregation across all rows."
                    ),
                    suggestion=(
                        "If duplicates are impossible or acceptable, use UNION ALL to skip "
                        "the deduplication shuffle. Keep UNION only when distinct rows are "
                        "actually required."
                    ),
                    fragment=self._fragment(union),
                )
            )
        return findings

    def _check_leading_wildcard_like(self, statement: exp.Expr) -> list[SQLFinding]:
        """Flag LIKE/ILIKE patterns that start with a wildcard.

        Only string-literal patterns are inspected; dynamic patterns (columns,
        parameters) are skipped.

        Args:
            statement: Parsed SQL statement.

        Returns:
            List of findings, one per leading-wildcard pattern.

        """
        findings: list[SQLFinding] = []
        for like in statement.find_all(exp.Like, exp.ILike):
            pattern = like.expression
            if not (isinstance(pattern, exp.Literal) and pattern.is_string):
                continue
            if not str(pattern.this).startswith(("%", "_")):
                continue
            findings.append(
                SQLFinding(
                    finding_type="sql_leading_wildcard_like",
                    severity=SeverityLevel.MEDIUM,
                    description=(
                        "LIKE pattern starting with a wildcard ('%' or '_') cannot be "
                        "rewritten into a prefix filter, so it defeats predicate pushdown "
                        "and forces a full scan with per-row matching."
                    ),
                    suggestion=(
                        "Anchor the pattern with a literal prefix (e.g. LIKE 'abc%') when "
                        "possible, precompute a reversed/normalized column for suffix "
                        "matches, or filter on other pushdown-friendly predicates first."
                    ),
                    fragment=self._fragment(like),
                )
            )
        return findings

    def _check_in_subquery(self, statement: exp.Expr) -> list[SQLFinding]:
        """Flag IN (SELECT ...) predicates (advisory).

        Spark often plans uncorrelated IN-subqueries efficiently (as semi
        joins), so this is a low-severity advisory finding.

        Args:
            statement: Parsed SQL statement.

        Returns:
            List of findings, one per IN-subquery.

        """
        findings: list[SQLFinding] = []
        for in_expr in statement.find_all(exp.In):
            if in_expr.args.get("query") is None:
                continue
            findings.append(
                SQLFinding(
                    finding_type="sql_in_subquery",
                    severity=SeverityLevel.LOW,
                    description=(
                        "IN (SELECT ...) subquery detected. Spark usually plans these as "
                        "semi joins, but EXISTS or an explicit JOIN can be clearer and "
                        "gives the optimizer more freedom, especially with NOT IN and NULLs."
                    ),
                    suggestion=(
                        "Consider rewriting as EXISTS (correlated) or as an explicit "
                        "JOIN/LEFT SEMI JOIN. Verify the query plan before and after - "
                        "this is advisory, not a guaranteed win."
                    ),
                    fragment=self._fragment(in_expr),
                )
            )
        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_star_projection(projection: exp.Expr) -> bool:
        """Check whether a SELECT projection is a star expression.

        Covers bare ``*`` and qualified ``t.*``. Stars nested inside function
        calls (e.g. ``COUNT(*)``) are not projections of the SELECT itself.

        Args:
            projection: One expression from a SELECT's projection list.

        Returns:
            True if the projection selects all columns.

        """
        return isinstance(projection, exp.Star) or (
            isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
        )

    def _has_explicit_cross_join(self, sql: str) -> bool:
        """Check whether the original SQL text contains a CROSS keyword token.

        Tokenizes the raw input (rather than regex-matching it) so CROSS inside
        string literals, comments, or identifiers does not count.

        Args:
            sql: Raw SQL text as written by the user.

        Returns:
            True if an explicit CROSS keyword appears in the token stream.

        """
        try:
            tokens = sqlglot.tokenize(sql, read=self.DIALECT)
        except SqlglotError:  # pragma: no cover - parse() succeeded, so this is unreachable
            return False
        return any(token.token_type == TokenType.CROSS for token in tokens)

    def _fragment(self, node: exp.Expr) -> str:
        """Render a node back to SQL, truncated for readability.

        Args:
            node: sqlglot expression node.

        Returns:
            SQL snippet for the node, truncated to a sensible length.

        """
        sql = node.sql(dialect=self.DIALECT)
        if len(sql) > _MAX_FRAGMENT_LENGTH:
            return sql[: _MAX_FRAGMENT_LENGTH - 3] + "..."
        return sql
