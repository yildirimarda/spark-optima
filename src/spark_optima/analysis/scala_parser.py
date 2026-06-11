# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Lightweight lexer-based parser for Scala Spark sources.

This module provides the ScalaCodeParser class that detects Spark
DataFrame/Dataset/RDD operations in Scala source code without a full Scala
compiler. Comments are stripped and string literals are masked *before*
scanning, so method names inside strings or comments never produce false
matches. The parser emits the same models as the Python parser
(:class:`~spark_optima.analysis.models.SparkOperation`,
:class:`~spark_optima.analysis.parser.ParseResult`), which lets every
operation-based smell detector work unchanged on Scala input.

Capabilities:
    - Spark method calls mapped through ``SPARK_METHOD_MAP`` (including the
      RDD operations groupByKey, reduceByKey and mapPartitions).
    - ``val``/``var`` assignment tracking for DataFrame variable lineage
      (``val x = df.join(...)`` registers the chain operations under ``x``).
    - Chained calls with chain positions in source order.
    - Argument text extraction per call via balanced-parenthesis scanning.
    - Recovery of plain string literals (including triple-quoted strings)
      so ``spark.sql("...")`` contents flow into SQL analysis.

Limitations:
    - Expressions inside string interpolations (``s"... ${df.count()}"``)
      are masked and therefore not scanned.
    - Parenthesis-less arity-0 calls (``df.cache``) are only recognized for
      reader/writer accessors (``read``, ``write``, ``readStream``,
      ``writeStream``).
    - Loop context is not tracked; ``SparkOperation.in_loop`` is always
      False for Scala operations.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from spark_optima.analysis.models import (
    CodeLocation,
    SparkOperation,
    SparkOperationType,
)
from spark_optima.analysis.parser import SPARK_METHOD_MAP, ParseResult

logger = logging.getLogger(__name__)

# Reader/writer accessors that start fluent chains without parentheses
_ATTRIBUTE_OPERATIONS: frozenset[str] = frozenset({"read", "readStream", "write", "writeStream"})

# Standalone function calls tracked exactly like the Python parser does
_STANDALONE_FUNCTIONS: dict[str, str] = {"broadcast": "broadcast_call", "udf": "udf_call"}

# Matches a (possibly whitespace-separated) method selection like ".join"
_METHOD_CALL_RE = re.compile(r"\.\s*([A-Za-z_]\w*)")

# Matches standalone broadcast(...)/udf(...) calls (not method selections)
_STANDALONE_CALL_RE = re.compile(r"(?<![\w.])(broadcast|udf)\s*\(")

# Matches val/var assignments: "val x =", "lazy val x: DataFrame =", ...
_ASSIGNMENT_RE = re.compile(
    r"^[ \t]*(?:final\s+)?(?:lazy\s+)?(?:val|var)\s+([A-Za-z_]\w*)(?:\s*:\s*[^=\n]+?)?\s*=(?!=)",
    re.MULTILINE,
)

# Escape sequences resolved inside regular (non-triple-quoted) Scala strings
_ESCAPE_MAP: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    '"': '"',
    "'": "'",
    "\\": "\\",
}

# Markers used by language auto-detection. The heuristics run on masked text
# (strings and comments blanked) so markers inside docstrings, string
# literals or comments never influence the decision. Scala classification
# requires a STRONG marker (val/var assignment or an org.apache.spark
# import); any Python marker keeps the source Python, and ambiguity
# defaults to Python.
_SCALA_STRONG_HINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:final\s+)?(?:lazy\s+)?(?:val|var)\s+[A-Za-z_]\w*(?:\s*:\s*[^=\n]+?)?\s*=(?!=)", re.MULTILINE),
    re.compile(r"^\s*import\s+org\.apache\.spark", re.MULTILINE),
)
_PYTHON_HINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:from|import)\s+pyspark\b", re.MULTILINE),
    # Python defs end the line with a colon; Scala defs continue with "= {..."
    re.compile(r"^\s*def\s+\w+\s*\([^)\n]*\)\s*(?:->\s*[^:\n]+)?\s*:\s*(?:#.*)?$", re.MULTILINE),
    re.compile(r"^\s*if\s+__name__\s*==", re.MULTILINE),
)

# Lightweight Python-style string/comment mask used by detect_language only.
# Triple-quoted strings are matched first so whole docstrings (the main
# source of misleading Scala markers) are blanked in one go.
_PYTHON_MASK_RE = re.compile(
    r"'''(?:\\.|[^\\])*?'''"
    r'|"""(?:\\.|[^\\])*?"""'
    r"|'(?:\\.|[^'\\\n])*'"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|#[^\n]*"
)


def _mask_python_like(source_code: str) -> str:
    """Blank string literals and ``#`` comments while keeping newlines.

    A deliberately lightweight, language-agnostic mask used only by
    :func:`detect_language`. It removes the text most likely to contain
    misleading markers (docstrings, string literals, ``#`` comments) from
    both Python and Scala sources before the heuristics run; line/column
    shape is preserved so the multiline marker regexes keep working.

    Args:
        source_code: Raw source text.

    Returns:
        Source text of identical shape with masked regions blanked.

    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return _PYTHON_MASK_RE.sub(blank, source_code)


def detect_language(source_code: str) -> str:
    """Guess whether source text is Python or Scala Spark code.

    The heuristics run on masked text (strings and comments blanked), so a
    Python file whose docstring quotes Scala examples is never misrouted.
    Python markers win over Scala markers, Scala requires a strong marker
    (a ``val``/``var`` assignment or an ``org.apache.spark`` import), and
    the default is "python", so existing Python behavior is never changed
    by ambiguous input.

    Args:
        source_code: Raw source text.

    Returns:
        "scala" when a strong Scala marker is present and no Python markers
        are found, "python" otherwise.

    """
    masked = _mask_python_like(source_code)
    if any(pattern.search(masked) for pattern in _PYTHON_HINTS):
        return "python"
    if any(pattern.search(masked) for pattern in _SCALA_STRONG_HINTS):
        return "scala"
    return "python"


@dataclass
class _StringLiteral:
    """A string literal found while masking Scala source.

    Attributes:
        token_start: Offset of the token start, including any interpolator
            prefix (the ``s`` in ``s"..."``).
        start: Offset of the opening quote character(s).
        end: Offset one past the closing quote character(s).
        content: Literal content; escape sequences are resolved for plain
            single-quoted strings, kept raw for triple-quoted strings.
        interpolated: True for interpolator-prefixed strings (``s"..."``,
            ``f"..."``, ``raw"..."``), whose content is not a plain literal.

    """

    token_start: int
    start: int
    end: int
    content: str
    interpolated: bool


def _unescape(text: str) -> str:
    """Resolve standard escape sequences in a plain Scala string literal.

    Args:
        text: Raw text between the quotes.

    Returns:
        Text with recognized backslash escapes replaced.

    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(_ESCAPE_MAP.get(text[i + 1], text[i + 1]))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _consume_interpolation(source: str, start: int) -> int:
    """Consume a balanced ``${...}`` block inside an interpolated string.

    Nested string literals (plain, interpolated and triple-quoted) and char
    literals inside the block are skipped, so quote and brace characters
    they contain neither unbalance the braces nor terminate the enclosing
    interpolated string.

    Args:
        source: Full source text.
        start: Offset of the ``$`` in ``${``.

    Returns:
        Offset one past the matching ``}``, or end of source if unbalanced.

    """
    n = len(source)
    depth = 0
    i = start + 1  # position of the opening "{"
    while i < n:
        ch = source[i]
        if ch == "{":
            depth += 1
            i += 1
        elif ch == "}":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        elif ch == '"':
            i = _consume_nested_string(source, i)
        elif ch == "'" and i + 2 < n and source[i + 1] != "\\" and source[i + 2] == "'":
            i += 3
        elif ch == "'" and i + 3 < n and source[i + 1] == "\\" and source[i + 3] == "'":
            i += 4
        else:
            i += 1
    return n


def _consume_nested_string(source: str, quote_pos: int) -> int:
    """Skip a string literal nested inside a ``${...}`` interpolation block.

    Handles plain, interpolated and triple-quoted strings; interpolated
    nested strings recurse back into :func:`_consume_interpolation` for
    their own ``${...}`` blocks.

    Args:
        source: Full source text.
        quote_pos: Offset of the opening quote character.

    Returns:
        Offset one past the closing quote(s), or end of source/line when
        the literal is unterminated.

    """
    n = len(source)
    p = quote_pos
    while p > 0 and (source[p - 1].isalnum() or source[p - 1] == "_"):
        p -= 1
    interpolated = p < quote_pos
    if source.startswith('"""', quote_pos):
        j = quote_pos + 3
        while j < n:
            if interpolated and source.startswith("$$", j):
                j += 2
            elif interpolated and source.startswith("${", j):
                j = _consume_interpolation(source, j)
            elif source.startswith('"""', j):
                return j + 3
            else:
                j += 1
        return n
    j = quote_pos + 1
    while j < n and source[j] != '"' and source[j] != "\n":
        if interpolated and source.startswith("$$", j):
            j += 2
        elif interpolated and source.startswith("${", j):
            j = _consume_interpolation(source, j)
        elif source[j] == "\\":
            j += 2
        else:
            j += 1
    return min(j + 1, n)


def _mask_source(source: str) -> tuple[str, list[_StringLiteral]]:
    """Blank out comments and string literals while preserving offsets.

    Line comments (``//``), nested block comments (``/* ... */``), char
    literals, plain strings, interpolated strings and triple-quoted strings
    are replaced by spaces (newlines are kept so line numbers stay valid).
    String literal contents are recorded so callers can recover them later.

    Args:
        source: Original Scala source code.

    Returns:
        Tuple of (masked source with identical length/offsets, recorded
        string literals in source order).

    """
    chars = list(source)
    literals: list[_StringLiteral] = []
    n = len(source)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if chars[k] != "\n":
                chars[k] = " "

    i = 0
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = source.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
        elif ch == "/" and nxt == "*":
            # Scala block comments nest
            depth, j = 1, i + 2
            while j < n and depth > 0:
                if source.startswith("/*", j):
                    depth += 1
                    j += 2
                elif source.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif ch == "'":
            # Char literals like 'x', '\n' or '"'; anything else (e.g. the
            # deprecated symbol syntax) is left untouched.
            if i + 2 < n and source[i + 1] != "\\" and source[i + 2] == "'":
                blank(i, i + 3)
                i += 3
            elif i + 3 < n and source[i + 1] == "\\" and source[i + 3] == "'":
                blank(i, i + 4)
                i += 4
            else:
                i += 1
        elif ch == '"':
            # Detect an interpolator prefix (identifier glued to the quote)
            p = i
            while p > 0 and (source[p - 1].isalnum() or source[p - 1] == "_"):
                p -= 1
            interpolated = p < i
            if source.startswith('"""', i):
                # Interpolated triple-quoted strings may embed ${...} blocks
                # whose nested literals must not terminate the outer string.
                j = i + 3
                close = -1
                while j < n:
                    if interpolated and source.startswith("$$", j):
                        j += 2
                    elif interpolated and source.startswith("${", j):
                        j = _consume_interpolation(source, j)
                    elif source.startswith('"""', j):
                        close = j
                        break
                    else:
                        j += 1
                if close == -1:
                    content, end = source[i + 3 :], n
                else:
                    content, end = source[i + 3 : close], close + 3
            else:
                j = i + 1
                while j < n and source[j] != '"' and source[j] != "\n":
                    if interpolated and source.startswith("$$", j):
                        j += 2
                    elif interpolated and source.startswith("${", j):
                        # Consume the balanced interpolation block, including
                        # any string literals it contains, before resuming
                        # the scan for the closing quote.
                        j = _consume_interpolation(source, j)
                    elif source[j] == "\\":
                        j += 2
                    else:
                        j += 1
                raw = source[i + 1 : j]
                end = min(j + 1, n)
                content = raw if interpolated else _unescape(raw)
            literals.append(
                _StringLiteral(
                    token_start=p if interpolated else i,
                    start=i,
                    end=end,
                    content=content,
                    interpolated=interpolated,
                )
            )
            blank(i, end)
            i = end
        else:
            i += 1
    return "".join(chars), literals


def _match_bracket(masked: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Find the offset of the bracket matching the one at ``open_pos``.

    Args:
        masked: Masked source (strings/comments already blanked).
        open_pos: Offset of the opening bracket.
        open_ch: Opening bracket character.
        close_ch: Closing bracket character.

    Returns:
        Offset of the matching closing bracket, or -1 when unbalanced.

    """
    depth = 0
    for k in range(open_pos, len(masked)):
        if masked[k] == open_ch:
            depth += 1
        elif masked[k] == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return -1


class ScalaCodeParser:
    """Parser for analyzing Scala Spark code with a lightweight lexer.

    Mirrors the :class:`~spark_optima.analysis.parser.SparkCodeParser` API
    (``parse_file``/``parse_source``) and produces the same
    :class:`~spark_optima.analysis.parser.ParseResult` model, tagged with
    ``language="scala"``.

    Attributes:
        source_code: The original source code being parsed.
        masked_source: Source with comments/strings blanked out.
        operations: List of detected Spark operations.
        dataframe_vars: Mapping of variable names to their operations,
            including ``val``/``var`` assignment lineage.
        string_literals: String literals recorded during masking.

    Example:
        >>> parser = ScalaCodeParser()
        >>> result = parser.parse_source('val out = df.join(small, Seq("id"))')
        >>> result.operations[0].method_name
        'join'

    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self.source_code: str = ""
        self.masked_source: str = ""
        self.operations: list[SparkOperation] = []
        self.dataframe_vars: dict[str, list[SparkOperation]] = {}
        self.string_literals: list[_StringLiteral] = []
        self._chain_counter: int = 0
        self._line_starts: list[int] = [0]
        self._op_offsets: list[tuple[int, int, SparkOperation]] = []
        self._chain_group_by_op: dict[int, int] = {}
        self._assignment_targets_by_op: dict[int, set[str]] = {}

    def parse_file(self, file_path: str | Path) -> ParseResult:
        """Parse a Scala file containing Spark code.

        Args:
            file_path: Path to the Scala file.

        Returns:
            ParseResult containing operations and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return self.parse_source(file_path.read_text(encoding="utf-8"))

    def parse_source(self, source_code: str) -> ParseResult:
        """Parse Scala source code containing Spark operations.

        Args:
            source_code: Scala source code string.

        Returns:
            ParseResult containing operations and metadata, with
            ``language`` set to "scala".

        """
        self.source_code = source_code
        self.operations = []
        self.dataframe_vars = {}
        self._chain_counter = 0
        self._op_offsets = []
        self._chain_group_by_op = {}
        self._assignment_targets_by_op = {}
        self._line_starts = [0] + [m.end() for m in re.finditer(r"\n", source_code)]

        self.masked_source, self.string_literals = _mask_source(source_code)

        candidates = self._scan_method_calls() + self._scan_standalone_calls()
        candidates.sort(key=lambda item: item[0])
        for offset, method_name, dataframe_var, arguments, end_offset in candidates:
            self._add_operation(offset, method_name, dataframe_var, arguments, end_offset)

        self._compute_statement_chains()
        self._apply_assignment_lineage()

        return ParseResult(
            operations=self.operations,
            dataframe_vars=self.dataframe_vars,
            operation_count=len(self.operations),
            language="scala",
        )

    def get_dataframe_lineage(self, var_name: str) -> list[SparkOperation]:
        """Get the operation lineage for a DataFrame variable.

        Args:
            var_name: Name of the DataFrame variable.

        Returns:
            List of operations in order of execution.

        """
        return self.dataframe_vars.get(var_name, [])

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_method_calls(self) -> list[tuple[int, str, str, list[str], int]]:
        """Scan the masked source for Spark method selections.

        Returns:
            List of (offset, method_name, dataframe_var, arguments,
            end_offset) candidate tuples in arbitrary order.

        """
        masked = self.masked_source
        found: list[tuple[int, str, str, list[str], int]] = []
        for match in _METHOD_CALL_RE.finditer(masked):
            name = match.group(1)
            if name not in SPARK_METHOD_MAP:
                continue

            # Skip an optional type-argument list, then look for the call parens
            j = self._skip_whitespace(match.end())
            if j < len(masked) and masked[j] == "[":
                close_bracket = _match_bracket(masked, j, "[", "]")
                if close_bracket == -1:
                    continue
                j = self._skip_whitespace(close_bracket + 1)

            dataframe_var = self._resolve_chain_root(match.start())
            if dataframe_var is None:
                continue

            if j < len(masked) and masked[j] == "(":
                close_paren = _match_bracket(masked, j, "(", ")")
                if close_paren == -1:
                    continue
                arguments = self._extract_call_arguments(j, close_paren)
                found.append((match.start(), name, dataframe_var, arguments, close_paren + 1))
            elif name in _ATTRIBUTE_OPERATIONS:
                # Reader/writer accessors are used without parentheses
                found.append((match.start(), name, dataframe_var, [], match.end()))
        return found

    def _scan_standalone_calls(self) -> list[tuple[int, str, str, list[str], int]]:
        """Scan the masked source for standalone broadcast()/udf() calls.

        Returns:
            List of (offset, method_name, dataframe_var, arguments,
            end_offset) candidate tuples in arbitrary order.

        """
        masked = self.masked_source
        found: list[tuple[int, str, str, list[str], int]] = []
        for match in _STANDALONE_CALL_RE.finditer(masked):
            name = match.group(1)
            open_paren = match.end() - 1
            close_paren = _match_bracket(masked, open_paren, "(", ")")
            if close_paren == -1:
                continue
            # Mirror the Python parser: broadcast carries no arguments
            arguments = [] if name == "broadcast" else self._extract_call_arguments(open_paren, close_paren)
            found.append((match.start(), name, _STANDALONE_FUNCTIONS[name], arguments, close_paren + 1))
        return found

    def _skip_whitespace(self, offset: int) -> int:
        """Advance an offset past whitespace in the masked source.

        Args:
            offset: Starting offset.

        Returns:
            Offset of the next non-whitespace character (or end of source).

        """
        masked = self.masked_source
        while offset < len(masked) and masked[offset] in " \t":
            offset += 1
        return offset

    def _resolve_chain_root(self, dot_pos: int) -> str | None:
        """Resolve the root variable of a fluent chain ending at a dot.

        For ``df.join(small, Seq("id")).select("a")`` the root of the
        ``.select`` selection is ``df``. Walks backwards over identifiers,
        dots and balanced parentheses/brackets.

        Args:
            dot_pos: Offset of the ``.`` starting the method selection.

        Returns:
            Root variable name, or None when the chain has no simple root.

        """
        masked = self.masked_source
        i = dot_pos - 1
        while i >= 0:
            while i >= 0 and masked[i] in " \t\r\n":
                i -= 1
            if i < 0:
                return None
            ch = masked[i]
            if ch in ")]":
                open_ch = "(" if ch == ")" else "["
                depth = 0
                while i >= 0:
                    if masked[i] == ch:
                        depth += 1
                    elif masked[i] == open_ch:
                        depth -= 1
                        if depth == 0:
                            i -= 1
                            break
                    i -= 1
                else:
                    return None
                continue
            if ch.isalnum() or ch == "_":
                j = i
                while j >= 0 and (masked[j].isalnum() or masked[j] == "_"):
                    j -= 1
                identifier = masked[j + 1 : i + 1]
                if identifier[0].isdigit():
                    return None
                k = j
                while k >= 0 and masked[k] in " \t\r\n":
                    k -= 1
                if k >= 0 and masked[k] == ".":
                    i = k - 1
                    continue
                return identifier
            return None
        return None

    def _extract_call_arguments(self, open_paren: int, close_paren: int) -> list[str]:
        """Extract argument text for a call via balanced-paren scanning.

        Top-level commas split arguments. Arguments that are exactly one
        plain string literal are rendered as their Python ``repr`` (matching
        the Python parser's output) so downstream detectors such as the SQL
        analyzer and ``select("*")`` checks work unchanged; all other
        arguments keep their original (whitespace-collapsed) source text.

        Args:
            open_paren: Offset of the opening parenthesis.
            close_paren: Offset of the matching closing parenthesis.

        Returns:
            List of argument string representations.

        """
        masked = self.masked_source
        spans: list[tuple[int, int]] = []
        depth = 0
        segment_start = open_paren + 1
        for k in range(open_paren + 1, close_paren):
            ch = masked[k]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                spans.append((segment_start, k))
                segment_start = k + 1
        if close_paren > open_paren + 1:
            spans.append((segment_start, close_paren))

        arguments: list[str] = []
        for start, end in spans:
            text = self.source_code[start:end]
            stripped = text.strip()
            if not stripped:
                continue
            trimmed_start = start + (len(text) - len(text.lstrip()))
            trimmed_end = end - (len(text) - len(text.rstrip()))
            literal = self._literal_at(trimmed_start, trimmed_end)
            if literal is not None and not literal.interpolated:
                arguments.append(repr(literal.content))
            else:
                arguments.append(" ".join(stripped.split()))
        return arguments

    def _literal_at(self, start: int, end: int) -> _StringLiteral | None:
        """Find a string literal exactly covering the given source span.

        Args:
            start: Span start offset (inclusive).
            end: Span end offset (exclusive).

        Returns:
            The matching literal, or None when the span is not exactly one
            string literal token.

        """
        for literal in self.string_literals:
            if literal.token_start == start and literal.end == end:
                return literal
            if literal.token_start > end:
                break
        return None

    # ------------------------------------------------------------------
    # Operation and lineage bookkeeping
    # ------------------------------------------------------------------

    def _add_operation(
        self,
        offset: int,
        method_name: str,
        dataframe_var: str,
        arguments: list[str],
        end_offset: int,
    ) -> SparkOperation:
        """Record a detected Spark operation.

        Args:
            offset: Source offset where the operation starts.
            method_name: Name of the Spark method.
            dataframe_var: Root variable of the call chain.
            arguments: Argument string representations.
            end_offset: Source offset one past the operation.

        Returns:
            The created SparkOperation.

        """
        op_type = SPARK_METHOD_MAP.get(method_name, SparkOperationType.TRANSFORMATION)
        operation = SparkOperation(
            operation_type=op_type,
            method_name=method_name,
            dataframe_var=dataframe_var,
            arguments=arguments,
            location=self._location_from_offsets(offset, end_offset),
            chain_position=self._chain_counter,
            in_loop=False,
        )
        self._chain_counter += 1
        self.operations.append(operation)
        self._op_offsets.append((offset, end_offset, operation))
        self.dataframe_vars.setdefault(dataframe_var, []).append(operation)
        return operation

    def _location_from_offsets(self, offset: int, end_offset: int) -> CodeLocation:
        """Build a CodeLocation from source offsets.

        Args:
            offset: Start offset.
            end_offset: End offset (exclusive).

        Returns:
            CodeLocation with 1-indexed lines and 0-indexed columns.

        """
        line_idx = bisect_right(self._line_starts, offset) - 1
        end_idx = bisect_right(self._line_starts, max(end_offset - 1, 0)) - 1
        return CodeLocation(
            line=line_idx + 1,
            column=offset - self._line_starts[line_idx],
            end_line=end_idx + 1,
            end_column=max(end_offset - 1, 0) - self._line_starts[end_idx] + 1,
        )

    def _apply_assignment_lineage(self) -> None:
        """Register val/var assignment targets in the variable lineage map.

        For ``val x = df.join(...)`` the operations found in the right-hand
        side are also recorded under ``x``, so later uses of ``x`` (e.g.
        ``x.write.parquet(...)``) can be linked back to how it was built.

        """
        masked = self.masked_source
        for match in _ASSIGNMENT_RE.finditer(masked):
            name = match.group(1)
            rhs_start = match.end()
            statement_end = self._find_statement_end(rhs_start)
            ops_in_statement = [op for offset, _end, op in self._op_offsets if rhs_start <= offset < statement_end]
            if not ops_in_statement:
                continue
            bucket = self.dataframe_vars.setdefault(name, [])
            for op in ops_in_statement:
                self._assignment_targets_by_op.setdefault(id(op), set()).add(name)
                if not any(existing is op for existing in bucket):
                    bucket.append(op)
            bucket.sort(key=lambda op: op.chain_position)

    def _compute_statement_chains(self) -> None:
        """Group operations that belong to one fluent statement chain.

        Two consecutive operations (in source order) share a chain group
        when the masked text between the end of the first call and the
        ``.`` selection of the second contains nothing but whitespace and
        chained-call glue such as ``.mode("overwrite")`` or ``.as[Foo]``
        (covering multi-line continuations). Operations nested inside
        another call's argument list stay in the enclosing chain's group.

        """
        masked = self.masked_source
        group = -1
        prev_end: int | None = None
        for offset, end_offset, op in self._op_offsets:
            nested = prev_end is not None and offset < prev_end
            same_chain = (
                prev_end is not None
                and offset >= prev_end
                and offset < len(masked)
                and masked[offset] == "."
                and self._gap_continues_chain(masked[prev_end:offset])
            )
            if not nested and not same_chain:
                group += 1
            self._chain_group_by_op[id(op)] = group
            prev_end = end_offset if prev_end is None else max(prev_end, end_offset)

    @staticmethod
    def _gap_continues_chain(gap: str) -> bool:
        """Check whether masked gap text only contains fluent-chain glue.

        Valid glue is a (possibly empty) sequence of ``.method``,
        ``.method(args)`` or ``.method[T](args)`` selections separated by
        whitespace — i.e. chained calls that are not Spark operations
        themselves, such as ``.mode("overwrite")``. Anything else (a new
        statement, a semicolon, an operator) breaks the chain.

        Args:
            gap: Masked source text between two operations.

        Returns:
            True when the gap keeps both operations in the same chain.

        """
        i, n = 0, len(gap)
        while i < n:
            if gap[i] in " \t\r\n":
                i += 1
                continue
            if gap[i] != ".":
                return False
            i += 1
            while i < n and gap[i] in " \t\r\n":
                i += 1
            name_start = i
            while i < n and (gap[i].isalnum() or gap[i] == "_"):
                i += 1
            if i == name_start:
                return False
            while i < n and gap[i] in " \t\r\n":
                i += 1
            # Skip any balanced argument/type-parameter groups
            while i < n and gap[i] in "([":
                open_ch = gap[i]
                close_ch = ")" if open_ch == "(" else "]"
                depth = 0
                while i < n:
                    if gap[i] == open_ch:
                        depth += 1
                    elif gap[i] == close_ch:
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                if depth != 0:
                    return False
                while i < n and gap[i] in " \t\r\n":
                    i += 1
        return True

    def same_statement_chain(self, op_a: SparkOperation, op_b: SparkOperation) -> bool:
        """Check whether two operations share one fluent statement chain.

        Args:
            op_a: First operation (from the latest parse).
            op_b: Second operation (from the latest parse).

        Returns:
            True when both operations were parsed from the same fluent
            statement chain; False otherwise (including operations that do
            not belong to the latest parse).

        """
        group_a = self._chain_group_by_op.get(id(op_a))
        group_b = self._chain_group_by_op.get(id(op_b))
        return group_a is not None and group_a == group_b

    def assignment_targets(self, op: SparkOperation) -> frozenset[str]:
        """Return the variables assigned from the statement containing an op.

        For ``val single = df.repartition(1)`` the repartition operation
        reports ``{"single"}``; operations outside any ``val``/``var``
        assignment report the empty set.

        Args:
            op: Operation from the latest parse.

        Returns:
            Frozen set of assignment target names for the operation.

        """
        return frozenset(self._assignment_targets_by_op.get(id(op), ()))

    def _find_statement_end(self, start: int) -> int:
        """Find the offset where the statement starting at ``start`` ends.

        Scala has no statement terminators, so the heuristic is: a statement
        ends at a newline when all brackets are balanced, the line does not
        end with a continuation token (operator, dot, comma, opening
        bracket) and the next line does not start with a ``.`` selection.

        Args:
            start: Offset where the right-hand side begins.

        Returns:
            Offset one past the statement's last character.

        """
        masked = self.masked_source
        n = len(masked)
        depth = 0
        i = start
        line_start = start
        while i < n:
            ch = masked[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    return i
                depth -= 1
            elif ch == "\n":
                if depth == 0:
                    line = masked[line_start:i].rstrip()
                    next_pos = i + 1
                    while next_pos < n and masked[next_pos] in " \t":
                        next_pos += 1
                    next_continues = next_pos < n and masked[next_pos] == "."
                    line_continues = line.endswith((".", "=", "+", ",", "&&", "||", "(", "{"))
                    if not next_continues and not line_continues:
                        return i
                line_start = i + 1
            i += 1
        return n


def parse_scala_code(source_code: str) -> ParseResult:
    """Convenience function to parse Scala Spark code.

    Args:
        source_code: Scala source code string.

    Returns:
        ParseResult containing detected operations (language="scala").

    Example:
        >>> code = '''
        ... val df = spark.read.parquet("data.parquet")
        ... val result = df.groupBy("col").count()
        ... result.show()
        ... '''
        >>> result = parse_scala_code(code)
        >>> result.language
        'scala'

    """
    parser = ScalaCodeParser()
    return parser.parse_source(source_code)
