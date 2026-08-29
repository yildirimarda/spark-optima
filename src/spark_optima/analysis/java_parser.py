# Copyright 2024 Spark Optima Contributors
# Licensed under the Apache License, Version 2.0

"""Lightweight lexer-based parser for Java Spark sources.

This module provides the JavaCodeParser class that detects Spark
DataFrame/Dataset/RDD operations in Java source code without a full Java
compiler. Comments are stripped and string literals are masked *before*
scanning. The parser produces the same :class:`~spark_optima.analysis.parser.ParseResult`
model as the Python and Scala parsers, tagged with ``language="java"``.

Java-specific concerns handled:
    - ``SparkSession.builder`` chaining (``.appName(...).getOrCreate()``)
    - Java UDF references (``org.apache.spark.api.java.function.MapFunction``)
    - Typed variable assignments (``Dataset<Row> df = ...``)
    - Plain double-quoted strings and triple-quoted text-block strings.

Limitations:
    - Expressions inside string interpolation are not supported (Java does
      not have native string interpolation in the versions targeted).
    - Parenthesis-less arity-0 calls are only recognized for reader/writer
      accessors.
    - Loop context (`in_loop`) is always ``False`` for Java operations.
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

# Matches standalone broadcast(...)/udf(...) calls
_STANDALONE_CALL_RE = re.compile(r"(?<![\w.])(broadcast|udf)\s*\(")

# Matches Java variable assignments: "Dataset<Row> df = ...", "df = ...", etc.
_ASSIGNMENT_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z_][\w<>\[\],\.\s]*?\s+)?([A-Za-z_]\w*)(?:\s*<[A-Za-z_]\w*>)?\s*=(?!=)",
    re.MULTILINE,
)

# Escape sequences resolved inside regular (non-triple-quoted) Java strings
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

# Java-specific strong markers used by language detection
_JAVA_STRONG_HINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*import\s+org\.apache\.spark", re.MULTILINE),
    re.compile(
        r"^\s*(?:final\s+)?(?:[A-Z][A-Za-z_]*|[A-Za-z_]\w*(?:<[A-Za-z_]\w*>)?)\s+[A-Za-z_]\w*\s*=(?!=)",
        re.MULTILINE,
    ),
    re.compile(r"SparkSession\.builder", re.MULTILINE),
    re.compile(r"org\.apache\.spark\.api\.java\.function", re.MULTILINE),
)

# Python markers that override Java classification
_PYTHON_HINTS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:from|import)\s+pyspark\b", re.MULTILINE),
    re.compile(r"^\s*def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:\n]+)?\s*:\s*(?:#.*)?$", re.MULTILINE),
    re.compile(r"^\s*if\s+__name__\s*==", re.MULTILINE),
)

# Lightweight mask used by detect_language
_PYTHON_MASK_RE = re.compile(
    r"'''(?:\\.|[^\\])*?'''"
    r'|"""(?:\\.|[^\\])*?"""'
    r"|'(?:\\.|[^'\\\n])*'"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|#[^\n]*"
)


def _mask_python_like(source_code: str) -> str:
    """Blank string literals and comments while keeping newlines."""

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return _PYTHON_MASK_RE.sub(blank, source_code)


def detect_language(source_code: str) -> str:
    """Guess whether source text is Python, Scala, or Java Spark code.

    Python markers win, Scala requires a strong Scala marker, Java
    requires a strong Java marker; ambiguity defaults to Python.

    Args:
        source_code: Raw source text.

    Returns:
        "java" when a strong Java marker is present and no Python/Scala
        markers override it, "scala" when Scala markers dominate,
        "python" otherwise.

    """
    masked = _mask_python_like(source_code)
    # Python always wins if present
    if any(pattern.search(masked) for pattern in _PYTHON_HINTS):
        return "python"
    # Scala requires a strong Scala-specific marker (val/var assignment,
    # object/def syntax, or typed val assignment with Scala-style syntax)
    scala_strong = (
        re.compile(
            r"^\s*(?:final\s+)?(?:lazy\s+)?(?:val|var)\s+[A-Za-z_]\w*(?:\s*:\s*[^=\n]+?)?\s*=(?!=)",
            re.MULTILINE,
        ),
        re.compile(r"^\s*import\s+org\.apache\.spark", re.MULTILINE),
        re.compile(r"\bobject\s+[A-Z]\w*", re.MULTILINE),
        re.compile(r"def\s+main\s*\(", re.MULTILINE),
    )
    has_scala_assignment = any(p.search(masked) for p in scala_strong[:1])
    has_scala_object = any(p.search(masked) for p in scala_strong[2:4])
    has_scala_import = any(p.search(masked) for p in scala_strong[1:2])
    has_java = any(p.search(masked) for p in _JAVA_STRONG_HINTS)
    # Unambiguous Scala markers override everything
    if has_scala_assignment:
        return "scala"
    if has_scala_object:
        return "scala"
    # Import-only: prefer Scala if no strong Java markers, else Java
    if has_scala_import:
        if not has_java:
            return "scala"
        # Shared import + strong java => java, else scala
        if has_java:
            return "java"
        return "scala"
    if has_java:
        return "java"
    return "python"


@dataclass
class _StringLiteral:
    token_start: int
    start: int
    end: int
    content: str
    interpolated: bool


def _unescape(text: str) -> str:
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


def _mask_source(source: str) -> tuple[str, list[_StringLiteral]]:
    """Blank out comments and string literals while preserving offsets."""
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
        elif ch == '"':
            if source.startswith('"""', i):
                j = i + 3
                close = -1
                while j < n:
                    if source.startswith('"""', j):
                        close = j
                        break
                    j += 1
                if close == -1:
                    content, end = source[i + 3 :], n
                else:
                    content, end = source[i + 3 : close], close + 3
                literals.append(
                    _StringLiteral(
                        token_start=i,
                        start=i,
                        end=end,
                        content=content,
                        interpolated=False,
                    )
                )
                blank(i, end)
                i = end
            else:
                j = i + 1
                while j < n and source[j] != '"' and source[j] != "\n":
                    if source[j] == "\\":
                        j += 2
                    else:
                        j += 1
                raw = source[i + 1 : j]
                end = min(j + 1, n)
                content = _unescape(raw)
                literals.append(
                    _StringLiteral(
                        token_start=i,
                        start=i,
                        end=end,
                        content=content,
                        interpolated=False,
                    )
                )
                blank(i, end)
                i = end
        else:
            i += 1
    return "".join(chars), literals


def _match_bracket(masked: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for k in range(open_pos, len(masked)):
        if masked[k] == open_ch:
            depth += 1
        elif masked[k] == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return -1


class JavaCodeParser:
    """Parser for analyzing Java Spark code with a lightweight lexer.

    Produces the same :class:`~spark_optima.analysis.parser.ParseResult`
    model tagged with ``language="java"``.
    """

    def __init__(self) -> None:
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
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return self.parse_source(file_path.read_text(encoding="utf-8"))

    def parse_source(self, source_code: str) -> ParseResult:
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
            language="java",
        )

    def get_dataframe_lineage(self, var_name: str) -> list[SparkOperation]:
        return self.dataframe_vars.get(var_name, [])

    def _scan_method_calls(self) -> list[tuple[int, str, str, list[str], int]]:
        masked = self.masked_source
        found: list[tuple[int, str, str, list[str], int]] = []
        for match in _METHOD_CALL_RE.finditer(masked):
            name = match.group(1)
            if name not in SPARK_METHOD_MAP:
                continue
            j = self._skip_whitespace(match.end())
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
                found.append((match.start(), name, dataframe_var, [], match.end()))
        return found

    def _scan_standalone_calls(self) -> list[tuple[int, str, str, list[str], int]]:
        masked = self.masked_source
        found: list[tuple[int, str, str, list[str], int]] = []
        for match in _STANDALONE_CALL_RE.finditer(masked):
            name = match.group(1)
            open_paren = match.end() - 1
            close_paren = _match_bracket(masked, open_paren, "(", ")")
            if close_paren == -1:
                continue
            arguments = [] if name == "broadcast" else self._extract_call_arguments(open_paren, close_paren)
            found.append((match.start(), name, _STANDALONE_FUNCTIONS[name], arguments, close_paren + 1))
        return found

    def _skip_whitespace(self, offset: int) -> int:
        masked = self.masked_source
        while offset < len(masked) and masked[offset] in " \t":
            offset += 1
        return offset

    def _resolve_chain_root(self, dot_pos: int) -> str | None:
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
        for literal in self.string_literals:
            if literal.token_start == start and literal.end == end:
                return literal
            if literal.token_start > end:
                break
        return None

    def _add_operation(
        self,
        offset: int,
        method_name: str,
        dataframe_var: str,
        arguments: list[str],
        end_offset: int,
    ) -> SparkOperation:
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
        line_idx = bisect_right(self._line_starts, offset) - 1
        end_idx = bisect_right(self._line_starts, max(end_offset - 1, 0)) - 1
        return CodeLocation(
            line=line_idx + 1,
            column=offset - self._line_starts[line_idx],
            end_line=end_idx + 1,
            end_column=max(end_offset - 1, 0) - self._line_starts[end_idx] + 1,
        )

    def _apply_assignment_lineage(self) -> None:
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
        group_a = self._chain_group_by_op.get(id(op_a))
        group_b = self._chain_group_by_op.get(id(op_b))
        return group_a is not None and group_a == group_b

    def assignment_targets(self, op: SparkOperation) -> frozenset[str]:
        return frozenset(self._assignment_targets_by_op.get(id(op), ()))

    def _find_statement_end(self, start: int) -> int:
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


def parse_java_code(source_code: str) -> ParseResult:
    """Convenience function to parse Java Spark code.

    Args:
        source_code: Java source code string.

    Returns:
        ParseResult containing detected operations (language="java").

    """
    parser = JavaCodeParser()
    return parser.parse_source(source_code)
