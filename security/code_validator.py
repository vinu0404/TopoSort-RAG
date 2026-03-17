"""
AST-based + regex-based pre-execution code validator.

Inspects LLM-generated Python code **before** it reaches ``subprocess``
to block dangerous imports, calls, and path references.

Usage::

    from security.code_validator import validate_and_log

    result = validate_and_log(code, agent_id="code_agent")
    if not result.is_safe:
        return {"error": "blocked", "violations": result.violations}
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple

logger = logging.getLogger("security.code_validator")

# ── Blocked imports ───────────────────────────────────────────────────────
DANGEROUS_IMPORTS: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "requests", "urllib", "httplib", "ftplib",
    "smtplib", "telnetlib",
    "pickle", "marshal", "shelve",
    "ctypes", "multiprocessing", "threading", "_thread",
    "importlib", "builtins",
    "code", "codeop", "pty", "tty", "termios",
    "fcntl", "resource", "gc", "inspect", "signal",
    "asyncio.subprocess",
}

# ── Explicitly allowed imports for data / chart / PDF work ────────────────
ALLOWED_IMPORTS: Set[str] = {
    "json", "math", "statistics", "datetime", "collections",
    "itertools", "functools", "re", "csv", "io", "textwrap",
    "string", "decimal", "fractions", "random", "copy",
    "pandas", "numpy",
    "matplotlib", "matplotlib.pyplot", "matplotlib.figure",
    "matplotlib.dates", "matplotlib.ticker",
    "reportlab", "reportlab.lib", "reportlab.lib.pagesizes",
    "reportlab.lib.styles", "reportlab.lib.units",
    "reportlab.lib.colors", "reportlab.lib.enums",
    "reportlab.platypus", "reportlab.platypus.tables",
    "reportlab.pdfgen", "reportlab.pdfgen.canvas",
}

# ── Dangerous function calls ──────────────────────────────────────────────
DANGEROUS_CALLS: Set[str] = {
    "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "input", "breakpoint", "exit", "quit",
}

# ── Regex patterns for string-level evasion ───────────────────────────────
DANGEROUS_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("os.system",          re.compile(r"os\.system\s*\(")),
    ("subprocess",         re.compile(r"subprocess\.")),
    ("Popen",              re.compile(r"Popen\s*\(")),
    ("popen",              re.compile(r"\.popen\s*\(")),
    ("shutil.rmtree",      re.compile(r"shutil\.rmtree")),
    ("shutil.move",        re.compile(r"shutil\.move")),
    ("socket",             re.compile(r"socket\.")),
    ("urllib.request",     re.compile(r"urllib\.request")),
    ("http_request",       re.compile(r"requests\.(get|post|put|delete|patch)")),
    ("pickle",             re.compile(r"pickle\.loads?")),
    ("builtins_access",    re.compile(r"__builtins__")),
    ("class_manip",        re.compile(r"__class__")),
    ("bases_access",       re.compile(r"__bases__")),
    ("subclasses",         re.compile(r"__subclasses__")),
    ("mro_access",         re.compile(r"__mro__")),
    ("globals_access",     re.compile(r"__globals__")),
    ("code_access",        re.compile(r"__code__")),
    ("rm_rf",              re.compile(r"rm\s+-rf")),
    ("sudo",               re.compile(r"sudo\s+")),
    ("chmod",              re.compile(r"chmod\s+")),
    ("chown",              re.compile(r"chown\s+")),
]

# ── Forbidden path fragments ─────────────────────────────────────────────
FORBIDDEN_PATHS: List[str] = [
    "/etc/", "/var/", "/usr/", "/bin/", "/sbin/",
    "/root/", "/home/",
    "C:\\Windows", "C:\\Program Files", "C:\\Users",
    "../", "..\\",
    ".env", "credentials", "secret", "password",
    "token", "private", "id_rsa", ".ssh", ".git",
]


@dataclass
class CodeValidationResult:
    """Outcome of a code-validation pass."""

    is_safe: bool
    violations: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low | medium | high | critical


class CodeValidator:
    """Validate LLM-generated Python code before execution."""

    def __init__(
        self,
        allowed_imports: Set[str] | None = None,
        max_code_length: int = 50_000,
    ):
        self.allowed_imports = allowed_imports or ALLOWED_IMPORTS
        self.max_code_length = max_code_length

    # ── public ────────────────────────────────────────────────────────────

    def validate(self, code: str) -> CodeValidationResult:
        violations: list[str] = []

        # 1. Length
        if len(code) > self.max_code_length:
            violations.append(
                f"Code length {len(code)} exceeds max {self.max_code_length}"
            )

        # 2. AST analysis
        try:
            tree = ast.parse(code)
            violations.extend(self._check_imports(tree))
            violations.extend(self._check_calls(tree))
        except SyntaxError as exc:
            violations.append(f"Syntax error: {exc}")

        # 3. Regex patterns
        violations.extend(self._check_patterns(code))

        # 4. Forbidden paths
        violations.extend(self._check_paths(code))

        # 5. Risk level
        n = len(violations)
        if n > 5:
            risk = "critical"
        elif n > 2:
            risk = "high"
        elif n > 0:
            risk = "medium"
        else:
            risk = "low"

        return CodeValidationResult(
            is_safe=(n == 0),
            violations=violations,
            risk_level=risk,
        )

    # ── private helpers ───────────────────────────────────────────────────

    def _check_imports(self, tree: ast.AST) -> list[str]:
        violations: list[str] = []
        allowed_roots = {m.split(".")[0] for m in self.allowed_imports}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in DANGEROUS_IMPORTS:
                        violations.append(f"Dangerous import: {alias.name}")
                    elif alias.name not in self.allowed_imports and root not in allowed_roots:
                        violations.append(f"Unapproved import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root in DANGEROUS_IMPORTS:
                    violations.append(f"Dangerous import from: {module}")

        return violations

    def _check_calls(self, tree: ast.AST) -> list[str]:
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name in DANGEROUS_CALLS:
                    violations.append(f"Dangerous call: {name}()")
        return violations

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    @staticmethod
    def _check_patterns(code: str) -> list[str]:
        return [
            f"Dangerous pattern: {desc}"
            for desc, rx in DANGEROUS_PATTERNS
            if rx.search(code)
        ]

    @staticmethod
    def _check_paths(code: str) -> list[str]:
        code_lower = code.lower()
        return [
            f"Forbidden path reference: {p}"
            for p in FORBIDDEN_PATHS
            if p.lower() in code_lower
        ]


# ── Module-level convenience functions ────────────────────────────────────
_default_validator = CodeValidator()


def validate_code(code: str) -> CodeValidationResult:
    """Validate code with the default validator instance."""
    return _default_validator.validate(code)


def validate_and_log(code: str, agent_id: str) -> CodeValidationResult:
    """Validate code and log any violations."""
    result = _default_validator.validate(code)
    if not result.is_safe:
        logger.warning(
            "[CodeValidator] agent=%s risk=%s violations=%s",
            agent_id,
            result.risk_level,
            result.violations[:5],
        )
    return result
