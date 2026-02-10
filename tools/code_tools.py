"""Code agent tools."""

from __future__ import annotations

import asyncio
import textwrap
from typing import Any, Dict, List

from tools import tool


@tool("code_agent")
async def execute_code(code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code in a sandboxed subprocess and return stdout/stderr/exit_code.
    """
    if language != "python":
        return {"error": f"Unsupported language: {language}", "exit_code": 1}

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "exit_code": proc.returncode,
        }
    except asyncio.TimeoutError:
        proc.kill() 
        return {"error": "Execution timed out", "exit_code": -1}
    except Exception as exc:
        return {"error": str(exc), "exit_code": -1}


@tool("code_agent")
async def code_linter(code: str, language: str = "python") -> List[Dict[str, Any]]:
    """
    Lint code and return a list of issues.
    """
    issues: List[Dict[str, Any]] = []
    try:
        compile(code, "<agent_code>", "exec")
    except SyntaxError as exc:
        issues.append(
            {
                "line": exc.lineno,
                "col": exc.offset,
                "message": exc.msg,
                "severity": "error",
            }
        )
    return issues
