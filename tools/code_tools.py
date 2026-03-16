"""Code agent tools — execute Python code and capture file artifacts."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import mimetypes
import os
import sys
import tempfile
from typing import Any, Dict, List

from tools import tool

logger = logging.getLogger("code_tools")


def _classify_artifact_type(filename: str, content_type: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "csv":
        return "csv"
    if ext == "pdf":
        return "pdf"
    if ext in ("png", "jpg", "jpeg", "gif", "svg", "webp"):
        return "chart" if any(k in filename.lower() for k in ("plot", "chart", "graph", "fig")) else "image"
    if ext in ("py", "js", "ts", "java", "go", "rs", "sql", "sh", "html", "css"):
        return "code"
    if ext == "json":
        return "json"
    if content_type and content_type.startswith("image/"):
        return "chart"
    return "text"


def _build_preview(filename: str, file_bytes: bytes, art_type: str) -> dict:
    """Build type-specific preview data (lightweight, for the card UI)."""
    preview: dict = {}
    try:
        if art_type == "csv":
            text = file_bytes.decode("utf-8", errors="replace")
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if rows:
                preview["headers"] = rows[0]
                preview["sample_rows"] = rows[1:6]
                preview["total_rows"] = len(rows) - 1
        elif art_type == "code":
            text = file_bytes.decode("utf-8", errors="replace")
            lines = text.split("\n")
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
            preview["snippet"] = "\n".join(lines[:20])
            preview["language"] = ext
            preview["total_lines"] = len(lines)
        elif art_type == "json":
            parsed = json.loads(file_bytes)
            preview["snippet"] = json.dumps(parsed, indent=2)[:500]
            preview["keys"] = list(parsed.keys()) if isinstance(parsed, dict) else None
        elif art_type in ("chart", "image"):
            preview["format"] = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        elif art_type == "pdf":
            preview["format"] = "pdf"
    except Exception:
        pass
    return preview


@tool("code_agent", requires_approval=True)
async def execute_code(code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code in a subprocess. Any files written to OUTPUT_DIR are
    captured as artifact previews (base64-encoded, NOT uploaded to S3).
    """
    if language != "python":
        return {"error": f"Unsupported language: {language}", "exit_code": 1}

    with tempfile.TemporaryDirectory(prefix="mrag_code_") as tmpdir:
        # Strip any LLM-generated OUTPUT_DIR reassignment so our preamble is authoritative
        import re
        cleaned_code = re.sub(r'^OUTPUT_DIR\s*=\s*.+$', f'# OUTPUT_DIR is pre-set', code, flags=re.MULTILINE)

        preamble = f'import os; OUTPUT_DIR = r"{tmpdir}"; os.makedirs(OUTPUT_DIR, exist_ok=True)\n'
        full_code = preamble + cleaned_code

        logger.debug("[execute_code] OUTPUT_DIR=%s", tmpdir)
        logger.debug("[execute_code] Generated code (first 500 chars):\n%s", cleaned_code[:500])

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                full_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "OUTPUT_DIR": tmpdir},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": "Execution timed out", "exit_code": -1}
        except Exception as exc:
            return {"error": str(exc), "exit_code": -1}

        # Scan tmpdir for produced files
        artifacts: List[Dict[str, Any]] = []
        try:
            tmpdir_contents = os.listdir(tmpdir)
            logger.info("[execute_code] tmpdir contents after run: %s", tmpdir_contents)
            for entry in tmpdir_contents:
                filepath = os.path.join(tmpdir, entry)
                if not os.path.isfile(filepath):
                    continue
                file_bytes = open(filepath, "rb").read()
                if len(file_bytes) == 0:
                    continue

                ct, _ = mimetypes.guess_type(entry)
                ct = ct or "application/octet-stream"
                art_type = _classify_artifact_type(entry, ct)
                preview = _build_preview(entry, file_bytes, art_type)
                b64 = base64.b64encode(file_bytes).decode("ascii")

                artifacts.append({
                    "filename": entry,
                    "artifact_type": art_type,
                    "content_type": ct,
                    "file_size_bytes": len(file_bytes),
                    "preview_data": preview,
                    "base64_data": b64,
                })
        except Exception:
            pass  # file scanning is best-effort

        return {
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
            "exit_code": proc.returncode,
            "artifacts": artifacts,
        }


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
