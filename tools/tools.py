"""Low-level workspace tools for Agent 7 subagents."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from harness import repo_index


class Workspace:
    """Encapsulates the repository root and enforces that all operations stay inside it."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Workspace does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Workspace is not a directory: {self.root}")

    def _resolve(self, path: str) -> Path:
        target = self.root.joinpath(path).resolve()
        if not str(target).startswith(str(self.root)):
            raise ValueError(f"Path escapes workspace: {path}")
        return target

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_text()

    def view_file(self, path: str) -> str:
        content = self.read_file(path)
        lines = content.splitlines()
        rendered = [f"File: {path} ({len(lines)} lines)"]
        for i, line in enumerate(lines, 1):
            rendered.append(f"{i:3} | {line}")
        return "\n".join(rendered)

    def list_files(self, path: str = ".", recursive: bool = False) -> list[str] | str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if recursive:
            entries = [str(p.relative_to(self.root)) for p in sorted(target.rglob("*"))]
            return "\n".join(entries)
        entries = [str(p.relative_to(self.root)) for p in sorted(target.iterdir())]
        return entries

    def search_files(self, path: str = ".", query: str = "", glob: str = "*") -> list[str]:
        target = self._resolve(path)
        matches = []
        for p in sorted(target.rglob(glob)):
            if p.is_file() and query in p.read_text(errors="ignore"):
                matches.append(str(p.relative_to(self.root)))
        return matches

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        repo_index.update_file_in_index(self.root, path, content)
        return f"Wrote {path}"

    def edit_file(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text().splitlines()
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            raise ValueError(f"Invalid line range {start_line}-{end_line}. File has {len(lines)} lines.")
        new_lines = new_content.splitlines()
        updated = lines[: start_line - 1] + new_lines + lines[end_line:]
        target.write_text("\n".join(updated) + "\n")
        repo_index.update_file_in_index(self.root, path)
        return f"Edited lines {start_line}-{end_line} in {path}"

    def query_repo_index(self, query: str, top_k: int = 10) -> str:
        results = repo_index.query_index(self.root, query, top_k=top_k)
        if not results:
            return "No matching files found."
        return "Relevant files:\n" + "\n".join(f"  - {p}" for p in results)

    def execute_command(self, command: str, cwd: str | None = None, timeout: int = 60) -> dict:
        run_dir = self._resolve(cwd) if cwd else self.root
        completed = subprocess.run(
            command,
            shell=True,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }

    def run_tests(self, command: str = "pytest -q", timeout: int = 120) -> dict:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as cache_dir:
            env["PYTEST_CACHE_DIR"] = cache_dir
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }
