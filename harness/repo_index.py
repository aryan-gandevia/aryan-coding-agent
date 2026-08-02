"""Lightweight repository indexing for Python workspaces.

The index maps files to their extracted symbols, imports, and docstrings, then
builds an inverted keyword index so the agent can quickly locate relevant files
without blind exploration.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


# Directories/files to skip during indexing.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    ".aryan-coding-agent",
    ".workspace",
    "build",
    "dist",
    ".egg-info",
    ".idea",
    ".vscode",
}
SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".class"}

# In-memory cache keyed by absolute workspace root path.
_INDEX_CACHE: dict[str, dict[str, Any]] = {}


def _index_cache_dir() -> Path:
    """Return the directory where persisted indices are stored."""
    cache = Path.home() / ".aryan-coding-agent" / "index_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _workspace_hash(root: Path) -> str:
    """Stable hash for a workspace root path."""
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def _persisted_index_path(root: Path) -> Path:
    """Path to the persisted index for a workspace."""
    return _index_cache_dir() / _workspace_hash(root) / "index.json"


def _split_identifier(name: str) -> list[str]:
    """Split a Python identifier into words, e.g. running_mean -> [running, mean]."""
    # Split snake_case
    parts = name.split("_")
    # Split CamelCase / camelCase boundaries
    words: list[str] = []
    for part in parts:
        words.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", part))
    return [w.lower() for w in words if w]


def _extract_python_info(content: str) -> dict[str, Any]:
    """Parse Python source and extract symbols, imports, and docstring."""
    info: dict[str, Any] = {
        "symbols": [],
        "imports": [],
        "docstring": "",
    }
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return info

    # Module-level docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        info["docstring"] = tree.body[0].value.value.strip()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info["symbols"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            info["symbols"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info["imports"].append(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                info["imports"].append(name)
                if module:
                    info["imports"].append(f"{module}.{name}")

    return info


def _collect_tokens(rel_path: str, info: dict[str, Any]) -> set[str]:
    """Collect all searchable tokens for a file."""
    tokens: set[str] = set()

    # Path tokens
    for part in Path(rel_path).parts:
        tokens.add(part.lower())
        tokens.update(_split_identifier(part))

    # Extension token
    ext = Path(rel_path).suffix.lower().lstrip(".")
    if ext:
        tokens.add(ext)

    # Symbols
    for symbol in info.get("symbols", []):
        tokens.add(symbol.lower())
        tokens.update(_split_identifier(symbol))

    # Imports
    for imp in info.get("imports", []):
        tokens.add(imp.lower())
        tokens.update(_split_identifier(imp))
        tokens.update(imp.lower().split("."))

    # Docstring words
    doc = info.get("docstring", "")
    tokens.update(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", doc.lower()))

    return tokens


def _build_file_entry(root: Path, file_path: Path) -> dict[str, Any]:
    """Create an index entry for a single file."""
    rel_path = str(file_path.relative_to(root))
    stat = file_path.stat()
    content = ""
    info: dict[str, Any] = {"symbols": [], "imports": [], "docstring": ""}

    if file_path.suffix == ".py":
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        info = _extract_python_info(content)

    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "symbols": info["symbols"],
        "imports": info["imports"],
        "docstring": info["docstring"],
        "tokens": sorted(_collect_tokens(rel_path, info)),
    }


def _should_skip(path: Path) -> bool:
    """Return True if a path should be skipped during indexing."""
    if any(part in SKIP_DIRS or part.startswith(".") for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    return False


def build_index(root: str | Path) -> dict[str, Any]:
    """Build a fresh index for the workspace at ``root``."""
    root_path = Path(root).resolve()
    files: dict[str, Any] = {}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Mutate dirnames in-place to avoid descending into skipped dirs.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if _should_skip(file_path):
                continue
            try:
                rel_path = str(file_path.relative_to(root_path))
                files[rel_path] = _build_file_entry(root_path, file_path)
            except (OSError, ValueError):
                continue

    inverted_index: dict[str, set[str]] = {}
    for rel_path, entry in files.items():
        for token in entry.get("tokens", []):
            inverted_index.setdefault(token, set()).add(rel_path)

    # Convert sets to sorted lists for JSON serialization.
    index: dict[str, Any] = {
        "workspace_root": str(root_path),
        "created_at": time.time(),
        "files": files,
        "inverted_index": {k: sorted(v) for k, v in sorted(inverted_index.items())},
    }

    _save_index(root_path, index)
    return index


def _save_index(root: Path, index: dict[str, Any]) -> None:
    """Persist the index to disk and the in-memory cache."""
    path = _persisted_index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    _INDEX_CACHE[str(root.resolve())] = index


def _is_stale(index: dict[str, Any], root: Path) -> bool:
    """Check whether the stored index still matches the filesystem."""
    stored_root = index.get("workspace_root")
    if stored_root != str(root.resolve()):
        return True

    files = index.get("files", {})
    for rel_path, entry in files.items():
        file_path = root / rel_path
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            return True
        if stat.st_mtime != entry.get("mtime") or stat.st_size != entry.get("size"):
            return True

    # Very coarse check: if there are files we didn't index, consider stale.
    # A full re-index is cheap enough for typical agent workspaces.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if _should_skip(file_path):
                continue
            try:
                rel_path = str(file_path.relative_to(root))
            except ValueError:
                continue
            if rel_path not in files:
                return True

    return False


def get_index(root: str | Path) -> dict[str, Any]:
    """Load or build the index for ``root``."""
    root_path = Path(root).resolve()
    key = str(root_path)

    if key in _INDEX_CACHE:
        if not _is_stale(_INDEX_CACHE[key], root_path):
            return _INDEX_CACHE[key]

    path = _persisted_index_path(root_path)
    if path.exists():
        try:
            index = json.loads(path.read_text(encoding="utf-8"))
            if not _is_stale(index, root_path):
                _INDEX_CACHE[key] = index
                return index
        except (json.JSONDecodeError, OSError):
            pass

    return build_index(root_path)


def invalidate_index(root: str | Path) -> None:
    """Drop the in-memory and disk index for ``root``."""
    root_path = Path(root).resolve()
    key = str(root_path)
    _INDEX_CACHE.pop(key, None)
    path = _persisted_index_path(root_path)
    if path.exists():
        path.unlink()


def update_file_in_index(root: str | Path, rel_path: str, content: str | None = None) -> None:
    """Update a single file entry in the cached index without a full rebuild."""
    root_path = Path(root).resolve()
    index = get_index(root_path)
    files = index.get("files", {})
    inverted_index: dict[str, list[str]] = index.get("inverted_index", {})

    # Remove old tokens for this file.
    old_entry = files.pop(rel_path, {})
    for token in old_entry.get("tokens", []):
        if token in inverted_index:
            inverted_index[token] = [p for p in inverted_index[token] if p != rel_path]
            if not inverted_index[token]:
                del inverted_index[token]

    file_path = root_path / rel_path
    if not file_path.exists():
        _save_index(root_path, index)
        return

    if content is None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            content = ""

    info = _extract_python_info(content) if file_path.suffix == ".py" else {"symbols": [], "imports": [], "docstring": ""}
    entry = _build_file_entry(root_path, file_path)
    files[rel_path] = entry

    for token in entry["tokens"]:
        inverted_index.setdefault(token, [])
        if rel_path not in inverted_index[token]:
            inverted_index[token].append(rel_path)
            inverted_index[token].sort()

    _save_index(root_path, index)


def query_index(root: str | Path, query: str, top_k: int = 10) -> list[str]:
    """Return the top ``top_k`` file paths matching ``query``."""
    root_path = Path(root).resolve()
    index = get_index(root_path)
    inverted_index = index.get("inverted_index", {})

    if not query.strip():
        return []

    # Tokenize the query: extract identifier-like words and split camelCase.
    query_tokens: set[str] = set()
    for raw in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower()):
        query_tokens.add(raw)
        query_tokens.update(_split_identifier(raw))

    scores: dict[str, int] = {}
    for token in query_tokens:
        for rel_path in inverted_index.get(token, []):
            scores[rel_path] = scores.get(rel_path, 0) + 1

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [rel_path for rel_path, _ in ranked[:top_k]]
