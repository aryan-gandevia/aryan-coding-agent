"""Tool schemas and dispatch builders for low-level subagent tools."""

import json
from typing import Callable

from tools.tools import Workspace


READ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file at the given workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "Read a file and display it with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the given workspace-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents for a query string within an optional glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["path", "query"],
            },
        },
    },
]


WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write or overwrite a file at a workspace-relative path with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Replace a contiguous block of lines (1-indexed, inclusive) with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "new_content": {"type": "string"},
            },
            "required": ["path", "start_line", "end_line", "new_content"],
        },
    },
}


EXECUTE_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Run a shell command inside the workspace. Use this for test runs or git.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}


RUN_TESTS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the project's test suite with pytest.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
        },
    },
}


QUERY_REPO_INDEX_TOOL = {
    "type": "function",
    "function": {
        "name": "query_repo_index",
        "description": (
            "Find the most relevant files for a natural-language query using the "
            "workspace's built-in code index. Use this first when exploring a "
            "workspace with more than a handful of files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}


def _build_dispatch(workspace: Workspace, allowed: set[str]) -> dict[str, Callable]:
    dispatch: dict[str, Callable] = {}
    tool_map = {
        "read_file": workspace.read_file,
        "view_file": workspace.view_file,
        "list_files": workspace.list_files,
        "search_files": workspace.search_files,
        "write_file": workspace.write_file,
        "edit_file": workspace.edit_file,
        "execute_command": workspace.execute_command,
        "run_tests": workspace.run_tests,
        "query_repo_index": workspace.query_repo_index,
    }
    for name in allowed:
        if name in tool_map:
            dispatch[name] = tool_map[name]
    return dispatch


def make_explorer_tools() -> list[dict]:
    """Read-only exploration tools plus index lookup and a constrained test runner."""
    return [*READ_TOOLS, QUERY_REPO_INDEX_TOOL, RUN_TESTS_TOOL]


def make_explorer_dispatch(workspace: Workspace) -> dict[str, Callable]:
    return _build_dispatch(
        workspace,
        {"read_file", "view_file", "list_files", "search_files", "query_repo_index", "run_tests"},
    )


def make_coder_tools() -> list[dict]:
    """Full toolset for code editing, exploration, and testing."""
    return [*READ_TOOLS, WRITE_TOOL, EDIT_TOOL, QUERY_REPO_INDEX_TOOL, EXECUTE_COMMAND_TOOL, RUN_TESTS_TOOL]


def make_coder_dispatch(workspace: Workspace) -> dict[str, Callable]:
    return _build_dispatch(
        workspace,
        {
            "read_file",
            "view_file",
            "list_files",
            "search_files",
            "query_repo_index",
            "write_file",
            "edit_file",
            "execute_command",
            "run_tests",
        },
    )


def make_reviewer_tools() -> list[dict]:
    """Reviewer reads files and runs verification commands, but does not edit."""
    return [*READ_TOOLS, EXECUTE_COMMAND_TOOL]


def make_reviewer_dispatch(workspace: Workspace) -> dict[str, Callable]:
    return _build_dispatch(
        workspace,
        {"read_file", "view_file", "list_files", "search_files", "execute_command"},
    )


def extract_tool_calls(message) -> list[dict]:
    """Convert OpenAI tool_calls into a list of plain dicts for storage."""
    calls = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
    return calls
