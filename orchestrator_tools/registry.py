"""Tool schemas and dispatch builder for the parent orchestrator's tools.

The parent agent does not call file-system tools directly. Instead it calls these
high-level tools, each of which dispatches a subagent or runs a high-level action.
"""

from functools import partial
from typing import Callable

from orchestrator_tools import tools as orchestrator_tools


ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "explore",
            "description": (
                "Dispatch an Explorer subagent to research a specific question about the repo. "
                "Use this when you need to understand files, tests, or current behavior."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The overall user task."},
                    "question": {"type": "string", "description": "The specific research question."},
                },
                "required": ["task", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": (
                "Call a Planner subagent to produce or update a concrete, numbered plan. "
                "Use this when you need to break down a complex task into steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The overall user task."},
                    "context": {"type": "string", "description": "Any relevant context discovered so far."},
                },
                "required": ["task", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": (
                "Dispatch a Coder subagent to implement a specific subtask on the given files. "
                "Provide a clear subtask and the files involved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The overall user task."},
                    "plan": {"type": "string", "description": "The plan or specific change to implement."},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files the coder should focus on.",
                    },
                },
                "required": ["task", "plan", "files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test",
            "description": (
                "Run the test suite directly. Use this after code changes to verify progress."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "description": "Optional test scope, e.g. 'tests/test_foo.py' or ''."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review",
            "description": (
                "Dispatch a Reviewer subagent to assess whether the current work satisfies the task. "
                "The reviewer will respond with PASS or REVISE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The overall user task."},
                    "summary": {"type": "string", "description": "A summary of what has been done and current test results."},
                },
                "required": ["task", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Mark the task as complete and provide a final summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Final summary of the work completed."},
                },
                "required": ["summary"],
            },
        },
    },
]


def make_orchestrator_tools_dispatch(workspace_root: str) -> dict[str, Callable]:
    """Return a dispatch table for orchestrator tools bound to a workspace root."""
    return {
        "explore": partial(orchestrator_tools.explore, workspace_root=workspace_root),
        "plan": partial(orchestrator_tools.plan, workspace_root=workspace_root),
        "code": partial(orchestrator_tools.code, workspace_root=workspace_root),
        "test": partial(orchestrator_tools.test, workspace_root=workspace_root),
        "review": partial(orchestrator_tools.review, workspace_root=workspace_root),
        "finish": partial(orchestrator_tools.finish, workspace_root=workspace_root),
    }
