"""High-level tools that the parent orchestrator uses to dispatch subagents.

Each function here is a code implementation. Some of them invoke a subagent LLM loop;
others (like ``test``) simply run a tool directly. The parent sees all of them as
uniform high-level actions through ``orchestrator_tools/registry.py``.
"""

import json

from models.model import call_model, call_text
from prompts.prompts import (
    CODER_SYSTEM_PROMPT,
    EXPLORER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    build_coder_prompt,
    build_explorer_prompt,
    build_planner_prompt,
    build_reviewer_prompt,
)
from tools.tools import Workspace
from tools.tools_registry import (
    make_coder_dispatch,
    make_coder_tools,
    make_explorer_dispatch,
    make_explorer_tools,
)


MAX_SUBAGENT_STEPS = 20


def _run_subagent(
    workspace_root: str,
    system_prompt: str,
    user_prompt: str,
    make_tools,
    make_dispatch,
    max_steps: int = MAX_SUBAGENT_STEPS,
) -> tuple[str, list[dict]]:
    """Run a generic tool-calling subagent loop.

    Returns the subagent's final text response and a short trace.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    workspace = Workspace(workspace_root)
    tools = make_tools()
    dispatch = make_dispatch(workspace)
    trace: list[dict] = []

    for step in range(max_steps):
        response = call_model(messages, tools)
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "(no response)", trace

        tool_call_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        }
        messages.append(tool_call_message)

        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            fn = dispatch.get(name)
            try:
                if fn is None:
                    raise ValueError(f"Unknown tool: {name}")
                result = fn(**args)
                success = True
            except Exception as exc:
                result = f"Error: {exc}"
                success = False

            trace.append(
                {
                    "step": step,
                    "tool": name,
                    "args": args,
                    "success": success,
                    "result": str(result)[:300],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "(subagent reached step limit)", trace


def explore(task: str, question: str, workspace_root: str) -> str:
    """Dispatch an Explorer subagent to research a specific question."""
    system_prompt = EXPLORER_SYSTEM_PROMPT
    user_prompt = build_explorer_prompt(task, question)
    result, trace = _run_subagent(
        workspace_root,
        system_prompt,
        user_prompt,
        make_explorer_tools,
        make_explorer_dispatch,
    )
    return f"EXPLORER RESULT:\n{result}\n\nTRACE:\n{json.dumps(trace, indent=2)}"


def plan(task: str, context: str, workspace_root: str) -> str:
    """Call a text-only Planner to produce or update a plan."""
    system_prompt = (
        "You are a Planner. Convert the user's task and context into a concrete, "
        "numbered plan. Each step should be small, specify files when possible, and "
        "note dependencies. Output only the plan."
    )
    user_prompt = build_planner_prompt(task, context)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_text(messages)


def code(task: str, plan: str, files: list[str], workspace_root: str) -> str:
    """Dispatch a Coder subagent to implement a specific subtask."""
    system_prompt = CODER_SYSTEM_PROMPT
    user_prompt = build_coder_prompt(task, plan, files)
    result, trace = _run_subagent(
        workspace_root,
        system_prompt,
        user_prompt,
        make_coder_tools,
        make_coder_dispatch,
    )
    return f"CODER RESULT:\n{result}\n\nTRACE:\n{json.dumps(trace, indent=2)}"


def test(scope: str, workspace_root: str) -> str:
    """Run the test suite directly (no LLM)."""
    workspace = Workspace(workspace_root)
    command = "pytest -q" if not scope else f"pytest -q {scope}"
    result = workspace.run_tests(command=command)
    return f"TEST RESULT:\n{json.dumps(result, indent=2)}"


def review(task: str, summary: str, workspace_root: str) -> str:
    """Dispatch a Reviewer subagent to assess current progress."""
    system_prompt = REVIEWER_SYSTEM_PROMPT
    user_prompt = build_reviewer_prompt(task, summary)
    result, trace = _run_subagent(
        workspace_root,
        system_prompt,
        user_prompt,
        make_explorer_tools,
        make_explorer_dispatch,
    )
    return f"REVIEWER RESULT:\n{result}\n\nTRACE:\n{json.dumps(trace, indent=2)}"


def finish(summary: str, workspace_root: str) -> str:
    """Signal that the task is complete."""
    return f"FINISH: {summary}"
