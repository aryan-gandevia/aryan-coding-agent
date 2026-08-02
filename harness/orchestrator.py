"""Parent orchestrator harness for Agent 7.

The parent is an LLM with a set of orchestrator tools. Each tool dispatches a
specialized subagent or runs a high-level action. The parent loops until it
calls the ``finish`` tool or reaches a step limit.
"""

import json

from models.model import MODEL, call_model
from orchestrator_tools.registry import ORCHESTRATOR_TOOLS, make_orchestrator_tools_dispatch
from prompts.prompts import PARENT_SYSTEM_PROMPT, build_parent_initial_prompt
from tokens import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    WARNING_THRESHOLD,
    count_message_tokens,
    format_tokens,
    usage_fraction,
)


def _report_tokens(messages: list[dict], max_tokens: int) -> None:
    used = count_message_tokens(messages, model_name=MODEL)
    fraction = usage_fraction(used, max_tokens)
    print(f"[Parent] Tokens: {format_tokens(used, max_tokens)} ({fraction:.1%})")
    if fraction >= WARNING_THRESHOLD:
        print(f"[Parent] Warning: context is {fraction:.0%} full. Consider compacting soon.")


def run_parent_loop(
    messages: list[dict],
    workspace_root: str,
    max_steps: int = 25,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> tuple[str, list[dict]]:
    """Run the parent orchestrator loop starting from an existing message list.

    Returns the final result string and the updated message list (including all
    assistant/tool messages from this run).
    """
    dispatch = make_orchestrator_tools_dispatch(workspace_root)

    for step in range(1, max_steps + 1):
        print(f"[Parent] Step {step}")
        response = call_model(messages, ORCHESTRATOR_TOOLS, tool_choice="required")
        message = response.choices[0].message
        content = message.content or ""

        if not message.tool_calls:
            print(f"[Parent] thought: {content[:200]}")
            messages.append({"role": "assistant", "content": content})
            _report_tokens(messages, max_context_tokens)
            continue

        # Record the assistant's tool call request.
        tool_calls_payload = [tc.model_dump() for tc in message.tool_calls]
        messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls_payload}
        )

        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"[Parent] action: {name}({json.dumps(args)})")

            if name == "finish":
                summary = args.get("summary", "Task completed.")
                _report_tokens(messages, max_context_tokens)
                return summary, messages

            fn = dispatch.get(name)
            try:
                if fn is None:
                    result = f"Error: unknown meta tool '{name}'"
                else:
                    result = fn(**args)
            except Exception as exc:
                result = f"Error: {exc}"

            # Keep the result manageable in the context window.
            result_str = str(result)
            if len(result_str) > 4000:
                result_str = result_str[:2000] + "\n... [truncated] ...\n" + result_str[-2000:]

            print(f"[Parent] observation ({name}): {result_str[:300]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_str}
            )

        _report_tokens(messages, max_context_tokens)

    _report_tokens(messages, max_context_tokens)
    return (
        f"Reached parent step limit ({max_steps}). Last assistant message: "
        f"{messages[-1].get('content', '')}"
    ), messages


def run(
    task: str,
    workspace_root: str,
    max_steps: int = 25,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    """Run the Agent 7 parent orchestrator on a task and workspace."""
    messages = [
        {"role": "system", "content": PARENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_parent_initial_prompt(task, workspace_root)},
    ]

    summary, messages = run_parent_loop(
        messages,
        workspace_root,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
    )
    used = count_message_tokens(messages, model_name=MODEL)
    return f"{summary}\n\nTokens used: {format_tokens(used, max_context_tokens)}"


if __name__ == "__main__":
    import sys

    task = sys.argv[1] if len(sys.argv) > 1 else "Fix the code so that all tests pass."
    workspace_root = sys.argv[2] if len(sys.argv) > 2 else "workspace"
    print(f"Task: {task}\nWorkspace: {workspace_root}\n")
    print(run(task, workspace_root))
