"""Entry point for the autonomous coding agent.

Provides a simple REPL. The agent runs each task independently against the
same workspace. Type /exit to quit.
"""

import sys

from harness.orchestrator import run


def run_once(task: str, workspace_root: str) -> None:
    print(f"Task: {task}\nWorkspace: {workspace_root}\n")
    print(run(task, workspace_root))


def run_repl(workspace_root: str) -> None:
    print(r"""
    _                              ____ _     ___ 
   / \   _ __ _   _  __ _ _ __    / ___| |   |_ _|
  / _ \ | '__| | | |/ _` | '_ \  | |   | |    | | 
 / ___ \| |  | |_| | (_| | | | | | |___| |___ | | 
/_/   \_\_|   \__, |\__,_|_| |_|  \____|_____|___|
              |___/                               
""")
    print("Aryan's Coding Agent")
    print(f"Workspace: {workspace_root}")
    print("Type your task or question, or /exit to quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Exiting.")
            break

        print()
        try:
            result = run(user_input, workspace_root)
            print(result)
        except Exception as exc:
            print(f"Error: {exc}")
        print()


def main() -> None:
    workspace_root = sys.argv[1] if len(sys.argv) > 1 else "workspace"
    run_repl(workspace_root)


if __name__ == "__main__":
    main()
