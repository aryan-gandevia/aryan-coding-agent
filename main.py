"""Entry point for the autonomous coding agent."""

import sys

from harness.orchestrator import run


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else "Fix the code so that all tests pass."
    workspace_root = sys.argv[2] if len(sys.argv) > 2 else "workspace"
    print(f"Task: {task}\nWorkspace: {workspace_root}\n")
    print(run(task, workspace_root))


if __name__ == "__main__":
    main()
