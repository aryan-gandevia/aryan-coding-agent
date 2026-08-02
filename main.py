"""Entry point for the autonomous coding agent.

Provides a simple REPL. Each turn is persisted in a session so context carries
over across turns. A new session is minted on the first turn; use /resume to
switch to an existing one. Type /exit to quit.
"""

import sys
from pathlib import Path

from harness.session import SessionManager
from harness.session_ui import select_session_to_resume


def _session_label(session) -> str:
    if session.title:
        return f"{session.title} (id {session.session_id})"
    return f"session {session.session_id}"


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
    print("Type your task or question, /resume to switch sessions, or /exit to quit.\n")

    repo_root = Path(__file__).resolve().parent
    manager = SessionManager(repo_root, workspace_root)
    session = None

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
            if user_input == "/resume":
                sessions = manager.list_sessions()
                # Don't offer the already-active session when resuming.
                if session is not None:
                    sessions = [s for s in sessions if s[0] != session.session_id]

                if not sessions:
                    print("No other sessions to resume.")
                    print()
                    continue

                selected_id = select_session_to_resume(sessions)
                if selected_id is None:
                    print("Resume cancelled.")
                else:
                    session = manager.get_session(selected_id)
                    session.load()
                    print(f"Resumed {_session_label(session)}.")
                    print(session.resume_summary())
                print()
                continue

            if session is None:
                session = manager.create_session()
                print(f"Created {_session_label(session)}.")

            result = session.run_turn(user_input)
            print(result)
        except Exception as exc:
            print(f"Error: {exc}")
        print()


def main() -> None:
    workspace_root = sys.argv[1] if len(sys.argv) > 1 else ".workspace"
    run_repl(workspace_root)


if __name__ == "__main__":
    main()
