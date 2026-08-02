"""Small terminal UI for selecting a session to resume.

Uses raw-terminal input so the user can navigate with the arrow keys. Falls back
to a plain numbered prompt if stdin is not a TTY.
"""

import os
import sys


def _terminal_width() -> int:
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 80


def _render_option(
    session: tuple[int, str, str], selected: bool, width: int
) -> None:
    """Print a single option row, overwriting the current line."""
    session_id, name, summary = session
    marker = ">" if selected else " "
    prefix = f"{marker} [{session_id}] {name}: "
    available = max(0, width - len(prefix) - 1)
    if len(summary) > available:
        summary = summary[: max(0, available - 3)] + "..."
    sys.stdout.write(f"\r\x1b[K{prefix}{summary}\n")
    sys.stdout.flush()


def _render_options(
    sessions: list[tuple[int, str, str]], selected: int, width: int
) -> None:
    """Print the selectable session list (prompt is handled separately)."""
    for i, session in enumerate(sessions):
        _render_option(session, i == selected, width)


def _tty_select(sessions: list[tuple[int, str, str]]) -> int | None:
    """Raw-terminal arrow-key selection.

    The prompt is printed once; only the option rows are redrawn when the
    selection moves.
    """
    import termios
    import tty

    selected = 0
    option_count = len(sessions)
    width = _terminal_width()

    sys.stdout.write(
        "Select a session to resume (↑/↓, Enter to select, q to cancel):\n"
    )
    _render_options(sessions, selected, width)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            key = sys.stdin.read(1)

            # Enter
            if key in ("\n", "\r"):
                return sessions[selected][0]

            # Ctrl-C or 'q' cancels
            if key == "\x03" or key == "q":
                return None

            # Arrow keys start with ESC
            if key == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    selected = (selected - 1) % len(sessions)
                elif seq == "[B":
                    selected = (selected + 1) % len(sessions)
                else:
                    continue
                # Move cursor to the first option row and overwrite each line.
                sys.stdout.write(f"\x1b[{option_count}A")
                _render_options(sessions, selected, width)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _fallback_select(sessions: list[tuple[int, str, str]]) -> int | None:
    """Plain numbered-list selection for non-TTY stdin."""
    print("Available sessions:")
    for i, (session_id, name, summary) in enumerate(sessions, 1):
        print(f"  {i}. [{session_id}] {name}: {summary}")
    while True:
        try:
            choice = input("Enter number (or q to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx][0]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def select_session_to_resume(
    sessions: list[tuple[int, str, str]],
) -> int | None:
    """Return the selected session ID, or None if the user cancels."""
    if not sessions:
        print("No saved sessions to resume.")
        return None

    if sys.stdin.isatty():
        return _tty_select(sessions)

    return _fallback_select(sessions)
